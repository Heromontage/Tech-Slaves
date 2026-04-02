"""
gnn_model.py — SentinelFlow Graph Neural Network
=================================================
Implements a multi-layer Graph Convolutional Network (GCN) for **node-level
delay classification** on the supply chain directed multigraph stored in Neo4j.

Problem statement
-----------------
Given the current state of the supply chain graph (port congestion, factory
output, sentiment risk), predict for every node whether it will experience a
delay in the next N hours:

    Node types  : Factory | Port | Warehouse | Retailer
    Edge type   : TRANSIT_ROUTE (directed, weighted by latency and cost)
    Target      : Binary per-node label — 0 = on-time, 1 = delayed

Why GCN?
--------
A standard feed-forward network sees only a node's own features. A GCN
propagates information across the graph by aggregating neighbour features at
each layer. This is exactly what supply chain risk analysis requires: a port's
congestion propagates delay risk to every downstream warehouse and retailer
connected by shipping lanes.

Architecture overview
---------------------

    Input node features (8-dim per node):
      [0]  container_volume_norm   — normalised TEU count              (Port)
      [1]  dwell_time_norm         — normalised average dwell hours     (Port)
      [2]  capacity_pct_norm       — port utilisation 0-200%           (Port)
      [3]  utilisation_pct_norm    — factory scheduled vs actual       (Factory)
      [4]  sentiment_risk_score    — NLP risk in [0, 1]                (all)
      [5]  node_type_port          — one-hot bit for Port              (all)
      [6]  node_type_factory       — one-hot bit for Factory           (all)
      [7]  node_type_warehouse     — one-hot bit for Warehouse         (all)
      (Retailer is the reference class → all three bits = 0)

    Input edge features (3-dim per edge, used in edge-enhanced variant):
      [0]  latency_norm      — current_latency / base_latency ratio
      [1]  cost_norm         — base_cost normalised by lane max
      [2]  mode_sea          — binary flag for Sea transit mode

    Layers:
      GCNConv(8  → 64)   + BatchNorm + ReLU + Dropout(0.3)
      GCNConv(64 → 64)   + BatchNorm + ReLU + Dropout(0.3)
      GCNConv(64 → 32)   + BatchNorm + ReLU
      Linear(32 → 2)     → log-softmax  →  {on-time, delayed}

    Optional EdgeConv variant (SentinelFlowGATWithEdge):
      Uses GATv2Conv with edge_dim=3 for richer aggregation.
      Preferred when edge latency data is reliably populated.

File layout
-----------
    SupplyChainGraphData     — PyG Data wrapper with feature engineering helpers
    SentinelFlowGCN          — Production GCN model (GCNConv backbone)
    SentinelFlowGATWithEdge  — Optional GAT variant with explicit edge features
    NodeDelayDataset         — In-memory PyG Dataset for offline training
    build_graph_from_snapshot— Factory: Neo4j snapshot → PyG Data object
    train_one_epoch          — Single training epoch
    evaluate                 — Validation / test evaluation
    run_training_loop        — Complete training pipeline (CLI entry-point)
    run_inference            — Single-graph inference for FastAPI integration

Requirements
------------
    torch>=2.3.0
    torch_geometric>=2.5.0       # pip install torch_geometric
    scikit-learn>=1.4.0          # for train/val/test masks

Install torch_geometric:
    pip install torch_geometric
    # GPU: also install torch_scatter, torch_sparse — see pyg.org/install

Environment variables
---------------------
    GNN_HIDDEN_DIM        Hidden layer width             (default: 64)
    GNN_NUM_LAYERS        Number of GCN layers           (default: 3)
    GNN_DROPOUT           Dropout rate                   (default: 0.3)
    GNN_LR                Adam learning rate             (default: 1e-3)
    GNN_WEIGHT_DECAY      L2 regularisation              (default: 5e-4)
    GNN_EPOCHS            Training epochs                (default: 200)
    GNN_PATIENCE          Early-stopping patience        (default: 20)
    GNN_CHECKPOINT_PATH   Path to save best model        (default: gnn_checkpoint.pt)
    GNN_DELAY_THRESHOLD_H Hours above baseline to label delayed (default: 36)
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── PyTorch Geometric imports ─────────────────────────────────────────────────
try:
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.nn import (
        BatchNorm,
        GATv2Conv,
        GCNConv,
        global_mean_pool,
    )
    from torch_geometric.utils import add_self_loops, to_undirected
    _PYG_AVAILABLE = True
except ImportError as _pyg_err:
    _PYG_AVAILABLE = False
    _PYG_IMPORT_ERROR = _pyg_err

logger = logging.getLogger("sentinelflow.gnn")


def _require_pyg() -> None:
    if not _PYG_AVAILABLE:
        raise ImportError(
            "torch_geometric is required for GNN features. "
            "Install it with: pip install torch_geometric\n"
            f"Original error: {_PYG_IMPORT_ERROR}"
        ) from _PYG_IMPORT_ERROR  # type: ignore[name-defined]


# ── Configuration ─────────────────────────────────────────────────────────────

HIDDEN_DIM        = int(os.getenv("GNN_HIDDEN_DIM",        "64"))
NUM_LAYERS        = int(os.getenv("GNN_NUM_LAYERS",         "3"))
DROPOUT           = float(os.getenv("GNN_DROPOUT",          "0.3"))
LR                = float(os.getenv("GNN_LR",               "1e-3"))
WEIGHT_DECAY      = float(os.getenv("GNN_WEIGHT_DECAY",     "5e-4"))
EPOCHS            = int(os.getenv("GNN_EPOCHS",             "200"))
PATIENCE          = int(os.getenv("GNN_PATIENCE",           "20"))
CHECKPOINT_PATH   = Path(os.getenv("GNN_CHECKPOINT_PATH",  "gnn_checkpoint.pt"))
DELAY_THRESHOLD_H = float(os.getenv("GNN_DELAY_THRESHOLD_H", "36.0"))

# Feature dimensions — must match build_graph_from_snapshot()
NODE_FEATURE_DIM = 8   # see docstring header
EDGE_FEATURE_DIM = 3   # latency_norm, cost_norm, mode_sea
NUM_CLASSES      = 2   # 0 = on-time, 1 = delayed

# ── Node / Edge metadata ──────────────────────────────────────────────────────

# Matches graph_ops._SAMPLE_NODES label taxonomy
NODE_LABELS = ("Factory", "Port", "Warehouse", "Retailer")
# Index in the one-hot slice [5, 6, 7] — Retailer is reference class
NODE_TYPE_INDEX = {"Port": 5, "Factory": 6, "Warehouse": 7}

TRANSIT_MODES = ("Sea", "Air", "Rail", "Road")
MODE_INDEX    = {m: i for i, m in enumerate(TRANSIT_MODES)}


# ═════════════════════════════════════════════════════════════════════════════
# Data representation
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeRecord:
    """
    Flattened representation of a supply chain graph node as returned by a
    Neo4j query or constructed from ORM data during training-set assembly.
    """
    node_id:          str                    # e.g. "PORT-CN-SHA"
    node_type:        str                    # Factory | Port | Warehouse | Retailer
    container_volume: float  = 0.0          # TEU count (Port only)
    dwell_time:       float  = 0.0          # hours (Port only)
    capacity_pct:     float  = 0.0          # 0-200 % (Port only)
    utilisation_pct:  float  = 100.0        # % (Factory only)
    sentiment_risk:   float  = 0.5          # [0, 1] from ml_pipeline
    is_delayed:       int    = 0            # ground-truth label (0 / 1)


@dataclass
class EdgeRecord:
    """
    Flattened representation of a TRANSIT_ROUTE relationship.
    """
    from_id:         str
    to_id:           str
    transit_mode:    str   = "Sea"
    base_cost:       float = 1000.0
    current_latency: float = 504.0


# ── Feature normalisation constants (fitted on sample network) ────────────────
# In production these would be computed from the full training corpus and
# saved alongside the model checkpoint.

_PORT_MAX_TEU       = 50_000.0   # design capacity of Port of Shanghai
_PORT_MAX_DWELL     = 200.0      # hours — extreme congestion ceiling
_FACTORY_UTIL_MAX   = 200.0      # allow 200% (overtime) as ceiling
_EDGE_MAX_LATENCY   = 900.0      # hours — longest realistic sea lane
_EDGE_MAX_COST      = 2_500.0    # USD per TEU — long-haul air benchmark


def _normalise(value: float, max_val: float, clip: bool = True) -> float:
    """Divide by max_val and optionally clip to [0, 1]."""
    n = value / max_val if max_val > 0 else 0.0
    return max(0.0, min(1.0, n)) if clip else n


def node_to_feature_vector(node: NodeRecord) -> list[float]:
    """
    Convert a NodeRecord to an 8-dimensional float feature vector.

    Dimension layout
    ----------------
    0  container_volume_norm   (Port only, 0 for others)
    1  dwell_time_norm         (Port only, 0 for others)
    2  capacity_pct_norm       (Port only, 0 for others)
    3  utilisation_pct_norm    (Factory only, 0 for others)
    4  sentiment_risk_score    (all node types)
    5  is_port                 (1 if Port, else 0)
    6  is_factory              (1 if Factory, else 0)
    7  is_warehouse            (1 if Warehouse, else 0)
    """
    is_port      = float(node.node_type == "Port")
    is_factory   = float(node.node_type == "Factory")
    is_warehouse = float(node.node_type == "Warehouse")

    return [
        _normalise(node.container_volume, _PORT_MAX_TEU)    * is_port,
        _normalise(node.dwell_time,       _PORT_MAX_DWELL)  * is_port,
        _normalise(node.capacity_pct,     200.0)            * is_port,
        _normalise(node.utilisation_pct,  _FACTORY_UTIL_MAX) * is_factory,
        max(0.0, min(1.0, node.sentiment_risk)),
        is_port,
        is_factory,
        is_warehouse,
    ]


def edge_to_feature_vector(edge: EdgeRecord) -> list[float]:
    """
    Convert an EdgeRecord to a 3-dimensional float feature vector.

    Dimension layout
    ----------------
    0  latency_ratio   current_latency / base_latency (congestion factor)
    1  cost_norm       base_cost / _EDGE_MAX_COST
    2  mode_sea        1 if Sea lane, else 0
    """
    base_latency = max(edge.current_latency * 0.8, 1.0)  # fallback if base missing
    return [
        min(edge.current_latency / base_latency, 5.0) / 5.0,   # cap at 5× slowdown
        _normalise(edge.base_cost, _EDGE_MAX_COST),
        float(edge.transit_mode == "Sea"),
    ]


def build_pyg_data(
    nodes: list[NodeRecord],
    edges: list[EdgeRecord],
    add_self_loops_flag: bool = True,
) -> "Data":
    """
    Convert raw NodeRecord / EdgeRecord lists into a PyTorch Geometric
    ``Data`` object ready for model input.

    The function handles:
    - Node feature matrix construction (N × NODE_FEATURE_DIM)
    - Edge index tensor (2 × E)  — directed, matching the TRANSIT_ROUTE direction
    - Edge attribute matrix (E × EDGE_FEATURE_DIM)
    - Node label tensor (N,)  — 0 = on-time, 1 = delayed
    - Optional self-loops (identity connections) for GCNConv stability

    Parameters
    ----------
    nodes               : Ordered list of NodeRecord objects.
    edges               : List of EdgeRecord objects referencing node_ids.
    add_self_loops_flag : If True, add self-loops so every node also attends
                          to its own features. Recommended for GCNConv.

    Returns
    -------
    torch_geometric.data.Data
        Populated graph data object. Access via:
          data.x          — node features  [N, 8]
          data.edge_index — COO edge index [2, E]
          data.edge_attr  — edge features  [E, 3]
          data.y          — node labels    [N]
    """
    _require_pyg()

    # ── Build node-id → integer index map ─────────────────────────────────────
    id_to_idx: dict[str, int] = {n.node_id: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # ── Node feature matrix ───────────────────────────────────────────────────
    x = torch.tensor(
        [node_to_feature_vector(n) for n in nodes],
        dtype=torch.float,
    )                                                   # shape: [N, NODE_FEATURE_DIM]

    # ── Label tensor ──────────────────────────────────────────────────────────
    y = torch.tensor([n.is_delayed for n in nodes], dtype=torch.long)  # [N]

    # ── Edge index + edge attributes ──────────────────────────────────────────
    src_list:      list[int]         = []
    dst_list:      list[int]         = []
    edge_features: list[list[float]] = []

    for edge in edges:
        src_idx = id_to_idx.get(edge.from_id)
        dst_idx = id_to_idx.get(edge.to_id)
        if src_idx is None or dst_idx is None:
            logger.debug(
                "Skipping edge %s → %s: one or both nodes not in node list.",
                edge.from_id, edge.to_id,
            )
            continue
        src_list.append(src_idx)
        dst_list.append(dst_idx)
        edge_features.append(edge_to_feature_vector(edge))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # [2, E]
        edge_attr  = torch.tensor(edge_features, dtype=torch.float)         # [E, 3]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float)

    # ── Self-loops ────────────────────────────────────────────────────────────
    if add_self_loops_flag and edge_index.size(1) > 0:
        edge_index, edge_attr = add_self_loops(
            edge_index,
            edge_attr=edge_attr,
            fill_value=torch.tensor([1.0, 0.0, 0.0]),  # latency=1 (no delay), cost=0, not sea
            num_nodes=N,
        )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = N

    logger.debug(
        "Built PyG Data: %d nodes  %d edges  class_balance=%s",
        N,
        edge_index.size(1),
        {0: int((y == 0).sum()), 1: int((y == 1).sum())},
    )
    return data


# ═════════════════════════════════════════════════════════════════════════════
# Model: SentinelFlowGCN  (primary)
# ═════════════════════════════════════════════════════════════════════════════

class SentinelFlowGCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network for supply chain node delay
    classification.

    Architecture
    ------------
    Each GCNConv layer computes:

        H^(l+1) = σ( D̃^(-½) Ã D̃^(-½) H^(l) W^(l) )

    where Ã = A + I (adjacency with self-loops), D̃ is the degree matrix,
    H^(l) is the node feature matrix at layer l, and W^(l) is a learnable
    weight matrix.

    Stacking three such layers allows every node to aggregate information from
    its 3-hop neighbourhood — sufficient to capture how congestion at a port
    propagates through intermediate warehouses to a retailer.

    Layer stack
    -----------
    GCNConv(in  → hidden)  → BatchNorm → ReLU → Dropout
    GCNConv(hidden → hidden) → BatchNorm → ReLU → Dropout
    ...repeated num_layers - 1 times...
    GCNConv(hidden → out)  → BatchNorm → ReLU
    Linear(out → num_classes) → log_softmax

    Parameters
    ----------
    in_channels   : Node feature dimensionality (default: 8).
    hidden_dim    : Width of all hidden GCN layers (default: 64).
    num_layers    : Total number of GCNConv layers (default: 3).
    num_classes   : Number of output classes (default: 2).
    dropout       : Dropout probability applied between layers (default: 0.3).
    improved      : Use the ``improved`` GCN variant (2Ã instead of Ã).
                    Slightly stronger self-loop signal — helps sparse graphs.

    Inputs (forward pass)
    ---------------------
    x          : Node feature matrix       [N, in_channels]
    edge_index : Edge connectivity in COO  [2, E]

    Output
    ------
    log-softmax scores over classes        [N, num_classes]
    """

    def __init__(
        self,
        in_channels:  int   = NODE_FEATURE_DIM,
        hidden_dim:   int   = HIDDEN_DIM,
        num_layers:   int   = NUM_LAYERS,
        num_classes:  int   = NUM_CLASSES,
        dropout:      float = DROPOUT,
        improved:     bool  = False,
    ) -> None:
        super().__init__()
        _require_pyg()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        self.dropout   = dropout
        self.num_layers = num_layers

        # ── GCN layer stack ───────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in_channels → hidden_dim
        self.convs.append(GCNConv(in_channels, hidden_dim, improved=improved))
        self.norms.append(BatchNorm(hidden_dim))

        # Intermediate layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, improved=improved))
            self.norms.append(BatchNorm(hidden_dim))

        # Final convolutional layer: hidden_dim → hidden_dim // 2
        out_conv_dim = max(hidden_dim // 2, num_classes)
        self.convs.append(GCNConv(hidden_dim, out_conv_dim, improved=improved))
        self.norms.append(BatchNorm(out_conv_dim))

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Linear(out_conv_dim, num_classes)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

        logger.info(
            "SentinelFlowGCN initialised: in=%d  hidden=%d  layers=%d  out=%d  classes=%d",
            in_channels, hidden_dim, num_layers, out_conv_dim, num_classes,
        )

    def _init_weights(self) -> None:
        """Kaiming uniform initialisation for the classification head."""
        nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))
        if self.classifier.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.classifier.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.classifier.bias, -bound, bound)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass — node-level classification.

        Parameters
        ----------
        x          : Node features [N, in_channels]
        edge_index : Edge index in COO format [2, E]

        Returns
        -------
        Tensor [N, num_classes]
            Log-softmax class scores. Apply ``torch.argmax(dim=-1)`` for
            hard predictions or ``torch.exp()`` for class probabilities.
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)        # neighbourhood aggregation
            x = norm(x)                    # stabilise activations
            x = F.relu(x)                  # non-linearity

            # Apply dropout between layers (not after the last conv)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Classification head ───────────────────────────────────────────────
        x = self.classifier(x)             # [N, num_classes]
        return F.log_softmax(x, dim=-1)    # numerically stable

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Return class probabilities (not log-probabilities).

        Useful for the FastAPI route handler to return a ``delay_probability``
        float alongside the hard predicted label.

        Returns
        -------
        Tensor [N, num_classes]  — probabilities summing to 1.0 per node.
        """
        self.eval()
        log_probs = self.forward(x, edge_index)
        return log_probs.exp()


# ═════════════════════════════════════════════════════════════════════════════
# Model: SentinelFlowGATWithEdge  (optional, edge-aware variant)
# ═════════════════════════════════════════════════════════════════════════════

class SentinelFlowGATWithEdge(nn.Module):
    """
    Graph Attention Network (GATv2) variant that incorporates explicit edge
    features (transit latency ratio, base cost, transport mode).

    Prefer this model over ``SentinelFlowGCN`` when:
    - Your training data has well-populated, reliable latency readings.
    - You want the model to learn *which edges matter more* (attention).

    GATv2Conv computes per-edge attention weights using:

        α_{ij} = softmax_j( a^T · LeakyReLU(W·[h_i ‖ h_j ‖ e_{ij}]) )

    where e_{ij} is the edge feature vector, allowing the model to amplify
    or suppress information from specific lanes (e.g., highly congested sea
    routes get higher attention weight).

    Parameters
    ----------
    in_channels   : Node feature dim (default: 8).
    edge_dim      : Edge feature dim (default: 3).
    hidden_dim    : Hidden width per attention head (default: 32).
    num_heads     : Multi-head attention count (default: 4).
    num_layers    : Number of GATv2 layers (default: 3).
    num_classes   : Output classes (default: 2).
    dropout       : Dropout rate (default: 0.3).

    Note: total hidden width = hidden_dim × num_heads.
    """

    def __init__(
        self,
        in_channels:  int   = NODE_FEATURE_DIM,
        edge_dim:     int   = EDGE_FEATURE_DIM,
        hidden_dim:   int   = HIDDEN_DIM // 2,   # 32 per head × 4 heads = 128 total
        num_heads:    int   = 4,
        num_layers:   int   = NUM_LAYERS,
        num_classes:  int   = NUM_CLASSES,
        dropout:      float = DROPOUT,
    ) -> None:
        super().__init__()
        _require_pyg()

        self.dropout    = dropout
        self.num_layers = num_layers
        self.num_heads  = num_heads

        total_dim = hidden_dim * num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in_channels → total_dim
        self.convs.append(
            GATv2Conv(
                in_channels, hidden_dim,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=True,
            )
        )
        self.norms.append(BatchNorm(total_dim))

        # Intermediate layers: total_dim → total_dim
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    total_dim, hidden_dim,
                    heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(BatchNorm(total_dim))

        # Final GAT layer: total_dim → hidden_dim (single head, concat=False)
        self.convs.append(
            GATv2Conv(
                total_dim, hidden_dim,
                heads=1,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=False,
            )
        )
        self.norms.append(BatchNorm(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)

        logger.info(
            "SentinelFlowGATWithEdge initialised: in=%d  edge_dim=%d  "
            "hidden=%d  heads=%d  layers=%d  classes=%d",
            in_channels, edge_dim, hidden_dim, num_heads, num_layers, num_classes,
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with optional edge attributes.

        Parameters
        ----------
        x          : [N, in_channels]
        edge_index : [2, E]
        edge_attr  : [E, edge_dim]  — pass None to ignore edge features.

        Returns
        -------
        Tensor [N, num_classes] — log-softmax scores.
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

    @torch.no_grad()
    def predict_proba(
        self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        self.eval()
        return self.forward(x, edge_index, edge_attr=edge_attr).exp()


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic training data generator
# ═════════════════════════════════════════════════════════════════════════════

def _make_sample_node(node_id: str, node_type: str, rng: random.Random) -> NodeRecord:
    """Generate a realistic synthetic NodeRecord for a given type."""
    congested = rng.random() < 0.25          # 25% base congestion probability

    if node_type == "Port":
        base_teu   = rng.randint(5_000, 40_000)
        congestion = rng.uniform(1.3, 2.0) if congested else rng.uniform(0.5, 0.95)
        volume     = int(base_teu * congestion)
        dwell      = rng.uniform(36, 120) if congested else rng.uniform(8, 30)
        capacity   = (volume / 50_000) * 100
        return NodeRecord(
            node_id=node_id, node_type=node_type,
            container_volume=volume, dwell_time=dwell, capacity_pct=capacity,
            utilisation_pct=100.0,
            sentiment_risk=rng.uniform(0.6, 0.95) if congested else rng.uniform(0.1, 0.45),
            is_delayed=int(congested),
        )

    elif node_type == "Factory":
        under_capacity = rng.random() < 0.20
        util = rng.uniform(40, 75) if under_capacity else rng.uniform(85, 115)
        return NodeRecord(
            node_id=node_id, node_type=node_type,
            utilisation_pct=util,
            sentiment_risk=rng.uniform(0.5, 0.80) if under_capacity else rng.uniform(0.1, 0.40),
            is_delayed=int(under_capacity),
        )

    elif node_type == "Warehouse":
        # Warehouses inherit delay from upstream ports
        delayed = congested
        return NodeRecord(
            node_id=node_id, node_type=node_type,
            sentiment_risk=rng.uniform(0.4, 0.75) if delayed else rng.uniform(0.1, 0.35),
            is_delayed=int(delayed),
        )

    else:  # Retailer
        return NodeRecord(
            node_id=node_id, node_type=node_type,
            sentiment_risk=rng.uniform(0.05, 0.30),
            is_delayed=0,
        )


def generate_synthetic_graph(
    num_ports:      int   = 7,
    num_factories:  int   = 4,
    num_warehouses: int   = 5,
    num_retailers:  int   = 4,
    avg_edges:      int   = 3,
    seed:           int   = 42,
) -> tuple[list[NodeRecord], list[EdgeRecord]]:
    """
    Generate a fully synthetic supply chain graph for prototyping and
    unit tests.

    Topology mirrors the SentinelFlow sample network from graph_ops.py:
    Factories → Ports → Warehouses → Retailers (directed DAG structure).

    Parameters
    ----------
    num_ports / num_factories / num_warehouses / num_retailers:
        Number of nodes of each type.
    avg_edges:
        Average out-degree per non-retailer node.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (nodes, edges) — ready to pass into ``build_pyg_data()``.
    """
    rng = random.Random(seed)

    nodes: list[NodeRecord] = []
    node_ids_by_type: dict[str, list[str]] = {t: [] for t in NODE_LABELS}

    # ── Create nodes ──────────────────────────────────────────────────────────
    for i in range(num_factories):
        nid = f"FACTORY-SYN-{i:03d}"
        nodes.append(_make_sample_node(nid, "Factory", rng))
        node_ids_by_type["Factory"].append(nid)

    for i in range(num_ports):
        nid = f"PORT-SYN-{i:03d}"
        nodes.append(_make_sample_node(nid, "Port", rng))
        node_ids_by_type["Port"].append(nid)

    for i in range(num_warehouses):
        nid = f"WAREHOUSE-SYN-{i:03d}"
        nodes.append(_make_sample_node(nid, "Warehouse", rng))
        node_ids_by_type["Warehouse"].append(nid)

    for i in range(num_retailers):
        nid = f"RETAILER-SYN-{i:03d}"
        nodes.append(_make_sample_node(nid, "Retailer", rng))
        node_ids_by_type["Retailer"].append(nid)

    # ── Create edges following supply chain topology ───────────────────────────
    # Factory → Port (Road / Rail)
    # Port    → Warehouse (Sea / Rail)
    # Warehouse → Retailer (Road / Air)

    edges: list[EdgeRecord] = []
    transit_mode_map = {
        ("Factory", "Port"):      "Road",
        ("Port",    "Warehouse"): "Sea",
        ("Warehouse","Retailer"): "Road",
    }

    def _add_edges(from_type: str, to_type: str) -> None:
        froms = node_ids_by_type[from_type]
        tos   = node_ids_by_type[to_type]
        if not froms or not tos:
            return
        mode = transit_mode_map.get((from_type, to_type), "Sea")

        for fid in froms:
            n_out = max(1, int(rng.gauss(avg_edges, 1)))
            targets = rng.choices(tos, k=min(n_out, len(tos)))
            for tid in set(targets):
                base_cost    = rng.uniform(200, 2_500)
                base_latency = rng.uniform(4, 800)
                # Inject congestion: delayed source node → high latency
                src_node = next(n for n in nodes if n.node_id == fid)
                congestion_factor = rng.uniform(1.3, 2.0) if src_node.is_delayed else 1.0
                edges.append(EdgeRecord(
                    from_id=fid, to_id=tid,
                    transit_mode=mode,
                    base_cost=base_cost,
                    current_latency=base_latency * congestion_factor,
                ))

    _add_edges("Factory", "Port")
    _add_edges("Port",    "Warehouse")
    _add_edges("Warehouse", "Retailer")

    # Propagate delay labels downstream:
    # A Warehouse is "delayed" if ANY of its upstream Ports is delayed.
    port_id_to_record: dict[str, NodeRecord] = {
        n.node_id: n for n in nodes if n.node_type == "Port"
    }
    for node in nodes:
        if node.node_type == "Warehouse":
            upstream_port_ids = {e.from_id for e in edges if e.to_id == node.node_id}
            if any(port_id_to_record.get(pid, NodeRecord("", "")).is_delayed
                   for pid in upstream_port_ids):
                node.is_delayed = 1
                node.sentiment_risk = max(node.sentiment_risk, 0.6)

    # Retailers inherit delay from Warehouses
    wh_id_to_record: dict[str, NodeRecord] = {
        n.node_id: n for n in nodes if n.node_type == "Warehouse"
    }
    for node in nodes:
        if node.node_type == "Retailer":
            upstream_wh_ids = {e.from_id for e in edges if e.to_id == node.node_id}
            if any(wh_id_to_record.get(wid, NodeRecord("", "")).is_delayed
                   for wid in upstream_wh_ids):
                node.is_delayed = 1

    class_counts = {0: sum(1 for n in nodes if n.is_delayed == 0),
                    1: sum(1 for n in nodes if n.is_delayed == 1)}
    logger.info(
        "Synthetic graph: %d nodes  %d edges  labels=%s",
        len(nodes), len(edges), class_counts,
    )
    return nodes, edges


# ═════════════════════════════════════════════════════════════════════════════
# Dataset wrapper
# ═════════════════════════════════════════════════════════════════════════════

class NodeDelayDataset(InMemoryDataset):
    """
    In-memory PyTorch Geometric dataset.

    Wraps a list of ``Data`` objects (one per graph snapshot) for use with
    DataLoader during batched training.

    For the SentinelFlow use case each "snapshot" is one complete state of the
    supply chain graph at a point in time.  In production you would generate
    multiple snapshots (e.g., one per hour) to create a large enough corpus.

    Usage
    -----
    # Build a dataset from multiple synthetic snapshots
    all_data = []
    for seed in range(50):
        nodes, edges = generate_synthetic_graph(seed=seed)
        all_data.append(build_pyg_data(nodes, edges))

    dataset = NodeDelayDataset(all_data)
    """

    def __init__(self, data_list: list["Data"]) -> None:
        _require_pyg()
        super().__init__(root=None)
        self.data, self.slices = self.collate(data_list)

    def _download(self) -> None:
        pass

    def _process(self) -> None:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Training utilities
# ═════════════════════════════════════════════════════════════════════════════

def _make_masks(
    num_nodes: int,
    train_ratio: float = 0.60,
    val_ratio:   float = 0.20,
    seed:        int   = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Randomly split node indices into train / val / test boolean masks.

    Returns
    -------
    (train_mask, val_mask, test_mask) — each a bool Tensor of shape [N].
    """
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=rng)

    n_train = int(num_nodes * train_ratio)
    n_val   = int(num_nodes * val_ratio)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train: n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    return train_mask, val_mask, test_mask


def _compute_class_weights(y: Tensor) -> Tensor:
    """
    Compute inverse-frequency class weights for the NLL loss.

    Supply chain graphs are typically imbalanced — far fewer delayed nodes
    than on-time nodes.  Weighting equalises the gradient signal.
    """
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    counts = counts.clamp(min=1)                       # avoid div-by-zero
    weights = counts.sum() / (NUM_CLASSES * counts)    # inverse frequency
    return weights


def train_one_epoch(
    model:      nn.Module,
    data:       "Data",
    optimiser:  torch.optim.Optimizer,
    criterion:  nn.Module,
    train_mask: Tensor,
    device:     torch.device,
) -> float:
    """
    Run one full gradient-update step over the training nodes.

    In a transductive node-classification setting (one big graph), all nodes
    share the same forward pass — the message-passing aggregation sees all
    neighbours — but the loss is computed only over ``train_mask`` nodes.

    Parameters
    ----------
    model      : GCN or GAT model in training mode.
    data       : PyG Data object for the graph.
    optimiser  : Configured torch Optimizer.
    criterion  : Loss function (NLLLoss recommended for log-softmax output).
    train_mask : Boolean mask [N] — True for training nodes.
    device     : Target device (cpu / cuda / mps).

    Returns
    -------
    float — training loss for this epoch.
    """
    model.train()
    optimiser.zero_grad()

    # Full graph forward pass (all nodes participate in message passing)
    if hasattr(model, "forward") and "edge_attr" in model.forward.__code__.co_varnames:
        # GAT variant — pass edge attributes
        out = model(
            data.x.to(device),
            data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
        )
    else:
        # GCN variant — no edge attributes
        out = model(data.x.to(device), data.edge_index.to(device))

    # Loss computed only over training nodes
    loss = criterion(
        out[train_mask.to(device)],
        data.y.to(device)[train_mask.to(device)],
    )
    loss.backward()
    optimiser.step()

    return float(loss.item())


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    data:     "Data",
    mask:     Tensor,
    device:   torch.device,
) -> dict[str, float]:
    """
    Evaluate model on the nodes selected by ``mask``.

    Returns
    -------
    dict with keys:
        accuracy    — % correctly classified
        precision   — TP / (TP + FP) for the "delayed" class
        recall      — TP / (TP + FN) for the "delayed" class
        f1          — harmonic mean of precision and recall
    """
    model.eval()

    if hasattr(model, "forward") and "edge_attr" in model.forward.__code__.co_varnames:
        out = model(
            data.x.to(device),
            data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
        )
    else:
        out = model(data.x.to(device), data.edge_index.to(device))

    pred   = out.argmax(dim=-1)                        # [N]
    labels = data.y.to(device)

    pred_m   = pred[mask.to(device)]
    labels_m = labels[mask.to(device)]

    correct = (pred_m == labels_m).sum().item()
    total   = mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0

    # Binary metrics for class 1 (delayed)
    tp = ((pred_m == 1) & (labels_m == 1)).sum().item()
    fp = ((pred_m == 1) & (labels_m == 0)).sum().item()
    fn = ((pred_m == 0) & (labels_m == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Complete training loop
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    """Summary of a completed training run."""
    best_val_f1:   float
    test_metrics:  dict[str, float]
    epochs_trained: int
    training_time_s: float
    checkpoint_path: Path
    loss_history:   list[float]       = field(default_factory=list)
    val_f1_history: list[float]       = field(default_factory=list)


def run_training_loop(
    data:            "Data",
    model:           Optional[nn.Module] = None,
    epochs:          int                 = EPOCHS,
    lr:              float               = LR,
    weight_decay:    float               = WEIGHT_DECAY,
    patience:        int                 = PATIENCE,
    checkpoint_path: Path                = CHECKPOINT_PATH,
    train_ratio:     float               = 0.60,
    val_ratio:       float               = 0.20,
    device_str:      str                 = "auto",
    seed:            int                 = 42,
    use_gat:         bool                = False,
    verbose:         bool                = True,
) -> TrainingResult:
    """
    Complete training pipeline for supply chain node delay classification.

    Steps
    -----
    1. Resolve device (CUDA > MPS > CPU).
    2. Build train / val / test masks via random split.
    3. Compute class weights to handle class imbalance.
    4. Instantiate model if not provided.
    5. Train with Adam + cosine annealing LR schedule.
    6. Early stopping on validation F1 score.
    7. Reload best checkpoint and evaluate on test set.
    8. Return TrainingResult summary.

    Parameters
    ----------
    data            : PyG Data object containing x, edge_index, edge_attr, y.
    model           : Pre-instantiated model; if None, a new one is created.
    epochs          : Maximum training epochs.
    lr              : Initial Adam learning rate.
    weight_decay    : L2 regularisation coefficient.
    patience        : Early-stopping patience (epochs without val F1 improvement).
    checkpoint_path : Where to save the best model state dict.
    train_ratio     : Fraction of nodes used for training.
    val_ratio       : Fraction of nodes used for validation.
    device_str      : "auto" | "cpu" | "cuda" | "mps".
    seed            : Random seed for mask generation.
    use_gat         : If True, use SentinelFlowGATWithEdge instead of GCN.
    verbose         : Print per-epoch progress.

    Returns
    -------
    TrainingResult
    """
    _require_pyg()

    # ── Device ────────────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info("Training device: %s", device)

    # ── Masks ─────────────────────────────────────────────────────────────────
    N = data.num_nodes
    train_mask, val_mask, test_mask = _make_masks(N, train_ratio, val_ratio, seed)
    logger.info(
        "Mask sizes — train:%d  val:%d  test:%d",
        train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item(),
    )

    # ── Class-weighted NLL loss ───────────────────────────────────────────────
    class_weights = _compute_class_weights(data.y).to(device)
    criterion = nn.NLLLoss(weight=class_weights)
    logger.info("Class weights: %s", class_weights.tolist())

    # ── Model ─────────────────────────────────────────────────────────────────
    if model is None:
        if use_gat:
            model = SentinelFlowGATWithEdge()
        else:
            model = SentinelFlowGCN()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d total  %d trainable", total_params, trainable)

    # ── Optimiser + LR scheduler ──────────────────────────────────────────────
    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1    = -1.0
    patience_count = 0
    loss_history:   list[float] = []
    val_f1_history: list[float] = []

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):

        loss = train_one_epoch(
            model, data, optimiser, criterion, train_mask, device
        )
        scheduler.step()
        loss_history.append(loss)

        val_metrics = evaluate(model, data, val_mask, device)
        val_f1 = val_metrics["f1"]
        val_f1_history.append(val_f1)

        # ── Save checkpoint on improvement ────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
            improved_str = "  ✓ saved"
        else:
            patience_count += 1
            improved_str = ""

        if verbose and (epoch % 10 == 0 or epoch == 1):
            logger.info(
                "Epoch %4d/%d  loss=%.4f  val_acc=%.4f  val_f1=%.4f  lr=%.6f%s",
                epoch, epochs,
                loss,
                val_metrics["accuracy"],
                val_f1,
                scheduler.get_last_lr()[0],
                improved_str,
            )

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_count >= patience:
            logger.info(
                "Early stopping at epoch %d — no improvement for %d epochs.",
                epoch, patience,
            )
            break

    training_time = time.perf_counter() - t0

    # ── Load best checkpoint and evaluate on test set ─────────────────────────
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, data, test_mask, device)

    logger.info("─" * 60)
    logger.info("Training complete in %.1f s", training_time)
    logger.info("Best val F1   : %.4f", best_val_f1)
    logger.info("Test metrics  : %s", test_metrics)
    logger.info("Checkpoint    : %s", checkpoint_path)
    logger.info("─" * 60)

    return TrainingResult(
        best_val_f1    = best_val_f1,
        test_metrics   = test_metrics,
        epochs_trained = len(loss_history),
        training_time_s= round(training_time, 2),
        checkpoint_path= checkpoint_path,
        loss_history   = loss_history,
        val_f1_history = val_f1_history,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Inference helper (FastAPI integration)
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(
    model:           nn.Module,
    nodes:           list[NodeRecord],
    edges:           list[EdgeRecord],
    device:          Optional[torch.device] = None,
    checkpoint_path: Optional[Path]        = None,
) -> list[dict]:
    """
    Run a single forward pass on a live graph snapshot and return per-node
    predictions.

    Designed to be called from ``main.py`` route handlers:

        from gnn_model import SentinelFlowGCN, run_inference
        from graph_ops import _SAMPLE_NODES, _SAMPLE_ROUTES  # or live Neo4j data

        model = SentinelFlowGCN()
        result = run_inference(model, nodes, edges, checkpoint_path=CHECKPOINT_PATH)

    Parameters
    ----------
    model           : Instantiated model (weights are loaded from checkpoint).
    nodes           : Current NodeRecord list from Neo4j / ORM layer.
    edges           : Current EdgeRecord list from Neo4j / ORM layer.
    device          : Inference device; auto-detects if None.
    checkpoint_path : If provided, load weights from this file before inference.

    Returns
    -------
    List of dicts, one per node:
        {
          "node_id":          str,
          "node_type":        str,
          "predicted_label":  int,   # 0 = on-time, 1 = delayed
          "delay_probability": float, # P(delayed) in [0, 1]
          "on_time_probability": float,
        }
    """
    _require_pyg()

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    if checkpoint_path and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.debug("Loaded checkpoint from %s", checkpoint_path)

    model = model.to(device)
    model.eval()

    data = build_pyg_data(nodes, edges, add_self_loops_flag=True)

    with torch.no_grad():
        if isinstance(model, SentinelFlowGATWithEdge):
            probs = model.predict_proba(
                data.x.to(device),
                data.edge_index.to(device),
                edge_attr=data.edge_attr.to(device),
            )
        else:
            probs = model.predict_proba(
                data.x.to(device),
                data.edge_index.to(device),
            )

    probs_np  = probs.cpu().numpy()
    predicted = probs_np.argmax(axis=1)

    results = []
    for i, node in enumerate(nodes):
        results.append({
            "node_id":             node.node_id,
            "node_type":           node.node_type,
            "predicted_label":     int(predicted[i]),
            "delay_probability":   round(float(probs_np[i, 1]), 4),
            "on_time_probability": round(float(probs_np[i, 0]), 4),
        })

    delayed_count = sum(1 for r in results if r["predicted_label"] == 1)
    logger.info(
        "Inference complete: %d nodes  %d predicted delayed (%.1f%%)",
        len(results), delayed_count,
        100 * delayed_count / max(len(results), 1),
    )
    return results


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry-point  (python gnn_model.py [--gat] [--epochs N] [--demo])
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    _require_pyg()

    # ── CLI argument parsing (minimal, no argparse dependency) ────────────────
    args          = sys.argv[1:]
    use_gat       = "--gat"    in args
    demo_mode     = "--demo"   in args
    epochs_arg    = next((args[i+1] for i, a in enumerate(args) if a == "--epochs"), None)
    cli_epochs    = int(epochs_arg) if epochs_arg else EPOCHS

    print("\n" + "═" * 72)
    print("  SentinelFlow — GNN Node Delay Classifier")
    print("═" * 72)
    print(f"  Model      : {'SentinelFlowGATWithEdge' if use_gat else 'SentinelFlowGCN'}")
    print(f"  Epochs     : {cli_epochs}")
    print(f"  Hidden dim : {HIDDEN_DIM}")
    print(f"  Layers     : {NUM_LAYERS}")
    print(f"  Dropout    : {DROPOUT}")
    print(f"  Device     : {'auto-detect'}")
    print("═" * 72 + "\n")

    # ── Build synthetic training graph ────────────────────────────────────────
    print("  Generating synthetic supply chain graph…")
    syn_nodes, syn_edges = generate_synthetic_graph(
        num_ports=12, num_factories=6, num_warehouses=8, num_retailers=5,
        avg_edges=3, seed=2024,
    )
    graph_data = build_pyg_data(syn_nodes, syn_edges, add_self_loops_flag=True)

    print(f"  Graph summary:")
    print(f"    Nodes      : {graph_data.num_nodes}")
    print(f"    Edges      : {graph_data.edge_index.size(1)}")
    print(f"    Node feats : {graph_data.x.shape}")
    print(f"    Edge feats : {graph_data.edge_attr.shape}")
    label_counts = torch.bincount(graph_data.y, minlength=2)
    print(f"    Labels     : on-time={label_counts[0].item()}  delayed={label_counts[1].item()}")
    print()

    if demo_mode:
        print("  [demo] Running a 5-epoch smoke test…")
        cli_epochs = 5

    # ── Train ─────────────────────────────────────────────────────────────────
    result = run_training_loop(
        data=graph_data,
        epochs=cli_epochs,
        use_gat=use_gat,
        verbose=True,
        checkpoint_path=CHECKPOINT_PATH,
    )

    print("\n" + "─" * 72)
    print("  Training summary:")
    print(f"    Epochs trained : {result.epochs_trained}")
    print(f"    Best val F1    : {result.best_val_f1:.4f}")
    print(f"    Test accuracy  : {result.test_metrics['accuracy']:.4f}")
    print(f"    Test precision : {result.test_metrics['precision']:.4f}")
    print(f"    Test recall    : {result.test_metrics['recall']:.4f}")
    print(f"    Test F1        : {result.test_metrics['f1']:.4f}")
    print(f"    Training time  : {result.training_time_s:.1f}s")
    print(f"    Checkpoint     : {result.checkpoint_path}")
    print("─" * 72)

    # ── Inference demo ─────────────────────────────────────────────────────────
    print("\n  Running inference on 5 sample nodes…")
    model = SentinelFlowGATWithEdge() if use_gat else SentinelFlowGCN()
    inference_results = run_inference(
        model=model,
        nodes=syn_nodes[:5],
        edges=[e for e in syn_edges if e.from_id in {n.node_id for n in syn_nodes[:5]}
               or e.to_id in {n.node_id for n in syn_nodes[:5]}],
        checkpoint_path=CHECKPOINT_PATH,
    )

    print()
    print(f"  {'Node ID':<28}  {'Type':<12}  {'Pred':>6}  {'P(delay)':>10}  {'Truth':>6}")
    print("  " + "─" * 70)
    for i, res in enumerate(inference_results):
        truth = syn_nodes[i].is_delayed
        pred  = res["predicted_label"]
        icon  = "✓" if pred == truth else "✗"
        print(
            f"  {res['node_id']:<28}  {res['node_type']:<12}  "
            f"{pred:>6}  {res['delay_probability']:>10.4f}  "
            f"{truth:>5} {icon}"
        )

    print()
