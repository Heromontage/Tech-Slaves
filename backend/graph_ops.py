"""
graph_ops.py — SentinelFlow Neo4j Graph Operations
====================================================
Provides functions to initialize, populate, and query the supply chain
directed multigraph stored in Neo4j.

Node Labels
-----------
  Factory    — A production facility (source node)
  Port       — A maritime/air/rail terminal hub
  Warehouse  — A distribution or storage center
  Retailer   — An end-delivery destination (sink node)

Relationship Types
------------------
  TRANSIT_ROUTE — A directed edge between any two supply chain nodes.
    Properties:
      transit_mode     : str   — "Sea" | "Air" | "Rail" | "Road"
      base_cost        : float — USD per TEU (baseline, no disruption)
      current_latency  : float — current transit time in hours

Usage
-----
    # Initialize the schema (run once, idempotent)
    asyncio.run(initialize_supply_chain_graph())

    # Create a transit route between two nodes
    asyncio.run(create_transit_route(
        from_label="Factory", from_id="FACTORY-CN-SH-001",
        to_label="Port",      to_id="PORT-CN-SHA",
        transit_mode="Road",  base_cost=120.0, current_latency=6.0,
    ))

    # Seed a sample network for development / testing
    asyncio.run(seed_sample_network())
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal

from database import get_neo4j_session

logger = logging.getLogger("sentinelflow.graph_ops")

# ── Type aliases ──────────────────────────────────────────────────────────────

NodeLabel = Literal["Factory", "Port", "Warehouse", "Retailer"]
TransitMode = Literal["Sea", "Air", "Rail", "Road"]

# ── Node definition dataclass ─────────────────────────────────────────────────

@dataclass
class SupplyChainNode:
    """
    Represents a node in the supply chain graph.

    Attributes
    ----------
    node_id      : Stable business-key identifier (used as the `id` property in Neo4j).
    label        : Neo4j label — one of Factory | Port | Warehouse | Retailer.
    name         : Human-readable display name shown in the SentinelFlow UI.
    region       : Geographic region tag for aggregated analytics queries.
    country_code : ISO 3166-1 alpha-2 country code.
    capacity     : Maximum TEU throughput per day (optional; omit if unknown).
    extra        : Arbitrary additional properties merged into the node on creation.
    """
    node_id: str
    label: NodeLabel
    name: str
    region: str
    country_code: str
    capacity: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class TransitRoute:
    """
    Represents a directed TRANSIT_ROUTE relationship between two supply chain nodes.

    Attributes
    ----------
    from_id        : node_id of the origin node.
    from_label     : Neo4j label of the origin node (needed for MATCH).
    to_id          : node_id of the destination node.
    to_label       : Neo4j label of the destination node.
    transit_mode   : Physical transport type — Sea | Air | Rail | Road.
    base_cost      : USD per TEU on this lane under normal operating conditions.
    current_latency: Current transit time in hours (updated by the monitoring pipeline).
    distance_km    : Great-circle or route distance in kilometres (optional).
    carrier        : Operating carrier or shipping line name (optional).
    """
    from_id: str
    from_label: NodeLabel
    to_id: str
    to_label: NodeLabel
    transit_mode: TransitMode
    base_cost: float
    current_latency: float
    distance_km: float | None = None
    carrier: str | None = None


# ── Schema initialization ─────────────────────────────────────────────────────

# Cypher constraint / index statements.
# Uniqueness constraints automatically create a backing index in Neo4j 4+.
_SCHEMA_STATEMENTS: list[str] = [
    # Uniqueness constraints — one per node label
    "CREATE CONSTRAINT factory_id_unique IF NOT EXISTS FOR (n:Factory) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT port_id_unique    IF NOT EXISTS FOR (n:Port)    REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT warehouse_id_unique IF NOT EXISTS FOR (n:Warehouse) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT retailer_id_unique  IF NOT EXISTS FOR (n:Retailer)  REQUIRE n.id IS UNIQUE",

    # Composite index on TRANSIT_ROUTE for the bottleneck-detection query pattern:
    # "find all high-latency Sea routes from a given node"
    "CREATE INDEX transit_route_mode_latency IF NOT EXISTS FOR ()-[r:TRANSIT_ROUTE]-() ON (r.transit_mode, r.current_latency)",

    # Index on region for geographic filter queries used in the dashboard map
    "CREATE INDEX factory_region   IF NOT EXISTS FOR (n:Factory)   ON (n.region)",
    "CREATE INDEX port_region      IF NOT EXISTS FOR (n:Port)      ON (n.region)",
    "CREATE INDEX warehouse_region IF NOT EXISTS FOR (n:Warehouse) ON (n.region)",
    "CREATE INDEX retailer_region  IF NOT EXISTS FOR (n:Retailer)  ON (n.region)",
]


async def initialize_supply_chain_graph() -> None:
    """
    Idempotently apply all schema constraints and indexes to the Neo4j database.

    Safe to call on every application startup — the ``IF NOT EXISTS`` guard
    on each statement means re-running causes no side effects.

    Raises
    ------
    RuntimeError
        If the Neo4j session cannot be established (propagated from
        ``get_neo4j_session``).
    """
    logger.info("Initializing supply chain graph schema…")

    async with get_neo4j_session() as session:
        for stmt in _SCHEMA_STATEMENTS:
            await session.run(stmt)
            logger.debug("Applied: %s", stmt[:80])

    logger.info(
        "Schema initialization complete — %d constraint/index statements applied.",
        len(_SCHEMA_STATEMENTS),
    )


# ── Node creation ─────────────────────────────────────────────────────────────

async def create_node(node: SupplyChainNode) -> None:
    """
    Upsert a single supply chain node (MERGE on id so re-runs are safe).

    All fields from ``SupplyChainNode`` are written as Neo4j properties.
    The ``extra`` dict is merged last, so it can override any field except
    ``id`` and the label itself.

    Parameters
    ----------
    node : SupplyChainNode
        The node to create or update.
    """
    props: dict = {
        "id": node.node_id,
        "name": node.name,
        "region": node.region,
        "country_code": node.country_code,
        **node.extra,
    }
    if node.capacity is not None:
        props["capacity"] = node.capacity

    # MERGE on (label, id) — creates the node if it does not exist,
    # then applies SET to keep all mutable properties current.
    cypher = (
        f"MERGE (n:{node.label} {{id: $id}}) "
        "SET n += $props "
        "RETURN n.id AS node_id"
    )

    async with get_neo4j_session() as session:
        result = await session.run(cypher, id=node.node_id, props=props)
        record = await result.single()
        logger.debug("Upserted %s node: %s", node.label, record["node_id"])


async def create_nodes_bulk(nodes: list[SupplyChainNode]) -> None:
    """
    Upsert a list of nodes in a single transaction using UNWIND for efficiency.

    Preferred over calling ``create_node`` in a loop when seeding large graphs.

    Parameters
    ----------
    nodes : list[SupplyChainNode]
        The nodes to upsert.  All nodes must share the same label.

    Raises
    ------
    ValueError
        If ``nodes`` contains more than one distinct label.
    """
    if not nodes:
        return

    labels = {n.label for n in nodes}
    if len(labels) > 1:
        raise ValueError(
            f"create_nodes_bulk requires a homogeneous label list; got {labels}. "
            "Call separately for each label."
        )
    label = next(iter(labels))

    rows = [
        {
            "id": n.node_id,
            "name": n.name,
            "region": n.region,
            "country_code": n.country_code,
            **({"capacity": n.capacity} if n.capacity is not None else {}),
            **n.extra,
        }
        for n in nodes
    ]

    cypher = (
        f"UNWIND $rows AS row "
        f"MERGE (n:{label} {{id: row.id}}) "
        "SET n += row "
        "RETURN count(n) AS upserted"
    )

    async with get_neo4j_session() as session:
        result = await session.run(cypher, rows=rows)
        record = await result.single()
        logger.info("Bulk upserted %d %s nodes.", record["upserted"], label)


# ── Relationship creation ─────────────────────────────────────────────────────

async def create_transit_route(
    from_label: NodeLabel,
    from_id: str,
    to_label: NodeLabel,
    to_id: str,
    transit_mode: TransitMode,
    base_cost: float,
    current_latency: float,
    distance_km: float | None = None,
    carrier: str | None = None,
) -> None:
    """
    Create (or update) a directed ``TRANSIT_ROUTE`` relationship between two
    supply chain nodes.

    The relationship is keyed on ``(from_node, to_node, transit_mode)`` so that
    multiple parallel routes with different modes can exist between the same
    pair of nodes (e.g., a Sea lane and an Air lane between Port A and
    Warehouse B).

    Parameters
    ----------
    from_label      : Label of the origin node (Factory | Port | Warehouse | Retailer).
    from_id         : ``id`` property of the origin node.
    to_label        : Label of the destination node.
    to_id           : ``id`` property of the destination node.
    transit_mode    : Physical transport mode — Sea | Air | Rail | Road.
    base_cost       : Baseline USD cost per TEU on this lane.
    current_latency : Current transit time in hours (may be updated by the
                      monitoring pipeline as congestion changes).
    distance_km     : Optional route distance in kilometres.
    carrier         : Optional operating carrier name.

    Raises
    ------
    LookupError
        If either the origin or destination node does not exist in the graph.
    """
    rel_props: dict = {
        "transit_mode": transit_mode,
        "base_cost": base_cost,
        "current_latency": current_latency,
    }
    if distance_km is not None:
        rel_props["distance_km"] = distance_km
    if carrier is not None:
        rel_props["carrier"] = carrier

    # MERGE on (from, to, transit_mode) to allow multiple mode-specific edges.
    cypher = (
        f"MATCH (a:{from_label} {{id: $from_id}}) "
        f"MATCH (b:{to_label}   {{id: $to_id}}) "
        "MERGE (a)-[r:TRANSIT_ROUTE {transit_mode: $transit_mode}]->(b) "
        "SET r += $props "
        "RETURN type(r) AS rel_type, r.transit_mode AS mode"
    )

    async with get_neo4j_session() as session:
        result = await session.run(
            cypher,
            from_id=from_id,
            to_id=to_id,
            transit_mode=transit_mode,
            props=rel_props,
        )
        record = await result.single()

        if record is None:
            raise LookupError(
                f"Could not create TRANSIT_ROUTE: one or both nodes not found "
                f"({from_label}:{from_id} → {to_label}:{to_id})."
            )
        logger.debug(
            "Upserted TRANSIT_ROUTE [%s] %s:%s → %s:%s",
            record["mode"], from_label, from_id, to_label, to_id,
        )


async def create_transit_routes_bulk(routes: list[TransitRoute]) -> None:
    """
    Upsert a list of ``TRANSIT_ROUTE`` relationships in a single transaction.

    Because UNWIND with heterogeneous MATCH labels is not natively supported
    in a single Cypher statement, this function batches routes by
    ``(from_label, to_label)`` pairs and issues one UNWIND per pair.

    Parameters
    ----------
    routes : list[TransitRoute]
        The relationships to create or update.
    """
    if not routes:
        return

    # Group by (from_label, to_label) so each batch has a uniform MATCH pattern.
    from collections import defaultdict
    batches: dict[tuple[str, str], list[TransitRoute]] = defaultdict(list)
    for r in routes:
        batches[(r.from_label, r.to_label)].append(r)

    async with get_neo4j_session() as session:
        for (from_label, to_label), batch in batches.items():
            rows = [
                {
                    "from_id": r.from_id,
                    "to_id": r.to_id,
                    "transit_mode": r.transit_mode,
                    "base_cost": r.base_cost,
                    "current_latency": r.current_latency,
                    **({"distance_km": r.distance_km} if r.distance_km else {}),
                    **({"carrier": r.carrier} if r.carrier else {}),
                }
                for r in batch
            ]
            cypher = (
                "UNWIND $rows AS row "
                f"MATCH (a:{from_label} {{id: row.from_id}}) "
                f"MATCH (b:{to_label}   {{id: row.to_id}}) "
                "MERGE (a)-[r:TRANSIT_ROUTE {transit_mode: row.transit_mode}]->(b) "
                "SET r += row "
                "RETURN count(r) AS upserted"
            )
            result = await session.run(cypher, rows=rows)
            record = await result.single()
            logger.info(
                "Bulk upserted %d TRANSIT_ROUTE edges (%s → %s).",
                record["upserted"], from_label, to_label,
            )


# ── Query helpers ─────────────────────────────────────────────────────────────

async def get_route_health(route_id: str) -> dict | None:
    """
    Return a lightweight health snapshot for all ``TRANSIT_ROUTE`` edges
    that originate from a node with the given ``id``.

    Used by ``GET /v2/routes/{route_id}/health`` in main.py.

    Returns None if the node does not exist.
    """
    cypher = (
        "MATCH (a {id: $route_id})-[r:TRANSIT_ROUTE]->(b) "
        "RETURN "
        "  a.id          AS origin_id, "
        "  b.id          AS dest_id, "
        "  labels(b)[0]  AS dest_label, "
        "  r.transit_mode      AS mode, "
        "  r.base_cost         AS base_cost, "
        "  r.current_latency   AS current_latency, "
        "  r.distance_km       AS distance_km, "
        "  r.carrier           AS carrier "
        "ORDER BY r.current_latency DESC"
    )

    async with get_neo4j_session() as session:
        result = await session.run(cypher, route_id=route_id)
        records = [r.data() async for r in result]

    if not records:
        return None

    # Derive a simple risk level from the worst-case latency delta
    max_latency = max(r["current_latency"] for r in records)
    avg_base = sum(r["base_cost"] for r in records) / len(records)
    risk_level = (
        "critical" if max_latency > 120
        else "high"  if max_latency > 72
        else "medium" if max_latency > 36
        else "low"
    )

    return {
        "route_id": route_id,
        "legs": records,
        "max_latency_hours": max_latency,
        "avg_base_cost_usd": round(avg_base, 2),
        "risk_level": risk_level,
    }


async def get_active_bottlenecks(
    severity: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Return nodes that the graph model has flagged as bottlenecks.

    A node is considered a bottleneck when the sum of ``current_latency``
    on its outbound edges exceeds twice the sum of ``base_cost`` weighted
    latency — a simple heuristic that works without requiring the full
    GNN inference pipeline.

    Parameters
    ----------
    severity : str | None
        Filter by risk level: "critical" | "high" | "medium" | "low".
        Pass None to return all severities.
    limit : int
        Maximum number of bottleneck nodes to return (default 50).
    """
    severity_filter = ""
    params: dict = {"limit": limit}

    if severity:
        # Map severity label to latency threshold
        thresholds = {"critical": 120, "high": 72, "medium": 36, "low": 0}
        threshold = thresholds.get(severity.lower(), 0)
        severity_filter = "AND max_latency > $threshold "
        params["threshold"] = threshold

    cypher = (
        "MATCH (n)-[r:TRANSIT_ROUTE]->(m) "
        "WITH n, "
        "     max(r.current_latency) AS max_latency, "
        "     avg(r.base_cost)       AS avg_base_cost, "
        "     count(r)               AS outbound_routes "
        "WHERE max_latency > avg_base_cost / 10 "   # latency-cost anomaly heuristic
        + severity_filter +
        "RETURN "
        "  n.id            AS node_id, "
        "  labels(n)[0]    AS node_type, "
        "  n.name          AS name, "
        "  n.region        AS region, "
        "  max_latency, "
        "  avg_base_cost, "
        "  outbound_routes "
        "ORDER BY max_latency DESC "
        "LIMIT $limit"
    )

    async with get_neo4j_session() as session:
        result = await session.run(cypher, **params)
        records = [r.data() async for r in result]

    # Attach a risk_level label to each record for the frontend
    for rec in records:
        lat = rec["max_latency"]
        rec["risk_level"] = (
            "critical" if lat > 120
            else "high"  if lat > 72
            else "medium" if lat > 36
            else "low"
        )

    return records


# ── Sample network seed ───────────────────────────────────────────────────────

#
# Sample topology (sourced from the SentinelFlow bottleneck.html demo data):
#
#   FACTORY-CN-GZ-001 (Guangzhou)
#       ──[Road, 6h]──► PORT-CN-SHA (Shanghai)
#       ──[Road, 8h]──► PORT-CN-SZX (Shenzhen)
#
#   PORT-CN-SHA (Shanghai)  ← currently congested at 92%
#       ──[Sea,  504h]──► PORT-US-LGB  (Long Beach)
#       ──[Sea,  792h]──► PORT-EU-HAM  (Hamburg)
#       ──[Sea,  120h]──► PORT-SG-SIN  (Singapore)
#       ──[Sea,  336h]──► PORT-AU-SYD  (Sydney)
#       ──[Rail,  72h]──► PORT-KR-BSN  (Busan)  ← proposed reroute
#
#   PORT-US-LGB (Long Beach)
#       ──[Road, 48h]──► WAREHOUSE-US-LA-001 (Los Angeles DC)
#
#   PORT-EU-HAM (Hamburg)
#       ──[Rail, 24h]──► WAREHOUSE-DE-BER-004 (Berlin DC)
#
#   WAREHOUSE-US-LA-001
#       ──[Road, 12h]──► RETAILER-US-NY-001 (New York Retailer Hub)
#
#   WAREHOUSE-DE-BER-004
#       ──[Road,  8h]──► RETAILER-EU-MUC-001 (Munich Retailer Hub)
#

_SAMPLE_NODES: list[SupplyChainNode] = [
    # Factories
    SupplyChainNode(
        node_id="FACTORY-CN-GZ-001",
        label="Factory",
        name="Guangzhou Assembly Plant #1",
        region="APAC",
        country_code="CN",
        capacity=15_000,
        extra={"timezone": "Asia/Shanghai", "product_type": "Electronics"},
    ),
    SupplyChainNode(
        node_id="FACTORY-VN-HCM-001",
        label="Factory",
        name="Ho Chi Minh Manufacturing Hub",
        region="APAC",
        country_code="VN",
        capacity=8_000,
        extra={"timezone": "Asia/Ho_Chi_Minh", "product_type": "Textiles"},
    ),

    # Ports
    SupplyChainNode(
        node_id="PORT-CN-SHA",
        label="Port",
        name="Port of Shanghai",
        region="APAC",
        country_code="CN",
        capacity=50_000,
        extra={"un_locode": "CNSHA", "congestion_pct": 92.0},
    ),
    SupplyChainNode(
        node_id="PORT-CN-SZX",
        label="Port",
        name="Port of Shenzhen (Yantian)",
        region="APAC",
        country_code="CN",
        capacity=30_000,
        extra={"un_locode": "CNSZX", "congestion_pct": 54.0},
    ),
    SupplyChainNode(
        node_id="PORT-KR-BSN",
        label="Port",
        name="Port of Busan",
        region="APAC",
        country_code="KR",
        capacity=22_000,
        extra={"un_locode": "KRBSN", "congestion_pct": 28.0},
    ),
    SupplyChainNode(
        node_id="PORT-SG-SIN",
        label="Port",
        name="Port of Singapore",
        region="APAC",
        country_code="SG",
        capacity=38_000,
        extra={"un_locode": "SGSIN", "congestion_pct": 41.0},
    ),
    SupplyChainNode(
        node_id="PORT-US-LGB",
        label="Port",
        name="Port of Long Beach",
        region="AMER",
        country_code="US",
        capacity=25_000,
        extra={"un_locode": "USLGB", "congestion_pct": 61.0},
    ),
    SupplyChainNode(
        node_id="PORT-EU-HAM",
        label="Port",
        name="Port of Hamburg",
        region="EMEA",
        country_code="DE",
        capacity=18_000,
        extra={"un_locode": "DEHAM", "congestion_pct": 37.0},
    ),
    SupplyChainNode(
        node_id="PORT-AU-SYD",
        label="Port",
        name="Port of Sydney",
        region="APAC",
        country_code="AU",
        capacity=7_000,
        extra={"un_locode": "AUSYD", "congestion_pct": 22.0},
    ),

    # Warehouses
    SupplyChainNode(
        node_id="WAREHOUSE-US-LA-001",
        label="Warehouse",
        name="Los Angeles Distribution Center",
        region="AMER",
        country_code="US",
        capacity=12_000,
        extra={"operator": "SentinelFlow Logistics AMER"},
    ),
    SupplyChainNode(
        node_id="WAREHOUSE-DE-BER-004",
        label="Warehouse",
        name="Berlin Distribution Center #4",
        region="EMEA",
        country_code="DE",
        capacity=9_000,
        extra={"operator": "SentinelFlow Logistics EMEA"},
    ),
    SupplyChainNode(
        node_id="WAREHOUSE-SG-SIN-002",
        label="Warehouse",
        name="Singapore Regional Hub",
        region="APAC",
        country_code="SG",
        capacity=14_000,
        extra={"operator": "SentinelFlow Logistics APAC"},
    ),

    # Retailers
    SupplyChainNode(
        node_id="RETAILER-US-NY-001",
        label="Retailer",
        name="New York Retailer Hub",
        region="AMER",
        country_code="US",
        extra={"channel": "eCommerce"},
    ),
    SupplyChainNode(
        node_id="RETAILER-EU-MUC-001",
        label="Retailer",
        name="Munich Retailer Hub",
        region="EMEA",
        country_code="DE",
        extra={"channel": "Brick & Mortar"},
    ),
    SupplyChainNode(
        node_id="RETAILER-AU-MEL-001",
        label="Retailer",
        name="Melbourne Retail Network",
        region="APAC",
        country_code="AU",
        extra={"channel": "Mixed"},
    ),
]

_SAMPLE_ROUTES: list[TransitRoute] = [
    # Factory → Port (Road / inland)
    TransitRoute("FACTORY-CN-GZ-001", "Factory", "PORT-CN-SHA", "Port",
                 "Road",  base_cost=320.0,  current_latency=6.0,    distance_km=1_200, carrier="SF Express"),
    TransitRoute("FACTORY-CN-GZ-001", "Factory", "PORT-CN-SZX", "Port",
                 "Road",  base_cost=180.0,  current_latency=3.5,    distance_km=150,   carrier="SF Express"),
    TransitRoute("FACTORY-VN-HCM-001", "Factory", "PORT-SG-SIN", "Port",
                 "Sea",   base_cost=420.0,  current_latency=72.0,   distance_km=1_170, carrier="Evergreen"),

    # Port → Port (trans-oceanic Sea lanes — latency inflated for Shanghai congestion)
    TransitRoute("PORT-CN-SHA", "Port", "PORT-US-LGB", "Port",
                 "Sea",   base_cost=1_800.0, current_latency=630.0,  distance_km=9_600, carrier="COSCO"),   # +126h due to congestion
    TransitRoute("PORT-CN-SHA", "Port", "PORT-EU-HAM", "Port",
                 "Sea",   base_cost=2_100.0, current_latency=890.0,  distance_km=19_500, carrier="Maersk"), # +98h due to congestion
    TransitRoute("PORT-CN-SHA", "Port", "PORT-SG-SIN", "Port",
                 "Sea",   base_cost=650.0,  current_latency=163.0,  distance_km=3_300, carrier="OOCL"),    # +43h due to congestion
    TransitRoute("PORT-CN-SHA", "Port", "PORT-AU-SYD", "Port",
                 "Sea",   base_cost=1_400.0, current_latency=413.0,  distance_km=7_800, carrier="MSC"),    # +77h due to congestion

    # Proposed reroute: Shanghai → Busan via Rail (Strategy #SF-001)
    TransitRoute("PORT-CN-SHA", "Port", "PORT-KR-BSN", "Port",
                 "Rail",  base_cost=580.0,  current_latency=24.0,   distance_km=1_050, carrier="Korea Rail Logistics"),

    # Port → Port (Busan onward — used when reroute is applied)
    TransitRoute("PORT-KR-BSN", "Port", "PORT-US-LGB", "Port",
                 "Sea",   base_cost=1_650.0, current_latency=504.0,  distance_km=8_900, carrier="HMM"),
    TransitRoute("PORT-KR-BSN", "Port", "PORT-EU-HAM", "Port",
                 "Sea",   base_cost=1_950.0, current_latency=792.0,  distance_km=18_200, carrier="Yang Ming"),

    # Port → Warehouse (last-mile inland)
    TransitRoute("PORT-US-LGB", "Port", "WAREHOUSE-US-LA-001", "Warehouse",
                 "Road",  base_cost=280.0,  current_latency=5.0,    distance_km=45,    carrier="XPO Logistics"),
    TransitRoute("PORT-EU-HAM", "Port", "WAREHOUSE-DE-BER-004", "Warehouse",
                 "Rail",  base_cost=390.0,  current_latency=22.0,   distance_km=290,   carrier="DB Cargo"),
    TransitRoute("PORT-SG-SIN", "Port", "WAREHOUSE-SG-SIN-002", "Warehouse",
                 "Road",  base_cost=120.0,  current_latency=2.0,    distance_km=25,    carrier="DHL"),
    TransitRoute("PORT-AU-SYD", "Port", "RETAILER-AU-MEL-001", "Retailer",
                 "Road",  base_cost=310.0,  current_latency=9.0,    distance_km=870,   carrier="Toll Group"),

    # Warehouse → Retailer (final delivery)
    TransitRoute("WAREHOUSE-US-LA-001", "Warehouse", "RETAILER-US-NY-001", "Retailer",
                 "Air",   base_cost=920.0,  current_latency=8.0,    distance_km=3_950, carrier="FedEx Express"),
    TransitRoute("WAREHOUSE-DE-BER-004", "Warehouse", "RETAILER-EU-MUC-001", "Retailer",
                 "Road",  base_cost=210.0,  current_latency=7.5,    distance_km=580,   carrier="DHL Freight"),
    TransitRoute("WAREHOUSE-SG-SIN-002", "Warehouse", "RETAILER-US-NY-001", "Retailer",
                 "Air",   base_cost=2_400.0, current_latency=22.0,  distance_km=15_300, carrier="Singapore Airlines Cargo"),
]


async def seed_sample_network() -> None:
    """
    Populate the Neo4j database with the full sample supply chain network.

    Execution order
    ---------------
    1. Schema constraints / indexes (idempotent).
    2. All nodes, grouped by label and bulk-upserted.
    3. All ``TRANSIT_ROUTE`` relationships, bulk-upserted.

    This function is idempotent — running it multiple times only updates
    properties, it does not duplicate nodes or edges.

    Typical call (one-time dev setup):
        asyncio.run(seed_sample_network())
    """
    logger.info("═" * 60)
    logger.info("  SentinelFlow — seeding sample supply chain network")
    logger.info("═" * 60)

    # Step 1 — ensure schema is ready
    await initialize_supply_chain_graph()

    # Step 2 — upsert nodes, grouped by label (bulk_create requires homogeneity)
    from collections import defaultdict
    by_label: dict[NodeLabel, list[SupplyChainNode]] = defaultdict(list)
    for node in _SAMPLE_NODES:
        by_label[node.label].append(node)

    for label, nodes in by_label.items():
        await create_nodes_bulk(nodes)
        logger.info("  ✓ %d %s nodes seeded.", len(nodes), label)

    # Step 3 — upsert relationships
    await create_transit_routes_bulk(_SAMPLE_ROUTES)
    logger.info("  ✓ %d TRANSIT_ROUTE edges seeded.", len(_SAMPLE_ROUTES))

    logger.info("═" * 60)
    logger.info("  Sample network ready. Node breakdown:")
    logger.info("    Factories   : %d", len(by_label["Factory"]))
    logger.info("    Ports       : %d", len(by_label["Port"]))
    logger.info("    Warehouses  : %d", len(by_label["Warehouse"]))
    logger.info("    Retailers   : %d", len(by_label["Retailer"]))
    logger.info("    Routes      : %d", len(_SAMPLE_ROUTES))
    logger.info("═" * 60)


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    commands = {
        "init":   initialize_supply_chain_graph,
        "seed":   seed_sample_network,
    }

    cmd = sys.argv[1] if len(sys.argv) > 1 else "seed"
    fn  = commands.get(cmd)

    if fn is None:
        print(f"Unknown command '{cmd}'. Available: {list(commands)}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(fn())
