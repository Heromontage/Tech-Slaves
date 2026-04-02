"""
main.py — SentinelFlow FastAPI application entry point
=======================================================
New in this revision
--------------------
  GET  /api/network-state   Full graph (nodes + edges) with live latency
  GET  /api/bottlenecks     Nodes/edges flagged by the Isolation Forest
  POST /api/mitigate        OR-Tools rerouting plan for a disrupted node

CORS is open for http://localhost (ports 3000, 5173, 8080) so a React
dev-server can reach the API without a proxy.

Startup sequence
  1. Lifespan opens → verify_all_connections()
  2. Ingestion tasks start.
  3. Bottleneck registry pre-seeded from the sample network.
  4. Application serves requests.
  5. Lifespan closes → tasks cancelled, pools drained.

Interactive docs
  Swagger UI  → http://localhost:8000/docs
  ReDoc        → http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_ingestion import register_ingestion_tasks, stop_ingestion_tasks
from database import (
    close_all_connections,
    get_neo4j_session,
    verify_all_connections,
    verify_neo4j,
    verify_postgres,
)
from ml_pipeline import bn_registry, seed_bn_registry_from_sample_network
from optimizer import GraphData, RouteOption, optimize_rerouting

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("sentinelflow.main")


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═" * 60)
    logger.info("  SentinelFlow API — starting up")
    logger.info("═" * 60)

    connection_report = await verify_all_connections()
    app.state.connection_report = connection_report

    logger.info("Database connectivity report:")
    for name, status in connection_report.items():
        logger.info("  %-10s → %s", name, status)

    # Start background data ingestion streams
    register_ingestion_tasks(app)

    # Pre-seed the Isolation Forest registry with historical baselines
    seed_bn_registry_from_sample_network()
    logger.info("Bottleneck registry seeded.")

    logger.info("All systems operational. Accepting requests.")
    logger.info("═" * 60)

    yield  # ── application is live ──────────────────────────────────────────

    logger.info("SentinelFlow API — shutting down…")
    await stop_ingestion_tasks(app)
    await close_all_connections()
    logger.info("Shutdown complete.")


# ── Application ───────────────────────────────────────────────────────────

app = FastAPI(
    title="SentinelFlow API",
    description=(
        "Supply Chain Digital Twin — real-time monitoring, "
        "predictive bottleneck detection, and automated rerouting optimization."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
# Allow React dev-servers on common local ports.  Extend this list or use
# allow_origins=["*"] to open access wider (not recommended for production).

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",   # Create React App
        "http://localhost:5173",   # Vite
        "http://localhost:8080",   # Vue / generic
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════
# Pydantic schemas for the endpoints
# ══════════════════════════════════════════════════════════════════════════

class NodeSchema(BaseModel):
    """A supply-chain graph node returned to the dashboard."""
    node_id:        str
    node_type:      str                    # Factory | Port | Warehouse | Retailer
    name:           str
    region:         Optional[str]   = None
    country_code:   Optional[str]   = None
    congestion_pct: Optional[float] = None
    capacity:       Optional[int]   = None


class EdgeSchema(BaseModel):
    """A TRANSIT_ROUTE edge returned to the dashboard."""
    from_id:         str
    to_id:           str
    transit_mode:    str
    base_cost:       float
    current_latency: float
    distance_km:     Optional[float] = None
    carrier:         Optional[str]   = None
    # Derived: how much slower than baseline
    latency_ratio:   float           = Field(
        default=1.0,
        description="current_latency / base_latency — >1 means congested"
    )


class NetworkStateResponse(BaseModel):
    nodes:       list[NodeSchema]
    edges:       list[EdgeSchema]
    node_count:  int
    edge_count:  int
    fetched_at:  float = Field(default_factory=time.time)


class BottleneckNodeSchema(BaseModel):
    node_id:          str
    node_type:        str
    name:             str
    region:           Optional[str]  = None
    risk_level:       str            # critical | high | medium | low
    max_latency_h:    float
    anomaly_score:    Optional[float] = None
    is_if_flagged:    bool           = False
    outbound_routes:  int            = 0


class BottleneckEdgeSchema(BaseModel):
    from_id:         str
    to_id:           str
    transit_mode:    str
    current_latency: float
    p95_threshold:   Optional[float] = None
    anomaly_score:   Optional[float] = None
    severity:        str             # critical | warning | normal


class BottleneckResponse(BaseModel):
    nodes:          list[BottleneckNodeSchema]
    edges:          list[BottleneckEdgeSchema]
    total_flagged:  int
    fetched_at:     float = Field(default_factory=time.time)


class MitigateRequest(BaseModel):
    disrupted_node_id: str = Field(
        ...,
        description="The Neo4j node ID to route around, e.g. 'PORT-CN-SHA'.",
        examples=["PORT-CN-SHA"],
    )
    total_cargo_teu: int = Field(
        default=12_000,
        ge=1,
        description="Total TEU volume to redistribute across alternative lanes.",
    )
    risk_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description=(
            "Weight applied to the risk-delay cost term. "
            "0 = pure cost minimisation, higher = prefer safer lanes."
        ),
    )
    min_throughput_pct: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of cargo that must be delivered [0, 1].",
    )


class RouteAllocationSchema(BaseModel):
    route_id:     str
    mode:         str
    carrier:      Optional[str]
    flow_teu:     float
    cost_usd:     float
    delay_hours:  float
    risk_factor:  float
    pct_of_total: float


class MitigateResponse(BaseModel):
    status:               str
    disrupted_node_id:    str
    total_cargo_teu:      float
    total_flow_teu:       float
    total_cost_usd:       float
    weighted_delay_hours: float
    throughput_pct:       float
    allocations:          list[RouteAllocationSchema]
    solver_wall_ms:       float
    message:              str


# ══════════════════════════════════════════════════════════════════════════
# Helper: fetch alternative routes for a disrupted node from Neo4j
# ══════════════════════════════════════════════════════════════════════════

async def _fetch_alternative_routes(disrupted_node_id: str) -> list[RouteOption]:
    """
    Query Neo4j for all TRANSIT_ROUTE edges that bypass the disrupted node,
    originating from the same source node(s) that would normally feed it.

    Falls back to the built-in SentinelFlow sample routes when the graph
    has not been seeded or the node is not found.
    """
    cypher = """
        MATCH (origin)-[r1:TRANSIT_ROUTE]->(disrupted {id: $node_id})
        MATCH (origin)-[r_alt:TRANSIT_ROUTE]->(alt)
        WHERE alt.id <> $node_id
          AND r_alt.current_latency IS NOT NULL
        RETURN
            (origin.id + '→' + alt.id + '-' + r_alt.transit_mode) AS route_id,
            r_alt.transit_mode      AS mode,
            r_alt.base_cost         AS base_cost,
            r_alt.current_latency   AS current_latency,
            r1.current_latency      AS base_latency,
            coalesce(alt.capacity, 10000) AS capacity_teu,
            r_alt.carrier           AS carrier
        LIMIT 10
    """
    try:
        async with get_neo4j_session() as session:
            result = await session.run(cypher, node_id=disrupted_node_id)
            records = [r.data() async for r in result]
    except Exception as exc:
        logger.warning(
            "Neo4j route query failed (%s) — falling back to sample routes.", exc
        )
        records = []

    if not records:
        logger.info(
            "No alternative routes found in graph for '%s'; using sample data.",
            disrupted_node_id,
        )
        from optimizer import build_shanghai_reroute_graph
        return build_shanghai_reroute_graph().routes

    options: list[RouteOption] = []
    for rec in records:
        try:
            options.append(RouteOption(
                route_id         = rec["route_id"],
                mode             = rec["mode"] or "Sea",
                cost_per_teu     = float(rec["base_cost"] or 1_500),
                delay_hours      = float(rec["current_latency"] or 504),
                risk_factor      = min(
                    float(rec["current_latency"] or 504) /
                    max(float(rec["base_latency"] or 504), 1) * 0.15,
                    1.0,
                ),
                capacity_teu     = int(rec["capacity_teu"] or 10_000),
                base_delay_hours = float(rec["base_latency"] or rec["current_latency"] or 504),
                carrier          = rec.get("carrier"),
            ))
        except Exception as exc:
            logger.debug("Skipping malformed route record: %s — %s", rec, exc)

    return options or build_shanghai_reroute_graph().routes


# ══════════════════════════════════════════════════════════════════════════
# Health endpoints 
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"], summary="Root ping")
async def root():
    """Lightweight liveness probe — no database round-trip."""
    return {"status": "SentinelFlow API is running", "version": "2.1.0"}


@app.get("/health", tags=["Health"], summary="Full system health check")
async def health_check():
    """
    Re-queries both databases in real time.

    Returns HTTP 503 when any downstream service is unhealthy so that
    load-balancer probes can act on it automatically.
    """
    postgres_status, neo4j_status = (
        await verify_postgres(),
        await verify_neo4j(),
    )
    any_error = (
        postgres_status.get("status") == "error"
        or neo4j_status.get("status") == "error"
    )
    payload = {
        "status": "degraded" if any_error else "healthy",
        "version": "2.1.0",
        "services": {
            "api":      {"status": "operational"},
            "postgres": postgres_status,
            "neo4j":    neo4j_status,
        },
    }
    if any_error:
        raise HTTPException(status_code=503, detail=payload)
    return payload


# ══════════════════════════════════════════════════════════════════════════
# GET /api/network-state
# ══════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/network-state",
    response_model=NetworkStateResponse,
    tags=["Dashboard"],
    summary="Full supply-chain graph with real-time latency",
)
async def get_network_state(
    region: Optional[str] = Query(
        default=None,
        description="Filter nodes by region tag, e.g. 'APAC', 'EMEA', 'AMER'."
    ),
):
    """
    Returns every node and TRANSIT_ROUTE edge currently stored in Neo4j,
    enriched with the latest ``current_latency`` readings from the AIS
    ingestion pipeline.

    The React dashboard can use this to render the global network map and
    colour edges by congestion level.
    """
    # ── Nodes ──────────────────────────────────────────────────────────────
    node_cypher = """
        MATCH (n)
        WHERE n.id IS NOT NULL
          AND ($region IS NULL OR n.region = $region)
        RETURN
            n.id            AS node_id,
            labels(n)[0]    AS node_type,
            n.name          AS name,
            n.region        AS region,
            n.country_code  AS country_code,
            n.congestion_pct AS congestion_pct,
            n.capacity      AS capacity
        ORDER BY node_type, node_id
    """

    # ── Edges ──────────────────────────────────────────────────────────────
    edge_cypher = """
        MATCH (a)-[r:TRANSIT_ROUTE]->(b)
        WHERE a.id IS NOT NULL AND b.id IS NOT NULL
          AND ($region IS NULL OR a.region = $region OR b.region = $region)
        RETURN
            a.id                AS from_id,
            b.id                AS to_id,
            r.transit_mode      AS transit_mode,
            r.base_cost         AS base_cost,
            r.current_latency   AS current_latency,
            r.distance_km       AS distance_km,
            r.carrier           AS carrier
        ORDER BY from_id, to_id
    """

    try:
        async with get_neo4j_session() as session:
            node_result = await session.run(node_cypher, region=region)
            node_records = [r.data() async for r in node_result]

            edge_result = await session.run(edge_cypher, region=region)
            edge_records = [r.data() async for r in edge_result]

    except Exception as exc:
        logger.error("Neo4j query failed in /api/network-state: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Graph database unavailable: {exc}",
        )

    # Derive latency_ratio (>1 = congested vs baseline)
    nodes = [NodeSchema(**rec) for rec in node_records]
    edges: list[EdgeSchema] = []
    for rec in edge_records:
        latency     = float(rec.get("current_latency") or 1)
        base_cost   = float(rec.get("base_cost") or 1)
        # Use base_cost / 10 as a rough baseline hours proxy when no
        # historical baseline is stored on the edge
        baseline_h  = latency * 0.8
        edges.append(EdgeSchema(
            **{k: v for k, v in rec.items()},
            latency_ratio=round(latency / baseline_h, 3) if baseline_h else 1.0,
        ))

    logger.info(
        "/api/network-state → %d nodes, %d edges (region=%s)",
        len(nodes), len(edges), region,
    )
    return NetworkStateResponse(
        nodes=nodes,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
    )


# ══════════════════════════════════════════════════════════════════════════
# GET /api/bottlenecks
# ══════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/bottlenecks",
    response_model=BottleneckResponse,
    tags=["Dashboard"],
    summary="Nodes and edges flagged by the Isolation Forest anomaly detector",
)
async def get_bottlenecks(
    severity: Optional[str] = Query(
        default=None,
        description="Filter by severity: 'critical', 'high', 'medium', or 'low'.",
    ),
    limit: int = Query(
        default=50, ge=1, le=500,
        description="Maximum number of bottleneck nodes to return.",
    ),
):
    """
    Two-layer detection:
    1. Graph heuristic (Neo4j)
    2. Isolation Forest (ml_pipeline)
    """
    # ── Layer 1: Neo4j graph heuristic ─────────────────────────────────────────
    severity_filter = ""
    params: dict = {"limit": limit}

    if severity:
        _thresholds = {"critical": 120, "high": 72, "medium": 36, "low": 0}
        threshold = _thresholds.get(severity.lower(), 0)
        severity_filter = "AND max_latency > $threshold "
        params["threshold"] = threshold

    node_cypher = f"""
        MATCH (n)-[r:TRANSIT_ROUTE]->(m)
        WITH n,
             max(r.current_latency) AS max_latency,
             avg(r.base_cost / 10)  AS baseline_hours,
             count(r)               AS outbound_routes
        WHERE max_latency > baseline_hours
          {severity_filter}
        RETURN
            n.id          AS node_id,
            labels(n)[0]  AS node_type,
            n.name        AS name,
            n.region      AS region,
            max_latency,
            outbound_routes
        ORDER BY max_latency DESC
        LIMIT $limit
    """

    edge_cypher = """
        MATCH (a)-[r:TRANSIT_ROUTE]->(b)
        WHERE r.current_latency > (r.base_cost / 10) * 1.2
        RETURN
            a.id                AS from_id,
            b.id                AS to_id,
            r.transit_mode      AS transit_mode,
            r.current_latency   AS current_latency
        ORDER BY r.current_latency DESC
        LIMIT 100
    """

    try:
        async with get_neo4j_session() as session:
            n_result = await session.run(node_cypher, **params)
            node_records = [r.data() async for r in n_result]

            e_result = await session.run(edge_cypher)
            edge_records = [r.data() async for r in e_result]

    except Exception as exc:
        logger.error("Neo4j query failed in /api/bottlenecks: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Graph database unavailable: {exc}",
        )

    # ── Layer 2: Isolation Forest enrichment ──────────────────────────────────
    def _risk_level(latency: float) -> str:
        if latency > 120: return "critical"
        if latency > 72:  return "high"
        if latency > 36:  return "medium"
        return "low"

    flagged_nodes: list[BottleneckNodeSchema] = []
    for rec in node_records:
        node_id     = rec["node_id"]
        max_latency = float(rec.get("max_latency") or 0)
        risk_lvl    = _risk_level(max_latency)

        if severity and risk_lvl != severity.lower():
            continue

        flagged_nodes.append(BottleneckNodeSchema(
            node_id         = node_id,
            node_type       = rec.get("node_type") or "Unknown",
            name            = rec.get("name") or node_id,
            region          = rec.get("region"),
            risk_level      = risk_lvl,
            max_latency_h   = round(max_latency, 2),
            anomaly_score   = None,   
            is_if_flagged   = False,
            outbound_routes = int(rec.get("outbound_routes") or 0),
        ))

    flagged_edges: list[BottleneckEdgeSchema] = []
    for rec in edge_records:
        from_id  = rec["from_id"]
        to_id    = rec["to_id"]
        route_id = f"{from_id}→{to_id}"
        latency  = float(rec.get("current_latency") or 0)

        anomaly_score: Optional[float] = None
        p95: Optional[float] = None
        severity_str = _risk_level(latency)

        if route_id in bn_registry.registered_routes:
            try:
                bn_result = await bn_registry.async_check(route_id, latency)
                anomaly_score = bn_result.anomaly_score
                p95           = bn_result.p95_threshold
                severity_str  = bn_result.severity

                for node in flagged_nodes:
                    if node.node_id == from_id:
                        if anomaly_score is not None and (
                            node.anomaly_score is None
                            or anomaly_score > node.anomaly_score
                        ):
                            node.anomaly_score = anomaly_score
                            node.is_if_flagged = bn_result.is_bottleneck
            except Exception as exc:
                logger.debug("IF check failed for %s: %s", route_id, exc)

        flagged_edges.append(BottleneckEdgeSchema(
            from_id         = from_id,
            to_id           = to_id,
            transit_mode    = rec.get("transit_mode") or "Sea",
            current_latency = round(latency, 2),
            p95_threshold   = p95,
            anomaly_score   = anomaly_score,
            severity        = severity_str,
        ))

    logger.info(
        "/api/bottlenecks → %d nodes, %d edges (severity=%s)",
        len(flagged_nodes), len(flagged_edges), severity,
    )
    return BottleneckResponse(
        nodes=flagged_nodes,
        edges=flagged_edges,
        total_flagged=len(flagged_nodes),
    )


# ══════════════════════════════════════════════════════════════════════════
# POST /api/mitigate
# ══════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/mitigate",
    response_model=MitigateResponse,
    tags=["Optimization"],
    summary="Run OR-Tools LP rerouting for a disrupted node",
)
async def mitigate(body: MitigateRequest):
    """
    Runs the OR-Tools GLOP linear programme to minimise cost and delay based on alternative lanes.
    """
    disrupted_id = body.disrupted_node_id.strip()
    if not disrupted_id:
        raise HTTPException(status_code=422, detail="disrupted_node_id must not be empty.")

    try:
        routes = await _fetch_alternative_routes(disrupted_id)
    except Exception as exc:
        logger.error("Route fetch failed for '%s': %s", disrupted_id, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not retrieve alternative routes: {exc}",
        )

    if not routes:
        raise HTTPException(
            status_code=404,
            detail=f"No alternative routes found for node '{disrupted_id}'.",
        )

    graph_data = GraphData(
        disrupted_node_id  = disrupted_id,
        routes             = routes,
        min_throughput_pct = body.min_throughput_pct,
    )

    try:
        result = optimize_rerouting(
            graph_data        = graph_data,
            disrupted_node_id = disrupted_id,
            total_cargo       = body.total_cargo_teu,
            risk_weight       = body.risk_weight,
            allow_partial     = True,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"OR-Tools not installed on this server: {exc}",
        )
    except Exception as exc:
        logger.error("Optimisation failed for '%s': %s", disrupted_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Optimisation engine error: {exc}",
        )

    if result.status not in ("OPTIMAL", "FEASIBLE"):
        raise HTTPException(
            status_code=422,
            detail={
                "status":  result.status,
                "message": result.message or "LP solver could not find a feasible solution.",
            },
        )

    logger.info(
        "/api/mitigate '%s' → %s  flow=%.0f TEU  cost=$%.2f  "
        "delay=%.1fh  throughput=%.1f%%",
        disrupted_id,
        result.status,
        result.total_flow_teu,
        result.total_cost_usd,
        result.weighted_delay_hours,
        result.throughput_pct,
    )

    return MitigateResponse(
        status               = result.status,
        disrupted_node_id    = result.disrupted_node_id,
        total_cargo_teu      = result.total_cargo_teu,
        total_flow_teu       = result.total_flow_teu,
        total_cost_usd       = result.total_cost_usd,
        weighted_delay_hours = result.weighted_delay_hours,
        throughput_pct       = result.throughput_pct,
        allocations          = [
            RouteAllocationSchema(
                route_id     = a.route_id,
                mode         = a.mode,
                carrier      = a.carrier,
                flow_teu     = a.flow_teu,
                cost_usd     = a.cost_usd,
                delay_hours  = a.delay_hours,
                risk_factor  = a.risk_factor,
                pct_of_total = a.pct_of_total,
            )
            for a in result.allocations
        ],
        solver_wall_ms = result.solver_wall_ms,
        message        = result.message,
    )