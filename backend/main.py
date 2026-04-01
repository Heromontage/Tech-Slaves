"""
main.py — SentinelFlow FastAPI application entry point
=======================================================
Startup sequence
  1. Lifespan context manager opens → verify_all_connections() runs.
     If either database is unreachable the process exits immediately with
     a clear error message rather than serving requests against a broken state.
  2. Application serves requests.
  3. Lifespan context manager closes → close_all_connections() drains pools.

Interactive docs
  Swagger UI  →  http://localhost:8000/docs
  ReDoc        →  http://localhost:8000/redoc
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import (
    close_all_connections,
    verify_all_connections,
    verify_neo4j,
    verify_postgres,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("sentinelflow.main")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async lifespan context manager — replaces the deprecated @app.on_event
    pattern introduced in FastAPI 0.93.

    Everything before `yield` runs at startup; everything after runs at
    shutdown.  Raising an exception before the yield aborts startup cleanly.
    """
    logger.info("═" * 60)
    logger.info("  SentinelFlow API — starting up")
    logger.info("═" * 60)

    # Verify both database connections concurrently.
    # RuntimeError is raised inside verify_all_connections() if either fails,
    # which propagates here and prevents the server from accepting traffic.
    connection_report = await verify_all_connections()

    logger.info("Database connectivity report:")
    for name, status in connection_report.items():
        logger.info("  %-10s → %s", name, status)

    logger.info("All systems operational. Accepting requests.")
    logger.info("═" * 60)

    # Store the report so the /health endpoint can return it without
    # re-querying the databases on every call.
    app.state.connection_report = connection_report

    yield   # ── application is live ──────────────────────────────────────────

    logger.info("SentinelFlow API — shutting down…")
    await close_all_connections()
    logger.info("Shutdown complete.")


# ── Application ───────────────────────────────────────────────────────────────

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Root ping")
async def root():
    """Lightweight liveness probe — no database round-trip."""
    return {"status": "SentinelFlow API is running"}


@app.get("/health", tags=["Health"], summary="Full system health check")
async def health_check():
    """
    Re-queries both databases in real time and returns their status.

    Response shape:

        {
          "status": "healthy" | "degraded",
          "version": "2.1.0",
          "services": {
            "api":      { "status": "operational" },
            "postgres": { "status": "connected", "pg_version": "...", ... },
            "neo4j":    { "status": "connected", "neo4j_version": "...", ... }
          }
        }

    HTTP 503 is returned when one or more downstream services are unhealthy,
    so load-balancer health checks can act on it automatically.
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
            "api": {"status": "operational"},
            "postgres": postgres_status,
            "neo4j": neo4j_status,
        },
    }

    if any_error:
        raise HTTPException(status_code=503, detail=payload)

    return payload


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v2/routes", tags=["Routes"], summary="List all monitored routes")
async def list_routes():
    """Returns all routes with health scores. Wire SQLAlchemy models to populate."""
    return {
        "routes": [],
        "total": 0,
        "message": "Connect SQLAlchemy Route model — see models/route.py.",
    }


@app.get(
    "/v2/routes/{route_id}/health",
    tags=["Routes"],
    summary="Get health score for a specific route",
)
async def get_route_health(route_id: str):
    """
    Returns live health score, risk level, and active bottlenecks for a route.
    Reads from PostgreSQL (time-series snapshot) and Neo4j (graph risk propagation).
    """
    return {
        "route_id": route_id,
        "health_score": None,
        "risk_level": None,
        "bottlenecks": [],
        "throughput": None,
        "last_updated": None,
        "message": "Wire get_db_session() + get_graph_session() to populate.",
    }


# ── Bottlenecks ───────────────────────────────────────────────────────────────

@app.get(
    "/v2/bottlenecks",
    tags=["Bottlenecks"],
    summary="List currently active bottlenecks",
)
async def list_bottlenecks(severity: str = None, limit: int = 50):
    """
    Queries the Neo4j graph for nodes flagged as bottlenecks.
    severity — one of: critical | warning | info
    limit    — max records returned (default 50, max 500)
    """
    return {
        "bottlenecks": [],
        "total": 0,
        "filters": {"severity": severity, "limit": limit},
        "message": "Wire get_graph_session() with Cypher MATCH (b:Bottleneck) to populate.",
    }


# ── Optimization ──────────────────────────────────────────────────────────────

@app.post(
    "/v2/optimize",
    tags=["Optimization"],
    summary="Run OR-Tools rerouting optimization",
)
async def run_optimization():
    """
    Triggers the linear-programming optimization engine.
    Reads the current graph topology from Neo4j, runs OR-Tools, and writes
    strategy records back to PostgreSQL.
    """
    return {
        "status": "queued",
        "strategies": [],
        "message": "Wire OR-Tools solver and get_graph_session() to populate.",
    }


# ── Streams ───────────────────────────────────────────────────────────────────

@app.get(
    "/v2/streams/status",
    tags=["Streams"],
    summary="Get status of all data ingestion streams",
)
async def get_stream_status():
    """Returns live throughput, latency, and uptime for each registered stream."""
    return {
        "streams": [],
        "total": 0,
        "message": "Wire stream monitor models from PostgreSQL to populate.",
    }
