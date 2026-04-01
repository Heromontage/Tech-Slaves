"""
database.py — SentinelFlow connection layer
============================================
Manages two independent database connections:

  1. PostgreSQL (async SQLAlchemy + asyncpg)
     Stores time-series operational data:
       - Factory output readings
       - Port manifesto entries
       - Stream ingestion logs
       - Route health snapshots

  2. Neo4j (official Python driver, async session)
     Stores the supply chain directed multigraph:
       - Nodes  → Ports, Warehouses, Factories, Carriers
       - Edges  → Routes, Handoffs, Dependencies
       - Labels → Bottleneck flags, risk scores

Both managers read credentials exclusively from environment variables
(populated via .env in development, injected secrets in production).
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from neo4j import AsyncGraphDatabase, AsyncDriver
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger("sentinelflow.database")


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All values are read from environment variables.
    In development, place a .env file in the backend/ directory.
    Variable names are case-insensitive.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL — async connection (asyncpg driver)
    postgres_user: str = "sentinelflow"
    postgres_password: str = "sentinelflow"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "sentinelflow"

    # SQLAlchemy pool tuning
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_pool_timeout: int = 30          # seconds before giving up on a conn
    postgres_pool_recycle: int = 1800        # recycle connections every 30 min

    # Neo4j — Bolt connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sentinelflow"
    neo4j_database: str = "neo4j"           # named database (Neo4j 4+ Enterprise)
                                             # use "neo4j" for Community Edition

    @property
    def postgres_dsn(self) -> str:
        """
        asyncpg DSN for SQLAlchemy async engine.
        Uses the postgresql+asyncpg:// scheme required by SQLAlchemy's async layer.
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()


# ── SQLAlchemy Base ───────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.
    Import this in your models/ modules:

        from database import Base

        class PortManifesto(Base):
            __tablename__ = "port_manifestos"
            ...
    """
    pass


# ── PostgreSQL async engine & session factory ─────────────────────────────────

def _build_postgres_engine() -> AsyncEngine:
    return create_async_engine(
        settings.postgres_dsn,
        echo=False,                          # set True to log all SQL statements
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
        pool_timeout=settings.postgres_pool_timeout,
        pool_recycle=settings.postgres_pool_recycle,
        pool_pre_ping=True,                  # verify connection before checkout
        connect_args={
            # asyncpg-specific: enforce a statement timeout of 30 s
            "command_timeout": 30,
            "server_settings": {
                "application_name": "sentinelflow_api",
                "timezone": "UTC",
            },
        },
    )


# Module-level singletons — initialised once on first import.
# Replace with None + lazy init pattern if you need deferred startup.
postgres_engine: AsyncEngine = _build_postgres_engine()

AsyncSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=postgres_engine,
    class_=AsyncSession,
    expire_on_commit=False,   # avoids lazy-load errors after commit in async ctx
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a transactional SQLAlchemy session.
    Commits on clean exit, rolls back on any exception.

    Usage — in a route or service:

        async with get_postgres_session() as session:
            result = await session.execute(select(PortManifesto))

    For FastAPI dependency injection, use get_db_session() below.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.

    Usage in a router:

        @router.get("/manifestos")
        async def list_manifestos(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(PortManifesto))
            return result.scalars().all()
    """
    async with get_postgres_session() as session:
        yield session


# ── Neo4j async driver ────────────────────────────────────────────────────────

def _build_neo4j_driver() -> AsyncDriver:
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=50,
        connection_timeout=15,               # seconds to establish Bolt connection
        max_transaction_retry_time=30,       # seconds to retry on transient errors
    )


neo4j_driver: AsyncDriver = _build_neo4j_driver()


@asynccontextmanager
async def get_neo4j_session():
    """
    Async context manager that yields a Neo4j async session bound to the
    configured database.

    Usage — in a service or route:

        async with get_neo4j_session() as session:
            result = await session.run(
                "MATCH (p:Port {id: $id}) RETURN p",
                id="SH-PORT-01",
            )
            record = await result.single()

    Note: Neo4j sessions are not thread-safe; create one per coroutine.
    """
    async with neo4j_driver.session(
        database=settings.neo4j_database,
        fetch_size=200,                      # records fetched per round-trip
    ) as session:
        yield session


async def get_graph_session():
    """
    FastAPI dependency — yields a Neo4j session for the lifetime of a request.

    Usage in a router:

        @router.get("/graph/routes")
        async def graph_routes(graph: AsyncSession = Depends(get_graph_session)):
            result = await graph.run("MATCH (r:Route) RETURN r LIMIT 25")
            ...
    """
    async with get_neo4j_session() as session:
        yield session


# ── Startup verification ──────────────────────────────────────────────────────

async def verify_postgres() -> dict:
    """
    Executes a lightweight query to confirm PostgreSQL is reachable and
    the connection pool is healthy. Returns a status dict consumed by
    the /health endpoint and the lifespan startup hook.
    """
    try:
        async with get_postgres_session() as session:
            result = await session.execute(
                text("SELECT version(), current_database(), now() AT TIME ZONE 'UTC'")
            )
            row = result.one()
            pg_version, db_name, server_time = row

        logger.info("PostgreSQL connected — db=%s server_time=%s", db_name, server_time)
        return {
            "status": "connected",
            "database": db_name,
            "server_time_utc": str(server_time),
            "pg_version": pg_version.split(",")[0],   # e.g. "PostgreSQL 16.2"
        }
    except Exception as exc:
        logger.error("PostgreSQL connection FAILED: %s", exc)
        return {"status": "error", "detail": str(exc)}


async def verify_neo4j() -> dict:
    """
    Runs a trivial Cypher query to confirm Neo4j is reachable and the
    Bolt driver can open a session. Returns a status dict.
    """
    try:
        async with get_neo4j_session() as session:
            result = await session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            )
            record = await result.single()
            neo4j_version = record["version"] if record else "unknown"
            edition = record["edition"] if record else "unknown"

        logger.info(
            "Neo4j connected — version=%s edition=%s db=%s",
            neo4j_version, edition, settings.neo4j_database,
        )
        return {
            "status": "connected",
            "database": settings.neo4j_database,
            "neo4j_version": neo4j_version,
            "edition": edition,
        }
    except Exception as exc:
        logger.error("Neo4j connection FAILED: %s", exc)
        return {"status": "error", "detail": str(exc)}


async def verify_all_connections() -> dict:
    """
    Runs both verification checks concurrently and returns a combined
    report. Raises RuntimeError if either service is unreachable, which
    will abort FastAPI startup in strict mode.
    """
    import asyncio

    postgres_status, neo4j_status = await asyncio.gather(
        verify_postgres(),
        verify_neo4j(),
        return_exceptions=False,
    )

    report = {
        "postgres": postgres_status,
        "neo4j": neo4j_status,
    }

    failed = [name for name, s in report.items() if s.get("status") == "error"]
    if failed:
        raise RuntimeError(
            f"Database connectivity check failed for: {', '.join(failed)}. "
            "Review logs above for details. Aborting startup."
        )

    return report


async def close_all_connections() -> None:
    """
    Gracefully drains the SQLAlchemy connection pool and closes the
    Neo4j Bolt driver. Called from the FastAPI lifespan shutdown hook.
    """
    logger.info("Closing PostgreSQL connection pool…")
    await postgres_engine.dispose()

    logger.info("Closing Neo4j Bolt driver…")
    await neo4j_driver.close()

    logger.info("All database connections closed.")
