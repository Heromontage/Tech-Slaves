"""
alembic/env.py — async-aware Alembic migration environment
===========================================================
Wired to the same async engine and declarative Base used by the application
so that `alembic revision --autogenerate` detects all models automatically.

Usage
-----
    # inside backend/
    alembic revision --autogenerate -m "initial tables"
    alembic upgrade head
    alembic downgrade -1
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine  # add this
from sqlalchemy import pool
from database import Base, settings
import models  # noqa

# Import Base (which has all models attached) and settings from the app layer.
# models.py must be imported so SQLAlchemy registers the table metadata.
from database import Base, settings  # noqa: F401 — triggers metadata registration
import models  # noqa: F401 — ensures all Table objects are attached to Base.metadata

# ── Alembic Config object ─────────────────────────────────────────────────────

config = context.config

# Interpret the config file for Python logging if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url from the application settings so we never hard-code
# credentials in alembic.ini.  asyncpg cannot be used by Alembic directly
# (it uses synchronous execution internally), so swap the driver to psycopg2.
config.set_main_option(
    "sqlalchemy.url",
    settings.postgres_dsn.replace("postgresql+asyncpg", "postgresql+psycopg2"),
)

target_metadata = Base.metadata


# ── Offline migrations (generate SQL scripts without a live DB) ───────────────

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,          # detect column type changes
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online migrations (apply directly against the database) ───────────────────

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # Build the asyncpg URL directly from settings — bypasses alembic.ini
    asyncpg_url = settings.postgres_dsn  # already postgresql+asyncpg://

    connectable = create_async_engine(
        asyncpg_url,
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
