"""
data_ingestion.py — SentinelFlow Async Background Ingestion Pipeline
=====================================================================
Simulates two real-time data streams that run as asyncio background tasks
for the duration of the FastAPI application's lifetime:

  Stream 1 — AIS Vessel Tracker
    Mocks Automatic Identification System (AIS) position reports for ships
    travelling between the ports defined in graph_ops.py.  Each tick
    generates a ``PortManifest`` record capturing the vessel's current port
    call, simulated TEU volume, and a dwell-time figure that drifts upward
    when a congestion event is active.

  Stream 2 — Media Sentiment Monitor
    Mocks news-wire and social-media posts about events that affect supply
    chain risk (labor strikes, weather, geopolitical incidents).  Each tick
    scores the synthesized headline with a DistilBERT-style risk score and
    persists a ``SentimentDatum`` record, complete with extracted named
    entities.

Integration with main.py
------------------------
Import and call ``register_ingestion_tasks(app)`` inside the FastAPI
lifespan context manager *after* the database connections have been
verified:

    # main.py
    from data_ingestion import register_ingestion_tasks

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await verify_all_connections()
        register_ingestion_tasks(app)   # ← add this line
        yield
        await stop_ingestion_tasks(app)
        await close_all_connections()

Both tasks are stored in ``app.state`` so they can be gracefully cancelled
on shutdown without leaving dangling coroutines.

Configuration (environment variables)
--------------------------------------
  AIS_TICK_SECONDS       Seconds between AIS position ticks    (default: 15)
  SENTIMENT_TICK_SECONDS Seconds between sentiment ticks        (default: 30)
  INGESTION_BURST_SIZE   Records written per tick per stream    (default:  3)
  CONGESTION_PORT_ID     Port ID that is currently "hot"        (default: PORT-CN-SHA)
  CONGESTION_FACTOR      Latency multiplier during congestion   (default: 1.8)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import insert

from database import AsyncSessionFactory
from models import PortManifest, SentimentDatum, SentimentSourceType

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("sentinelflow.ingestion")

# ── Configuration ─────────────────────────────────────────────────────────────

AIS_TICK_SECONDS       = float(os.getenv("AIS_TICK_SECONDS",       "15"))
SENTIMENT_TICK_SECONDS = float(os.getenv("SENTIMENT_TICK_SECONDS", "30"))
INGESTION_BURST_SIZE   = int(os.getenv("INGESTION_BURST_SIZE",     "3"))
CONGESTION_PORT_ID     = os.getenv("CONGESTION_PORT_ID",            "PORT-CN-SHA")
CONGESTION_FACTOR      = float(os.getenv("CONGESTION_FACTOR",      "1.8"))

# ── AIS vessel / port reference data ─────────────────────────────────────────

@dataclass(frozen=True)
class PortMeta:
    """Static metadata for a port node, used as the AIS position anchor."""
    port_id:      str          # matches graph_ops.py node IDs
    name:         str
    un_locode:    str
    lat:          float        # decimal degrees
    lon:          float        # decimal degrees
    country_code: str
    design_teu:   int          # design capacity in TEU/day


# The seven ports from the SentinelFlow sample network
PORTS: list[PortMeta] = [
    PortMeta("PORT-CN-SHA", "Port of Shanghai",    "CNSHA",  31.3680, 121.6230, "CN", 50_000),
    PortMeta("PORT-CN-SZX", "Port of Shenzhen",    "CNSZX",  22.5229, 113.9430, "CN", 30_000),
    PortMeta("PORT-KR-BSN", "Port of Busan",        "KRBSN",  35.0960, 129.0403, "KR", 22_000),
    PortMeta("PORT-SG-SIN", "Port of Singapore",   "SGSIN",   1.2644, 103.8200, "SG", 38_000),
    PortMeta("PORT-US-LGB", "Port of Long Beach",  "USLGB",  33.7701, -118.1937, "US", 25_000),
    PortMeta("PORT-EU-HAM", "Port of Hamburg",     "DEHAM",  53.5404,   9.9662, "DE", 18_000),
    PortMeta("PORT-AU-SYD", "Port of Sydney",      "AUSYD", -33.8676, 151.2070, "AU",  7_000),
]
_PORT_BY_ID: dict[str, PortMeta] = {p.port_id: p for p in PORTS}


@dataclass
class VesselState:
    """
    Mutable in-memory state for a single simulated vessel.

    The vessel oscillates between two ports (``origin`` → ``destination``
    and back) at a configurable speed.  Position is interpolated linearly
    along the great-circle path so that lat/lon values in the AIS feed
    look realistic without requiring a full geodesic library.
    """
    mmsi:        str            # Maritime Mobile Service Identity (9-digit mock)
    vessel_name: str
    origin:      PortMeta
    destination: PortMeta
    progress:    float = 0.0   # 0.0 = at origin, 1.0 = at destination
    speed_knots: float = 14.0  # average container ship speed
    teu_capacity: int = 12_000

    # Internal: direction flag (True = origin→dest, False = dest→origin)
    _outbound: bool = field(default=True, repr=False)

    @property
    def current_port(self) -> PortMeta | None:
        """Returns the port the vessel is currently docked at, or None if at sea."""
        if self.progress <= 0.02:
            return self.origin if self._outbound else self.destination
        if self.progress >= 0.98:
            return self.destination if self._outbound else self.origin
        return None

    @property
    def lat(self) -> float:
        """Interpolated latitude (linear approximation of great-circle arc)."""
        a, b = (self.origin, self.destination) if self._outbound else (self.destination, self.origin)
        return a.lat + (b.lat - a.lat) * self.progress

    @property
    def lon(self) -> float:
        """Interpolated longitude (does NOT cross antimeridian correctly — acceptable for mock data)."""
        a, b = (self.origin, self.destination) if self._outbound else (self.destination, self.origin)
        return a.lon + (b.lon - a.lon) * self.progress

    def tick(self, elapsed_hours: float) -> None:
        """Advance the vessel's position by ``elapsed_hours`` of travel time."""
        # Approximate great-circle distance in nautical miles using the
        # Equirectangular approximation (fast, acceptable for ≤ 10 000 nm routes)
        a = self.origin if self._outbound else self.destination
        b = self.destination if self._outbound else self.origin
        dlat = math.radians(b.lat - a.lat)
        dlon = math.radians(b.lon - a.lon)
        mid_lat = math.radians((a.lat + b.lat) / 2)
        dist_nm = math.sqrt((dlat * 3440.065) ** 2 + (dlon * math.cos(mid_lat) * 3440.065) ** 2)
        if dist_nm < 1:
            dist_nm = 1  # avoid division by zero for same-port pairs

        travel_fraction = (self.speed_knots * elapsed_hours) / dist_nm
        self.progress = min(1.0, self.progress + travel_fraction)

        if self.progress >= 1.0:
            # Arrived — wait at destination then turn around
            self.progress = 0.0
            self._outbound = not self._outbound


# ── Vessel fleet (one per major lane in the sample network) ──────────────────

def _build_fleet() -> list[VesselState]:
    sha = _PORT_BY_ID["PORT-CN-SHA"]
    lgb = _PORT_BY_ID["PORT-US-LGB"]
    ham = _PORT_BY_ID["PORT-EU-HAM"]
    sin = _PORT_BY_ID["PORT-SG-SIN"]
    bsn = _PORT_BY_ID["PORT-KR-BSN"]
    syd = _PORT_BY_ID["PORT-AU-SYD"]
    szx = _PORT_BY_ID["PORT-CN-SZX"]

    return [
        VesselState("563012340", "MV ORIENT STAR",    sha, lgb, progress=0.10, speed_knots=16, teu_capacity=14_000),
        VesselState("477012341", "MV PACIFIC CROWN",  sha, lgb, progress=0.55, speed_knots=15, teu_capacity=12_500),
        VesselState("477012342", "MV BLUE HORIZON",   sha, ham, progress=0.25, speed_knots=14, teu_capacity=11_000),
        VesselState("566012343", "MV EUROPA BRIDGE",  sha, ham, progress=0.70, speed_knots=13, teu_capacity=13_000),
        VesselState("563012344", "MV SILK ROAD",      sha, sin, progress=0.40, speed_knots=17, teu_capacity= 8_000),
        VesselState("440012345", "MV BUSAN EXPRESS",  sha, bsn, progress=0.15, speed_knots=20, teu_capacity= 6_000),
        VesselState("503012346", "MV SOUTHERN CROSS", sha, syd, progress=0.60, speed_knots=14, teu_capacity= 9_500),
        VesselState("477012347", "MV PEARL DELTA",    szx, sin, progress=0.30, speed_knots=16, teu_capacity= 7_000),
    ]


# Module-level fleet — mutated in place on every AIS tick
_FLEET: list[VesselState] = _build_fleet()

# ── AIS position report generator ────────────────────────────────────────────

@dataclass
class AISPositionReport:
    """A single simulated AIS vessel position message."""
    mmsi:           str
    vessel_name:    str
    lat:            float
    lon:            float
    speed_knots:    float
    heading_deg:    float           # 0–359, approximate
    port_id:        str | None      # non-None only when at berth
    port_name:      str | None
    un_locode:      str | None
    teu_onboard:    int
    dwell_hours:    float           # hours at current port (0.0 if at sea)
    timestamp:      datetime


def _generate_ais_reports(
    tick_elapsed_hours: float,
    burst_size: int,
) -> list[AISPositionReport]:
    """
    Advance every vessel in the fleet and return ``burst_size`` position reports,
    sampled with replacement from vessels currently at a port (prioritised) or
    at sea.

    The congestion scenario inflates ``dwell_hours`` for vessels berthed at
    ``CONGESTION_PORT_ID`` by ``CONGESTION_FACTOR``.
    """
    now = datetime.now(timezone.utc)
    at_port: list[tuple[VesselState, PortMeta]] = []
    at_sea:  list[VesselState] = []

    for vessel in _FLEET:
        vessel.tick(tick_elapsed_hours)
        port = vessel.current_port
        if port:
            at_port.append((vessel, port))
        else:
            at_sea.append(vessel)

    reports: list[AISPositionReport] = []
    pool = at_port if at_port else [(v, None) for v in at_sea]

    for _ in range(burst_size):
        if at_port and random.random() < 0.7:
            vessel, port = random.choice(at_port)
        elif at_sea:
            vessel = random.choice(at_sea)
            port = None
        else:
            vessel, port = random.choice(pool)  # type: ignore[misc]
            port = port  # keep mypy happy

        # TEU load: random between 60–95% of capacity, with slight noise
        teu_onboard = int(vessel.teu_capacity * random.uniform(0.60, 0.95))

        # Dwell time — inflated at the congested port
        base_dwell = random.uniform(12, 48) if port else 0.0
        if port and port.port_id == CONGESTION_PORT_ID:
            base_dwell *= CONGESTION_FACTOR

        # Approximate heading from current progress direction
        heading = random.uniform(0, 360)  # simplified — no geodesic bearing calc

        reports.append(AISPositionReport(
            mmsi          = vessel.mmsi,
            vessel_name   = vessel.vessel_name,
            lat           = round(vessel.lat + random.uniform(-0.002, 0.002), 6),
            lon           = round(vessel.lon + random.uniform(-0.002, 0.002), 6),
            speed_knots   = vessel.speed_knots + random.uniform(-1.5, 1.5) if not port else 0.0,
            heading_deg   = heading,
            port_id       = port.port_id  if port else None,
            port_name     = port.name     if port else None,
            un_locode     = port.un_locode if port else None,
            teu_onboard   = teu_onboard,
            dwell_hours   = round(base_dwell, 2),
            timestamp     = now,
        ))

    return reports


async def _persist_ais_reports(reports: list[AISPositionReport]) -> None:
    """
    Write AIS position reports that have an active port call to ``port_manifests``.

    Only port-berthed reports are persisted — open-ocean pings are logged
    but not written to the DB (no port-level aggregation makes sense at sea).
    """
    rows_to_insert = []
    for r in reports:
        if r.port_id is None:
            logger.debug(
                "AIS | %s at sea (%.4f, %.4f) %.1f kn",
                r.vessel_name, r.lat, r.lon, r.speed_knots,
            )
            continue

        port = _PORT_BY_ID.get(r.port_id)
        design_teu = port.design_teu if port else 25_000
        capacity_pct = round((r.teu_onboard / design_teu) * 100, 2)

        rows_to_insert.append({
            "port_name":        r.port_name,
            "port_code":        r.un_locode,
            "container_volume": r.teu_onboard,
            "dwell_time":       r.dwell_hours,
            "capacity_pct":     min(capacity_pct, 200.0),  # cap at 200%
            "timestamp":        r.timestamp,
        })

        logger.info(
            "AIS | %s berthed at %s — %d TEU  dwell=%.1fh  cap=%.1f%%",
            r.vessel_name, r.port_name, r.teu_onboard, r.dwell_hours, capacity_pct,
        )

    if not rows_to_insert:
        return

    async with AsyncSessionFactory() as session:
        await session.execute(insert(PortManifest), rows_to_insert)
        await session.commit()
        logger.debug("AIS | Persisted %d port manifest records.", len(rows_to_insert))


# ── Sentiment corpus ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeadlineTemplate:
    """A parameterised news / social headline template."""
    template:     str            # use {port}, {carrier}, {region} as placeholders
    source_type:  SentimentSourceType
    base_risk:    float          # baseline risk score in [0, 1]
    risk_spread:  float          # ± random noise applied to base_risk
    language:     str
    entity_keys:  list[str]      # which placeholders become named entities


_HEADLINE_TEMPLATES: list[HeadlineTemplate] = [
    # ── Labor / strike events (high risk) ──────────────────────────────────
    HeadlineTemplate(
        template    = "Dockworkers at {port} announce 72-hour strike, threatening {carrier} schedules",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.85,
        risk_spread = 0.08,
        language    = "en",
        entity_keys = ["port", "carrier"],
    ),
    HeadlineTemplate(
        template    = "Union members at {port} vote to walk out over wage dispute — {carrier} issues force-majeure notice",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.82,
        risk_spread = 0.07,
        language    = "en",
        entity_keys = ["port", "carrier"],
    ),
    HeadlineTemplate(
        template    = "BREAKING: Customs officers at {port} begin work-to-rule; clearance times up 300%",
        source_type = SentimentSourceType.SOCIAL,
        base_risk   = 0.78,
        risk_spread = 0.10,
        language    = "en",
        entity_keys = ["port"],
    ),
    HeadlineTemplate(
        template    = "Labor unrest spreading across {region} ports — freight forwarders warn of 'systemic gridlock'",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.80,
        risk_spread = 0.06,
        language    = "en",
        entity_keys = ["region"],
    ),
    HeadlineTemplate(
        template    = "{port} crane operators reject latest offer; strike entering week two",
        source_type = SentimentSourceType.SOCIAL,
        base_risk   = 0.88,
        risk_spread = 0.05,
        language    = "en",
        entity_keys = ["port"],
    ),

    # ── Extreme weather events (medium-high risk) ──────────────────────────
    HeadlineTemplate(
        template    = "Typhoon Amber forces closure of {port}; {carrier} diverts 14 vessels to {port}",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.74,
        risk_spread = 0.09,
        language    = "en",
        entity_keys = ["port", "carrier"],
    ),
    HeadlineTemplate(
        template    = "Dense fog halts pilotage at {port} for third consecutive day",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.52,
        risk_spread = 0.12,
        language    = "en",
        entity_keys = ["port"],
    ),
    HeadlineTemplate(
        template    = "Winter storm DELTA grounds air freight at {port}; road connections blocked",
        source_type = SentimentSourceType.SOCIAL,
        base_risk   = 0.63,
        risk_spread = 0.10,
        language    = "en",
        entity_keys = ["port"],
    ),
    HeadlineTemplate(
        template    = "{region} faces flooding: rail links disrupted, {carrier} suspends inland deliveries",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.70,
        risk_spread = 0.08,
        language    = "en",
        entity_keys = ["region", "carrier"],
    ),

    # ── Geopolitical / regulatory (medium risk) ────────────────────────────
    HeadlineTemplate(
        template    = "New tariff regime at {port} raises import costs by 18% for {carrier} customers",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.55,
        risk_spread = 0.11,
        language    = "en",
        entity_keys = ["port", "carrier"],
    ),
    HeadlineTemplate(
        template    = "Sanctions update: {carrier} suspends {region} service amid compliance review",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.72,
        risk_spread = 0.07,
        language    = "en",
        entity_keys = ["carrier", "region"],
    ),
    HeadlineTemplate(
        template    = "Customs reform at {port}: new digitisation rules delay clearances by 36h",
        source_type = SentimentSourceType.SOCIAL,
        base_risk   = 0.48,
        risk_spread = 0.12,
        language    = "en",
        entity_keys = ["port"],
    ),

    # ── Positive / recovery signals (low risk) ─────────────────────────────
    HeadlineTemplate(
        template    = "{port} congestion easing as berth allocation reform takes effect",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.18,
        risk_spread = 0.06,
        language    = "en",
        entity_keys = ["port"],
    ),
    HeadlineTemplate(
        template    = "{carrier} reports record on-time performance in {region} corridor this quarter",
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.10,
        risk_spread = 0.05,
        language    = "en",
        entity_keys = ["carrier", "region"],
    ),
    HeadlineTemplate(
        template    = "Strike at {port} called off after last-minute agreement — operations resuming",
        source_type = SentimentSourceType.SOCIAL,
        base_risk   = 0.22,
        risk_spread = 0.08,
        language    = "en",
        entity_keys = ["port"],
    ),

    # ── Mandarin social-media (language variety) ───────────────────────────
    HeadlineTemplate(
        template    = "{port}港工人罢工，{carrier}货运延误预计超过48小时",   # "{port} port worker strike, {carrier} delays expected to exceed 48h"
        source_type = SentimentSourceType.SOCIAL,
        base_risk   = 0.83,
        risk_spread = 0.08,
        language    = "zh",
        entity_keys = ["port", "carrier"],
    ),
    HeadlineTemplate(
        template    = "台风警告：{port}港关闭，{region}航运受严重影响",       # "Typhoon warning: {port} closed, {region} shipping severely affected"
        source_type = SentimentSourceType.NEWS,
        base_risk   = 0.76,
        risk_spread = 0.09,
        language    = "zh",
        entity_keys = ["port", "region"],
    ),
]

# Substitution pools for template placeholders
_PLACEHOLDER_POOLS: dict[str, list[str]] = {
    "port": [
        "Shanghai", "Shenzhen", "Busan", "Singapore",
        "Long Beach", "Hamburg", "Rotterdam", "Antwerp",
    ],
    "carrier": [
        "Maersk", "MSC", "COSCO", "Evergreen",
        "CMA CGM", "Hapag-Lloyd", "Yang Ming", "HMM",
    ],
    "region": [
        "Asia-Pacific", "North Europe", "Mediterranean",
        "Trans-Pacific", "North America West Coast",
        "Southeast Asia", "Middle East Gulf",
    ],
}


def _render_headline(template: HeadlineTemplate) -> tuple[str, list[str]]:
    """
    Resolve all placeholders in a template string and return both the
    rendered headline and the list of substituted named entities.
    """
    text = template.template
    entities: list[str] = []

    for key in _PLACEHOLDER_POOLS:
        placeholder = f"{{{key}}}"
        if placeholder in text:
            value = random.choice(_PLACEHOLDER_POOLS[key])
            text = text.replace(placeholder, value, 1)  # first occurrence only
            if key in template.entity_keys:
                entities.append(value)
            # Replace any second occurrence (e.g., {port} appears twice) with a different value
            while placeholder in text:
                alt_value = random.choice(_PLACEHOLDER_POOLS[key])
                text = text.replace(placeholder, alt_value, 1)
                if key in template.entity_keys:
                    entities.append(alt_value)

    return text, entities


def _generate_sentiment_events(burst_size: int) -> list[dict]:
    """
    Generate ``burst_size`` synthetic sentiment events, weighted toward
    high-risk templates when the congestion scenario is active.

    Returns a list of dicts ready to be bulk-inserted into ``sentiment_data``.
    """
    now = datetime.now(timezone.utc)
    rows: list[dict] = []

    # During an active congestion event, 60% of events should be high-risk
    congestion_active = True  # always True in simulation; replace with runtime flag in prod
    if congestion_active:
        high_risk = [t for t in _HEADLINE_TEMPLATES if t.base_risk >= 0.70]
        low_risk  = [t for t in _HEADLINE_TEMPLATES if t.base_risk <  0.70]
        pool = random.choices(
            [high_risk, low_risk],
            weights=[60, 40],
            k=1,
        )[0]
    else:
        pool = _HEADLINE_TEMPLATES  # type: ignore[assignment]

    for _ in range(burst_size):
        tpl = random.choice(pool)
        text, entities = _render_headline(tpl)

        # Clamp risk score to [0, 1]
        noise = random.uniform(-tpl.risk_spread, tpl.risk_spread)
        risk_score = round(max(0.0, min(1.0, tpl.base_risk + noise)), 4)
        confidence = round(random.uniform(0.70, 0.98), 4)

        rows.append({
            "source_type":  tpl.source_type,
            "source_url":   None,           # no real URL in simulation
            "language":     tpl.language,
            "risk_score":   risk_score,
            "confidence":   confidence,
            "raw_text":     text[:8096],    # enforce schema max_length
            "entities":     json.dumps(entities, ensure_ascii=False),
            "timestamp":    now,
        })

        logger.info(
            "SENT | [%.3f / %s] %s",
            risk_score,
            tpl.source_type.value,
            text[:90] + ("…" if len(text) > 90 else ""),
        )

    return rows


async def _persist_sentiment_events(rows: list[dict]) -> None:
    """Bulk-insert a batch of sentiment rows into ``sentiment_data``."""
    if not rows:
        return

    async with AsyncSessionFactory() as session:
        await session.execute(insert(SentimentDatum), rows)
        await session.commit()
        logger.debug("SENT | Persisted %d sentiment records.", len(rows))


# ── Background task loops ─────────────────────────────────────────────────────

async def _ais_ingestion_loop() -> None:
    """
    Continuously emit AIS position reports at ``AIS_TICK_SECONDS`` intervals.

    Elapsed-time tracking ensures that vessel progress is proportional to
    real wall-clock time even if the event loop is briefly saturated.
    """
    logger.info(
        "AIS ingestion loop started — tick=%.0fs  burst=%d  congestion_port=%s",
        AIS_TICK_SECONDS, INGESTION_BURST_SIZE, CONGESTION_PORT_ID,
    )
    last_tick = asyncio.get_event_loop().time()

    while True:
        await asyncio.sleep(AIS_TICK_SECONDS)
        now = asyncio.get_event_loop().time()
        elapsed_seconds = now - last_tick
        last_tick = now

        # Convert wall-clock seconds to simulated hours
        # Scale factor: 1 real second ≈ 0.5 simulated hours (30× compression)
        sim_hours = (elapsed_seconds / 3600) * 30

        try:
            reports = _generate_ais_reports(
                tick_elapsed_hours=sim_hours,
                burst_size=INGESTION_BURST_SIZE,
            )
            await _persist_ais_reports(reports)
        except asyncio.CancelledError:
            logger.info("AIS ingestion loop cancelled — shutting down cleanly.")
            raise
        except Exception as exc:
            # Log and continue — a single failed tick must not kill the loop
            logger.exception("AIS tick failed: %s", exc)


async def _sentiment_ingestion_loop() -> None:
    """
    Continuously emit media sentiment events at ``SENTIMENT_TICK_SECONDS`` intervals.
    """
    logger.info(
        "Sentiment ingestion loop started — tick=%.0fs  burst=%d",
        SENTIMENT_TICK_SECONDS, INGESTION_BURST_SIZE,
    )

    while True:
        await asyncio.sleep(SENTIMENT_TICK_SECONDS)

        try:
            rows = _generate_sentiment_events(burst_size=INGESTION_BURST_SIZE)
            await _persist_sentiment_events(rows)
        except asyncio.CancelledError:
            logger.info("Sentiment ingestion loop cancelled — shutting down cleanly.")
            raise
        except Exception as exc:
            logger.exception("Sentiment tick failed: %s", exc)


# ── FastAPI integration helpers ───────────────────────────────────────────────

_TASK_KEY_AIS  = "ingestion_task_ais"
_TASK_KEY_SENT = "ingestion_task_sentiment"


def register_ingestion_tasks(app: "FastAPI") -> None:
    """
    Spawn both background ingestion tasks and attach them to ``app.state``.

    Call this *inside* the FastAPI lifespan context manager, after the
    database connections have been verified:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await verify_all_connections()
            register_ingestion_tasks(app)   # ← here
            yield
            await stop_ingestion_tasks(app)
            await close_all_connections()

    Parameters
    ----------
    app : FastAPI
        The running FastAPI application instance.
    """
    loop = asyncio.get_event_loop()

    ais_task  = loop.create_task(_ais_ingestion_loop(),       name="ais_ingestion")
    sent_task = loop.create_task(_sentiment_ingestion_loop(), name="sentiment_ingestion")

    setattr(app.state, _TASK_KEY_AIS,  ais_task)
    setattr(app.state, _TASK_KEY_SENT, sent_task)

    logger.info("Ingestion tasks registered: ais_ingestion, sentiment_ingestion")


async def stop_ingestion_tasks(app: "FastAPI") -> None:
    """
    Gracefully cancel both ingestion tasks and wait for them to finish.

    Call this during the FastAPI lifespan shutdown (after the ``yield``):

        await stop_ingestion_tasks(app)
    """
    for key in (_TASK_KEY_AIS, _TASK_KEY_SENT):
        task: asyncio.Task | None = getattr(app.state, key, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("Ingestion task cancelled: %s", task.get_name())


# ── Diagnostic helpers (useful from the REPL / test suite) ───────────────────

async def emit_one_ais_batch(burst_size: int = INGESTION_BURST_SIZE) -> list[AISPositionReport]:
    """
    Emit a single AIS burst without waiting for the tick interval.
    Useful for integration tests and manual inspection.

        reports = asyncio.run(emit_one_ais_batch(burst_size=5))
        for r in reports: print(r)
    """
    reports = _generate_ais_reports(tick_elapsed_hours=0.25, burst_size=burst_size)
    await _persist_ais_reports(reports)
    return reports


async def emit_one_sentiment_batch(burst_size: int = INGESTION_BURST_SIZE) -> list[dict]:
    """
    Emit a single sentiment burst without waiting for the tick interval.

        rows = asyncio.run(emit_one_sentiment_batch(burst_size=5))
        for r in rows: print(r["raw_text"], r["risk_score"])
    """
    rows = _generate_sentiment_events(burst_size=burst_size)
    await _persist_sentiment_events(rows)
    return rows


# ── Integration patch for existing main.py (lifespan drop-in) ────────────────

LIFESPAN_PATCH = '''
# ── Paste this into main.py to activate the ingestion pipeline ────────────────
#
# 1. Add the import at the top of main.py:
#
#    from data_ingestion import register_ingestion_tasks, stop_ingestion_tasks
#
# 2. Replace the existing lifespan function with this one:

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SentinelFlow API — starting up")

    connection_report = await verify_all_connections()
    app.state.connection_report = connection_report

    # ── Start background ingestion ──
    register_ingestion_tasks(app)

    logger.info("All systems operational. Accepting requests.")
    yield

    # ── Shutdown ──
    await stop_ingestion_tasks(app)
    await close_all_connections()
    logger.info("Shutdown complete.")
'''


# ── Standalone runner (python data_ingestion.py) ──────────────────────────────

if __name__ == "__main__":
    """
    Run both ingestion loops standalone (no FastAPI required).
    Useful for local development when you only want to populate the DB.

        python data_ingestion.py
        python data_ingestion.py --once     # emit exactly one burst then exit
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    once_mode = "--once" in sys.argv

    async def _main() -> None:
        if once_mode:
            logger.info("One-shot mode — emitting a single burst from each stream.")
            ais_reports = await emit_one_ais_batch(burst_size=5)
            sent_rows   = await emit_one_sentiment_batch(burst_size=5)
            logger.info(
                "Done. AIS records: %d  Sentiment records: %d",
                len([r for r in ais_reports if r.port_id]),
                len(sent_rows),
            )
        else:
            logger.info("Continuous mode — Ctrl-C to stop.")
            ais_task  = asyncio.create_task(_ais_ingestion_loop())
            sent_task = asyncio.create_task(_sentiment_ingestion_loop())
            try:
                await asyncio.gather(ais_task, sent_task)
            except (KeyboardInterrupt, asyncio.CancelledError):
                ais_task.cancel()
                sent_task.cancel()
                logger.info("Ingestion stopped.")

    asyncio.run(_main())
