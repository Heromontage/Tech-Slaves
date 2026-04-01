"""
schemas.py — SentinelFlow Pydantic v2 request / response schemas
================================================================
Every table in models.py has a matching schema family:

  <Model>Create   — validated inbound payload for POST endpoints
  <Model>Update   — partial validated payload for PATCH endpoints (all fields optional)
  <Model>Response — outbound shape returned from GET/POST responses
  <Model>Page     — paginated list wrapper

Design rules
------------
* All timestamps are serialised as ISO-8601 strings with UTC offset ("Z" suffix).
* UUIDs are serialised as plain strings for JSON-friendly consumption.
* Numeric fields coming from SQLAlchemy Numeric columns are coerced to float
  in the response schemas; the raw Decimal is not JSON-serialisable.
* `from_attributes = True` on every Response model so Pydantic can build them
  directly from ORM instances without an explicit .model_validate() call.
* Strict validators (Annotated constraints) are applied at the field level to
  catch bad data before it reaches the database layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Generic, List, Optional, TypeVar

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from models import SentimentSourceType


# ── Shared config ─────────────────────────────────────────────────────────────

class _BaseSchema(BaseModel):
    """
    Shared Pydantic config inherited by every schema in this module.
    • populate_by_name  — allows both alias and field name as input keys
    • str_strip_whitespace — trims accidental leading/trailing spaces on strings
    """
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class _BaseResponse(_BaseSchema):
    """Extended config for outbound schemas that are built from ORM instances."""
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        from_attributes=True,          # enables .model_validate(orm_instance)
    )


# ── Pagination wrapper ────────────────────────────────────────────────────────

T = TypeVar("T")


class Page(_BaseSchema, Generic[T]):
    """
    Generic paginated response wrapper.

    Usage in a route:

        return Page[PortManifestResponse](
            items=results,
            total=total_count,
            page=page,
            page_size=page_size,
        )
    """
    items: List[T]
    total: int = Field(..., ge=0, description="Total records matching the query.")
    page: int  = Field(1,  ge=1, description="Current page number (1-indexed).")
    page_size: int = Field(50, ge=1, le=500, description="Records per page.")

    @property
    def total_pages(self) -> int:
        if self.page_size == 0:
            return 0
        return max(1, -(-self.total // self.page_size))  # ceiling division


# ── Annotated field types (reusable constraints) ──────────────────────────────

# TEU volume: non-negative integer, realistic upper bound (world's largest port
# handles ~50 M TEU / year ≈ 137 K TEU / day; 10 M gives a generous snapshot cap)
TEUVolume = Annotated[int, Field(ge=0, le=10_000_000)]

# Dwell time in hours: 0 h to 90 days
DwellTimeHours = Annotated[float, Field(ge=0.0, le=2160.0)]

# Capacity percentage: 0–200 % (>100 means over-capacity)
CapacityPct = Annotated[Optional[float], Field(default=None, ge=0.0, le=200.0)]

# Production units: positive integers up to 1 billion
ProductionUnits = Annotated[int, Field(ge=0, le=1_000_000_000)]

# Normalised score in [0, 1]
NormalisedScore = Annotated[float, Field(ge=0.0, le=1.0)]

# Shift duration: 15 minutes to 30 days
ShiftHours = Annotated[Optional[float], Field(default=None, ge=0.25, le=720.0)]

# ISO 639-1 language code
LanguageCode = Annotated[Optional[str], Field(default=None, min_length=2, max_length=8)]

# UN/LOCODE (5 chars) or IATA (3 chars); allow up to 10 for extended codes
PortCode = Annotated[Optional[str], Field(default=None, min_length=2, max_length=10)]


# ═════════════════════════════════════════════════════════════════════════════
# PortManifest schemas
# ═════════════════════════════════════════════════════════════════════════════

class PortManifestCreate(_BaseSchema):
    """
    Validated payload for POST /v2/port-manifests.
    Sent by the AIS / port-authority ingestion worker.
    """

    port_name: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Human-readable port name, e.g. 'Port of Shanghai'.",
        examples=["Port of Shanghai", "Port of Los Angeles"],
    )
    port_code: PortCode = Field(
        default=None,
        description="UN/LOCODE or IATA code for graph node matching, e.g. 'CNSHA'.",
        examples=["CNSHA", "USLAX"],
    )
    container_volume: TEUVolume = Field(
        ...,
        description="Snapshot TEU count at the port at the time of the manifest.",
        examples=[12_400, 87_300],
    )
    dwell_time: DwellTimeHours = Field(
        ...,
        description="Average container dwell time in hours.",
        examples=[18.5, 72.0],
    )
    capacity_pct: CapacityPct = Field(
        default=None,
        description="(container_volume / port design capacity) × 100. Omit if unknown.",
        examples=[92.3, 107.8],
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the port manifest was recorded.",
        examples=["2026-03-31T16:00:00Z"],
    )

    @field_validator("port_code")
    @classmethod
    def normalise_port_code(cls, v: str | None) -> str | None:
        """Store UN/LOCODE in uppercase; strip whitespace."""
        return v.strip().upper() if v else None


class PortManifestUpdate(_BaseSchema):
    """
    Partial payload for PATCH /v2/port-manifests/{id}.
    Every field is optional — only supplied fields are updated.
    """
    port_name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    port_code: PortCode = None
    container_volume: Optional[TEUVolume] = None
    dwell_time: Optional[DwellTimeHours] = None
    capacity_pct: CapacityPct = None
    timestamp: Optional[datetime] = None

    @field_validator("port_code")
    @classmethod
    def normalise_port_code(cls, v: str | None) -> str | None:
        return v.strip().upper() if v else None


class PortManifestResponse(_BaseResponse):
    """
    Outbound shape for GET /v2/port-manifests and POST /v2/port-manifests.
    Built directly from a PortManifest ORM instance.
    """
    id: uuid.UUID
    port_name: str
    port_code: Optional[str]
    container_volume: int
    dwell_time: float
    capacity_pct: Optional[float]
    timestamp: datetime
    created_at: datetime
    deleted_at: Optional[datetime]

    # Derived field computed at serialisation time — not stored in DB
    is_over_capacity: bool = Field(
        default=False,
        description="True when capacity_pct exceeds 100 %.",
    )

    @model_validator(mode="after")
    def compute_derived(self) -> "PortManifestResponse":
        self.is_over_capacity = (
            self.capacity_pct is not None and self.capacity_pct > 100.0
        )
        return self


class PortManifestPage(Page[PortManifestResponse]):
    """Paginated list of port manifests."""
    pass


# ═════════════════════════════════════════════════════════════════════════════
# FactoryOutput schemas
# ═════════════════════════════════════════════════════════════════════════════

class FactoryOutputCreate(_BaseSchema):
    """
    Validated payload for POST /v2/factory-outputs.
    Typically sent by the Factory IoT MQTT → HTTP bridge worker.
    """

    factory_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Stable factory identifier matching the MQTT client_id.",
        examples=["FACTORY-CN-GZ-001", "FACTORY-VN-HCM-007"],
    )
    factory_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional human-readable label set at device registration.",
        examples=["Guangzhou Assembly Plant #1"],
    )
    region: Optional[str] = Field(
        default=None,
        max_length=80,
        description="Geographic region tag, e.g. 'APAC', 'EMEA', 'AMER'.",
        examples=["APAC", "EMEA"],
    )
    capacity_scheduled: ProductionUnits = Field(
        ...,
        gt=0,
        description="Planned production units for this interval. Must be > 0.",
        examples=[10_000, 250_000],
    )
    actual_output: ProductionUnits = Field(
        ...,
        description="Units actually produced. May exceed capacity_scheduled on OT runs.",
        examples=[9_850, 261_000],
    )
    shift_duration_hrs: ShiftHours = Field(
        default=None,
        description="Length of the production interval in hours.",
        examples=[8.0, 12.5],
    )
    timestamp: datetime = Field(
        ...,
        description="UTC start-of-shift / interval timestamp.",
        examples=["2026-03-31T08:00:00Z"],
    )

    @model_validator(mode="after")
    def compute_utilisation(self) -> "FactoryOutputCreate":
        """
        Attach utilisation_pct for downstream use; route handlers will write it
        to the DB column so queries avoid repeated division.
        """
        self._utilisation_pct: float = round(
            (self.actual_output / self.capacity_scheduled) * 100, 2
        )
        return self

    @property
    def utilisation_pct(self) -> float:
        return self._utilisation_pct


class FactoryOutputUpdate(_BaseSchema):
    """Partial payload for PATCH /v2/factory-outputs/{id}."""
    factory_name: Optional[str] = Field(default=None, max_length=200)
    region: Optional[str] = Field(default=None, max_length=80)
    capacity_scheduled: Optional[int] = Field(default=None, gt=0)
    actual_output: Optional[int] = Field(default=None, ge=0)
    shift_duration_hrs: ShiftHours = None
    timestamp: Optional[datetime] = None


class FactoryOutputResponse(_BaseResponse):
    """
    Outbound shape for GET /v2/factory-outputs and POST /v2/factory-outputs.
    """
    id: uuid.UUID
    factory_id: str
    factory_name: Optional[str]
    region: Optional[str]
    capacity_scheduled: int
    actual_output: int
    utilisation_pct: Optional[float]
    shift_duration_hrs: Optional[float]
    timestamp: datetime
    created_at: datetime
    deleted_at: Optional[datetime]

    # Human-readable utilisation label for dashboard display
    utilisation_label: str = Field(
        default="",
        description="Bucketed label derived from utilisation_pct: under | nominal | over.",
    )

    @model_validator(mode="after")
    def compute_label(self) -> "FactoryOutputResponse":
        pct = self.utilisation_pct
        if pct is None:
            self.utilisation_label = "unknown"
        elif pct < 70:
            self.utilisation_label = "under"
        elif pct <= 100:
            self.utilisation_label = "nominal"
        else:
            self.utilisation_label = "over"
        return self


class FactoryOutputPage(Page[FactoryOutputResponse]):
    """Paginated list of factory output records."""
    pass


# ═════════════════════════════════════════════════════════════════════════════
# SentimentData schemas
# ═════════════════════════════════════════════════════════════════════════════

# Maximum raw_text length enforced at the API boundary (8 096 chars ≈ 8 KB)
_RAW_TEXT_MAX_LEN = 8_096


class SentimentDataCreate(_BaseSchema):
    """
    Validated payload for POST /v2/sentiment.
    Sent by the DistilBERT (news) and VADER (social) NLP ingestion workers.
    """

    source_type: SentimentSourceType = Field(
        ...,
        description="Origin stream: 'news' for wire-service articles, 'social' for posts.",
        examples=["news", "social"],
    )
    source_url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Source URL or MQTT topic. Omit for aggregated / privacy-scrubbed data.",
        examples=["https://reuters.com/article/port-shanghai-congestion"],
    )
    language: LanguageCode = Field(
        default=None,
        description="ISO 639-1 language code of raw_text.",
        examples=["en", "zh", "ja"],
    )
    risk_score: NormalisedScore = Field(
        ...,
        description="Normalised supply-chain risk score in [0.0, 1.0]. 1.0 = maximum risk.",
        examples=[0.82, 0.14],
    )
    confidence: Optional[NormalisedScore] = Field(
        default=None,
        description="Model classification confidence in [0.0, 1.0]. Omit if unavailable.",
        examples=[0.94, 0.67],
    )
    raw_text: str = Field(
        ...,
        min_length=1,
        max_length=_RAW_TEXT_MAX_LEN,
        description=(
            "PII-scrubbed article excerpt or social post. "
            f"Hard limit: {_RAW_TEXT_MAX_LEN} characters."
        ),
        examples=["Port of Shanghai facing severe congestion as vessel backlog hits 92%…"],
    )
    entities: Optional[Any] = Field(
        default=None,
        description=(
            "Named entities extracted by the NLP pipeline. "
            "Pass as a JSON-serialisable list; stored as TEXT in PostgreSQL."
        ),
        examples=[["Port of Shanghai", "Evergreen Marine", "CNSHA"]],
    )
    timestamp: datetime = Field(
        ...,
        description="UTC publication / post time of the source document.",
        examples=["2026-03-31T14:22:00Z"],
    )

    @field_validator("raw_text")
    @classmethod
    def scrub_pii_placeholder(cls, v: str) -> str:
        """
        Placeholder hook: in production this would call a PII-scrubbing service.
        For now, strip null bytes that would break PostgreSQL TEXT storage.
        """
        return v.replace("\x00", "")

    @field_validator("language")
    @classmethod
    def normalise_language(cls, v: str | None) -> str | None:
        return v.strip().lower() if v else None


class SentimentDataUpdate(_BaseSchema):
    """Partial payload for PATCH /v2/sentiment/{id}."""
    risk_score: Optional[NormalisedScore] = None
    confidence: Optional[NormalisedScore] = None
    entities: Optional[Any] = None


class SentimentDataResponse(_BaseResponse):
    """
    Outbound shape for GET /v2/sentiment and POST /v2/sentiment.
    raw_text is truncated to 500 chars in list responses to keep payloads small;
    the full text is returned on single-record GET /v2/sentiment/{id}.
    """
    id: uuid.UUID
    source_type: SentimentSourceType
    source_url: Optional[str]
    language: Optional[str]
    risk_score: float
    confidence: Optional[float]
    raw_text: str
    entities: Optional[str]      # raw JSON string as stored in DB
    timestamp: datetime
    created_at: datetime
    deleted_at: Optional[datetime]

    # Derived: human-readable severity tier for the dashboard
    risk_tier: str = Field(
        default="",
        description="low | medium | high | critical — derived from risk_score.",
    )

    @model_validator(mode="after")
    def compute_risk_tier(self) -> "SentimentDataResponse":
        score = self.risk_score
        if score < 0.25:
            self.risk_tier = "low"
        elif score < 0.50:
            self.risk_tier = "medium"
        elif score < 0.75:
            self.risk_tier = "high"
        else:
            self.risk_tier = "critical"
        return self


class SentimentDataListResponse(_BaseResponse):
    """
    Truncated variant used in paginated list responses.
    Keeps payloads lightweight by capping raw_text at 500 chars.
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    source_type: SentimentSourceType
    language: Optional[str]
    risk_score: float
    risk_tier: str = ""
    timestamp: datetime

    # Truncated preview only
    raw_text_preview: str = Field(
        default="",
        description="First 500 characters of raw_text for list display.",
    )

    @model_validator(mode="after")
    def compute_fields(self) -> "SentimentDataListResponse":
        if self.risk_score < 0.25:
            self.risk_tier = "low"
        elif self.risk_score < 0.50:
            self.risk_tier = "medium"
        elif self.risk_score < 0.75:
            self.risk_tier = "high"
        else:
            self.risk_tier = "critical"
        return self


class SentimentDataPage(Page[SentimentDataListResponse]):
    """Paginated list of sentiment records (truncated view)."""
    pass


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # Shared
    "Page",
    # PortManifest
    "PortManifestCreate",
    "PortManifestUpdate",
    "PortManifestResponse",
    "PortManifestPage",
    # FactoryOutput
    "FactoryOutputCreate",
    "FactoryOutputUpdate",
    "FactoryOutputResponse",
    "FactoryOutputPage",
    # SentimentData
    "SentimentDataCreate",
    "SentimentDataUpdate",
    "SentimentDataResponse",
    "SentimentDataListResponse",
    "SentimentDataPage",
]
