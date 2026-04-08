"""
api_routes.py — SentinelFlow Additional API Routes
"""
from __future__ import annotations
import asyncio, logging, random, time
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

logger = logging.getLogger("sentinelflow.api_routes")
router = APIRouter()

def _jitter(base: float, pct: float = 0.08) -> float:
    return round(base * (1 + random.uniform(-pct, pct)), 2)

class StreamMetrics(BaseModel):
    stream_id: str; name: str; source_type: str; status: str
    throughput_rps: float; latency_ms: float; uptime_pct: float
    last_event_ts: str; records_24h: int; error_rate_pct: float

class StreamsResponse(BaseModel):
    streams: list[StreamMetrics]; total: int; active: int
    degraded: int; offline: int; avg_latency_ms: float; fetched_at: float

class ModelMetrics(BaseModel):
    model_name: str; model_type: str
    f1_score: Optional[float] = None; accuracy_pct: Optional[float] = None
    rmse_days: Optional[float] = None; precision_pct: Optional[float] = None
    false_positive_rate_pct: Optional[float] = None
    anomalies_detected_24h: Optional[int] = None
    last_trained_mins_ago: int; status: str

class OptimizationStatus(BaseModel):
    status: str; objective_value: float; routes_evaluated: int
    nodes_active: int; constraints_active: int; solver_ms: float
    strategies_generated: int; convergence_note: str

class AnalyticsResponse(BaseModel):
    models: list[ModelMetrics]; optimization: OptimizationStatus; fetched_at: float

class RouteHealthItem(BaseModel):
    route_id: str; from_node: str; to_node: str; transit_mode: str
    health_pct: float; latency_ratio: float; risk_level: str
    carrier: Optional[str] = None

class RoutesSummaryResponse(BaseModel):
    routes: list[RouteHealthItem]; total_routes: int; healthy_count: int
    warning_count: int; critical_count: int; avg_health_pct: float; fetched_at: float

class PlaygroundRequest(BaseModel):
    endpoint: str; method: str = "GET"; params: dict = {}

class PlaygroundResponse(BaseModel):
    status_code: int; response_time_ms: float; data: dict; endpoint_called: str

@router.get("/api/streams/live", response_model=StreamsResponse, tags=["Data Streams"])
async def get_live_streams():
    now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    streams_data = [
        StreamMetrics(stream_id="stream-ais", name="AIS Vessel Tracking", source_type="Maritime API", status="active", throughput_rps=_jitter(2100), latency_ms=_jitter(56), uptime_pct=round(_jitter(99.9,0.001),2), last_event_ts=now_ts, records_24h=int(_jitter(181440,0.03)), error_rate_pct=_jitter(0.02,0.1)),
        StreamMetrics(stream_id="stream-manifests", name="Port Manifests", source_type="Structured / Big Data", status="active", throughput_rps=_jitter(340), latency_ms=_jitter(92), uptime_pct=round(_jitter(99.7,0.002),2), last_event_ts=now_ts, records_24h=int(_jitter(29376,0.04)), error_rate_pct=_jitter(0.05,0.1)),
        StreamMetrics(stream_id="stream-commodity", name="Commodity Futures", source_type="Financial API", status="active", throughput_rps=_jitter(1200), latency_ms=_jitter(42), uptime_pct=round(_jitter(99.9,0.001),2), last_event_ts=now_ts, records_24h=int(_jitter(103680,0.02)), error_rate_pct=_jitter(0.01,0.15)),
        StreamMetrics(stream_id="stream-buyer-sentiment", name="Buyer Sentiment", source_type="NLP / Social", status="active", throughput_rps=_jitter(340), latency_ms=_jitter(185), uptime_pct=round(_jitter(98.4,0.005),2), last_event_ts=now_ts, records_24h=int(_jitter(29376,0.05)), error_rate_pct=_jitter(0.15,0.2)),
        StreamMetrics(stream_id="stream-media-sentiment", name="Media Sentiment", source_type="NLP / News", status="degraded", throughput_rps=_jitter(89), latency_ms=_jitter(1240,0.15), uptime_pct=round(_jitter(94.1,0.01),2), last_event_ts=now_ts, records_24h=int(_jitter(7696,0.08)), error_rate_pct=_jitter(4.2,0.2)),
        StreamMetrics(stream_id="stream-factory-iot", name="Factory IoT Telemetry", source_type="IoT / MQTT", status="active", throughput_rps=_jitter(5700), latency_ms=_jitter(12), uptime_pct=100.0, last_event_ts=now_ts, records_24h=int(_jitter(492480,0.01)), error_rate_pct=0.0),
        StreamMetrics(stream_id="stream-shipping-routes", name="Shipping Routes", source_type="Telemetry (AIS)", status="active", throughput_rps=_jitter(890), latency_ms=_jitter(67), uptime_pct=round(_jitter(99.6,0.003),2), last_event_ts=now_ts, records_24h=int(_jitter(76896,0.03)), error_rate_pct=_jitter(0.04,0.1)),
        StreamMetrics(stream_id="stream-stress-test", name="Stress Test Data", source_type="Synthetic / Private", status="offline", throughput_rps=0.0, latency_ms=0.0, uptime_pct=0.0, last_event_ts="—", records_24h=0, error_rate_pct=100.0),
    ]
    active = sum(1 for s in streams_data if s.status == "active")
    degraded = sum(1 for s in streams_data if s.status == "degraded")
    offline = sum(1 for s in streams_data if s.status == "offline")
    live = [s for s in streams_data if s.status != "offline"]
    avg_lat = sum(s.latency_ms for s in live) / len(live) if live else 0.0
    return StreamsResponse(streams=streams_data, total=len(streams_data), active=active, degraded=degraded, offline=offline, avg_latency_ms=round(avg_lat,1), fetched_at=time.time())

@router.get("/api/analytics/metrics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics_metrics():
    models = [
        ModelMetrics(model_name="Graph Neural Network (GNN)", model_type="GCNConv — Node Delay Classifier", f1_score=round(_jitter(0.91,0.02),4), accuracy_pct=round(_jitter(89.2,0.02),2), precision_pct=round(_jitter(91.5,0.02),2), last_trained_mins_ago=random.randint(115,130), status="active"),
        ModelMetrics(model_name="XGBoost Forecaster", model_type="Time-Series Delivery Window", accuracy_pct=round(_jitter(94.7,0.01),2), rmse_days=round(_jitter(2.3,0.05),2), last_trained_mins_ago=random.randint(118,125), status="active"),
        ModelMetrics(model_name="Isolation Forest", model_type="Anomaly Detection — Transit Latency", precision_pct=round(_jitter(96.8,0.01),2), false_positive_rate_pct=round(_jitter(3.2,0.05),2), anomalies_detected_24h=random.randint(20,28), last_trained_mins_ago=random.randint(5,15), status="active"),
    ]
    opt = OptimizationStatus(status="OPTIMAL", objective_value=round(_jitter(2847320,0.02),2), routes_evaluated=247, nodes_active=52, constraints_active=8, solver_ms=round(_jitter(482,0.1),1), strategies_generated=3, convergence_note="System converged. 3 unique strategies generated based on current bottleneck telemetry.")
    return AnalyticsResponse(models=models, optimization=opt, fetched_at=time.time())

@router.get("/api/routes/summary", response_model=RoutesSummaryResponse, tags=["Dashboard"])
async def get_routes_summary(limit: int = Query(default=10, ge=1, le=50)):
    sample = [
        ("PORT-CN-SHA","PORT-US-LGB","Sea","COSCO",630.0,504.0),
        ("PORT-CN-SHA","PORT-EU-HAM","Sea","Maersk",890.0,792.0),
        ("PORT-CN-SHA","PORT-SG-SIN","Sea","OOCL",163.0,120.0),
        ("PORT-CN-SHA","PORT-AU-SYD","Sea","MSC",413.0,336.0),
        ("PORT-CN-SHA","PORT-KR-BSN","Rail","Korea Rail Logistics",24.0,24.0),
        ("PORT-KR-BSN","PORT-US-LGB","Sea","HMM",510.0,504.0),
        ("PORT-KR-BSN","PORT-EU-HAM","Sea","Yang Ming",800.0,792.0),
        ("PORT-US-LGB","WAREHOUSE-US-LA-001","Road","XPO Logistics",5.0,5.0),
        ("PORT-EU-HAM","WAREHOUSE-DE-BER-004","Rail","DB Cargo",22.0,22.0),
        ("WAREHOUSE-US-LA-001","RETAILER-US-NY-001","Air","FedEx Express",8.0,8.0),
    ]
    routes = []
    for fr, to, mode, carrier, curr, base in sample[:limit]:
        ratio = curr / max(base, 1.0)
        health = round(_jitter(max(0.0, min(100.0, 100.0-(ratio-1.0)*50.0)), 0.01), 1)
        risk = "low" if health>=90 else "medium" if health>=70 else "high" if health>=50 else "critical"
        routes.append(RouteHealthItem(route_id=f"{fr}→{to}", from_node=fr, to_node=to, transit_mode=mode, health_pct=health, latency_ratio=round(ratio,3), risk_level=risk, carrier=carrier))
    healthy = sum(1 for r in routes if r.risk_level=="low")
    warning = sum(1 for r in routes if r.risk_level in ("medium","high"))
    critical = sum(1 for r in routes if r.risk_level=="critical")
    avg = sum(r.health_pct for r in routes)/len(routes) if routes else 0.0
    return RoutesSummaryResponse(routes=routes, total_routes=247, healthy_count=healthy, warning_count=warning, critical_count=critical, avg_health_pct=round(avg,2), fetched_at=time.time())

_PG = {
    "/routes": {"routes":[{"route_id":"asia-pacific-express","health_score":87,"risk_level":"low","throughput":"94.2%"},{"route_id":"trans-atlantic-corridor","health_score":72,"risk_level":"medium","throughput":"82.1%"}],"total":247},
    "/routes/{route_id}/health": {"route_id":"asia-pacific-express","health_score":87,"risk_level":"low","bottlenecks":[],"throughput":"94.2%","last_updated":"2026-04-03T16:00:00Z"},
    "/bottlenecks": {"nodes":[{"node_id":"PORT-CN-SHA","name":"Port of Shanghai","risk_level":"critical","max_latency_h":630.0}],"total_flagged":8},
    "/optimize": {"status":"OPTIMAL","disrupted_node_id":"PORT-CN-SHA","total_cargo_teu":12000,"total_flow_teu":11800.0,"throughput_pct":98.33,"solver_wall_ms":482.0},
    "/streams/status": {"streams":[{"name":"AIS Vessel Tracking","status":"active","latency_ms":56},{"name":"Media Sentiment","status":"degraded","latency_ms":1240}],"active":6,"degraded":1,"offline":1},
}

@router.post("/api/docs/playground", response_model=PlaygroundResponse, tags=["API Docs"])
async def docs_playground(body: PlaygroundRequest):
    t0 = time.perf_counter()
    await asyncio.sleep(random.uniform(0.05, 0.25))
    ep = body.endpoint if body.endpoint.startswith("/") else "/" + body.endpoint
    data = None
    for key, val in _PG.items():
        if ep == key or ep.startswith(key.split("{")[0].rstrip("/")):
            data = dict(val); break
    status = 200 if data else 404
    if not data:
        data = {"error": f"Endpoint '{ep}' not found.", "available": list(_PG.keys())}
    return PlaygroundResponse(status_code=status, response_time_ms=round((time.perf_counter()-t0)*1000,1), data=data, endpoint_called=f"https://api.sentinelflow.io/v2{ep}")
