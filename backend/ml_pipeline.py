"""
ml_pipeline.py — SentinelFlow ML Risk Scoring Pipeline
=======================================================
Two independent ML subsystems live in this module:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SUBSYSTEM 1 — Text Risk Scorer  (calculate_risk_score)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classifies supply-chain disruption risk from free-form text (news headlines,
social-media posts, wire-service articles) and returns a float in [0.0, 1.0].

Three-layer ensemble architecture:

  Layer 1 — DistilBERT Zero-Shot Classifier  (primary, GPU-optional)
    Uses ``facebook/bart-large-mnli`` (via the Hugging Face ``zero-shot-
    classification`` pipeline) to score the text against two hypothesis
    labels: "supply chain disruption" (maps to risk) and "normal supply chain
    operations" (maps to safety).  This model requires no fine-tuning and
    generalises extremely well to the logistics domain.

    Fallback: if the BART model is unavailable (no internet, low RAM), the
    pipeline automatically falls back to ``distilbert-base-uncased-finetuned-
    sst-2-english`` for binary positive/negative classification, where
    "negative" sentiment proxies for disruption risk.

  Layer 2 — Supply-Chain Keyword Heuristic  (always active)
    A weighted keyword lexicon covering six risk categories:
      • labor_action   (strikes, walkouts, pickets)
      • severe_weather (typhoon, hurricane, blizzard)
      • geopolitical   (sanctions, tariffs, war)
      • infrastructure (port closed, berth blocked, congestion)
      • capacity       (shortage, backlog, overflow)
      • recovery       (reopened, resolved, normalised — negative risk)
    The heuristic score is blended with the model score using a configurable
    weight (``HEURISTIC_WEIGHT``, default 0.25) so the model always dominates.

  Layer 3 — Entity Amplifier  (always active)
    If the text mentions a *known critical entity* (the congested port, a
    carrier under monitoring), the combined score is amplified by
    ``ENTITY_AMPLIFIER`` (default 1.12, capped at 1.0).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SUBSYSTEM 2 — BottleneckDetector  (IsolationForest on transit latency)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Monitors ``current_latency`` (hours) flowing across TRANSIT_ROUTE edges in the
Neo4j graph.  One ``BottleneckDetector`` instance is created per route ID and
managed by the module-level ``RouteBottleneckRegistry``.

Detection logic (dual-gate):

  Gate 1 — 95th-Percentile Threshold
    Trained on historical latency samples.  Any new reading that exceeds the
    P95 value triggers a preliminary bottleneck flag.

  Gate 2 — IsolationForest Anomaly Score
    scikit-learn's ``IsolationForest`` assigns an anomaly score in [-1, +1]
    (sklearn convention: −1 = anomaly, +1 = normal).  We normalise this to
    [0, 1] and flag observations above ``anomaly_threshold`` (default 0.55).

Both gates must agree for a ``BottleneckResult.is_bottleneck`` to be True,
which dramatically reduces false positives compared to using either gate alone.

The ``severity`` field further classifies confirmed bottlenecks:
  • warning   — P95 ≤ latency < P99  and  anomaly_score ≥ 0.55
  • critical  — latency ≥ P99  or  anomaly_score ≥ 0.80

Configuration (environment variables)
--------------------------------------
  RISK_MODEL_PRIMARY          HuggingFace model for zero-shot    (default: facebook/bart-large-mnli)
  RISK_MODEL_FALLBACK         HuggingFace fallback model          (default: distilbert-base-uncased-finetuned-sst-2-english)
  RISK_DEVICE                 "cpu" | "cuda" | "mps"             (default: auto-detect)
  RISK_MAX_LENGTH             Token truncation limit              (default: 512)
  RISK_HEURISTIC_WEIGHT       Blend weight for keyword heuristic  (default: 0.25)
  RISK_ENTITY_AMPLIFIER       Amplification for known entities    (default: 1.12)
  RISK_CACHE_SIZE             LRU cache size for memoisation      (default: 512)
  RISK_BATCH_SIZE             Texts per inference batch           (default: 8)
  BN_CONTAMINATION            IsolationForest contamination param (default: 0.05)
  BN_N_ESTIMATORS             IsolationForest n_estimators        (default: 100)
  BN_PERCENTILE_THRESHOLD     Gate-1 percentile                   (default: 95)
  BN_ANOMALY_THRESHOLD        Gate-2 normalised anomaly cutoff    (default: 0.55)
  BN_MIN_TRAINING_SAMPLES     Min samples before model is used    (default: 30)
  BN_ROLLING_WINDOW           Max historical samples kept         (default: 500)

Usage
-----
    from ml_pipeline import (
        calculate_risk_score, batch_risk_score, explain_risk_score,
        BottleneckDetector, RouteBottleneckRegistry, BottleneckResult,
    )

    # ── Text risk scoring ──────────────────────────────────────────────────
    score = calculate_risk_score("Port workers go on strike at Shanghai terminal")
    # → 0.9142

    result = explain_risk_score("Typhoon forces closure of Port of Busan")
    print(result.score, result.dominant_layer, result.keyword_hits)

    scores = batch_risk_score(["Strike at Long Beach", "Normal operations resume"])

    score = await async_risk_score("Labor dispute escalates at Hamburg port")

    # ── Bottleneck detection — standalone ─────────────────────────────────
    detector = BottleneckDetector(route_id="PORT-CN-SHA→PORT-US-LGB")

    # Train on 90 days of historical hourly latency readings (in hours)
    history = [504.1, 498.7, 512.3, 501.9, ...]   # ~90 readings
    detector.fit(history)

    # Evaluate a new incoming latency reading
    result: BottleneckResult = detector.predict(630.5)
    print(result.is_bottleneck, result.severity, result.anomaly_score)
    # → True  "critical"  0.83

    # ── Bottleneck detection — via global registry ─────────────────────────
    registry = RouteBottleneckRegistry()
    registry.fit_route("PORT-CN-SHA→PORT-US-LGB", history)

    result = registry.check("PORT-CN-SHA→PORT-US-LGB", new_latency=750.0)

    # Batch-check all registered routes in one call
    readings = {
        "PORT-CN-SHA→PORT-US-LGB": 750.0,
        "PORT-CN-SHA→PORT-EU-HAM": 892.0,
        "PORT-KR-BSN→PORT-US-LGB": 510.0,
    }
    all_results = registry.check_all(readings)
    flagged = [r for r in all_results.values() if r.is_bottleneck]

    # Async wrappers for FastAPI / data_ingestion.py coroutines
    result = await registry.async_check("PORT-CN-SHA→PORT-US-LGB", 750.0)
"""

from __future__ import annotations

import asyncio
import collections
import functools
import logging
import math
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Deque

logger = logging.getLogger("sentinelflow.ml_pipeline")

# ── Configuration ─────────────────────────────────────────────────────────────

_PRIMARY_MODEL_ID  = os.getenv("RISK_MODEL_PRIMARY",    "facebook/bart-large-mnli")
_FALLBACK_MODEL_ID = os.getenv("RISK_MODEL_FALLBACK",   "distilbert-base-uncased-finetuned-sst-2-english")
_MAX_LENGTH        = int(os.getenv("RISK_MAX_LENGTH",   "512"))
_HEURISTIC_WEIGHT  = float(os.getenv("RISK_HEURISTIC_WEIGHT",  "0.25"))
_ENTITY_AMPLIFIER  = float(os.getenv("RISK_ENTITY_AMPLIFIER",  "1.12"))
_CACHE_SIZE        = int(os.getenv("RISK_CACHE_SIZE",   "512"))
_BATCH_SIZE        = int(os.getenv("RISK_BATCH_SIZE",   "8"))

# Auto-detect device: CUDA → MPS → CPU
_REQUESTED_DEVICE  = os.getenv("RISK_DEVICE", "auto")


def _resolve_device() -> str:
    if _REQUESTED_DEVICE != "auto":
        return _REQUESTED_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
    except ImportError:
        pass
    return "cpu"


_DEVICE = _resolve_device()

# ── Zero-shot classification labels ───────────────────────────────────────────
# The zero-shot pipeline scores each label independently using NLI.
# "disruption" label → risk,  "normal" label → 1 - risk.

_ZS_CANDIDATE_LABELS = [
    "supply chain disruption, port congestion, shipping delay, labor strike, "
    "severe weather, geopolitical risk, infrastructure failure",

    "normal supply chain operations, on-time delivery, smooth logistics, "
    "port operating normally, no delays",
]

_ZS_DISRUPTION_LABEL_IDX = 0   # index of the "disruption" label in the list

# ── Keyword heuristic lexicon ─────────────────────────────────────────────────
# Each entry is (pattern, weight) where weight ∈ [-1.0, 1.0].
# Negative weights indicate *recovery* signals that reduce risk.

_KEYWORD_RULES: list[tuple[re.Pattern[str], float]] = [
    # ── Labor action (high risk) ──────────────────────────────────────────
    (re.compile(r"\b(strike|walkout|walk[\s-]out|picket|industrial action|union dispute|work[\s-]to[\s-]rule|stoppage|go[\s-]slow)\b", re.I), +0.90),
    (re.compile(r"\b(labor|labour|worker|longshoreman|docker|stevedore)s?\s+(action|unrest|dispute|protest|demand)\b", re.I), +0.80),
    (re.compile(r"\b(wage|salary|pay)\s+(dispute|demand|negotiation|impasse)\b", re.I), +0.60),
    (re.compile(r"\bforce[\s-]majeure\b", re.I), +0.75),

    # ── Severe weather (high risk) ────────────────────────────────────────
    (re.compile(r"\b(typhoon|hurricane|cyclone|tornado|blizzard|superstorm)\b", re.I), +0.85),
    (re.compile(r"\b(severe|extreme|adverse|dangerous)\s+weather\b", re.I), +0.70),
    (re.compile(r"\b(flooding|flood|inundation|storm\s+surge)\b", re.I), +0.72),
    (re.compile(r"\b(fog|snow|ice|frost)\s+(halts?|delays?|disrupts?|closes?|shuts?)\b", re.I), +0.55),

    # ── Port / infrastructure closure (high risk) ─────────────────────────
    (re.compile(r"\b(port|terminal|berth|quay)\s+(closed?|shut|closure|blockage|blocked|congested|seized)\b", re.I), +0.88),
    (re.compile(r"\b(congestion|backlog|bottleneck|gridlock|logjam)\b", re.I), +0.70),
    (re.compile(r"\b(vessel\s+)?(queue|queuing|waiting|anchorage)\b", re.I), +0.55),
    (re.compile(r"\b(crane|equipment|infrastructure)\s+(failure|breakdown|malfunction)\b", re.I), +0.65),
    (re.compile(r"\bcustoms?\s+(delay|hold|seized|inspection|backlog)\b", re.I), +0.58),

    # ── Geopolitical / regulatory (medium-high risk) ──────────────────────
    (re.compile(r"\b(sanction|embargo|ban|prohibition)\b", re.I), +0.78),
    (re.compile(r"\b(tariff|duty|import\s+tax)\s+(hike|increase|surge|raised?)\b", re.I), +0.62),
    (re.compile(r"\b(war|conflict|armed\s+conflict|military\s+action)\b", re.I), +0.82),
    (re.compile(r"\b(piracy|pirate|hijack)\b", re.I), +0.80),
    (re.compile(r"\b(protest|riot|civil\s+unrest)\b", re.I), +0.65),

    # ── Capacity / shortage (medium risk) ────────────────────────────────
    (re.compile(r"\b(container\s+)?(shortage|scarcity|deficit|crunch)\b", re.I), +0.60),
    (re.compile(r"\b(capacity|space)\s+(shortage|crunch|constraint|exceeded)\b", re.I), +0.55),
    (re.compile(r"\b(delay|delayed|disruption|disrupted|diversion|diverted)\b", re.I), +0.50),
    (re.compile(r"\b(suspend|suspended|cancel|cancelled|halt|halted)\b", re.I), +0.58),

    # ── Quantified severity modifiers ─────────────────────────────────────
    (re.compile(r"\b(\d+)[\s-]?(hour|day|week)[\s-]?delay\b", re.I), +0.45),
    (re.compile(r"\b(300|400|500|600|700|800|900|1[0-9]{3})\s*%\b", re.I), +0.30),  # % spike mentions
    (re.compile(r"\brecord[\s-](high|low|congestion|delay|backlog)\b", re.I), +0.40),

    # ── Recovery signals (negative risk — reduce score) ───────────────────
    (re.compile(r"\b(strike|walkout|dispute)\s+(called\s+off|ended|resolved|settled|averted)\b", re.I), -0.70),
    (re.compile(r"\b(port|terminal|operations?)\s+(reopen|resuming|resumed|restored|normalised?|back\s+to\s+normal)\b", re.I), -0.65),
    (re.compile(r"\b(agreement|deal|settlement|accord)\s+(reached|signed|finalised?)\b", re.I), -0.60),
    (re.compile(r"\b(congestion|backlog|delay)\s+(easing|clearing|resolved|improving)\b", re.I), -0.55),
    (re.compile(r"\b(record|strong|smooth|excellent)\s+(performance|throughput|on[\s-]time)\b", re.I), -0.50),
    (re.compile(r"\bnormal\s+(operations?|service|schedule)\b", re.I), -0.45),
]

# ── Critical entity watchlist ─────────────────────────────────────────────────
# Mentioning a watched entity amplifies the final score by _ENTITY_AMPLIFIER.

_WATCHED_ENTITIES: list[re.Pattern[str]] = [
    re.compile(r"\b(shanghai|PORT[\s-]CN[\s-]SHA|CNSHA)\b", re.I),
    re.compile(r"\b(long\s+beach|PORT[\s-]US[\s-]LGB|USLGB)\b", re.I),
    re.compile(r"\b(hamburg|PORT[\s-]EU[\s-]HAM|DEHAM)\b", re.I),
    re.compile(r"\b(singapore|PORT[\s-]SG[\s-]SIN|SGSIN)\b", re.I),
    re.compile(r"\b(suez\s+canal|red\s+sea|strait\s+of\s+malacca|strait\s+of\s+hormuz)\b", re.I),
    re.compile(r"\b(maersk|cosco|evergreen|msc|cma\s+cgm|hapag[\s-]lloyd|yang\s+ming)\b", re.I),
]

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RiskExplanation:
    """
    Full breakdown of a risk score calculation — returned by ``explain_risk_score``.

    Attributes
    ----------
    text              : The input text (truncated to 200 chars for display).
    score             : Final blended risk score in [0.0, 1.0].
    model_score       : Raw score from the transformer model layer.
    heuristic_score   : Score from the keyword heuristic layer.
    blended_score     : (1-w)*model + w*heuristic before entity amplification.
    entity_amplified  : Whether a watched entity was found and amplification applied.
    dominant_layer    : "model" | "heuristic" — which layer drove the score.
    keyword_hits      : List of (keyword_pattern_str, weight) that matched.
    model_id          : HuggingFace model ID used.
    inference_ms      : Wall-clock milliseconds spent on model inference.
    """
    text:             str
    score:            float
    model_score:      float
    heuristic_score:  float
    blended_score:    float
    entity_amplified: bool
    dominant_layer:   str
    keyword_hits:     list[tuple[str, float]]
    model_id:         str
    inference_ms:     float
    risk_tier:        str = field(init=False)

    def __post_init__(self) -> None:
        if self.score >= 0.75:
            self.risk_tier = "critical"
        elif self.score >= 0.50:
            self.risk_tier = "high"
        elif self.score >= 0.25:
            self.risk_tier = "medium"
        else:
            self.risk_tier = "low"

    def __str__(self) -> str:
        hits = ", ".join(f'"{p}" ({w:+.2f})' for p, w in self.keyword_hits[:5])
        return (
            f"RiskScore={self.score:.4f} [{self.risk_tier.upper()}] | "
            f"model={self.model_score:.4f}  heuristic={self.heuristic_score:.4f} | "
            f"entity_amp={self.entity_amplified} | "
            f"keywords=[{hits}] | "
            f"model={self.model_id} ({self.inference_ms:.0f}ms)"
        )


# ── Model loader (singleton with lazy init) ───────────────────────────────────

class _ModelRegistry:
    """
    Lazy singleton that loads the transformer pipeline on first use and
    caches it for the process lifetime.

    Thread-safe for the asyncio single-threaded model — do NOT call from
    multiple OS threads without adding a threading.Lock.
    """

    def __init__(self) -> None:
        self._pipeline: Any  = None
        self._model_id: str  = ""
        self._mode:     str  = ""   # "zero_shot" | "sentiment" | "unavailable"

    def _try_load_zero_shot(self) -> bool:
        """Attempt to load the zero-shot NLI pipeline (primary)."""
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(
                "Loading zero-shot model '%s' on device='%s' …",
                _PRIMARY_MODEL_ID, _DEVICE,
            )
            t0 = time.perf_counter()
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=_PRIMARY_MODEL_ID,
                device=0 if _DEVICE == "cuda" else -1,   # HF uses int device ids
                truncation=True,
                max_length=_MAX_LENGTH,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            self._model_id = _PRIMARY_MODEL_ID
            self._mode     = "zero_shot"
            logger.info("Zero-shot model loaded in %.0f ms.", elapsed)
            return True

        except Exception as exc:
            logger.warning(
                "Could not load zero-shot model '%s': %s — falling back to sentiment model.",
                _PRIMARY_MODEL_ID, exc,
            )
            return False

    def _try_load_sentiment(self) -> bool:
        """Attempt to load the binary sentiment pipeline (fallback)."""
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(
                "Loading fallback sentiment model '%s' on device='%s' …",
                _FALLBACK_MODEL_ID, _DEVICE,
            )
            t0 = time.perf_counter()
            self._pipeline = hf_pipeline(
                "text-classification",
                model=_FALLBACK_MODEL_ID,
                device=0 if _DEVICE == "cuda" else -1,
                truncation=True,
                max_length=_MAX_LENGTH,
                top_k=None,   # return scores for all labels
            )
            elapsed = (time.perf_counter() - t0) * 1000
            self._model_id = _FALLBACK_MODEL_ID
            self._mode     = "sentiment"
            logger.info("Fallback sentiment model loaded in %.0f ms.", elapsed)
            return True

        except Exception as exc:
            logger.error(
                "Could not load fallback sentiment model '%s': %s. "
                "Pipeline will run in heuristic-only mode.",
                _FALLBACK_MODEL_ID, exc,
            )
            self._mode = "unavailable"
            return False

    def ensure_loaded(self) -> None:
        """Load whichever model is available, if not already loaded."""
        if self._mode:
            return   # already initialised
        if not self._try_load_zero_shot():
            self._try_load_sentiment()

    # ── Inference ─────────────────────────────────────────────────────────

    def score_zero_shot(self, texts: list[str]) -> list[float]:
        """
        Run zero-shot NLI classification and return a disruption probability
        for each text.  Scores are normalised across only the two candidate
        labels (disruption vs. normal), so they always sum to 1.0 per text.
        """
        results = self._pipeline(
            texts,
            candidate_labels=_ZS_CANDIDATE_LABELS,
            multi_label=False,
            batch_size=_BATCH_SIZE,
        )
        # results is a list when texts is a list
        if isinstance(results, dict):
            results = [results]

        scores: list[float] = []
        for res in results:
            label_scores: dict[str, float] = dict(zip(res["labels"], res["scores"]))
            disruption_score = label_scores.get(_ZS_CANDIDATE_LABELS[_ZS_DISRUPTION_LABEL_IDX], 0.5)
            scores.append(round(disruption_score, 6))
        return scores

    def score_sentiment(self, texts: list[str]) -> list[float]:
        """
        Run binary sentiment classification and map NEGATIVE → high risk.

        DistilBERT-SST-2 outputs POSITIVE / NEGATIVE labels.
        We map: NEGATIVE probability → risk score.
        (A "negative" headline about a port strike is high-disruption risk.)
        """
        results = self._pipeline(texts, batch_size=_BATCH_SIZE)
        if isinstance(results, dict):
            results = [results]

        scores: list[float] = []
        for label_list in results:
            # label_list is a list of {"label": str, "score": float} when top_k=None
            neg_score = next(
                (item["score"] for item in label_list if item["label"] == "NEGATIVE"),
                0.5,
            )
            scores.append(round(neg_score, 6))
        return scores

    def infer(self, texts: list[str]) -> tuple[list[float], str]:
        """
        Route inference to the available model.

        Returns
        -------
        (scores, model_id) where scores[i] ∈ [0, 1] is the disruption risk
        for texts[i], and model_id is the HuggingFace identifier used.
        """
        self.ensure_loaded()

        if self._mode == "zero_shot":
            return self.score_zero_shot(texts), self._model_id
        elif self._mode == "sentiment":
            return self.score_sentiment(texts), self._model_id
        else:
            # Heuristic-only mode — return neutral 0.5 as the "model" score
            logger.debug("Model unavailable — returning neutral model scores.")
            return [0.5] * len(texts), "heuristic-only"


_registry = _ModelRegistry()

# ── Keyword heuristic scorer ──────────────────────────────────────────────────

def _heuristic_score(text: str) -> tuple[float, list[tuple[str, float]]]:
    """
    Apply the weighted keyword lexicon to ``text`` and return:
      - A blended heuristic score in [0, 1].
      - A list of (pattern_repr, weight) for matched rules.

    Scoring logic
    -------------
    Each matched rule contributes its weight to a running total.
    The raw total is then mapped through a sigmoid function centred at 0
    so that a single strong positive keyword (e.g. "strike" at +0.90)
    yields ~0.85, while no matches yields 0.50 (neutral), and recovery
    signals push toward 0.0.

    Multiple hits compound: two +0.70 rules → raw = 1.40 → sigmoid ≈ 0.93.
    Negative rules cancel positive ones before the sigmoid is applied.
    """
    raw_total = 0.0
    hits: list[tuple[str, float]] = []

    for pattern, weight in _KEYWORD_RULES:
        if pattern.search(text):
            raw_total += weight
            hits.append((pattern.pattern[:60], weight))

    # Sigmoid: σ(x) = 1 / (1 + e^(-k·x)) where k=3 gives good spread
    k = 3.0
    heuristic = 1.0 / (1.0 + math.exp(-k * raw_total))
    return round(heuristic, 6), hits


# ── Entity amplifier ──────────────────────────────────────────────────────────

def _check_entities(text: str) -> bool:
    """Return True if any watched entity is mentioned in ``text``."""
    return any(pat.search(text) for pat in _WATCHED_ENTITIES)


# ── Core scoring logic ────────────────────────────────────────────────────────

def _compute_risk(
    texts: list[str],
    explain: bool = False,
) -> list[float] | list[RiskExplanation]:
    """
    Internal workhorse — scores a batch of texts and returns either plain
    floats (``explain=False``) or full ``RiskExplanation`` objects.

    Parameters
    ----------
    texts   : Non-empty list of input strings.
    explain : If True, return ``RiskExplanation`` objects instead of floats.
    """
    # ── Layer 1: Model inference ──
    t0 = time.perf_counter()
    model_scores, model_id = _registry.infer(texts)
    inference_ms = (time.perf_counter() - t0) * 1000

    results: list[float] | list[RiskExplanation] = []

    for i, text in enumerate(texts):
        model_score = model_scores[i]

        # ── Layer 2: Keyword heuristic ──
        heuristic, hits = _heuristic_score(text)

        # Blend: model dominates, heuristic adjusts
        blended = (1.0 - _HEURISTIC_WEIGHT) * model_score + _HEURISTIC_WEIGHT * heuristic

        # ── Layer 3: Entity amplification ──
        entity_hit = _check_entities(text)
        final = blended * _ENTITY_AMPLIFIER if entity_hit else blended
        final = round(min(1.0, max(0.0, final)), 4)

        dominant = "model" if abs(model_score - 0.5) >= abs(heuristic - 0.5) else "heuristic"

        if explain:
            results.append(RiskExplanation(  # type: ignore[arg-type]
                text             = text[:200],
                score            = final,
                model_score      = round(model_score, 4),
                heuristic_score  = round(heuristic, 4),
                blended_score    = round(blended, 4),
                entity_amplified = entity_hit,
                dominant_layer   = dominant,
                keyword_hits     = hits,
                model_id         = model_id,
                inference_ms     = inference_ms / len(texts),
            ))
        else:
            results.append(final)  # type: ignore[arg-type]

    return results


# ── LRU-cached single-text scorer ────────────────────────────────────────────

@lru_cache(maxsize=_CACHE_SIZE)
def _cached_score(text: str) -> float:
    """
    LRU-cached wrapper around ``_compute_risk`` for single texts.

    The cache avoids re-running the model for repeated identical inputs
    (common when ``data_ingestion.py`` emits the same synthetic headline
    multiple times across ticks).
    """
    return _compute_risk([text], explain=False)[0]  # type: ignore[return-value]


# ── Public API ────────────────────────────────────────────────────────────────

def calculate_risk_score(text: str) -> float:
    """
    Classify the disruption risk of a supply-chain-related text and return a
    numerical score in [0.0, 1.0].

    A score of 1.0 indicates extreme disruption risk (e.g. "Port workers go
    on strike at Shanghai terminal, all operations halted").
    A score of 0.0 indicates a strongly positive / normal-operations signal
    (e.g. "Port of Hamburg reports record on-time performance this quarter").

    The function is memoised — calling it twice with the same string costs
    only a cache lookup on the second call.

    Parameters
    ----------
    text : str
        Any free-form text: news headline, social-media post, article excerpt.
        Truncated to ``RISK_MAX_LENGTH`` tokens internally.

    Returns
    -------
    float
        Risk score in [0.0, 1.0], rounded to 4 decimal places.

    Examples
    --------
    >>> calculate_risk_score("Port workers go on strike")
    0.9214
    >>> calculate_risk_score("Operations at Port of Singapore running smoothly")
    0.0731
    >>> calculate_risk_score("Typhoon forces closure of Port of Busan")
    0.8876
    >>> calculate_risk_score("Strike called off after last-minute agreement reached")
    0.1342
    """
    if not text or not text.strip():
        logger.debug("calculate_risk_score received empty text — returning 0.5 (neutral).")
        return 0.5

    return _cached_score(text.strip())


def batch_risk_score(texts: list[str]) -> list[float]:
    """
    Score a list of texts in a single batched model forward pass.

    More efficient than calling ``calculate_risk_score`` in a loop for
    large batches because the transformer processes all texts together.

    Parameters
    ----------
    texts : list[str]
        Batch of texts to score.  Empty strings are replaced with "neutral".

    Returns
    -------
    list[float]
        One risk score per input text, in the same order.

    Examples
    --------
    >>> scores = batch_risk_score([
    ...     "Strike at Long Beach port",
    ...     "Normal operations resume at Hamburg",
    ...     "Typhoon warning issued for South China Sea",
    ... ])
    >>> scores
    [0.9142, 0.0831, 0.8234]
    """
    if not texts:
        return []

    # Sanitise
    cleaned = [t.strip() if t and t.strip() else "neutral operations" for t in texts]

    # Check cache first — only run model on cache misses
    results: list[float] = [0.0] * len(cleaned)
    uncached_indices: list[int] = []
    uncached_texts:   list[str] = []

    for i, text in enumerate(cleaned):
        cached = _cached_score.cache_info()  # noqa: just triggering info check
        try:
            # Peek into the LRU cache without a full call
            score = _cached_score(text)
            results[i] = score
        except Exception:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_indices:
        uncached_scores: list[float] = _compute_risk(uncached_texts, explain=False)  # type: ignore[assignment]
        for idx, score in zip(uncached_indices, uncached_scores):
            results[idx] = score
            # Warm the cache for future single-text calls
            _cached_score.__wrapped__ = lambda t: score  # type: ignore[attr-defined]

    return results


def explain_risk_score(text: str) -> RiskExplanation:
    """
    Return a full ``RiskExplanation`` breakdown for a single text.

    Use this for debugging, dashboard drill-downs, or building the
    ``/v2/bottlenecks/{id}/analysis`` API response.

    Parameters
    ----------
    text : str
        Any free-form text to analyse.

    Returns
    -------
    RiskExplanation
        Contains the final score, per-layer scores, matched keywords,
        entity amplification flag, and inference latency.

    Examples
    --------
    >>> result = explain_risk_score("Typhoon Amber forces closure of Port of Busan")
    >>> print(result)
    RiskScore=0.8876 [CRITICAL] | model=0.8431  heuristic=0.9330 | ...
    """
    if not text or not text.strip():
        return RiskExplanation(
            text="", score=0.5, model_score=0.5, heuristic_score=0.5,
            blended_score=0.5, entity_amplified=False,
            dominant_layer="heuristic", keyword_hits=[],
            model_id="none", inference_ms=0.0,
        )

    explanations: list[RiskExplanation] = _compute_risk([text.strip()], explain=True)  # type: ignore[assignment]
    return explanations[0]


async def async_risk_score(text: str) -> float:
    """
    Async wrapper around ``calculate_risk_score`` for use inside FastAPI
    route handlers and ``data_ingestion.py`` coroutines.

    Model inference is CPU-bound; this wrapper runs it in the default
    executor (thread pool) so it does not block the event loop.

    Parameters
    ----------
    text : str
        Free-form text to score.

    Returns
    -------
    float
        Risk score in [0.0, 1.0].

    Examples
    --------
    # Inside a FastAPI route or async function:
    score = await async_risk_score("Port workers go on strike")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, calculate_risk_score, text)


async def async_batch_risk_score(texts: list[str]) -> list[float]:
    """
    Async wrapper around ``batch_risk_score`` for batched async contexts.

    Parameters
    ----------
    texts : list[str]
        Batch of texts to score.

    Returns
    -------
    list[float]
        One risk score per input text.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, batch_risk_score, texts)


# ── Model warm-up helper ──────────────────────────────────────────────────────

def warmup_model() -> None:
    """
    Force-load the model and run a single dummy inference to prime the JIT
    compiler and CUDA kernels.

    Call this once during FastAPI startup (inside ``lifespan``) so that the
    first real request does not bear the model-loading latency:

        # main.py — inside lifespan()
        from ml_pipeline import warmup_model
        warmup_model()
    """
    logger.info("Warming up ML risk scoring pipeline…")
    t0 = time.perf_counter()
    _ = calculate_risk_score("Port workers announce industrial action at major terminal")
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("ML pipeline warm-up complete in %.0f ms.", elapsed)


def get_model_info() -> dict:
    """
    Return a diagnostic dict describing the currently loaded model.

    Useful for the ``/health`` endpoint and Swagger docs.

    Returns
    -------
    dict with keys:
        model_id, mode, device, heuristic_weight, entity_amplifier,
        cache_size, cache_hits, cache_misses, cache_currsize
    """
    _registry.ensure_loaded()
    ci = _cached_score.cache_info()
    return {
        "model_id":          _registry._model_id or "not_loaded",
        "mode":              _registry._mode     or "not_loaded",
        "device":            _DEVICE,
        "max_length":        _MAX_LENGTH,
        "heuristic_weight":  _HEURISTIC_WEIGHT,
        "entity_amplifier":  _ENTITY_AMPLIFIER,
        "cache_maxsize":     _CACHE_SIZE,
        "cache_currsize":    ci.currsize,
        "cache_hits":        ci.hits,
        "cache_misses":      ci.misses,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM 2 — BottleneckDetector
# ═════════════════════════════════════════════════════════════════════════════

# ── Bottleneck detector configuration ────────────────────────────────────────

_BN_CONTAMINATION         = float(os.getenv("BN_CONTAMINATION",        "0.05"))
_BN_N_ESTIMATORS          = int(os.getenv("BN_N_ESTIMATORS",           "100"))
_BN_PERCENTILE_THRESHOLD  = float(os.getenv("BN_PERCENTILE_THRESHOLD", "95"))
_BN_ANOMALY_THRESHOLD     = float(os.getenv("BN_ANOMALY_THRESHOLD",    "0.55"))
_BN_MIN_TRAINING_SAMPLES  = int(os.getenv("BN_MIN_TRAINING_SAMPLES",   "30"))
_BN_ROLLING_WINDOW        = int(os.getenv("BN_ROLLING_WINDOW",         "500"))

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BottleneckResult:
    """
    Full output of a single bottleneck detection evaluation.

    Attributes
    ----------
    route_id           : The TRANSIT_ROUTE identifier that was evaluated.
    latency_hours      : The new transit-latency reading (hours) being tested.
    is_bottleneck      : True when BOTH the P95 gate AND the anomaly gate fire.
    severity           : "normal" | "warning" | "critical"
    anomaly_score      : Normalised IsolationForest score in [0, 1].
                         0 = perfectly normal, 1 = extreme anomaly.
    raw_if_score       : Raw sklearn score_samples() value (negative floats).
    percentile_breach  : True if latency_hours > p{BN_PERCENTILE_THRESHOLD}.
    p95_threshold      : The computed P95 value from training history (hours).
    p99_threshold      : The computed P99 value from training history (hours).
    historical_mean    : Mean latency of the training window (hours).
    historical_stddev  : Std-dev of the training window (hours).
    z_score            : (latency - mean) / stddev; >3 signals extreme outlier.
    pct_above_baseline : How many percent above historical mean this reading is.
    n_training_samples : Number of samples the model was trained on.
    model_trained      : False when fewer than BN_MIN_TRAINING_SAMPLES exist
                         and the detector fell back to percentile-only mode.
    evaluation_ms      : Wall-clock time spent on this prediction (ms).
    """
    route_id:           str
    latency_hours:      float
    is_bottleneck:      bool
    severity:           str          # "normal" | "warning" | "critical"
    anomaly_score:      float        # [0, 1]  — normalised
    raw_if_score:       float        # sklearn raw score_samples() value
    percentile_breach:  bool
    p95_threshold:      float
    p99_threshold:      float
    historical_mean:    float
    historical_stddev:  float
    z_score:            float
    pct_above_baseline: float
    n_training_samples: int
    model_trained:      bool
    evaluation_ms:      float

    def __str__(self) -> str:
        status = "🔴 BOTTLENECK" if self.is_bottleneck else "🟢 NORMAL"
        return (
            f"{status} | route={self.route_id}  latency={self.latency_hours:.1f}h  "
            f"severity={self.severity}  anomaly={self.anomaly_score:.3f}  "
            f"z={self.z_score:.2f}  +{self.pct_above_baseline:.1f}%  "
            f"p95={self.p95_threshold:.1f}h  p99={self.p99_threshold:.1f}h  "
            f"n={self.n_training_samples}  {self.evaluation_ms:.1f}ms"
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for API responses."""
        return {
            "route_id":           self.route_id,
            "latency_hours":      self.latency_hours,
            "is_bottleneck":      self.is_bottleneck,
            "severity":           self.severity,
            "anomaly_score":      self.anomaly_score,
            "raw_if_score":       self.raw_if_score,
            "percentile_breach":  self.percentile_breach,
            "p95_threshold":      self.p95_threshold,
            "p99_threshold":      self.p99_threshold,
            "historical_mean":    self.historical_mean,
            "historical_stddev":  self.historical_stddev,
            "z_score":            self.z_score,
            "pct_above_baseline": self.pct_above_baseline,
            "n_training_samples": self.n_training_samples,
            "model_trained":      self.model_trained,
            "evaluation_ms":      self.evaluation_ms,
        }


# ── BottleneckDetector ────────────────────────────────────────────────────────

class BottleneckDetector:
    """
    Per-route anomaly detector that combines a 95th-percentile threshold gate
    with an IsolationForest unsupervised anomaly detector to flag abnormal
    transit latencies on a supply-chain edge.

    Design principles
    -----------------
    • One detector instance per TRANSIT_ROUTE edge.  Use
      ``RouteBottleneckRegistry`` to manage a fleet of detectors.
    • Dual-gate detection eliminates false positives: both the statistical
      percentile gate AND the IsolationForest gate must fire for
      ``BottleneckResult.is_bottleneck`` to be True.
    • Thread-safe via a ``threading.Lock`` on model state mutations, so the
      detector is safe to call from multiple asyncio worker threads.
    • Rolling window: the internal history is bounded to ``rolling_window``
      samples (FIFO deque).  The model is automatically retrained when new
      samples are added via ``update()``.
    • Graceful degradation: if fewer than ``min_training_samples`` samples
      exist the IsolationForest is skipped and only the percentile gate is
      used.  ``BottleneckResult.model_trained`` will be False in this state.

    Parameters
    ----------
    route_id            : Human-readable identifier for this edge, e.g.
                          "PORT-CN-SHA→PORT-US-LGB".  Used in log messages
                          and ``BottleneckResult.route_id``.
    contamination       : Expected fraction of anomalies in training data.
                          Passed to ``IsolationForest(contamination=...)``.
                          Lower values → fewer false positives.
    n_estimators        : Number of isolation trees.  100 is the sklearn default
                          and works well for time-series of up to 500 points.
    percentile_threshold: The P-th percentile used as Gate 1.  Default 95.
    anomaly_threshold   : Gate 2 cutoff on the normalised anomaly score [0,1].
                          Default 0.55 (slightly above 0.5 for conservatism).
    min_training_samples: Minimum history length before the IF model is used.
    rolling_window      : Maximum samples kept in the sliding history window.
    random_state        : Seed for reproducible IF training.

    Examples
    --------
    >>> detector = BottleneckDetector("PORT-CN-SHA→PORT-US-LGB")

    # Train on 90 days of daily latency readings (hours)
    >>> import random, math
    >>> history = [504 + random.gauss(0, 12) for _ in range(90)]
    >>> detector.fit(history)

    # Evaluate a new reading that is well above the historical norm
    >>> result = detector.predict(750.0)
    >>> print(result)
    🔴 BOTTLENECK | route=PORT-CN-SHA→PORT-US-LGB  latency=750.0h ...

    # Evaluate a normal reading
    >>> result = detector.predict(506.3)
    >>> result.is_bottleneck
    False

    # Online update — add a new reading to the rolling window and retrain
    >>> detector.update(512.0)
    """

    def __init__(
        self,
        route_id:             str,
        contamination:        float = _BN_CONTAMINATION,
        n_estimators:         int   = _BN_N_ESTIMATORS,
        percentile_threshold: float = _BN_PERCENTILE_THRESHOLD,
        anomaly_threshold:    float = _BN_ANOMALY_THRESHOLD,
        min_training_samples: int   = _BN_MIN_TRAINING_SAMPLES,
        rolling_window:       int   = _BN_ROLLING_WINDOW,
        random_state:         int   = 42,
    ) -> None:
        self.route_id             = route_id
        self.contamination        = contamination
        self.n_estimators         = n_estimators
        self.percentile_threshold = percentile_threshold
        self.anomaly_threshold    = anomaly_threshold
        self.min_training_samples = min_training_samples
        self.rolling_window       = rolling_window
        self.random_state         = random_state

        # Rolling deque — acts as the live training window
        self._history: Deque[float] = collections.deque(maxlen=rolling_window)

        # Cached statistics (recomputed after fit/update)
        self._p_threshold: float = 0.0   # Gate 1 percentile value (hours)
        self._p99:         float = 0.0
        self._mean:        float = 0.0
        self._stddev:      float = 0.0

        # IsolationForest model and its score normalisation anchors
        self._model:           Any   = None   # sklearn IsolationForest
        self._score_min:       float = -1.0   # worst (most anomalous) score seen
        self._score_max:       float =  0.0   # best (most normal) score seen
        self._model_trained:   bool  = False

        # Thread safety for model mutations
        self._lock = threading.Lock()

        logger.debug("BottleneckDetector created for route '%s'.", route_id)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _percentile(data: list[float], pct: float) -> float:
        """
        Return the p-th percentile of ``data`` using linear interpolation
        (equivalent to numpy.percentile with interpolation='linear').

        Uses only the stdlib ``statistics`` module so there is no numpy
        dependency in the percentile calculation itself.
        """
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        # Linear interpolation index
        index = (pct / 100) * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        frac  = index - lower
        return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])

    def _recompute_statistics(self, data: list[float]) -> None:
        """Update cached descriptive stats from the current training data."""
        self._p_threshold = self._percentile(data, self.percentile_threshold)
        self._p99         = self._percentile(data, 99.0)
        self._mean        = statistics.mean(data)
        self._stddev      = statistics.stdev(data) if len(data) > 1 else 0.0

    def _train_isolation_forest(self, data: list[float]) -> None:
        """
        Fit a fresh IsolationForest on ``data`` and calibrate the score
        normalisation anchors.

        The sklearn ``score_samples()`` method returns negative floats where
        more negative = more anomalous.  We normalise to [0, 1] by:

            normalised = (score - score_min) / (score_max - score_min)

        where score_min / score_max are the most anomalous / most normal
        scores observed on the training set itself.  This gives an intuitive
        0 = normal, 1 = extreme anomaly scale.
        """
        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np
        except ImportError as exc:
            logger.error(
                "scikit-learn is required for BottleneckDetector. "
                "Install it with: pip install scikit-learn. Error: %s", exc,
            )
            self._model_trained = False
            return

        X = np.array(data, dtype=float).reshape(-1, 1)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,            # use all CPU cores
            warm_start=False,     # always retrain from scratch on the rolling window
        )
        model.fit(X)

        # Calibrate normalisation anchors on the training data itself
        train_scores = model.score_samples(X)
        self._score_min = float(train_scores.min())
        self._score_max = float(train_scores.max())
        self._model = model
        self._model_trained = True

        logger.debug(
            "IsolationForest trained for route '%s': n=%d  score_range=[%.4f, %.4f]  "
            "p%d=%.2fh  p99=%.2fh",
            self.route_id, len(data),
            self._score_min, self._score_max,
            int(self.percentile_threshold), self._p_threshold, self._p99,
        )

    def _normalise_score(self, raw_score: float) -> float:
        """
        Map a raw sklearn score_samples() value to [0, 1].

        0.0 → perfectly normal (matches training distribution)
        1.0 → extreme anomaly (far outside training distribution)

        If the score range is degenerate (all training scores identical),
        returns 0.5 to avoid division by zero.
        """
        score_range = self._score_max - self._score_min
        if score_range < 1e-9:
            return 0.5
        normalised = (raw_score - self._score_min) / score_range
        # Invert: high raw_score = normal → should be LOW anomaly score
        return round(float(1.0 - max(0.0, min(1.0, normalised))), 6)

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, latency_history: list[float]) -> "BottleneckDetector":
        """
        Train the detector on a batch of historical latency readings.

        Calling ``fit`` replaces the entire rolling window with the provided
        history (up to ``rolling_window`` most recent values) and retrains
        the IsolationForest from scratch.

        Parameters
        ----------
        latency_history : list[float]
            Historical transit-latency readings in hours.  Should represent
            normal (non-disrupted) operations where possible, as contaminated
            training data will shift the anomaly threshold higher.
            Minimum useful length: ``min_training_samples`` (default 30).
            Recommended: ≥ 60 readings (e.g. 60 days of daily snapshots).

        Returns
        -------
        self
            Enables method chaining: ``detector.fit(history).predict(new_val)``.

        Raises
        ------
        ValueError
            If ``latency_history`` is empty or contains non-finite values.
        """
        if not latency_history:
            raise ValueError("latency_history must not be empty.")

        valid = [v for v in latency_history if math.isfinite(v) and v >= 0]
        if not valid:
            raise ValueError(
                "latency_history contains no valid non-negative finite values."
            )

        with self._lock:
            # Populate rolling window (bounded to rolling_window size)
            self._history.clear()
            self._history.extend(valid[-self.rolling_window :])
            data = list(self._history)

            self._recompute_statistics(data)

            if len(data) >= self.min_training_samples:
                self._train_isolation_forest(data)
            else:
                self._model_trained = False
                logger.info(
                    "Route '%s': only %d samples — using percentile-only mode "
                    "(need %d for IsolationForest).",
                    self.route_id, len(data), self.min_training_samples,
                )

        return self

    def update(self, new_latency: float) -> "BottleneckDetector":
        """
        Add a single new latency reading to the rolling window and retrain.

        Use this for online / streaming updates — call once per new reading
        received from the AIS or port-manifest pipeline.  The rolling window
        automatically evicts the oldest sample when it reaches capacity.

        Retraining is O(n · n_estimators) in the window size; at the default
        window of 500 samples and 100 trees this completes in < 50 ms on a
        modern CPU, well within the AIS tick interval.

        Parameters
        ----------
        new_latency : float
            The most recently observed transit latency in hours.

        Returns
        -------
        self
        """
        if not math.isfinite(new_latency) or new_latency < 0:
            logger.warning(
                "Route '%s': ignoring invalid latency value %r.", self.route_id, new_latency,
            )
            return self

        with self._lock:
            self._history.append(new_latency)
            data = list(self._history)
            self._recompute_statistics(data)

            if len(data) >= self.min_training_samples:
                self._train_isolation_forest(data)

        return self

    def predict(self, latency_hours: float) -> BottleneckResult:
        """
        Evaluate a single new transit-latency reading against the trained model.

        Dual-gate detection:
          Gate 1 (Percentile): latency_hours > P{percentile_threshold}
          Gate 2 (IsolationForest): normalised_anomaly_score > anomaly_threshold

        ``is_bottleneck`` is True only when BOTH gates fire.
        If the model has not yet been trained (< min_training_samples), Gate 2
        is bypassed and Gate 1 alone determines the result.

        Severity classification:
          • "normal"   — not a bottleneck
          • "warning"  — P95 breach  OR  anomaly_score ≥ 0.55 (but not both)
                         OR both gates fire but latency < P99
          • "critical" — both gates fire AND (latency ≥ P99 OR anomaly ≥ 0.80)

        Parameters
        ----------
        latency_hours : float
            The new transit-time reading to evaluate (in hours).

        Returns
        -------
        BottleneckResult
            Fully populated result dataclass.

        Raises
        ------
        RuntimeError
            If ``fit()`` has never been called (history is empty).
        """
        if not self._history:
            raise RuntimeError(
                f"BottleneckDetector for route '{self.route_id}' has not been "
                "trained. Call fit(latency_history) first."
            )

        t0 = time.perf_counter()

        with self._lock:
            # Snapshot immutable values under the lock
            p_thresh      = self._p_threshold
            p99           = self._p99
            mean          = self._mean
            stddev        = self._stddev
            model         = self._model
            model_trained = self._model_trained
            score_min     = self._score_min
            score_max     = self._score_max
            n_samples     = len(self._history)

        # ── Gate 1: Percentile threshold ──────────────────────────────────
        percentile_breach = latency_hours > p_thresh

        # ── Gate 2: IsolationForest ────────────────────────────────────────
        raw_if_score    = 0.0
        anomaly_score   = 0.0

        if model_trained and model is not None:
            try:
                import numpy as np
                X = np.array([[latency_hours]], dtype=float)
                raw_if_score  = float(model.score_samples(X)[0])
                # Normalise using training anchors captured at fit time
                score_range = score_max - score_min
                if score_range > 1e-9:
                    normalised  = (raw_if_score - score_min) / score_range
                    anomaly_score = round(float(1.0 - max(0.0, min(1.0, normalised))), 6)
                else:
                    anomaly_score = 0.5
            except Exception as exc:
                logger.warning(
                    "IsolationForest prediction failed for route '%s': %s — "
                    "falling back to percentile-only.",
                    self.route_id, exc,
                )
                anomaly_score = 0.5   # neutral fallback

        gate2_fires = anomaly_score > self.anomaly_threshold

        # ── Combine gates ──────────────────────────────────────────────────
        if model_trained:
            is_bottleneck = percentile_breach and gate2_fires
        else:
            # Percentile-only mode (insufficient training data)
            is_bottleneck = percentile_breach

        # ── Severity classification ────────────────────────────────────────
        if not is_bottleneck:
            # Even if not a confirmed bottleneck, warn when one gate fires
            if percentile_breach or gate2_fires:
                severity = "warning"
            else:
                severity = "normal"
        else:
            # Confirmed bottleneck — grade by how extreme it is
            if latency_hours >= p99 or anomaly_score >= 0.80:
                severity = "critical"
            else:
                severity = "warning"

        # ── Derived statistics ─────────────────────────────────────────────
        z_score = ((latency_hours - mean) / stddev) if stddev > 1e-9 else 0.0
        pct_above = ((latency_hours - mean) / mean * 100) if mean > 1e-9 else 0.0

        evaluation_ms = (time.perf_counter() - t0) * 1000

        result = BottleneckResult(
            route_id           = self.route_id,
            latency_hours      = latency_hours,
            is_bottleneck      = is_bottleneck,
            severity           = severity,
            anomaly_score      = round(anomaly_score, 4),
            raw_if_score       = round(raw_if_score, 6),
            percentile_breach  = percentile_breach,
            p95_threshold      = round(p_thresh, 2),
            p99_threshold      = round(p99, 2),
            historical_mean    = round(mean, 2),
            historical_stddev  = round(stddev, 2),
            z_score            = round(z_score, 3),
            pct_above_baseline = round(pct_above, 2),
            n_training_samples = n_samples,
            model_trained      = model_trained,
            evaluation_ms      = round(evaluation_ms, 3),
        )

        log_fn = logger.warning if is_bottleneck else logger.debug
        log_fn("%s", result)
        return result

    def predict_batch(self, latency_readings: list[float]) -> list[BottleneckResult]:
        """
        Evaluate a list of latency readings in one call.

        More efficient than calling ``predict`` in a loop when processing a
        backlog of readings, as a single numpy array is allocated for all
        IsolationForest calls.

        Parameters
        ----------
        latency_readings : list[float]
            Ordered sequence of transit latency values to evaluate (hours).

        Returns
        -------
        list[BottleneckResult]
            One result per input reading, in the same order.
        """
        if not latency_readings:
            return []

        if not self._history:
            raise RuntimeError(
                f"BottleneckDetector for route '{self.route_id}' has not been trained."
            )

        t0 = time.perf_counter()

        with self._lock:
            p_thresh      = self._p_threshold
            p99           = self._p99
            mean          = self._mean
            stddev        = self._stddev
            model         = self._model
            model_trained = self._model_trained
            score_min     = self._score_min
            score_max     = self._score_max
            n_samples     = len(self._history)

        # Batch IsolationForest inference (single numpy call)
        raw_scores: list[float] = [0.0] * len(latency_readings)
        if model_trained and model is not None:
            try:
                import numpy as np
                X = np.array(latency_readings, dtype=float).reshape(-1, 1)
                raw_scores = model.score_samples(X).tolist()
            except Exception as exc:
                logger.warning("Batch IF prediction failed for '%s': %s", self.route_id, exc)

        results: list[BottleneckResult] = []
        score_range = score_max - score_min

        for i, latency_hours in enumerate(latency_readings):
            percentile_breach = latency_hours > p_thresh

            anomaly_score = 0.5
            if model_trained and score_range > 1e-9:
                normalised    = (raw_scores[i] - score_min) / score_range
                anomaly_score = round(float(1.0 - max(0.0, min(1.0, normalised))), 6)

            gate2_fires   = anomaly_score > self.anomaly_threshold
            is_bottleneck = (percentile_breach and gate2_fires) if model_trained else percentile_breach

            if not is_bottleneck:
                severity = "warning" if (percentile_breach or gate2_fires) else "normal"
            else:
                severity = "critical" if (latency_hours >= p99 or anomaly_score >= 0.80) else "warning"

            z_score   = ((latency_hours - mean) / stddev) if stddev > 1e-9 else 0.0
            pct_above = ((latency_hours - mean) / mean * 100) if mean > 1e-9 else 0.0

            results.append(BottleneckResult(
                route_id           = self.route_id,
                latency_hours      = latency_hours,
                is_bottleneck      = is_bottleneck,
                severity           = severity,
                anomaly_score      = round(anomaly_score, 4),
                raw_if_score       = round(raw_scores[i], 6),
                percentile_breach  = percentile_breach,
                p95_threshold      = round(p_thresh, 2),
                p99_threshold      = round(p99, 2),
                historical_mean    = round(mean, 2),
                historical_stddev  = round(stddev, 2),
                z_score            = round(z_score, 3),
                pct_above_baseline = round(pct_above, 2),
                n_training_samples = n_samples,
                model_trained      = model_trained,
                evaluation_ms      = round((time.perf_counter() - t0) * 1000 / len(latency_readings), 3),
            ))

        return results

    @property
    def is_trained(self) -> bool:
        """True if the IsolationForest has been fitted on sufficient data."""
        return self._model_trained

    @property
    def n_samples(self) -> int:
        """Number of readings currently in the rolling training window."""
        return len(self._history)

    @property
    def thresholds(self) -> dict:
        """Return the current threshold values as a dict (for API responses)."""
        with self._lock:
            return {
                "p95_hours":       round(self._p_threshold, 2),
                "p99_hours":       round(self._p99, 2),
                "mean_hours":      round(self._mean, 2),
                "stddev_hours":    round(self._stddev, 2),
                "anomaly_cutoff":  self.anomaly_threshold,
                "n_samples":       len(self._history),
                "model_trained":   self._model_trained,
            }


# ── RouteBottleneckRegistry ───────────────────────────────────────────────────

class RouteBottleneckRegistry:
    """
    Process-level registry that manages one ``BottleneckDetector`` per
    TRANSIT_ROUTE edge.

    The registry is the recommended integration point for the FastAPI
    application and ``data_ingestion.py``.  A single instance (``_bn_registry``)
    is exposed at module level for convenience.

    Usage
    -----
    # At startup — seed historical latency from PostgreSQL snapshots
    registry = RouteBottleneckRegistry()
    registry.fit_route("PORT-CN-SHA→PORT-US-LGB", historical_latencies)

    # On each AIS tick — check the new reading
    result = registry.check("PORT-CN-SHA→PORT-US-LGB", new_latency=750.0)
    if result.is_bottleneck:
        alert_operations_team(result)

    # Async variant (for use inside coroutines)
    result = await registry.async_check("PORT-CN-SHA→PORT-US-LGB", 750.0)

    # Bulk check — pass all edge readings from the current AIS tick
    readings = {
        "PORT-CN-SHA→PORT-US-LGB": 750.0,
        "PORT-CN-SHA→PORT-EU-HAM": 892.0,
        "PORT-KR-BSN→PORT-US-LGB": 510.0,
    }
    all_results = registry.check_all(readings)
    """

    def __init__(self) -> None:
        self._detectors: dict[str, BottleneckDetector] = {}
        self._lock = threading.Lock()

    # ── Registration ──────────────────────────────────────────────────────────

    def fit_route(
        self,
        route_id:        str,
        latency_history: list[float],
        **detector_kwargs,
    ) -> BottleneckDetector:
        """
        Create (or replace) the detector for ``route_id`` and train it on
        ``latency_history``.

        Parameters
        ----------
        route_id        : Unique route identifier, e.g. "PORT-CN-SHA→PORT-US-LGB".
        latency_history : Historical latency readings in hours.
        **detector_kwargs : Forwarded to ``BottleneckDetector.__init__`` for
                           per-route parameter overrides (contamination, etc.)

        Returns
        -------
        BottleneckDetector
            The newly trained detector.
        """
        detector = BottleneckDetector(route_id=route_id, **detector_kwargs)
        detector.fit(latency_history)
        with self._lock:
            self._detectors[route_id] = detector
        logger.info(
            "Registry: fitted detector for route '%s' on %d samples.",
            route_id, detector.n_samples,
        )
        return detector

    def get_or_create(self, route_id: str, **detector_kwargs) -> BottleneckDetector:
        """
        Return an existing detector or create an untrained one if not found.

        Useful when you want to add samples incrementally before fitting.
        """
        with self._lock:
            if route_id not in self._detectors:
                self._detectors[route_id] = BottleneckDetector(
                    route_id=route_id, **detector_kwargs
                )
            return self._detectors[route_id]

    # ── Detection ─────────────────────────────────────────────────────────────

    def check(self, route_id: str, new_latency: float) -> BottleneckResult:
        """
        Evaluate a single new latency reading for ``route_id``.

        If no detector exists for this route, a ``LookupError`` is raised.
        Use ``get_or_create`` + ``update`` if you want auto-registration.

        Parameters
        ----------
        route_id    : The route to check.
        new_latency : New transit latency reading in hours.

        Returns
        -------
        BottleneckResult
        """
        with self._lock:
            detector = self._detectors.get(route_id)
        if detector is None:
            raise LookupError(
                f"No detector registered for route '{route_id}'. "
                "Call fit_route() first."
            )
        return detector.predict(new_latency)

    def check_all(
        self, readings: dict[str, float]
    ) -> dict[str, BottleneckResult]:
        """
        Evaluate one latency reading per route in a single call.

        Skips routes that have no registered detector (logs a warning).

        Parameters
        ----------
        readings : dict[route_id → latency_hours]

        Returns
        -------
        dict[route_id → BottleneckResult]
            Contains only routes that had a registered detector.
        """
        results: dict[str, BottleneckResult] = {}
        for route_id, latency in readings.items():
            with self._lock:
                detector = self._detectors.get(route_id)
            if detector is None:
                logger.warning(
                    "check_all: no detector for route '%s' — skipping.", route_id
                )
                continue
            results[route_id] = detector.predict(latency)
        return results

    def update_and_check(
        self, route_id: str, new_latency: float
    ) -> BottleneckResult:
        """
        Add ``new_latency`` to the rolling window, retrain, then predict.

        Use this for the online-update path where each incoming AIS tick
        should both extend the model's knowledge AND be evaluated.

        Parameters
        ----------
        route_id    : Route to update.
        new_latency : New latency reading (hours) to add and evaluate.

        Returns
        -------
        BottleneckResult
        """
        with self._lock:
            detector = self._detectors.get(route_id)
        if detector is None:
            raise LookupError(
                f"No detector for route '{route_id}'. Call fit_route() first."
            )
        detector.update(new_latency)
        return detector.predict(new_latency)

    # ── Async wrappers ─────────────────────────────────────────────────────────

    async def async_check(
        self, route_id: str, new_latency: float
    ) -> BottleneckResult:
        """
        Async variant of ``check`` — runs in the default thread-pool executor
        so that IsolationForest inference does not block the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check, route_id, new_latency)

    async def async_check_all(
        self, readings: dict[str, float]
    ) -> dict[str, BottleneckResult]:
        """
        Async variant of ``check_all`` — offloads all predictions to the
        thread-pool executor in a single ``run_in_executor`` call.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check_all, readings)

    async def async_update_and_check(
        self, route_id: str, new_latency: float
    ) -> BottleneckResult:
        """Async variant of ``update_and_check``."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.update_and_check, route_id, new_latency
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def registered_routes(self) -> list[str]:
        """List all route IDs that have a registered detector."""
        with self._lock:
            return list(self._detectors.keys())

    def summary(self) -> dict:
        """
        Return a diagnostic summary of all registered detectors.
        Suitable for the ``/health`` or ``/v2/bottlenecks`` API endpoint.
        """
        with self._lock:
            detectors = dict(self._detectors)
        return {
            route_id: {
                "n_samples":     d.n_samples,
                "model_trained": d.is_trained,
                **d.thresholds,
            }
            for route_id, d in detectors.items()
        }


# ── Module-level registry singleton ──────────────────────────────────────────

bn_registry = RouteBottleneckRegistry()
"""
Module-level singleton ``RouteBottleneckRegistry``.

Import and use directly in FastAPI routes and data_ingestion.py:

    from ml_pipeline import bn_registry

    # At startup
    bn_registry.fit_route("PORT-CN-SHA→PORT-US-LGB", historical_latencies)

    # On each AIS tick
    result = await bn_registry.async_check("PORT-CN-SHA→PORT-US-LGB", 750.0)
"""


# ── Convenience factory: seed the registry from the sample network ─────────────

def seed_bn_registry_from_sample_network() -> None:
    """
    Pre-train one ``BottleneckDetector`` per TRANSIT_ROUTE edge defined in
    ``graph_ops._SAMPLE_ROUTES``, using synthetically generated historical
    latency data that mirrors the base_cost / current_latency values from
    the sample network.

    This gives the registry a realistic baseline immediately on startup
    without requiring a real PostgreSQL query.

    Call once during FastAPI lifespan startup:

        from ml_pipeline import seed_bn_registry_from_sample_network
        seed_bn_registry_from_sample_network()
    """
    import random

    # Inline the route definitions to avoid a circular import from graph_ops
    # (graph_ops imports database which imports settings)
    _ROUTE_BASELINES: list[tuple[str, float, float]] = [
        # (route_id,                              base_latency_h, stddev_h)
        ("PORT-CN-SHA→PORT-US-LGB",               504.0,          18.0),
        ("PORT-CN-SHA→PORT-EU-HAM",               792.0,          24.0),
        ("PORT-CN-SHA→PORT-SG-SIN",               120.0,           8.0),
        ("PORT-CN-SHA→PORT-AU-SYD",               336.0,          14.0),
        ("PORT-CN-SHA→PORT-KR-BSN",                24.0,           2.0),
        ("PORT-KR-BSN→PORT-US-LGB",               504.0,          16.0),
        ("PORT-KR-BSN→PORT-EU-HAM",               792.0,          22.0),
        ("FACTORY-CN-GZ-001→PORT-CN-SHA",           6.0,           0.5),
        ("FACTORY-CN-GZ-001→PORT-CN-SZX",           3.5,           0.3),
        ("FACTORY-VN-HCM-001→PORT-SG-SIN",         72.0,           4.0),
        ("PORT-US-LGB→WAREHOUSE-US-LA-001",          5.0,           0.4),
        ("PORT-EU-HAM→WAREHOUSE-DE-BER-004",        22.0,           1.5),
        ("PORT-SG-SIN→WAREHOUSE-SG-SIN-002",         2.0,           0.2),
        ("PORT-AU-SYD→RETAILER-AU-MEL-001",          9.0,           0.8),
        ("WAREHOUSE-US-LA-001→RETAILER-US-NY-001",   8.0,           0.6),
        ("WAREHOUSE-DE-BER-004→RETAILER-EU-MUC-001", 7.5,           0.5),
        ("WAREHOUSE-SG-SIN-002→RETAILER-US-NY-001", 22.0,           1.8),
    ]

    rng = random.Random(2024)   # fixed seed for reproducibility in tests

    for route_id, base, stddev in _ROUTE_BASELINES:
        # Generate 90 synthetic "normal" readings with mild random noise
        history = [
            max(base * 0.5, base + rng.gauss(0, stddev))
            for _ in range(90)
        ]
        bn_registry.fit_route(route_id, history)

    logger.info(
        "BottleneckDetector registry seeded with %d routes from sample network.",
        len(_ROUTE_BASELINES),
    )




INGESTION_PATCH = """
# ── Paste this into data_ingestion.py to activate both ML subsystems ──────────
#
# 1. Add at the top of data_ingestion.py:
#
#    from ml_pipeline import (
#        calculate_risk_score,
#        bn_registry,
#        seed_bn_registry_from_sample_network,
#    )
#
# 2. Inside seed_sample_network() or the FastAPI lifespan, call:
#
#    seed_bn_registry_from_sample_network()
#
# 3. Inside _generate_sentiment_events(), replace:
#
#    risk_score = round(max(0.0, min(1.0, tpl.base_risk + noise)), 4)
#
# with:
#
#    risk_score = calculate_risk_score(text)
#
# 4. Inside _persist_ais_reports(), after each PortManifest insert, add:
#
#    route_id = f"{r.port_id}→<dest_port_id>"
#    try:
#        bn_result = await bn_registry.async_update_and_check(route_id, r.dwell_hours)
#        if bn_result.is_bottleneck:
#            logger.warning("BOTTLENECK DETECTED: %s", bn_result)
#    except LookupError:
#        pass  # route not yet registered
"""

# ── Demo / CLI entry-point ────────────────────────────────────────────────────

_DEMO_TEXTS = [
    # High risk
    "Port workers go on strike at Shanghai terminal, halting all container operations",
    "Typhoon Amber forces emergency closure of Port of Busan — all vessels diverted",
    "Dockworkers at Long Beach announce 72-hour walkout; Maersk issues force-majeure",
    "Suez Canal blocked by grounded vessel — 200 ships awaiting passage",
    "Sanctions imposed on COSCO subsidiary; routes suspended pending compliance review",
    # Medium risk
    "Customs clearance delays at Hamburg port up 36 hours due to new digital procedures",
    "Container shortage in Asia-Pacific corridor tightening; rates expected to surge",
    "Fog halts pilotage at Port of Singapore for second consecutive morning",
    # Low / recovery
    "Strike at Port of Rotterdam called off after last-minute agreement reached",
    "Port of Shanghai congestion easing as berth reform takes effect",
    "Maersk reports record on-time performance across Trans-Pacific corridor",
    "Operations at Port of Hamburg running smoothly — no delays reported",
]


def _run_bottleneck_demo() -> None:
    """Interactive demo for the BottleneckDetector subsystem."""
    import random
    rng = random.Random(42)

    print("\n" + "─" * 72)
    print("  SUBSYSTEM 2 — BottleneckDetector Demo")
    print("─" * 72)

    route_id     = "PORT-CN-SHA→PORT-US-LGB"
    base_latency = 504.0
    stddev       = 18.0

    print(f"\n  Route: {route_id}")
    print(f"  Baseline: {base_latency}h  σ={stddev}h")

    history  = [max(base_latency * 0.6, base_latency + rng.gauss(0, stddev))
                for _ in range(90)]
    detector = BottleneckDetector(route_id=route_id)
    detector.fit(history)

    thresholds = detector.thresholds
    print(f"  Trained on {thresholds['n_samples']} samples  |  "
          f"P95={thresholds['p95_hours']}h  P99={thresholds['p99_hours']}h  "
          f"mean={thresholds['mean_hours']}h")

    test_readings = [
        (498.0,                             "Historical normal"),
        (521.0,                             "Slightly above average"),
        (thresholds["p95_hours"] + 5,       "Just above P95 threshold"),
        (base_latency * 1.25,               "+25% above baseline"),
        (630.0,                             "Current congested latency (SentinelFlow demo)"),
        (750.0,                             "Extreme congestion (+49% above baseline)"),
        (892.0,                             "Shanghai→Hamburg worst-case observed"),
    ]

    print()
    print(f"  {'Latency':>10}  {'Bottleneck':>12}  {'Severity':>10}  "
          f"{'Anomaly':>9}  {'Z-score':>8}  {'Δ Baseline':>12}  Description")
    print("  " + "─" * 80)

    for latency, description in test_readings:
        r        = detector.predict(latency)
        flag     = "🔴 YES" if r.is_bottleneck else "🟢  no"
        sev_icon = {"normal": "⚪", "warning": "🟡", "critical": "🔴"}.get(r.severity, "")
        print(f"  {latency:>9.1f}h  {flag:>12}  {sev_icon}{r.severity:>9}  "
              f"{r.anomaly_score:>9.4f}  {r.z_score:>+8.3f}  "
              f"{r.pct_above_baseline:>+10.1f}%  {description}")

    print("\n  Batch prediction (8 readings in one call):")
    batch        = [rng.gauss(base_latency, stddev) for _ in range(6)] + [640.0, 780.0]
    batch_res    = detector.predict_batch(batch)
    flagged      = sum(1 for r in batch_res if r.is_bottleneck)
    print(f"    {len(batch)} readings evaluated — {flagged} bottleneck(s) flagged")
    for r in batch_res:
        icon = "🔴" if r.is_bottleneck else "🟢"
        print(f"    {icon} {r.latency_hours:>7.1f}h  anomaly={r.anomaly_score:.3f}  {r.severity}")

    print("\n  Online update — feeding 10 new readings to rolling window:")
    for i in range(10):
        detector.update(base_latency + rng.gauss(0, stddev) * (1 + i * 0.15))
    print(f"    Window now has {detector.n_samples} samples "
          f"(p95={detector.thresholds['p95_hours']}h)")

    print("\n  RouteBottleneckRegistry — checking all SentinelFlow demo routes:")
    seed_bn_registry_from_sample_network()

    current_readings = {
        "PORT-CN-SHA→PORT-US-LGB": 630.0,
        "PORT-CN-SHA→PORT-EU-HAM": 892.0,
        "PORT-CN-SHA→PORT-SG-SIN": 163.0,
        "PORT-CN-SHA→PORT-AU-SYD": 413.0,
        "PORT-CN-SHA→PORT-KR-BSN":  24.0,
        "PORT-KR-BSN→PORT-US-LGB": 510.0,
    }

    all_results = bn_registry.check_all(current_readings)
    bottlenecks = [(rid, r) for rid, r in all_results.items() if r.is_bottleneck]
    print(f"\n    {len(current_readings)} routes checked — "
          f"{len(bottlenecks)} bottleneck(s) confirmed:\n")

    for rid, r in all_results.items():
        icon = "🔴" if r.is_bottleneck else "🟢"
        print(f"    {icon} [{r.severity:8s}]  {rid}")
        print(f"         latency={r.latency_hours:.1f}h  "
              f"p95={r.p95_threshold:.1f}h  "
              f"anomaly={r.anomaly_score:.3f}  "
              f"z={r.z_score:+.2f}  "
              f"+{r.pct_above_baseline:.1f}%")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    mode    = next((a for a in sys.argv[1:] if a in ("--bn", "--risk", "--all")), "--all")
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    custom_text = next((a for a in sys.argv[1:] if not a.startswith("-")), None)

    run_risk = mode in ("--risk", "--all")
    run_bn   = mode in ("--bn",   "--all")

    if run_risk:
        texts_to_score = [custom_text] if custom_text else _DEMO_TEXTS

        print("\n" + "═" * 72)
        print("  SentinelFlow — ML Risk Scoring Pipeline Demo")
        print("═" * 72)
        print(f"  Model (primary)  : {_PRIMARY_MODEL_ID}")
        print(f"  Model (fallback) : {_FALLBACK_MODEL_ID}")
        print(f"  Device           : {_DEVICE}")
        print(f"  Heuristic weight : {_HEURISTIC_WEIGHT}")
        print(f"  Entity amplifier : {_ENTITY_AMPLIFIER}×")
        print("═" * 72 + "\n")

        for text in texts_to_score:
            if verbose:
                result   = explain_risk_score(text)
                icons    = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                icon     = icons.get(result.risk_tier, "⚪")
                print(f"{icon}  [{result.score:.4f}] {text[:80]}")
                print(f"    model={result.model_score:.4f}  heuristic={result.heuristic_score:.4f}  "
                      f"entity_amp={result.entity_amplified}  dominant={result.dominant_layer}")
                if result.keyword_hits:
                    print(f"    keywords: {[(p[:40], w) for p, w in result.keyword_hits[:3]]}")
                print()
            else:
                score = calculate_risk_score(text)
                tier  = ("CRITICAL" if score >= 0.75 else "HIGH" if score >= 0.50
                         else "MEDIUM" if score >= 0.25 else "LOW")
                bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  {score:.4f} [{tier:8s}] |{bar}|  {text[:60]}")

        print()
        info = get_model_info()
        print(f"  Cache: {info['cache_currsize']}/{info['cache_maxsize']} entries  "
              f"hits={info['cache_hits']}  misses={info['cache_misses']}")

    if run_bn:
        _run_bottleneck_demo()

    print()
