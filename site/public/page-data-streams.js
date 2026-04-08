/**
 * page-data-streams.js
 * =====================================================================
 * Wires data-streams.html to GET /api/streams/live
 * Auto-refreshes every 15 seconds. Updates:
 *   - Summary stat cards (total, active, degraded, offline, avg latency)
 *   - Individual stream cards (status badge, throughput, latency, uptime)
 *   - Live system log entries
 *   - "Last Updated" timestamp
 * =====================================================================
 */

document.addEventListener("DOMContentLoaded", () => {

  // ── Map stream_id → which card DOM element to update ────────────────
  const STREAM_CARD_MAP = {
    "stream-commodity":       "card-commodity",
    "stream-buyer-sentiment": "card-buyer-sentiment",
    "stream-media-sentiment": "card-media-sentiment",
    "stream-factory-iot":     "card-factory-iot",
    "stream-ais":             "card-ais",
    "stream-stress-test":     "card-stress-test",
    "stream-manifests":       "card-manifests",
    "stream-shipping-routes": "card-shipping-routes",
  };

  // ── Summary counters ─────────────────────────────────────────────────
  function updateSummaryCards(data) {
    SF.setText("stat-total-streams",  data.total);
    SF.setText("stat-active-streams", data.active);
    SF.setText("stat-degraded-streams", data.degraded);
    SF.setText("stat-offline-streams", data.offline);
    SF.setText("stat-avg-latency", `${data.avg_latency_ms}`);
    SF.pulse("stat-avg-latency");
  }

  // ── Per-stream card update ────────────────────────────────────────────
  function updateStreamCard(stream) {
    const cardId = STREAM_CARD_MAP[stream.stream_id];
    if (!cardId) return;

    const statusEl = document.getElementById(`${cardId}-status`);
    if (statusEl) statusEl.innerHTML = SF.statusBadge(stream.status);

    SF.setText(`${cardId}-throughput`, `${stream.throughput_rps.toLocaleString()} r/s`);
    SF.setText(`${cardId}-latency`, `${stream.latency_ms}ms`);
    SF.setText(`${cardId}-uptime`, `${stream.uptime_pct}%`);

    const latEl = document.getElementById(`${cardId}-latency`);
    if (latEl) {
      latEl.className = "font-mono text-white";
      if (stream.status === "degraded") {
        latEl.className = "font-mono text-amber-400 font-bold";
      } else if (stream.status === "offline") {
        latEl.className = "font-mono text-red-400/60";
      }
    }

    // Dim the entire card if offline
    const card = document.getElementById(cardId);
    if (card) {
      card.classList.toggle("opacity-60", stream.status === "offline");
      card.classList.toggle("grayscale", stream.status === "offline");
    }
  }

  // ── Live log injection ────────────────────────────────────────────────
  function buildLogEntry(stream) {
    const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
    const colorClass = {
      active:   "text-cyan-400",
      degraded: "text-amber-400",
      offline:  "text-red-400",
    }[stream.status] || "text-slate-400";

    const msgs = {
      active:   `${stream.records_24h.toLocaleString()} records ingested — ${stream.throughput_rps} ops/s`,
      degraded: `Connection latency spike detected (${stream.latency_ms}ms) — monitoring`,
      offline:  `Stream terminated (SIGTERM) — reconnecting...`,
    };

    return `<p><span class="text-slate-500">[${ts}]</span> <span class="${colorClass}">${stream.name}:</span> ${msgs[stream.status]}</p>`;
  }

  function updateLiveLog(streams) {
    const logEl = document.getElementById("live-log-container");
    if (!logEl) return;

    // Keep existing entries + prepend 3 new ones
    const newEntries = streams
      .filter((_, i) => i < 5)
      .map(buildLogEntry)
      .join("");

    logEl.innerHTML = newEntries + logEl.innerHTML;

    // Trim to 12 entries max
    const entries = logEl.querySelectorAll("p");
    if (entries.length > 12) {
      Array.from(entries).slice(12).forEach((el) => el.remove());
    }
  }

  // ── Last updated timestamp ────────────────────────────────────────────
  function updateTimestamp() {
    const el = document.getElementById("last-updated-ts");
    if (el) el.textContent = "Just now";
    setTimeout(() => {
      if (el) el.textContent = "< 15s ago";
    }, 2000);
  }

  // ── Refresh button ────────────────────────────────────────────────────
  const refreshBtn = document.getElementById("btn-refresh-all");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      refreshBtn.classList.add("sf-loading");
      loadStreams().finally(() => refreshBtn.classList.remove("sf-loading"));
    });
  }

  // ── Main load function ────────────────────────────────────────────────
  async function loadStreams() {
    try {
      const data = await SF.api.streams();
      updateSummaryCards(data);
      data.streams.forEach(updateStreamCard);
      updateLiveLog(data.streams.filter((s) => s.status !== "offline"));
      updateTimestamp();
    } catch (e) {
      window.sfToast("Streams API unavailable — showing last known state");
    }
  }

  SF.autoRefresh(loadStreams, 15_000);
});
