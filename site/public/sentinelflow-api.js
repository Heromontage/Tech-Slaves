/**
 * sentinelflow-api.js
 * =====================================================================
 * Central API client for SentinelFlow frontend <-> backend integration.
 *
 * USAGE (include before any page-specific script):
 *   <script src="sentinelflow-api.js"></script>
 *
 * CONFIG: Change API_BASE_URL if your backend runs on a different port.
 * =====================================================================
 */

const SF = (() => {
  // ── Configuration ────────────────────────────────────────────────────
  const API_BASE_URL = "http://localhost:8000";
  const REFRESH_INTERVAL_MS = 15_000; // auto-refresh every 15 seconds

  // ── Internal helpers ─────────────────────────────────────────────────
  async function _fetch(path, options = {}) {
    try {
      const res = await fetch(`${API_BASE_URL}${path}`, {
        headers: { "Content-Type": "application/json", ...options.headers },
        ...options,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (e) {
      console.warn(`[SentinelFlow API] ${path} failed:`, e.message);
      throw e;
    }
  }

  // ── DOM helpers ──────────────────────────────────────────────────────
  function setText(id, text, fallback = "—") {
    const el = document.getElementById(id);
    if (el) el.textContent = text ?? fallback;
  }

  function setHTML(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  }

  function pulse(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.add("sf-updated");
    setTimeout(() => el.classList.remove("sf-updated"), 600);
  }

  // Risk level → Tailwind CSS colour classes
  function riskColor(risk) {
    return {
      critical: { text: "text-red-400",    bg: "bg-red-500/10",    border: "border-red-500/30" },
      high:     { text: "text-orange-400", bg: "bg-orange-500/10", border: "border-orange-500/30" },
      medium:   { text: "text-amber-400",  bg: "bg-amber-500/10",  border: "border-amber-500/30" },
      low:      { text: "text-emerald-400",bg: "bg-emerald-500/10",border: "border-emerald-500/30" },
    }[risk] || { text: "text-slate-400", bg: "bg-slate-500/10", border: "border-slate-500/30" };
  }

  function statusBadge(status) {
    const map = {
      active:   `<span class="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[10px] font-bold uppercase border border-emerald-500/20 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>Active</span>`,
      degraded: `<span class="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 text-[10px] font-bold uppercase border border-amber-500/20 flex items-center gap-1"><span class="material-symbols-outlined text-xs">warning</span>Degraded</span>`,
      offline:  `<span class="px-2 py-0.5 rounded-full bg-red-500/10 text-red-400 text-[10px] font-bold uppercase border border-red-500/20 flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-red-400"></span>Offline</span>`,
    };
    return map[status] || map.offline;
  }

  // ── Public API endpoints ─────────────────────────────────────────────
  const api = {
    health: () => _fetch("/health"),
    networkState: (region) => _fetch(`/api/network-state${region ? `?region=${region}` : ""}`),
    bottlenecks: (severity, limit = 50) =>
      _fetch(`/api/bottlenecks${severity ? `?severity=${severity}&limit=${limit}` : `?limit=${limit}`}`),
    streams: () => _fetch("/api/streams/live"),
    analytics: () => _fetch("/api/analytics/metrics"),
    routesSummary: (limit = 10) => _fetch(`/api/routes/summary?limit=${limit}`),
    mitigate: (nodeId, cargo = 12000, riskWeight = 1.0) =>
      _fetch("/api/mitigate", {
        method: "POST",
        body: JSON.stringify({
          disrupted_node_id: nodeId,
          total_cargo_teu: cargo,
          risk_weight: riskWeight,
          min_throughput_pct: 0.9,
        }),
      }),
    playground: (endpoint, method = "GET") =>
      _fetch("/api/docs/playground", {
        method: "POST",
        body: JSON.stringify({ endpoint, method }),
      }),
  };

  // ── Auto-refresh registry ────────────────────────────────────────────
  const _timers = [];

  function autoRefresh(fn, intervalMs = REFRESH_INTERVAL_MS) {
    fn(); // run immediately
    const id = setInterval(fn, intervalMs);
    _timers.push(id);
    return id;
  }

  function stopAll() {
    _timers.forEach(clearInterval);
    _timers.length = 0;
  }

  // ── Expose public interface ──────────────────────────────────────────
  return { api, autoRefresh, stopAll, setText, setHTML, pulse, riskColor, statusBadge };
})();

// Inject a tiny CSS pulse animation once
(function injectCSS() {
  if (document.getElementById("sf-pulse-style")) return;
  const s = document.createElement("style");
  s.id = "sf-pulse-style";
  s.textContent = `
    .sf-updated { animation: sf-flash 0.6s ease; }
    @keyframes sf-flash {
      0%,100% { opacity:1; }
      30%      { opacity:0.4; }
    }
    .sf-loading { opacity:0.5; pointer-events:none; }
    .sf-error-toast {
      position:fixed; bottom:1.5rem; right:1.5rem; z-index:9999;
      background:#1b1f2c; border:1px solid rgba(239,68,68,0.4);
      color:#fca5a5; padding:0.75rem 1.25rem; border-radius:0.75rem;
      font-size:0.75rem; font-family:'JetBrains Mono',monospace;
      box-shadow:0 0 20px rgba(239,68,68,0.15);
      animation: sf-slide-in 0.3s ease;
    }
    @keyframes sf-slide-in {
      from { transform:translateY(1rem); opacity:0; }
      to   { transform:translateY(0);    opacity:1; }
    }
  `;
  document.head.appendChild(s);
})();

// Global error toast helper
window.sfToast = function (msg, duration = 4000) {
  const div = document.createElement("div");
  div.className = "sf-error-toast";
  div.textContent = `⚠ ${msg}`;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), duration);
};
