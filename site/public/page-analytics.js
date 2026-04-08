/**
 * page-analytics.js
 * =====================================================================
 * Wires analytics.html to:
 *   GET  /api/analytics/metrics   — ML model stats + optimization status
 *   POST /api/mitigate            — "Apply Strategy" buttons
 * Auto-refreshes every 15 seconds.
 * =====================================================================
 */

document.addEventListener("DOMContentLoaded", () => {

  // ── Model metrics renderer ────────────────────────────────────────────
  function renderModelCards(models) {
    models.forEach((m) => {
      if (m.model_name.includes("GNN")) {
        SF.setText("gnn-f1-score", m.f1_score?.toFixed(2) ?? "—");
        SF.setText("gnn-accuracy", m.accuracy_pct ? `${m.accuracy_pct}%` : "—");
        SF.setText("gnn-precision", m.precision_pct ? `${m.precision_pct}%` : "—");
        SF.setText("gnn-trained-ago", `${m.last_trained_mins_ago}m ago`);
        SF.pulse("gnn-f1-score");
      }

      if (m.model_name.includes("XGBoost")) {
        SF.setText("xgb-rmse", m.rmse_days ? `${m.rmse_days} days` : "—");
        SF.setText("xgb-accuracy", m.accuracy_pct ? `${m.accuracy_pct}%` : "—");
        SF.setText("xgb-trained-ago", `${m.last_trained_mins_ago}m ago`);
        SF.pulse("xgb-accuracy");
      }

      if (m.model_name.includes("Isolation")) {
        SF.setText("if-anomalies", m.anomalies_detected_24h ?? "—");
        SF.setText("if-precision", m.precision_pct ? `${m.precision_pct}%` : "—");
        SF.setText("if-fpr", m.false_positive_rate_pct ? `${m.false_positive_rate_pct}%` : "—");
        SF.setText("if-trained-ago", `${m.last_trained_mins_ago}m ago`);
        SF.pulse("if-anomalies");

        // Update the detection bar width (precision as %)
        const bar = document.getElementById("if-precision-bar");
        if (bar && m.precision_pct) bar.style.width = `${m.precision_pct}%`;
      }
    });
  }

  // ── Optimization status renderer ──────────────────────────────────────
  function renderOptimization(opt) {
    SF.setText("opt-status", opt.status);
    SF.setText("opt-routes", opt.routes_evaluated);
    SF.setText("opt-nodes", opt.nodes_active);
    SF.setText("opt-constraints", opt.constraints_active);
    SF.setText("opt-solver-ms", `${opt.solver_ms}ms`);
    SF.setText("opt-strategies", opt.strategies_generated);
    SF.setText("opt-note", opt.convergence_note);
    SF.pulse("opt-solver-ms");

    const badge = document.getElementById("opt-status-badge");
    if (badge) {
      badge.textContent = opt.status;
      badge.className = opt.status === "OPTIMAL"
        ? "px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-bold border border-emerald-500/20 uppercase"
        : "px-3 py-1 rounded-full bg-amber-500/10 text-amber-400 text-xs font-bold border border-amber-500/20 uppercase";
    }
  }

  // ── "Apply Strategy" handler ──────────────────────────────────────────
  async function handleApplyStrategy(btn, nodeId, cargo, riskWeight) {
    const origHTML = btn.innerHTML;
    btn.innerHTML = `<span class="animate-pulse">Running optimizer…</span>`;
    btn.disabled = true;

    try {
      const result = await SF.api.mitigate(nodeId, cargo, riskWeight);

      // Show result in a result panel if it exists
      const resultEl = document.getElementById("strategy-result-panel");
      if (resultEl) {
        resultEl.innerHTML = `
          <div class="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/20 text-sm">
            <p class="text-emerald-400 font-bold mb-2">✓ Strategy Applied — ${result.status}</p>
            <div class="grid grid-cols-3 gap-4 font-mono text-xs text-slate-300">
              <div><span class="text-slate-500 block">Flow</span>${result.total_flow_teu?.toLocaleString()} TEU</div>
              <div><span class="text-slate-500 block">Cost</span>$${result.total_cost_usd?.toLocaleString()}</div>
              <div><span class="text-slate-500 block">Throughput</span>${result.throughput_pct?.toFixed(1)}%</div>
            </div>
            <p class="text-slate-500 text-[10px] mt-2 font-mono">Solver: ${result.solver_wall_ms}ms · ${result.allocations?.length ?? 0} route(s) allocated</p>
          </div>`;
        resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } else {
        // Toast if no panel
        const toast = document.createElement("div");
        toast.className = "sf-error-toast";
        toast.style.borderColor = "rgba(16,185,129,0.4)";
        toast.style.color = "#6ee7b7";
        toast.innerHTML = `✓ ${result.status}: ${result.total_flow_teu?.toLocaleString()} TEU routed at ${result.throughput_pct?.toFixed(1)}% throughput`;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
      }
    } catch (e) {
      window.sfToast(`Optimizer error: ${e.message}`);
    } finally {
      btn.innerHTML = origHTML;
      btn.disabled = false;
    }
  }

  // ── Bind strategy buttons ─────────────────────────────────────────────
  function bindStrategyButtons() {
    // Strategy 1: Reroute Shanghai → Busan
    const btn1 = document.getElementById("btn-apply-sf001");
    if (btn1) {
      btn1.addEventListener("click", () =>
        handleApplyStrategy(btn1, "PORT-CN-SHA", 12000, 1.0)
      );
    }

    // Strategy 2: Split Mediterranean
    const btn2 = document.getElementById("btn-apply-sf002");
    if (btn2) {
      btn2.addEventListener("click", () =>
        handleApplyStrategy(btn2, "PORT-CN-SHA", 8000, 0.5)
      );
    }

    // Strategy 3: West Coast capacity
    const btn3 = document.getElementById("btn-apply-sf003");
    if (btn3) {
      btn3.addEventListener("click", () =>
        handleApplyStrategy(btn3, "PORT-US-LGB", 6000, 0.3)
      );
    }
  }

  // ── Main load ─────────────────────────────────────────────────────────
  async function loadAnalytics() {
    try {
      const data = await SF.api.analytics();
      renderModelCards(data.models);
      renderOptimization(data.optimization);
    } catch (e) {
      window.sfToast("Analytics API unavailable — showing last known state");
    }
  }

  bindStrategyButtons();
  SF.autoRefresh(loadAnalytics, 15_000);
});
