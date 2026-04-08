/**
 * page-bottleneck.js
 * =====================================================================
 * Wires bottleneck.html to:
 *   GET  /api/bottlenecks         — flagged nodes + edges
 *   GET  /api/routes/summary      — affected route health bars
 *   POST /api/mitigate            — "Apply Suggestion" button
 * Auto-refreshes every 15 seconds.
 * =====================================================================
 */

document.addEventListener("DOMContentLoaded", () => {

  // ── Summary stat cards ────────────────────────────────────────────────
  function updateSummaryCards(data) {
    SF.setText("bn-total-flagged", data.total_flagged ?? 0);

    const critical = data.nodes.filter((n) => n.risk_level === "critical");
    const warnings = data.nodes.filter((n) => n.risk_level !== "critical");
    SF.setText("bn-critical-count", critical.length);
    SF.setText("bn-warning-count", warnings.length);
    SF.pulse("bn-total-flagged");

    if (critical.length > 0) {
      SF.setText("bn-primary-node", critical[0].name);
      SF.setText("bn-primary-latency", `${critical[0].max_latency_h}h delay impact`);
    }
  }

  // ── Affected routes table ────────────────────────────────────────────
  function renderAffectedRoutes(edges) {
    const tbody = document.getElementById("affected-routes-tbody");
    if (!tbody) return;

    if (!edges || edges.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="px-8 py-6 text-center text-slate-500 font-mono text-xs">No affected routes detected</td></tr>`;
      return;
    }

    tbody.innerHTML = edges.map((edge) => {
      const sev = edge.severity || "warning";
      const c = SF.riskColor(sev === "critical" ? "critical" : "medium");
      const lat = edge.current_latency ? `+${(edge.current_latency / 24).toFixed(1)}d` : "—";
      return `
        <tr class="hover:bg-white/5 transition-colors">
          <td class="px-8 py-4 text-cyan-400 font-bold font-mono text-xs">${edge.from_id?.replace("PORT-CN-", "CN-").replace("PORT-US-", "US-").replace("PORT-EU-", "EU-") ?? "—"}</td>
          <td class="px-8 py-4 text-slate-300 font-mono text-xs">${formatNodeName(edge.from_id)}</td>
          <td class="px-8 py-4 text-slate-300 font-mono text-xs">${formatNodeName(edge.to_id)}</td>
          <td class="px-8 py-4 text-slate-400 font-mono text-xs">${edge.transit_mode ?? "Sea"}</td>
          <td class="px-8 py-4 ${c.text} font-bold font-mono text-xs">${lat}</td>
          <td class="px-8 py-4">
            <span class="${c.bg} ${c.text} px-2 py-0.5 rounded-full border ${c.border} text-[10px] font-bold uppercase">${sev.toUpperCase()}</span>
          </td>
          <td class="px-8 py-4 text-center">
            <span class="material-symbols-outlined ${sev === "critical" ? "text-cyan-400" : "text-slate-600"} text-sm">${sev === "critical" ? "check_circle" : "cancel"}</span>
          </td>
        </tr>`;
    }).join("");

    // Update the showing count
    SF.setText("affected-routes-count", `Showing ${edges.length} affected`);
  }

  // ── Sentiment & anomaly displays ──────────────────────────────────────
  function updateAnalysisPanel(data) {
    if (!data.nodes || data.nodes.length === 0) return;
    const primary = data.nodes[0];

    // Anomaly score bar
    const scoreEl = document.getElementById("primary-anomaly-score");
    if (scoreEl && primary.anomaly_score != null) {
      const pct = Math.round(primary.anomaly_score * 100);
      scoreEl.textContent = `${pct}%`;
    }

    // IF flagged indicator
    const ifBadge = document.getElementById("primary-if-badge");
    if (ifBadge) {
      ifBadge.textContent = primary.is_if_flagged ? "Isolation Forest: FLAGGED" : "Isolation Forest: Monitoring";
      ifBadge.className = primary.is_if_flagged
        ? "text-[10px] font-mono text-red-400 uppercase"
        : "text-[10px] font-mono text-amber-400 uppercase";
    }
  }

  // ── "Apply Suggestion" button ─────────────────────────────────────────
  const applyBtn = document.getElementById("btn-apply-suggestion");
  if (applyBtn) {
    applyBtn.addEventListener("click", async () => {
      const origHTML = applyBtn.innerHTML;
      applyBtn.innerHTML = `<span class="animate-pulse">Running OR-Tools optimizer…</span>`;
      applyBtn.disabled = true;

      try {
        const result = await SF.api.mitigate("PORT-CN-SHA", 12000, 1.0);
        const resultEl = document.getElementById("mitigation-result");
        if (resultEl) {
          resultEl.innerHTML = `
            <div class="p-4 rounded-lg bg-cyan-500/5 border border-cyan-500/20 mt-4">
              <p class="text-cyan-400 font-bold text-sm mb-2">✓ Rerouting Applied — ${result.status}</p>
              <div class="grid grid-cols-2 gap-4 font-mono text-xs text-slate-300">
                <div><span class="text-slate-500 block mb-1">THROUGHPUT</span>${result.throughput_pct?.toFixed(1)}%</div>
                <div><span class="text-slate-500 block mb-1">SOLVER TIME</span>${result.solver_wall_ms}ms</div>
                <div><span class="text-slate-500 block mb-1">FLOW ALLOCATED</span>${result.total_flow_teu?.toLocaleString()} TEU</div>
                <div><span class="text-slate-500 block mb-1">TOTAL COST</span>$${result.total_cost_usd?.toLocaleString()}</div>
              </div>
              ${result.allocations?.length ? `
              <div class="mt-3 pt-3 border-t border-white/5 space-y-1">
                ${result.allocations.map(a => `
                  <div class="flex justify-between text-[10px] font-mono text-slate-400">
                    <span>${a.route_id}</span>
                    <span class="text-cyan-300">${a.flow_teu?.toLocaleString()} TEU (${a.pct_of_total}%)</span>
                  </div>`).join("")}
              </div>` : ""}
            </div>`;
          resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      } catch (e) {
        window.sfToast(`Optimizer failed: ${e.message}`);
      } finally {
        applyBtn.innerHTML = origHTML;
        applyBtn.disabled = false;
      }
    });
  }

  // ── Filter dropdowns ──────────────────────────────────────────────────
  document.querySelectorAll("[data-severity-filter]").forEach((sel) => {
    sel.addEventListener("change", () => loadBottlenecks());
  });

  // ── Helpers ───────────────────────────────────────────────────────────
  function formatNodeName(nodeId) {
    if (!nodeId) return "—";
    const map = {
      "PORT-CN-SHA": "Shanghai (PVG)",
      "PORT-US-LGB": "Long Beach (LGB)",
      "PORT-EU-HAM": "Hamburg (HAM)",
      "PORT-SG-SIN": "Singapore (SIN)",
      "PORT-AU-SYD": "Sydney (SYD)",
      "PORT-KR-BSN": "Busan (BSN)",
    };
    return map[nodeId] || nodeId.replace(/^(PORT|WAREHOUSE|FACTORY|RETAILER)-/, "").replace(/-/g, " ");
  }

  // ── Main load ─────────────────────────────────────────────────────────
  async function loadBottlenecks() {
    try {
      const [bnData] = await Promise.all([
        SF.api.bottlenecks(),
      ]);
      updateSummaryCards(bnData);
      renderAffectedRoutes(bnData.edges);
      updateAnalysisPanel(bnData);
    } catch (e) {
      window.sfToast("Bottleneck API unavailable — showing last known state");
    }
  }

  SF.autoRefresh(loadBottlenecks, 15_000);
});
