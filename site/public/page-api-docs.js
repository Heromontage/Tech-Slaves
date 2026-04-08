/**
 * page-api-docs.js
 * =====================================================================
 * Wires api.html to:
 *   POST /api/docs/playground  — "Try it Live" button
 *   GET  /health               — API status indicator
 * =====================================================================
 */

document.addEventListener("DOMContentLoaded", () => {

  // ── API Status indicator ──────────────────────────────────────────────
  async function checkApiStatus() {
    const dot = document.getElementById("api-status-dot");
    const label = document.getElementById("api-status-label");

    try {
      const health = await SF.api.health();
      if (dot) {
        dot.className = "w-2 h-2 rounded-full bg-emerald-400";
      }
      if (label) {
        label.textContent = `API Status: ${health.status === "healthy" ? "Operational" : "Degraded"}`;
        label.className = health.status === "healthy" ? "text-[10px] font-mono text-slate-400" : "text-[10px] font-mono text-amber-400";
      }
    } catch {
      if (dot) dot.className = "w-2 h-2 rounded-full bg-red-400";
      if (label) {
        label.textContent = "API Status: Offline";
        label.className = "text-[10px] font-mono text-red-400";
      }
    }
  }

  // ── Playground endpoint tabs ──────────────────────────────────────────
  const ENDPOINT_EXAMPLES = {
    "/routes":                   { method: "GET",  label: "List all monitored routes" },
    "/routes/{route_id}/health": { method: "GET",  label: "Get route health score" },
    "/bottlenecks":              { method: "GET",  label: "List active bottlenecks" },
    "/optimize":                 { method: "POST", label: "Run rerouting optimizer" },
    "/streams/status":           { method: "GET",  label: "Data stream status" },
  };

  let selectedEndpoint = "/routes/{route_id}/health";

  // Highlight active endpoint in the sidebar list
  function setActiveEndpoint(endpoint) {
    selectedEndpoint = endpoint;
    document.querySelectorAll("[data-endpoint]").forEach((el) => {
      const isActive = el.dataset.endpoint === endpoint;
      el.classList.toggle("border-l-2", isActive);
      el.classList.toggle("border-primary-container", isActive);
      el.classList.toggle("bg-surface-container", isActive);
    });

    // Update the playground input if it exists
    const input = document.getElementById("playground-endpoint-input");
    if (input) input.value = endpoint;

    // Update method badge
    const methodEl = document.getElementById("playground-method");
    if (methodEl && ENDPOINT_EXAMPLES[endpoint]) {
      const m = ENDPOINT_EXAMPLES[endpoint].method;
      methodEl.textContent = m;
      methodEl.className = `px-3 py-1 rounded text-[10px] font-mono font-bold w-16 text-center ${m === "GET" ? "bg-emerald-500/10 text-emerald-400" : "bg-cyan-500/10 text-cyan-400"}`;
    }
  }

  // Bind endpoint list items
  document.querySelectorAll("[data-endpoint]").forEach((el) => {
    el.style.cursor = "pointer";
    el.addEventListener("click", () => {
      setActiveEndpoint(el.dataset.endpoint);
    });
  });

  // ── Code example language tabs ────────────────────────────────────────
  const CODE_EXAMPLES = {
    curl: (ep) => `curl -X GET \\
  "https://api.sentinelflow.io/v2${ep}" \\
  -H "Authorization: Bearer YOUR_API_TOKEN" \\
  -H "Content-Type: application/json"`,
    python: (ep) => `import httpx

client = httpx.Client(
    base_url="https://api.sentinelflow.io/v2",
    headers={"Authorization": "Bearer YOUR_API_TOKEN"}
)

response = client.get("${ep}")
data = response.json()
print(data)`,
    nodejs: (ep) => `const response = await fetch(
  "https://api.sentinelflow.io/v2${ep}",
  {
    method: "GET",
    headers: {
      "Authorization": "Bearer YOUR_API_TOKEN",
      "Content-Type": "application/json"
    }
  }
);
const data = await response.json();
console.log(data);`,
  };

  let activeLang = "curl";

  function renderCodeExample() {
    const codeEl = document.getElementById("code-example-block");
    if (!codeEl) return;
    const fn = CODE_EXAMPLES[activeLang];
    if (fn) {
      codeEl.textContent = fn(selectedEndpoint);
    }
  }

  document.querySelectorAll("[data-lang]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeLang = btn.dataset.lang;
      document.querySelectorAll("[data-lang]").forEach((b) => {
        b.className = "text-xs font-medium text-slate-500 hover:text-slate-300";
      });
      btn.className = "text-xs font-bold text-primary-container border-b border-primary-container pb-1";
      renderCodeExample();
    });
  });

  // ── Copy code button ──────────────────────────────────────────────────
  const copyBtn = document.getElementById("btn-copy-code");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const codeEl = document.getElementById("code-example-block");
      if (codeEl) {
        navigator.clipboard.writeText(codeEl.textContent).then(() => {
          copyBtn.querySelector(".material-symbols-outlined").textContent = "check";
          setTimeout(() => {
            if (copyBtn.querySelector(".material-symbols-outlined")) {
              copyBtn.querySelector(".material-symbols-outlined").textContent = "content_copy";
            }
          }, 2000);
        });
      }
    });
  }

  // ── Try it Live button ────────────────────────────────────────────────
  const tryLiveBtn = document.getElementById("btn-try-live");
  if (tryLiveBtn) {
    tryLiveBtn.addEventListener("click", async () => {
      const resultEl = document.getElementById("playground-result");
      const statusEl = document.getElementById("playground-status");
      const timeEl = document.getElementById("playground-time");

      tryLiveBtn.textContent = "Sending…";
      tryLiveBtn.disabled = true;

      // Show loading state
      if (resultEl) {
        resultEl.textContent = "// Waiting for response…";
        resultEl.className = "font-mono text-slate-500 text-xs leading-relaxed";
      }

      try {
        const ep = selectedEndpoint || "/routes/{route_id}/health";
        const method = ENDPOINT_EXAMPLES[ep]?.method || "GET";
        const data = await SF.api.playground(ep, method);

        // Status badge
        if (statusEl) {
          statusEl.textContent = `${data.status_code} ${data.status_code === 200 ? "OK" : "ERROR"}`;
          statusEl.className = data.status_code === 200
            ? "px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-mono text-[10px] font-bold"
            : "px-2 py-0.5 rounded bg-red-500/10 text-red-400 font-mono text-[10px] font-bold";
        }

        if (timeEl) timeEl.textContent = `${data.response_time_ms}ms`;

        // Pretty-print response
        if (resultEl) {
          resultEl.textContent = JSON.stringify(data.data, null, 2);
          resultEl.className = "font-mono text-cyan-300 text-xs leading-relaxed whitespace-pre-wrap";
        }

        // Show the result panel
        const panel = document.getElementById("playground-result-panel");
        if (panel) panel.classList.remove("hidden");

      } catch (e) {
        if (resultEl) {
          resultEl.textContent = `// Error: ${e.message}`;
          resultEl.className = "font-mono text-red-400 text-xs";
        }
        window.sfToast(`Playground error: ${e.message}`);
      } finally {
        tryLiveBtn.textContent = "Try it Live";
        tryLiveBtn.disabled = false;
      }
    });
  }

  // ── Init ──────────────────────────────────────────────────────────────
  checkApiStatus();
  setActiveEndpoint(selectedEndpoint);
  renderCodeExample();

  // Re-check API status every 30 seconds
  setInterval(checkApiStatus, 30_000);
});
