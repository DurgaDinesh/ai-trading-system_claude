// NiftySniper Dashboard — Client-side JavaScript

// ── Cumulative P&L Chart ────────────────────────────────────────────────────
async function renderPnLChart() {
  const canvas = document.getElementById("pnlChart");
  if (!canvas) return;

  const resp = await fetch("/api/trades?limit=100");
  const trades = await resp.json();

  const closed = trades.filter(t => t.status === "CLOSED" && t.net_pnl !== null)
    .reverse();

  let cumulative = 0;
  const labels = [];
  const data = [];
  const bgColors = [];

  for (const trade of closed) {
    cumulative += trade.net_pnl;
    labels.push(trade.instrument ? trade.instrument.slice(-12) : "");
    data.push(cumulative.toFixed(2));
    bgColors.push(cumulative >= 0 ? "rgba(0, 212, 170, 0.8)" : "rgba(233, 69, 96, 0.8)");
  }

  if (window._pnlChart) window._pnlChart.destroy();

  window._pnlChart = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Cumulative Net P&L (₹)",
        data,
        borderColor: cumulative >= 0 ? "#00d4aa" : "#e94560",
        backgroundColor: "rgba(0, 212, 170, 0.05)",
        fill: true,
        tension: 0.3,
        pointRadius: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#eaeaea" } },
        tooltip: {
          callbacks: {
            label: ctx => `₹${Number(ctx.parsed.y).toLocaleString("en-IN", {maximumFractionDigits: 2})}`,
          }
        }
      },
      scales: {
        x: { ticks: { color: "#7a7a9a", maxTicksLimit: 10 }, grid: { color: "#2a2a4a" } },
        y: {
          ticks: {
            color: "#7a7a9a",
            callback: v => `₹${Number(v).toLocaleString("en-IN")}`,
          },
          grid: { color: "#2a2a4a" },
        },
      },
    },
  });
}

// ── Auto-refresh status every 30s ───────────────────────────────────────────
async function refreshStatus() {
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();
    const dot = document.querySelector(".status-dot");
    const text = document.querySelector(".status-text");
    if (dot && text) {
      if (data.halted) {
        dot.className = "status-dot dot-red";
        text.textContent = "HALTED";
        text.className = "status-text text-red";
      } else if (data.paused) {
        dot.className = "status-dot dot-yellow";
        text.textContent = "PAUSED";
        text.className = "status-text text-yellow";
      } else {
        dot.className = "status-dot dot-green";
        text.textContent = "RUNNING";
        text.className = "status-text text-green";
      }
    }
  } catch (e) {
    console.warn("Status refresh failed:", e);
  }
}

// ── Initialize ───────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  renderPnLChart();
  setInterval(renderPnLChart, 60000);   // Refresh chart every minute
  setInterval(refreshStatus, 30000);
});
