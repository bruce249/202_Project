const materialSelect = document.getElementById("material");
const nAvramiSelect = document.getElementById("nAvrami");
const nVirtualSelect = document.getElementById("nVirtual");
const runBtn = document.getElementById("runBtn");

const physicsT = document.getElementById("physicsT");
const mlT = document.getElementById("mlT");
const physicsTime = document.getElementById("physicsTime");
const mlTime = document.getElementById("mlTime");

let rateChart;
let tttChart;

const chartTheme = {
  ticks: { color: "#d9e8f6" },
  grid: { color: "rgba(126, 163, 198, 0.2)" },
};

function formatTemp(v) {
  return Number.isFinite(v) ? `${v.toFixed(1)} C` : "-";
}

function formatTime(v) {
  return Number.isFinite(v) ? `${v.toExponential(3)} s` : "-";
}

function unpackXY(series) {
  return {
    x: series.map((pair) => pair[0]),
    y: series.map((pair) => pair[1]),
  };
}

function makeLineDataset(label, series, color, dash = []) {
  const xy = unpackXY(series);
  return {
    label,
    data: xy.x.map((x, i) => ({ x, y: xy.y[i] })),
    borderColor: color,
    pointRadius: 0,
    tension: 0.08,
    borderDash: dash,
  };
}

function renderRateChart(data) {
  const datasets = [
    makeLineDataset("Nucleation", data.rate_curves.nucleation, "#58a6ff"),
    makeLineDataset("Growth", data.rate_curves.growth, "#f78166", [8, 6]),
    makeLineDataset("Overall", data.rate_curves.overall, "#3fb950", [4, 4]),
  ];

  if (rateChart) rateChart.destroy();
  rateChart = new Chart(document.getElementById("rateChart"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ...chartTheme,
          title: { display: true, text: "Normalized rate", color: "#92abc2" },
        },
        y: {
          ...chartTheme,
          reverse: true,
          title: { display: true, text: "Temperature (C)", color: "#92abc2" },
        },
      },
      plugins: { legend: { labels: { color: "#d9e8f6" } } },
    },
  });
}

function renderTTTChart(data) {
  const datasets = [
    makeLineDataset("Physics 1%", data.ttt_curves.x_1pct, "#58a6ff"),
    makeLineDataset("Physics 50%", data.ttt_curves.x_50pct, "#d2a8ff", [7, 5]),
    makeLineDataset("Physics 99%", data.ttt_curves.x_99pct, "#f78166"),
    makeLineDataset("ML 1%", data.ttt_curves.ml_1pct, "#ffa657", [6, 4]),
  ];

  if (tttChart) tttChart.destroy();
  tttChart = new Chart(document.getElementById("tttChart"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ...chartTheme,
          type: "logarithmic",
          title: { display: true, text: "Time (s)", color: "#92abc2" },
        },
        y: {
          ...chartTheme,
          title: { display: true, text: "Temperature (C)", color: "#92abc2" },
        },
      },
      plugins: { legend: { labels: { color: "#d9e8f6" } } },
    },
  });
}

async function loadMaterials() {
  const res = await fetch("/api/materials");
  const data = await res.json();
  const preferred = "Zinc";

  materialSelect.innerHTML = "";
  data.materials.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    if (name === preferred) option.selected = true;
    materialSelect.appendChild(option);
  });
}

async function runAnalysis() {
  runBtn.disabled = true;
  runBtn.textContent = "Running...";
  try {
    const params = new URLSearchParams({
      material: materialSelect.value,
      n_avrami: nAvramiSelect.value,
      n_virtual: nVirtualSelect.value,
    });
    const res = await fetch(`/api/analyze?${params.toString()}`);
    if (!res.ok) throw new Error("API request failed");
    const data = await res.json();

    physicsT.textContent = formatTemp(data.nose.physics_temperature_c);
    mlT.textContent = formatTemp(data.nose.ml_temperature_c);
    physicsTime.textContent = formatTime(data.nose.physics_time_s);
    mlTime.textContent = formatTime(data.nose.ml_time_s);

    renderRateChart(data);
    renderTTTChart(data);
  } catch (err) {
    physicsT.textContent = "Error";
    mlT.textContent = "Error";
    physicsTime.textContent = "Error";
    mlTime.textContent = "Error";
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run Analysis";
  }
}

runBtn.addEventListener("click", runAnalysis);

(async () => {
  await loadMaterials();
  await runAnalysis();
})();
