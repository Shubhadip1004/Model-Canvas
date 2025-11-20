// frontend/app.js
let currentDataset = null;
let currentAlgo = document.getElementById("algoSelect").value;

// wire controls
document.getElementById("algoSelect").addEventListener("change", (e) => {
  currentAlgo = e.target.value;
  if (currentDataset) {
    loadAndPlot(currentDataset, currentAlgo);
  }
});

function selectDataset(name) {
  currentDataset = name;
  document.getElementById("meta").innerText = `Loaded dataset: ${name}`;
  loadAndPlot(name, currentAlgo);
}

async function loadAndPlot(dataset, algo) {
  setStatus(`Training ${algo} on ${dataset} ...`);
  try {
    const API = "https://model-canvas-backend.onrender.com";
    const resp = await fetch(`${API}/train?dataset=${dataset}&algo=${algo}`);
    if (!resp.ok) {
      const err = await resp.json();
      setStatus(`Error: ${err.error || resp.statusText}`);
      return;
    }
    const data = await resp.json();
    setStatus(`Trained ${algo} on ${dataset}`);
    showMetrics(data);
    renderPlot(data);
  } catch (err) {
    setStatus("Network error: " + err.message);
  }
}

function setStatus(txt) {
  document.getElementById("meta").innerText = txt;
}

function showMetrics(data) {
  document.getElementById("metrics").innerText = `Accuracy: ${data.accuracy}  |  Classes: ${data.classes.join(", ")}`;
}

// delegating plotting to plot.js
function renderPlot(data) {
  window.renderModelCanvasPlot(data);
}
