/**
 * Visual Hierarchy & Attention Predictor - Client Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const statusIndicator = document.querySelector(".status-indicator");
  const statusText = document.getElementById("statusText");
  const modelToggle = document.getElementById("modelToggle");
  const colormapSelect = document.getElementById("colormapSelect");
  const opacitySlider = document.getElementById("opacitySlider");
  const opacityVal = document.getElementById("opacityVal");
  const compareToggle = document.getElementById("compareToggle");

  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const samplesGrid = document.getElementById("samplesGrid");

  const loadingState = document.getElementById("loadingState");
  const resultsSection = document.getElementById("resultsSection");
  const viewerTabs = document.getElementById("viewerTabs");
  const singleImageViewer = document.getElementById("singleImageViewer");
  const dualImageViewer = document.getElementById("dualImageViewer");
  const primaryDisplayImg = document.getElementById("primaryDisplayImg");
  const baselineDisplayImg = document.getElementById("baselineDisplayImg");
  const finetunedDisplayImg = document.getElementById("finetunedDisplayImg");
  const comparisonTabBtn = document.getElementById("comparisonTabBtn");
  const downloadBtn = document.getElementById("downloadBtn");

  const statPeak = document.getElementById("statPeak");
  const statMean = document.getElementById("statMean");
  const statTop3 = document.getElementById("statTop3");
  const statElements = document.getElementById("statElements");
  const hierarchyTableBody = document.getElementById("hierarchyTableBody");

  // State
  let currentModel = "model2";
  let activeTab = "overlay";
  let currentPredictionData = null;
  let activeFile = null;
  let activeSampleId = null;

  // Initialize
  checkSystemStatus();
  loadSampleGallery();

  // Model Selection Toggle
  modelToggle.querySelectorAll(".segment-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      modelToggle.querySelectorAll(".segment-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentModel = btn.dataset.model;
      if (activeFile || activeSampleId) {
        runPrediction();
      }
    });
  });

  // Colormap and Opacity Changes
  colormapSelect.addEventListener("change", () => {
    if (activeFile || activeSampleId) runPrediction();
  });

  opacitySlider.addEventListener("input", (e) => {
    opacityVal.textContent = `${e.target.value}%`;
  });

  opacitySlider.addEventListener("change", () => {
    if (activeFile || activeSampleId) runPrediction();
  });

  // Comparison toggle
  compareToggle.addEventListener("change", (e) => {
    if (e.target.checked) {
      comparisonTabBtn.classList.remove("hidden");
      switchTab("comparison");
    } else {
      comparisonTabBtn.classList.add("hidden");
      if (activeTab === "comparison") switchTab("overlay");
    }
    if (activeFile || activeSampleId) runPrediction();
  });

  // Browse and Drag/Drop
  browseBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Viewer Tab Switching
  viewerTabs.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab(btn.dataset.tab);
    });
  });

  // Download Map
  downloadBtn.addEventListener("click", () => {
    if (!currentPredictionData) return;
    const a = document.createElement("a");
    a.href = primaryDisplayImg.src;
    a.download = `attention_prediction_${activeTab}.png`;
    a.click();
  });

  function switchTab(tabName) {
    activeTab = tabName;
    viewerTabs.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tabName);
    });

    if (!currentPredictionData) return;

    if (tabName === "comparison") {
      singleImageViewer.classList.add("hidden");
      dualImageViewer.classList.remove("hidden");
      baselineDisplayImg.src = currentPredictionData.images.baseline_overlay || currentPredictionData.images.overlay;
      finetunedDisplayImg.src = currentPredictionData.images.overlay;
    } else {
      dualImageViewer.classList.add("hidden");
      singleImageViewer.classList.remove("hidden");

      if (tabName === "overlay") {
        primaryDisplayImg.src = currentPredictionData.images.overlay;
      } else if (tabName === "hierarchy") {
        primaryDisplayImg.src = currentPredictionData.images.hierarchy;
      } else if (tabName === "pure_heatmap") {
        primaryDisplayImg.src = currentPredictionData.images.heatmap;
      } else if (tabName === "original") {
        primaryDisplayImg.src = currentPredictionData.images.original;
      }
    }
  }

  function handleFileSelected(file) {
    activeFile = file;
    activeSampleId = null;
    runPrediction();
  }

  function handleSampleSelected(sampleId) {
    activeSampleId = sampleId;
    activeFile = null;
    runPrediction();
  }

  // API: Status Check
  async function checkSystemStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.status === "ready") {
        statusIndicator.style.backgroundColor = "var(--accent-emerald)";
        statusText.textContent = `Model 2 Ready · Val CC: ${data.model2.val_cc.toFixed(4)} · Device: ${data.device.toUpperCase()}`;
      } else {
        statusText.textContent = "Engine Standby";
      }
    } catch (err) {
      statusIndicator.style.backgroundColor = "var(--accent-gold)";
      statusText.textContent = "Connecting to Local Backend...";
    }
  }

  // API: Load Sample Gallery
  async function loadSampleGallery() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      samplesGrid.innerHTML = "";
      if (data.samples && data.samples.length > 0) {
        data.samples.forEach((sample) => {
          const card = document.createElement("div");
          card.className = "sample-card";
          card.title = sample.title;
          card.innerHTML = `<img src="${sample.thumbnail_url}" alt="${sample.title}" loading="lazy">`;
          card.addEventListener("click", () => handleSampleSelected(sample.id));
          samplesGrid.appendChild(card);
        });
      }
    } catch (err) {
      console.warn("Failed to load sample gallery:", err);
    }
  }

  // API: Run Attention Prediction
  async function runPrediction() {
    loadingState.classList.remove("hidden");
    resultsSection.classList.add("hidden");

    const formData = new FormData();
    if (activeFile) {
      formData.append("file", activeFile);
    } else if (activeSampleId) {
      formData.append("sample_id", activeSampleId);
    } else {
      loadingState.classList.add("hidden");
      return;
    }

    formData.append("model", currentModel);
    formData.append("colormap", colormapSelect.value);
    formData.append("alpha", (opacitySlider.value / 100).toString());
    formData.append("compare_baseline", compareToggle.checked ? "true" : "false");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Prediction request failed.");
      }

      const data = await res.json();
      currentPredictionData = data;
      renderResults(data);
    } catch (err) {
      alert(`Prediction Error: ${err.message}`);
    } finally {
      loadingState.classList.add("hidden");
      resultsSection.classList.remove("hidden");
      resultsSection.scrollIntoView({ behavior: "smooth" });
    }
  }

  // Render Predictions, Stats, and Hierarchy Table
  function renderResults(data) {
    // 1. Stats Cards
    statPeak.textContent = data.stats.peak_attention.toFixed(2);
    statMean.textContent = data.stats.mean_attention.toFixed(3);
    statTop3.textContent = `${data.stats.top_3_focus_share_pct}%`;
    statElements.textContent = data.stats.elements_detected;

    // 2. Render Hierarchy Leaderboard Table
    hierarchyTableBody.innerHTML = "";
    if (data.elements && data.elements.length > 0) {
      data.elements.forEach((el) => {
        const tr = document.createElement("tr");
        const rankClass = el.rank <= 3 ? `rank-${el.rank}` : "rank-sub";

        tr.innerHTML = `
          <td>
            <span class="rank-badge ${rankClass}">#${el.rank}</span>
          </td>
          <td>
            <strong>${el.category}</strong>
          </td>
          <td>
            <div class="progress-bar-wrap">
              <div class="progress-bar">
                <div class="progress-fill" style="width: ${el.attention_percentage}%"></div>
              </div>
              <span>${el.attention_percentage}%</span>
            </div>
          </td>
          <td><code>${el.mean_score.toFixed(4)}</code></td>
          <td>${el.feedback}</td>
        `;

        // Interactive row hover: switch to hierarchy view to inspect bounding box
        tr.addEventListener("mouseenter", () => {
          if (activeTab !== "hierarchy" && activeTab !== "comparison") {
            // Soft preview indicator
          }
        });

        hierarchyTableBody.appendChild(tr);
      });
    }

    // 3. Update Display Image in Active Tab
    switchTab(activeTab);
  }
});
