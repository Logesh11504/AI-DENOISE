document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const dropZone = document.getElementById("dropZone");
  const processBtn = document.getElementById("processBtn");
  const statusBadge = document.getElementById("statusBadge");
  const resultsGrid = document.getElementById("resultsGrid");

  console.log("Fingerprint AI System Initialized");

  initDefault();

  async function initDefault() {
    console.log("Loading default prediction...");
    dropZone.querySelector(".drop-zone__prompt").textContent =
      "Demo_Fingerprint_Image.png";
    await processFingerprint(true);
  }

  dropZone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      processBtn.disabled = false;
      dropZone.querySelector(".drop-zone__prompt").textContent =
        fileInput.files[0].name;
      resultsGrid.style.opacity = "0.5";
    }
  });

  async function processFingerprint(isDefault = false) {
    const formData = new FormData();

    if (!isDefault) {
      const file = fileInput.files[0];
      if (!file) return;
      formData.append("file", file);
      processBtn.textContent = "Analyzing User Image...";
    } else {
      processBtn.textContent = "Loading System Default...";
    }

    processBtn.disabled = true;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: isDefault ? new FormData() : formData,
      });

      if (!response.ok) throw new Error(`Server Error: ${response.status}`);

      const data = await response.json();

      resultsGrid.style.display = "grid";
      resultsGrid.style.opacity = "1";

      document.getElementById(
        "inputPreview"
      ).innerHTML = `<img src="${data.original}">`;
      document.getElementById(
        "noisyPreview"
      ).innerHTML = `<img src="${data.noisy}">`;
      document.getElementById(
        "outputPreview"
      ).innerHTML = `<img src="${data.denoised}">`;

      const score = data.match_score;
      let performanceLabel = "";
      let statusClass = "";

      if (score > 25) {
        performanceLabel = "EXCELLENT RECONSTRUCTION (High Confidence)";
        statusClass = "status-success";
      } else if (score > 10) {
        performanceLabel = "SUCCESSFUL RECOVERY (Medium Confidence)";
        statusClass = "status-success";
      } else {
        performanceLabel = "RECONSTRUCTION FAILED (Low Feature Match)";
        statusClass = "status-error";
      }

      statusBadge.innerHTML = `
    <div class="score-label">ORB Feature Match Score</div>
    <div class="score-value">${score}</div>
    <div class="performance-text">${performanceLabel}</div>
`;

      statusBadge.className = "status-badge " + statusClass;
      statusBadge.style.display = "flex";
    } catch (error) {
      console.error("Process Error:", error);
      if (!isDefault) alert("Error: " + error.message);
    } finally {
      processBtn.textContent = "Initialize Reconstruction";
      processBtn.disabled = isDefault;
    }
  }

  processBtn.onclick = () => processFingerprint(false);
});
