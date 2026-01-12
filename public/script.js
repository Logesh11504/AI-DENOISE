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
    // We call the API with a special flag or empty form to trigger default logic
    await processFingerprint(true);
  }

  // Trigger file input when clicking the drop zone
  dropZone.addEventListener("click", () => fileInput.click());

  // Enable button and show filename when a file is selected
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      processBtn.disabled = false;
      dropZone.querySelector(".drop-zone__prompt").textContent =
        fileInput.files[0].name;
      // Clear previous results visually when a new file is picked
      resultsGrid.style.opacity = "0.5";
    }
  });

  /**
   * Main function to send the image to the FastAPI backend,
   * receive the processed results, and update the UI.
   */
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
      // If isDefault is true, we send a request without a file
      const response = await fetch("/api/predict", {
        method: "POST",
        body: isDefault ? new FormData() : formData,
      });

      if (!response.ok) throw new Error(`Server Error: ${response.status}`);

      const data = await response.json();

      // UI Update Logic
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

      statusBadge.style.display = "block";
      statusBadge.innerHTML = `VERIFICATION SCORE: ${data.match_score}`;
      statusBadge.style.color = data.match_score > 10 ? "#00ff88" : "#ff4d4d";
    } catch (error) {
      console.error("Process Error:", error);
      if (!isDefault) alert("Error: " + error.message);
    } finally {
      processBtn.textContent = "Initialize Reconstruction";
      processBtn.disabled = isDefault; // Keep disabled if no file selected
    }
  }

  processBtn.onclick = () => processFingerprint(false);
});
