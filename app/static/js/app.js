/**
 * app/static/js/app.js
 * =====================
 * Frontend logic cho Emotion AI Web Demo.
 *
 * Chức năng:
 *   - Webcam mode: capture frames → POST /api/predict → render bbox + chart
 *   - Upload mode: drag & drop / file select → POST /api/upload → render results
 *   - Chart.js: probability bar chart (real-time update)
 *   - Canvas overlay: bounding box + emotion label
 */

"use strict";

/* ============================================================
   GLOBAL STATE
   ============================================================ */
const CFG = window.FER_CONFIG || { emotions: [], colors: {}, emojis: {}, mode: "webcam" };

// Webcam state
let webcamStream = null;
let isProcessing = false;
let animFrameId  = null;
let probChart    = null;
let fpsCounter   = { frames: 0, lastTime: 0 };

// Upload state
let selectedFile  = null;
let resultImageB64 = null;


/* ============================================================
   CHART.JS — Probability Bar Chart
   ============================================================ */

function initChart(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;

  const labels = CFG.emotions;
  const bgColors = labels.map(e => hexToRgba(CFG.colors[e] || "#6366f1", 0.7));
  const borderColors = labels.map(e => CFG.colors[e] || "#6366f1");

  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Confidence",
        data: new Array(labels.length).fill(0),
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${(ctx.raw * 100).toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          min: 0, max: 1,
          grid: { color: "rgba(255,255,255,0.06)" },
          ticks: {
            color: "#94a3b8",
            callback: v => `${(v*100).toFixed(0)}%`,
            font: { size: 11 }
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            color: "#f1f5f9",
            font: { size: 11, weight: "600" },
            callback: (_, idx) => `${CFG.emojis[CFG.emotions[idx]] || ""} ${CFG.emotions[idx]}`
          }
        }
      }
    }
  });
}

function updateChart(chart, probabilities) {
  if (!chart) return;
  chart.data.datasets[0].data = CFG.emotions.map(e => probabilities[e] || 0);
  chart.update("none");
}

function hexToRgba(hex, alpha) {
  const h = hex.replace("#","");
  const r = parseInt(h.slice(0,2),16);
  const g = parseInt(h.slice(2,4),16);
  const b = parseInt(h.slice(4,6),16);
  return `rgba(${r},${g},${b},${alpha})`;
}


/* ============================================================
   WEBCAM MODE
   ============================================================ */

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: {ideal:640}, height: {ideal:480}, facingMode: "user" },
      audio: false
    });

    const video = document.getElementById("webcam-video");
    video.srcObject = webcamStream;
    await video.play();

    // Init overlay canvas
    const canvas = document.getElementById("overlay-canvas");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;

    // UI updates
    document.getElementById("video-placeholder").classList.add("hidden");
    document.getElementById("btn-start").classList.add("hidden");
    document.getElementById("btn-stop").classList.remove("hidden");
    setStatus("active", "Camera đang hoạt động...");

    // Init chart
    probChart = initChart("prob-chart");

    // Start prediction loop
    fpsCounter = { frames: 0, lastTime: performance.now() };
    scheduleCapture();

  } catch (err) {
    console.error("Camera error:", err);
    setStatus("error", "Không thể truy cập camera: " + err.message);
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  const video = document.getElementById("webcam-video");
  video.srcObject = null;

  // Clear canvas
  const canvas = document.getElementById("overlay-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Reset UI
  document.getElementById("video-placeholder").classList.remove("hidden");
  document.getElementById("btn-start").classList.remove("hidden");
  document.getElementById("btn-stop").classList.add("hidden");
  document.getElementById("fps-badge").textContent = "0 FPS";
  document.getElementById("face-count-badge").textContent = "0 faces";
  setStatus("idle", "Chờ kết nối camera...");
  resetEmotionCard();
}

function scheduleCapture() {
  const INTERVAL_MS = 30;  // Giảm xuống 30ms để tăng FPS (nhanh nhất có thể, nhưng phụ thuộc vào backend)

  const loop = async () => {
    if (!webcamStream) return;
    if (!isProcessing) {
      isProcessing = true;
      await captureAndPredict();
      isProcessing = false;
    }
    animFrameId = setTimeout(() => requestAnimationFrame(loop), INTERVAL_MS);
  };
  requestAnimationFrame(loop);
}

async function captureAndPredict() {
  const video  = document.getElementById("webcam-video");
  const canvas = document.getElementById("overlay-canvas");
  if (!video.videoWidth) return;

  // Capture frame
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width  = video.videoWidth;
  tmpCanvas.height = video.videoHeight;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.drawImage(video, 0, 0);
  const b64 = tmpCanvas.toDataURL("image/jpeg", 0.8);

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64 })
    });
    const data = await resp.json();

    if (data.success) {
      // Draw bounding boxes
      drawBoundingBoxes(canvas, data.faces, video.videoWidth, video.videoHeight);
      updateWebcamResults(data.faces);
      updateFPS();
      document.getElementById("face-count-badge").textContent =
        `${data.face_count} face${data.face_count !== 1 ? "s" : ""}`;
    }
  } catch (err) {
    console.warn("Predict error:", err);
  }
}

function drawBoundingBoxes(canvas, faces, vw, vh) {
  const ctx = canvas.getContext("2d");
  canvas.width  = vw;
  canvas.height = vh;
  ctx.clearRect(0, 0, vw, vh);

  faces.forEach((face, idx) => {
    const { x, y, w, h } = face.bbox;
    const color = CFG.colors[face.emotion] || "#6366f1";
    const conf  = (face.confidence * 100).toFixed(0);
    const label = `${CFG.emojis[face.emotion] || ""} ${face.emotion} ${conf}%`;

    // ---- Box ----
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 8;
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur = 0;

    // ---- Corner accents ----
    const len = 18;
    ctx.lineWidth = 3;
    [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]].forEach(([cx,cy,sx,sy]) => {
      ctx.beginPath();
      ctx.moveTo(cx, cy + sy * len);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx + sx * len, cy);
      ctx.stroke();
    });

    // ---- Label background ----
    ctx.font = "bold 13px Inter, sans-serif";
    const textW = ctx.measureText(label).width;
    const labelY = y > 30 ? y - 8 : y + h + 22;
    ctx.fillStyle = hexToRgba(color, 0.85);
    ctx.beginPath();
    ctx.roundRect(x - 1, labelY - 18, textW + 12, 22, 4);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 5, labelY - 2);
  });
}

function updateWebcamResults(faces) {
  const facesList = document.getElementById("faces-list");

  if (!faces || faces.length === 0) {
    facesList.innerHTML = '<p class="empty-state">Chưa phát hiện khuôn mặt...</p>';
    resetEmotionCard();
    return;
  }

  // Update top emotion card (face chính — lớn nhất / đầu tiên)
  const topFace = faces[0];
  updateEmotionCard(topFace);

  // Update chart
  updateChart(probChart, topFace.probabilities);

  // Update faces list
  facesList.innerHTML = faces.map((f, idx) => `
    <div class="face-item">
      <div class="face-idx">${idx + 1}</div>
      <span class="face-emoji">${CFG.emojis[f.emotion] || "😐"}</span>
      <div class="face-info">
        <div class="face-emotion" style="color:${CFG.colors[f.emotion] || '#6366f1'}">${f.emotion}</div>
        <div class="face-conf">${(f.confidence * 100).toFixed(1)}%</div>
        <div class="face-mini-bar" style="width:${f.confidence * 100}%; background:${CFG.colors[f.emotion] || '#6366f1'}"></div>
      </div>
    </div>
  `).join("");
}

function updateEmotionCard(face) {
  const color = CFG.colors[face.emotion] || "#6366f1";
  document.getElementById("emotion-emoji").textContent  = CFG.emojis[face.emotion] || "😐";
  document.getElementById("emotion-label").textContent  = face.emotion;
  document.getElementById("emotion-label").style.backgroundImage =
    `linear-gradient(135deg, ${color}, ${color}aa)`;
  document.getElementById("emotion-label").style["-webkit-background-clip"] = "text";
  document.getElementById("emotion-label").style["-webkit-text-fill-color"] = "transparent";
  const pct = Math.round(face.confidence * 100);
  document.getElementById("conf-bar").style.width       = `${pct}%`;
  document.getElementById("conf-bar").style.background  =
    `linear-gradient(90deg, ${color}, ${color}88)`;
  document.getElementById("conf-text").textContent = `${pct}% confidence`;
}

function resetEmotionCard() {
  document.getElementById("emotion-emoji").textContent = "😐";
  document.getElementById("emotion-label").textContent = "—";
  document.getElementById("conf-bar").style.width = "0%";
  document.getElementById("conf-text").textContent = "—";
}

function updateFPS() {
  fpsCounter.frames++;
  const now  = performance.now();
  const diff = now - fpsCounter.lastTime;
  if (diff >= 1000) {
    const fps = Math.round(fpsCounter.frames * 1000 / diff);
    document.getElementById("fps-badge").textContent = `${fps} FPS`;
    fpsCounter.frames = 0;
    fpsCounter.lastTime = now;
  }
}

function setStatus(state, text) {
  const dot  = document.getElementById("status-dot");
  const span = document.getElementById("status-text");
  if (!dot || !span) return;
  dot.className = "status-dot" + (state === "active" ? " active" : state === "error" ? " error" : "");
  span.textContent = text;
}


/* ============================================================
   UPLOAD MODE
   ============================================================ */

function handleDragOver(e) {
  e.preventDefault();
  document.getElementById("drop-zone").classList.add("dragover");
}
function handleDragLeave(e) {
  document.getElementById("drop-zone").classList.remove("dragover");
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById("drop-zone").classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) processFile(files[0]);
}
function handleFileSelect(e) {
  const files = e.target.files;
  if (files.length > 0) processFile(files[0]);
}

function processFile(file) {
  if (!file.type.startsWith("image/")) {
    showToast("Chỉ hỗ trợ file ảnh (PNG, JPG, JPEG, WEBP).");
    return;
  }
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById("preview-img");
    img.src = e.target.result;
    document.getElementById("preview-wrapper").classList.remove("hidden");
    document.getElementById("drop-zone").style.display = "none";
    document.getElementById("btn-analyze").disabled = false;
    // Reset result
    document.getElementById("result-image-card").classList.add("hidden");
    document.getElementById("chart-card").classList.add("hidden");
    resetFacesDetail();
  };
  reader.readAsDataURL(file);
}

async function analyzeImage() {
  if (!selectedFile) return;

  const btn = document.getElementById("btn-analyze");
  btn.disabled = true;
  document.getElementById("btn-analyze-text").textContent = "Đang phân tích...";
  document.getElementById("btn-analyze-icon").textContent = "⏳";
  document.getElementById("progress-bar").classList.remove("hidden");
  document.getElementById("progress-fill").style.width = "60%";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const resp = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await resp.json();

    document.getElementById("progress-fill").style.width = "100%";

    if (!data.success) {
      showToast(data.error || "Phân tích thất bại.");
      return;
    }

    // Show annotated image
    if (data.annotated_image) {
      resultImageB64 = data.annotated_image;
      document.getElementById("result-image").src = data.annotated_image;
      document.getElementById("result-image-card").classList.remove("hidden");
    }

    // Show face details
    renderFaceDetails(data.faces);

    // Show chart (of first / most confident face)
    if (data.faces.length > 0) {
      document.getElementById("chart-card").classList.remove("hidden");
      if (!probChart) probChart = initChart("prob-chart");
      updateChart(probChart, data.faces[0].probabilities);
    }

    document.getElementById("face-count-badge").textContent =
      `${data.face_count} khuôn mặt`;

  } catch (err) {
    showToast("Lỗi kết nối server: " + err.message);
  } finally {
    btn.disabled = false;
    document.getElementById("btn-analyze-text").textContent = "Phân tích cảm xúc";
    document.getElementById("btn-analyze-icon").textContent = "🔍";
    setTimeout(() => {
      document.getElementById("progress-bar").classList.add("hidden");
      document.getElementById("progress-fill").style.width = "0%";
    }, 600);
  }
}

function renderFaceDetails(faces) {
  const container = document.getElementById("faces-detail-list");
  if (!faces || faces.length === 0) {
    container.innerHTML = `
      <div class="empty-state-upload">
        <div class="empty-icon">🔍</div>
        <p>Không phát hiện khuôn mặt nào trong ảnh.<br>Thử ảnh khác với mặt người rõ hơn.</p>
      </div>`;
    return;
  }

  container.innerHTML = faces.map((face, idx) => {
    const color = CFG.colors[face.emotion] || "#6366f1";
    const emoji = CFG.emojis[face.emotion] || "😐";
    const conf  = (face.confidence * 100).toFixed(1);

    const probBars = CFG.emotions.map(em => {
      const pct  = ((face.probabilities[em] || 0) * 100).toFixed(1);
      const barW = (face.probabilities[em] || 0) * 100;
      const c    = CFG.colors[em] || "#6366f1";
      return `
        <div class="prob-row">
          <span class="prob-name">${CFG.emojis[em] || ""} ${em}</span>
          <div class="prob-track">
            <div class="prob-fill" style="width:${barW}%;background:${c}"></div>
          </div>
          <span class="prob-val">${pct}%</span>
        </div>`;
    }).join("");

    return `
      <div class="face-detail-item">
        <div class="face-detail-header">
          <div class="face-detail-num" style="background:${color}">${idx + 1}</div>
          <div>
            <div class="face-detail-emotion" style="color:${color}">${emoji} ${face.emotion}</div>
            <div class="face-detail-conf">${conf}% confidence</div>
          </div>
        </div>
        <div class="prob-bars">${probBars}</div>
      </div>`;
  }).join("");
}

function resetFacesDetail() {
  document.getElementById("faces-detail-list").innerHTML = `
    <div class="empty-state-upload">
      <div class="empty-icon">🔮</div>
      <p>Upload ảnh và nhấn Phân tích để xem kết quả</p>
    </div>`;
  document.getElementById("face-count-badge").textContent = "0 khuôn mặt";
}

function downloadResult() {
  if (!resultImageB64) return;
  const a = document.createElement("a");
  a.href = resultImageB64;
  a.download = `emotion_result_${Date.now()}.jpg`;
  a.click();
}


/* ============================================================
   TOAST NOTIFICATION
   ============================================================ */
function showToast(message) {
  const toast = document.getElementById("error-toast");
  if (!toast) return;
  document.getElementById("toast-message").textContent = message;
  toast.classList.remove("hidden");
  setTimeout(hideToast, 5000);
}
function hideToast() {
  const toast = document.getElementById("error-toast");
  if (toast) toast.classList.add("hidden");
}


/* ============================================================
   INIT
   ============================================================ */
document.addEventListener("DOMContentLoaded", () => {
  if (CFG.mode === "webcam") {
    setStatus("idle", "Chờ kết nối camera...");
  }
  if (CFG.mode === "upload") {
    probChart = null;  // lazy init khi cần
  }

  // Navbar scroll effect
  const navbar = document.getElementById("navbar");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 10) {
      navbar.style.boxShadow = "0 4px 24px rgba(0,0,0,0.4)";
    } else {
      navbar.style.boxShadow = "none";
    }
  });
});
