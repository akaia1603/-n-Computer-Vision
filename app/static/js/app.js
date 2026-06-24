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
   GLOBAL STATE & EMOTION SVGS
   ============================================================ */
const CFG = window.FER_CONFIG || { emotions: [], colors: {}, emojis: {}, mode: "webcam" };

// Sleek SVG line-art icons for each emotion
const EMOTION_SVGS = {
  Surprise: `<svg class="emo-icon" viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="14.5" r="2.5"></circle><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`,
  Fear: `<svg class="emo-icon" viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 15.5c1.5 0 2-1 2-1s-0.5-0.5-2-0.5-2 0.5-2 0.5 0.5 1 2 1z"></path><path d="M8 8.5c1-0.5 2-0.5 2.5 0"></path><path d="M16 8.5c-1-0.5-2-0.5-2.5 0"></path><line x1="9" y1="10.5" x2="9.01" y2="10.5"></line><line x1="15" y1="10.5" x2="15.01" y2="10.5"></line></svg>`,
  Disgust: `<svg class="emo-icon" viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M8 15.5c1.5-1 2.5-1 4 0s2.5 1 4 0"></path><path d="M8 9.5l1.5-0.5"></path><path d="M16 9.5l-1.5-0.5"></path><line x1="9" y1="11.5" x2="9.01" y2="11.5"></line><line x1="15" y1="11.5" x2="15.01" y2="11.5"></line></svg>`,
  Happiness: `<svg class="emo-icon" viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`,
  Sadness: `<svg class="emo-icon" viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M16 16s-1.5-2-4-2-4 2-4 2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`,
  Anger: `<svg class="emo-icon" viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M16 16s-1.5-2-4-2-4 2-4 2"></path><path d="M7.5 8.5L10 10"></path><path d="M16.5 8.5L14 10"></path><line x1="9" y1="11" x2="9.01" y2="11"></line><line x1="15" y1="11" x2="15.01" y2="11"></line></svg>`,
  Neutral: `<svg class="emo-icon" viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="15" x2="16" y2="15"></line><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>`
};

// Webcam state
let webcamStream = null;
let isProcessing = false;
let animFrameId  = null;
let probChart    = null;
let fpsCounter   = { frames: 0, lastTime: 0 };
let trackedFaces = [];
let frameSkip    = 0;           // skip every other frame for FPS
let isFirstFrame = true;        // reset smoother on first frame
let saveCounter  = 0;

// Upload state
let selectedFile  = null;
let resultImageB64 = null;

/**
 * Thuật toán Face Tracking và EMA Smoothing
 * Giúp triệt tiêu hoàn toàn rung giật bounding box và nhảy giật cục cảm xúc giữa các frames
 */
function trackAndSmoothFaces(newFaces) {
  const alpha = 0.35; // Hệ số làm mịn EMA cảm xúc (cao hơn = phản hồi nhanh hơn)
  const bboxSmoothAlpha = 0.30; // Hệ số làm mịn EMA tọa độ hộp
  const maxDistance = 150; // Khoảng cách tối đa để khớp cùng 1 khuôn mặt

  const updatedTrackedFaces = [];

  for (const face of newFaces) {
    const centerNew = {
      x: face.bbox.x + face.bbox.w / 2,
      y: face.bbox.y + face.bbox.h / 2
    };

    let bestMatch = null;
    let minDistance = Infinity;

    for (const tracked of trackedFaces) {
      const centerTracked = {
        x: tracked.bbox.x + tracked.bbox.w / 2,
        y: tracked.bbox.y + tracked.bbox.h / 2
      };
      const dist = Math.hypot(centerNew.x - centerTracked.x, centerNew.y - centerTracked.y);
      if (dist < minDistance && dist < maxDistance) {
        minDistance = dist;
        bestMatch = tracked;
      }
    }

    if (bestMatch) {
      // Làm mịn tọa độ bounding box
      const smoothedBbox = {
        x: bboxSmoothAlpha * face.bbox.x + (1 - bboxSmoothAlpha) * bestMatch.bbox.x,
        y: bboxSmoothAlpha * face.bbox.y + (1 - bboxSmoothAlpha) * bestMatch.bbox.y,
        w: bboxSmoothAlpha * face.bbox.w + (1 - bboxSmoothAlpha) * bestMatch.bbox.w,
        h: bboxSmoothAlpha * face.bbox.h + (1 - bboxSmoothAlpha) * bestMatch.bbox.h
      };

      // Làm mịn xác suất phân phối cảm xúc
      const smoothedProbabilities = {};
      let maxProb = -1;
      let topEmotion = face.emotion;

      for (const emotion in face.probabilities) {
        const prevProb = bestMatch.probabilities[emotion] || 0;
        const newProb = face.probabilities[emotion];
        const smoothed = alpha * newProb + (1 - alpha) * prevProb;
        smoothedProbabilities[emotion] = smoothed;
        if (smoothed > maxProb) {
          maxProb = smoothed;
          topEmotion = emotion;
        }
      }

      const updatedFace = {
        bbox: smoothedBbox,
        emotion: topEmotion,
        confidence: maxProb,
        probabilities: smoothedProbabilities,
        id: bestMatch.id,
        framesUnseen: 0
      };
      updatedTrackedFaces.push(updatedFace);
      
      // Loại khỏi hàng chờ của frame trước
      trackedFaces = trackedFaces.filter(t => t.id !== bestMatch.id);
    } else {
      // Khuôn mặt mới
      const newTrackedFace = {
        bbox: { ...face.bbox },
        emotion: face.emotion,
        confidence: face.confidence,
        probabilities: { ...face.probabilities },
        id: Math.random().toString(36).substring(2, 9),
        framesUnseen: 0
      };
      updatedTrackedFaces.push(newTrackedFace);
    }
  }

  // Giữ lại các mặt bị khuất tạm thời tối đa 5 frames
  for (const tracked of trackedFaces) {
    if (tracked.framesUnseen < 5) {
      tracked.framesUnseen++;
      updatedTrackedFaces.push(tracked);
    }
  }

  trackedFaces = updatedTrackedFaces;
  return trackedFaces.filter(f => f.framesUnseen === 0);
}


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
          grid: { color: "rgba(0, 0, 0, 0.05)" },
          ticks: {
            color: "#64748b",
            callback: v => `${(v*100).toFixed(0)}%`,
            font: { size: 11, family: "Inter, sans-serif" }
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            color: "#1e293b",
            font: { size: 11, weight: "600", family: "Inter, sans-serif" },
            callback: (_, idx) => CFG.emotions[idx]
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
    isFirstFrame = true;
    scheduleCapture();

  } catch (err) {
    console.error("Camera error:", err);
    setStatus("error", "Không thể truy cập camera: " + err.message);
  }
}

function stopWebcam() {
  trackedFaces = [];
  isFirstFrame = true;
  frameSkip = 0;
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
  const INTERVAL_MS = 50;  // ~20 FPS target

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

  // Capture frame at lower resolution + quality for faster transfer
  const tmpCanvas = document.createElement("canvas");
  const CAPTURE_W = 320;
  const CAPTURE_H = 240;
  tmpCanvas.width  = CAPTURE_W;
  tmpCanvas.height = CAPTURE_H;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, CAPTURE_W, CAPTURE_H);
  const b64 = tmpCanvas.toDataURL("image/jpeg", 0.4);

  const modelSelect = document.getElementById("model-select");
  const model_type = modelSelect ? modelSelect.value : "keras";

  // Reset server-side smoother on first frame after start
  const reset_smoother = isFirstFrame;
  isFirstFrame = false;

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64, model_type: model_type, reset_smoother: reset_smoother })
    });
    const data = await resp.json();

    if (data.success) {
      const smoothedFaces = trackAndSmoothFaces(data.faces);

      drawBoundingBoxes(canvas, smoothedFaces, video.videoWidth, video.videoHeight);
      updateWebcamResults(smoothedFaces);
      updateFPS();
      document.getElementById("face-count-badge").textContent =
        `${data.face_count} face${data.face_count !== 1 ? "s" : ""}`;

      // Auto-save to history (every 20 frames)
      if (smoothedFaces.length > 0) {
        saveCounter++;
        if (saveCounter >= 20) {
          saveCounter = 0;
          for (const face of smoothedFaces) {
            if (face.low_confidence) continue;
            fetch("/api/save-result", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                emotion: face.emotion,
                confidence: face.confidence,
                probabilities: face.probabilities,
                model_type: model_type,
                face_count: data.face_count,
                source: "webcam",
              }),
            }).catch(() => {});
          }
        }
      }
    }
  } catch (err) {
    console.warn("Predict error:", err);
  }
}

function drawBoundingBoxes(canvas, faces, vw, vh) {
  const ctx = canvas.getContext("2d");
  const CAPTURE_W = 320;
  const CAPTURE_H = 240;
  const scaleX = vw / CAPTURE_W;
  const scaleY = vh / CAPTURE_H;
  canvas.width  = vw;
  canvas.height = vh;
  ctx.clearRect(0, 0, vw, vh);

  faces.forEach((face) => {
    let { x, y, w, h } = face.bbox;
    // Mirror X since video is displayed with scaleX(-1)
    x = CAPTURE_W - x - w;
    // Scale from capture resolution to canvas display resolution
    x = x * scaleX;
    y = y * scaleY;
    w = w * scaleX;
    h = h * scaleY;

    const color = CFG.colors[face.emotion] || "#6366f1";
    const conf  = (face.confidence * 100).toFixed(0);
    const label = `${face.emotion} ${conf}%`;

    // ---- Box ----
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 8;
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur = 0;

    // ---- Corner accents ----
    const len = 14;
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

  const topFace = faces[0];
  if (topFace.low_confidence) {
    document.getElementById("emotion-emoji").innerHTML = `<svg class="emo-icon" viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v.01"/><path d="M12 13V8"/></svg>`;
    document.getElementById("emotion-emoji").style.color = "var(--text-muted)";
    document.getElementById("emotion-label").textContent = "Analyzing...";
    document.getElementById("emotion-label").style.backgroundImage = "none";
    document.getElementById("emotion-label").style.webkitTextFillColor = "var(--text-muted)";
    document.getElementById("conf-bar").style.width = "0%";
    document.getElementById("conf-text").textContent = "Low confidence";
    probChart && updateChart(probChart, topFace.probabilities);
  } else {
    updateEmotionCard(topFace);
    updateChart(probChart, topFace.probabilities);
  }

  facesList.innerHTML = faces.map((f, idx) => `
    <div class="face-item">
      <div class="face-idx">${idx + 1}</div>
      <span class="face-emoji-svg" style="color:${CFG.colors[f.emotion] || '#6366f1'}">${f.low_confidence ? EMOTION_SVGS.Neutral : (EMOTION_SVGS[f.emotion] || EMOTION_SVGS.Neutral)}</span>
      <div class="face-info">
        <div class="face-emotion" style="color:${f.low_confidence ? 'var(--text-muted)' : (CFG.colors[f.emotion] || '#6366f1')}">${f.low_confidence ? 'Analyzing...' : f.emotion}</div>
        <div class="face-conf">${f.low_confidence ? '—' : (f.confidence * 100).toFixed(1) + '%'}</div>
        <div class="face-mini-bar" style="width:${f.low_confidence ? 5 : f.confidence * 100}%; background:${f.low_confidence ? 'var(--text-muted)' : (CFG.colors[f.emotion] || '#6366f1')}"></div>
      </div>
    </div>
  `).join("");
}

function updateEmotionCard(face) {
  const color = CFG.colors[face.emotion] || "#6366f1";
  document.getElementById("emotion-emoji").innerHTML = EMOTION_SVGS[face.emotion] || EMOTION_SVGS.Neutral;
  document.getElementById("emotion-emoji").style.color = color;
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
  document.getElementById("emotion-emoji").innerHTML = EMOTION_SVGS.Neutral;
  document.getElementById("emotion-emoji").style.color = "var(--text-muted)";
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

  const modelSelect = document.getElementById("model-select");
  const model_type = modelSelect ? modelSelect.value : "keras";

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("model_type", model_type);

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
        <div class="empty-icon">
          <svg viewBox="0 0 24 24" width="36" height="36" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </div>
        <p>Không phát hiện khuôn mặt nào trong ảnh.<br>Thử ảnh khác với mặt người rõ hơn.</p>
      </div>`;
    return;
  }

  container.innerHTML = faces.map((face, idx) => {
    const color = CFG.colors[face.emotion] || "#6366f1";
    const conf  = (face.confidence * 100).toFixed(1);

    const probBars = CFG.emotions.map(em => {
      const pct  = ((face.probabilities[em] || 0) * 100).toFixed(1);
      const barW = (face.probabilities[em] || 0) * 100;
      const c    = CFG.colors[em] || "#6366f1";
      return `
        <div class="prob-row">
          <span class="prob-name" style="display:flex; align-items:center; gap:6px; color:${c};">
            ${EMOTION_SVGS[em] || EMOTION_SVGS.Neutral} 
            <span>${em}</span>
          </span>
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
            <div class="face-detail-emotion" style="color:${color}; display:flex; align-items:center; gap:8px;">
              ${EMOTION_SVGS[face.emotion] || EMOTION_SVGS.Neutral} 
              <span>${face.emotion}</span>
            </div>
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
      <div class="empty-icon">
        <svg viewBox="0 0 24 24" width="36" height="36" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
          <circle cx="8.5" cy="8.5" r="1.5"></circle>
          <polyline points="21 15 16 10 5 21"></polyline>
        </svg>
      </div>
      <p>Tải lên ảnh và nhấn Phân tích để xem kết quả</p>
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
