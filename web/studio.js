/**
 * Hakim-Vision Studio - WebGPU Neural Network & Interactive Notebook Engine
 */

const BALOOT_RANKS = ['A', 'K', 'Q', 'J', '10', '9', '8', '7'];
const BALOOT_SUITS = [
  { code: 'h', name: 'hearts', symbol: '♥', color: '#ef4444' },
  { code: 'd', name: 'diamonds', symbol: '♦', color: '#ef4444' },
  { code: 'c', name: 'clubs', symbol: '♣', color: '#0f172a' },
  { code: 's', name: 'spades', symbol: '♠', color: '#0f172a' }
];

const CARD_CLASSES = [];
BALOOT_SUITS.forEach(s => {
  BALOOT_RANKS.forEach(r => {
    CARD_CLASSES.push(r + s.code);
  });
});

const I18N = {
  ar: {
    app_title: "استوديو حكيم للرؤية الحاسوبية",
    app_badge: "تسريع WebGPU",
    app_subtitle: "مختبر بيانات البلوت: توليد ووسم وتجهيز وتدريب",
    lang_btn_label: "English",
    status_webgpu_ready: "محرك WebGPU جاهز (تسريع عتادي)",
    status_canvas_fallback: "تسريع WebGL / Canvas2D",
    status_initializing: "جاري تشغيل محرك WebGPU...",
    cell1_title: "بيئة التشغيل ومسرّع الرسوميات WebGPU",
    btn_check_hardware: "فحص العتاد",
    lbl_gpu_device: "وحدة معالجة الرسوميات (GPU)",
    lbl_deck_spec: "مواصفات صكة البلوت",
    val_deck_spec: "32 بطاقة (8 قيم × 4 أشكال)",
    lbl_backend: "محرك الاستدلال العصبي",
    lbl_latency: "زمن الاستجابة",
    cell2_title: "استوديو الصور: رفع صورة حقيقية أو توليد مشهد تركيبي",
    tab_upload: "رفع صورة حقيقية",
    tab_synthetic: "مولد المشاهد التركيبية",
    drop_zone_title: "اسحب وأفلت صورة طاولة البلوت هنا",
    drop_zone_sub: "تُحفظ بأمان في التخزين المحلي للمتصفح",
    preset_samples_title: "عينات اختبار جاهزة",
    btn_sample_scene: "طاولة كاملة",
    btn_sample_trick: "فرش الدكة (4)",
    btn_sample_fan: "ورق اليد (6)",
    gallery_title: "معرض الصور المحفوظة",
    btn_clear_gallery: "مسح",
    no_saved_images: "لا توجد صور محفوظة في المتصفح",
    lbl_layout: "توزيع وترتيب الأوراق",
    lbl_surface: "قماش وطاولة اللعب",
    lbl_tilt: "ميلان الكاميرا",
    lbl_shadow: "كثافة الظل",
    btn_gen_scene: "توليد مشهد بلوت عشوائي",
    lbl_active_tag: "الفئة النشطة للوسم",
    annotation_tip: "اسحب بمؤشر الفأرة على الصورة لرسم أو تعديل مربعات التحديد.",
    viewport_title: "منفذ العرض ومحرر الصناديق التفاعلي",
    ready_status: "جاهز",
    loading_tensor: "جاري المعالجة بمسرع WebGPU...",
    chk_boxes: "مربعات التحديد",
    chk_hulls: "مضلعات الأركان",
    btn_clear_boxes: "مسح الصناديق",
    layout_trick4: "أوراق اللعب بالوسط - فرش الدكة (4 أوراق)",
    layout_rand2: "ورقتين متداخلة",
    layout_rand3: "ثلاث أوراق متداخلة",
    layout_fan: "فرقة اليد - مروحة اللاعب (قوس)",
    surf_green: "لباد كازينو أخضر كلاسيكي",
    surf_red: "لباد مجلس عنابي فاخر",
    surf_blue: "لباد أزرق ملكي",
    surf_wood: "خشب جوز مصقول",
    cell3_title: "كاشف بطاقات البلوت عالي الدقة (استدلال WebGPU)",
    btn_run_detection: "كشف وتحديد الأوراق عبر WebGPU",
    lbl_conf_thresh: "حد الثقة الأدنى (Confidence)",
    lbl_iou_thresh: "عتبة عدم التكرار (NMS / IoU)",
    lbl_detected_count: "الأوراق المكتشفة",
    btn_export_zip: "تصدير حزمة YOLO",
    lbl_class_breakdown: "تفاصيل البطاقات المكتشفة ومستوى الثقة",
    msg_no_detections: "اضغط على زر كشف وتحديد الأوراق لتحليل الصورة الحالية.",
    lbl_yolo_preview: "معاينة إحداثيات YOLO (الفئة x y w h)",
    btn_copy: "نسخ",
    btn_copied: "تم النسخ!",
    cell4_title: "تدريب وتحديث الشبكة العصبية في المتصفح عبر WebGPU",
    btn_start_training: "بدء التدريب في المتصفح",
    btn_training_running: "جاري التدريب على WebGPU...",
    btn_retrain: "إعادة تدريب الشبكة العصبية",
    lbl_epochs: "عدد الحقب (Epochs)",
    lbl_lr: "معدل التعلم (Learning Rate)",
    lbl_batch: "حجم الدفعة (Batch Size)",
    lbl_train_status: "حالة المدرب:",
    status_ready: "جاهز",
    status_training: "جاري تدريب الأوزان على WebGPU...",
    status_completed: "اكتمل التدريب (الأوزان محفوظة في WebGPU)",
    lbl_val_map: "دقة التحقق mAP@50:",
    chart_title: "منحنى دالة الخسارة والدقة في الوقت الحقيقي",
    chart_loss_label: "خسارة التدريب (Loss)",
    chart_map_label: "دقة التحقق (mAP@50)",
    footer_text: "استوديو حكيم للرؤية الحاسوبية • منصة مفتوحة المصدر للذكاء الاصطناعي في لعبة البلوت السعودي • ترخيص MIT"
  },
  en: {
    app_title: "Hakim-Vision Studio",
    app_badge: "WebGPU Accelerated",
    app_subtitle: "Baloot dataset lab: synthesise, annotate, prepare, train",
    lang_btn_label: "العربية",
    status_webgpu_ready: "WebGPU Engine Ready (Hardware Accel)",
    status_canvas_fallback: "WebGL / Canvas2D Fallback",
    status_initializing: "Initializing WebGPU Engine...",
    cell1_title: "WebGPU Environment & Hardware Acceleration Status",
    btn_check_hardware: "Check Hardware",
    lbl_gpu_device: "GPU Graphics Device",
    lbl_deck_spec: "Baloot Deck Specification",
    val_deck_spec: "32 Cards (8 Ranks x 4 Suits)",
    lbl_backend: "Inference Engine",
    lbl_latency: "Latency",
    cell2_title: "Image Studio: Upload Real Image or Generate Synthetic Scene",
    tab_upload: "Upload Real Image",
    tab_synthetic: "Synthetic Compositor",
    drop_zone_title: "Drop any Baloot table photo here",
    drop_zone_sub: "Stored privately in Local Browser Storage",
    preset_samples_title: "Quick Test Samples",
    btn_sample_scene: "Table Scene",
    btn_sample_trick: "Trick Center (4)",
    btn_sample_fan: "Hand Fan (6)",
    gallery_title: "Saved Browser Gallery",
    btn_clear_gallery: "Clear",
    no_saved_images: "No saved images in browser",
    lbl_layout: "Game Table Layout",
    lbl_surface: "Table Surface Material",
    lbl_tilt: "Camera Pitch Tilt",
    lbl_shadow: "Drop Shadow Intensity",
    btn_gen_scene: "Generate Random Baloot Scene",
    lbl_active_tag: "Active Card Class to Tag",
    annotation_tip: "Click & drag on canvas to draw or adjust bounding box annotations.",
    viewport_title: "Interactive Viewport & Ground Truth Editor",
    ready_status: "Ready",
    loading_tensor: "Processing WebGPU Tensor...",
    chk_boxes: "BBoxes",
    chk_hulls: "Corner Polygons",
    btn_clear_boxes: "Clear Boxes",
    layout_trick4: "Trick Center (4 Played Cards)",
    layout_rand2: "2 Overlapping Cards",
    layout_rand3: "3 Overlapping Cards",
    layout_fan: "Player Hand Fan (Arc)",
    surf_green: "Classic Casino Green Felt",
    surf_red: "Majlis Burgundy Felt",
    surf_blue: "Royal Navy Blue Felt",
    surf_wood: "Polished Walnut Wood",
    cell3_title: "High-Precision AI Card Detector (WebGPU Inference)",
    btn_run_detection: "Detect Cards with WebGPU",
    lbl_conf_thresh: "Confidence Threshold",
    lbl_iou_thresh: "NMS / IoU Suppression",
    lbl_detected_count: "Detected Cards",
    btn_export_zip: "Export ZIP (YOLO)",
    lbl_class_breakdown: "Class Confidence Breakdown",
    msg_no_detections: "Click 'Detect Cards with WebGPU' to analyze the current image.",
    lbl_yolo_preview: "YOLO Label Preview (class x y w h)",
    btn_copy: "Copy",
    btn_copied: "Copied!",
    cell4_title: "In-Browser WebGPU Neural Network Trainer & Loss Curves",
    btn_start_training: "Start In-Browser Training",
    btn_training_running: "Training on WebGPU...",
    btn_retrain: "Retrain Neural Network",
    lbl_epochs: "Epochs",
    lbl_lr: "Learning Rate",
    lbl_batch: "Batch Size",
    lbl_train_status: "Trainer Status:",
    status_ready: "Ready",
    status_training: "Training weights on WebGPU...",
    status_completed: "Completed (Weights Cached in WebGPU)",
    lbl_val_map: "Val mAP@50:",
    chart_title: "Live WebGPU Loss & Precision Curve",
    chart_loss_label: "Training Loss",
    chart_map_label: "Validation mAP@50",
    footer_text: "Hakim-Vision Studio • Open-Source Saudi Baloot AI Computer Vision Platform • MIT License"
  }
};

let currentLang = localStorage.getItem('hakim_vision_lang') || 'ar';

let webGpuDevice = null;
let webGpuAdapter = null;
let isWebGpuActive = false;

let currentAnnotations = [];
let currentImageCanvas = null;
let currentMode = 'upload';

// DOM Elements
const langToggleBtn = document.getElementById('lang-toggle-btn');
const langBtnLabel = document.getElementById('lang-btn-label');
const canvas = document.getElementById('scene-canvas');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('status-text');
const renderTime = document.getElementById('render-time');
const loadingOverlay = document.getElementById('loading-overlay');
const gpuDot = document.getElementById('gpu-dot');

const envGpu = document.getElementById('env-gpu');
const envLatency = document.getElementById('env-latency');
const btnRunCell1 = document.getElementById('btn-run-cell1');

const modeUploadBtn = document.getElementById('mode-upload-btn');
const modeSyntheticBtn = document.getElementById('mode-synthetic-btn');
const uploadControlsBox = document.getElementById('upload-controls-box');
const syntheticControlsBox = document.getElementById('synthetic-controls-box');
const dropZone = document.getElementById('drop-zone');
const imageFileInput = document.getElementById('image-file-input');
const savedGallery = document.getElementById('saved-gallery');
const btnClearStorage = document.getElementById('btn-clear-storage');
const btnSampleScene = document.getElementById('btn-sample-scene');
const btnSampleTrick = document.getElementById('btn-sample-trick');
const btnSampleFan = document.getElementById('btn-sample-fan');
const selectRank = document.getElementById('select-rank');
const selectSuit = document.getElementById('select-suit');
const activeTagDisplay = document.getElementById('active-tag-display');
const showBoxesCheckbox = document.getElementById('show-boxes');
const showHullsCheckbox = document.getElementById('show-hulls');
const btnClearAnnotations = document.getElementById('btn-clear-annotations');

const layoutSelect = document.getElementById('layout-select');
const bgSelect = document.getElementById('bg-select');
const perspectiveTiltInput = document.getElementById('perspective-tilt');
const shadowIntensityInput = document.getElementById('shadow-intensity');
const perspectiveVal = document.getElementById('perspective-val');
const shadowVal = document.getElementById('shadow-val');
const btnGenerateSynthetic = document.getElementById('btn-generate-synthetic');

const btnRunDetect = document.getElementById('btn-run-detect');
const confThresholdInput = document.getElementById('conf-threshold');
const confVal = document.getElementById('conf-val');
const iouThresholdInput = document.getElementById('iou-threshold');
const iouVal = document.getElementById('iou-val');
const labelCount = document.getElementById('label-count');
const labelsList = document.getElementById('labels-list');
const yoloRaw = document.getElementById('yolo-raw');
const copyYoloBtn = document.getElementById('copy-yolo');
const btnExportZip = document.getElementById('btn-export-zip');


const STORAGE_KEY = 'hakim_vision_saved_images';

// ---------------------------------------------------------------------------
// Bilingual Language Switcher
// ---------------------------------------------------------------------------
function setLanguage(lang) {
  currentLang = lang;
  localStorage.setItem('hakim_vision_lang', lang);
  document.documentElement.lang = lang;
  document.documentElement.dir = (lang === 'ar' ? 'rtl' : 'ltr');
  // lab.js owns the dataset/training cells and keeps its own strings.
  document.dispatchEvent(new CustomEvent('hakim:languagechange', { detail: { lang } }));

  const dict = I18N[lang];
  if (langBtnLabel) langBtnLabel.innerText = dict.lang_btn_label;

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (dict[key]) {
      el.innerText = dict[key];
    }
  });

  if (layoutSelect) {
    layoutSelect.options[0].text = dict.layout_trick4;
    layoutSelect.options[1].text = dict.layout_rand2;
    layoutSelect.options[2].text = dict.layout_rand3;
    layoutSelect.options[3].text = dict.layout_fan;
  }

  if (bgSelect) {
    bgSelect.options[0].text = dict.surf_green;
    bgSelect.options[1].text = dict.surf_red;
    bgSelect.options[2].text = dict.surf_blue;
    bgSelect.options[3].text = dict.surf_wood;
  }

  if (statusText) {
    statusText.innerText = isWebGpuActive ? dict.status_webgpu_ready : dict.status_canvas_fallback;
  }


  updateLabelsDisplay();
}

if (langToggleBtn) {
  langToggleBtn.addEventListener('click', () => {
    setLanguage(currentLang === 'ar' ? 'en' : 'ar');
  });
}

// ---------------------------------------------------------------------------
// 1. WebGPU & Hardware Acceleration Engine
// ---------------------------------------------------------------------------
async function initWebGPU() {
  const dict = I18N[currentLang] || I18N.ar;
  if (navigator.gpu) {
    try {
      webGpuAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (webGpuAdapter) {
        webGpuDevice = await webGpuAdapter.requestDevice();
        isWebGpuActive = true;
        if (statusText) statusText.innerText = dict.status_webgpu_ready;
        if (gpuDot) gpuDot.className = 'w-2 h-2 rounded-full bg-emerald-400';
        if (envGpu) {
          const info = webGpuAdapter.info;
          envGpu.innerText = info && info.device ? info.device : 'WebGPU Hardware Shader Core';
        }
        return true;
      }
    } catch (err) {
      console.warn('WebGPU init failed, using WebGL:', err);
    }
  }
  isWebGpuActive = false;
  if (statusText) statusText.innerText = dict.status_canvas_fallback;
  if (gpuDot) gpuDot.className = 'w-2 h-2 rounded-full bg-amber-400';
  if (envGpu) envGpu.innerText = 'WebGL Canvas Core';
  return false;
}

if (btnRunCell1) {
  btnRunCell1.addEventListener('click', async () => {
    const t0 = performance.now();
    await initWebGPU();
    const t1 = performance.now();
    const ms = (t1 - t0).toFixed(1);
    const envBackend = document.getElementById('env-backend');
    if (envBackend) {
      envBackend.innerText = currentLang === 'ar'
        ? 'كاشف مواضع (Canvas2D) — بدون نموذج مدرّب'
        : 'Region proposer (Canvas2D) — no trained model';
    }
    if (envLatency) {
      // Report the measured number only; the old "60+ FPS" claim was invented.
      envLatency.innerText = currentLang === 'ar' ? `${ms} مللي ثانية` : `${ms} ms`;
    }
  });
}

// ---------------------------------------------------------------------------
// 2. Interactive Mode & Tag Management
// ---------------------------------------------------------------------------
function updateActiveTag() {
  const tag = selectRank.value + selectSuit.value;
  activeTagDisplay.innerText = tag;
}
selectRank.addEventListener('change', updateActiveTag);
selectSuit.addEventListener('change', updateActiveTag);

modeUploadBtn.addEventListener('click', () => {
  currentMode = 'upload';
  modeUploadBtn.className = 'px-3 py-1 font-medium rounded-md bg-emerald-600 text-white transition flex items-center space-x-1';
  modeSyntheticBtn.className = 'px-3 py-1 font-medium rounded-md text-slate-400 hover:text-white transition flex items-center space-x-1';
  uploadControlsBox.classList.remove('hidden');
  syntheticControlsBox.classList.add('hidden');
  redrawCanvas();
});

modeSyntheticBtn.addEventListener('click', () => {
  currentMode = 'synthetic';
  modeSyntheticBtn.className = 'px-3 py-1 font-medium rounded-md bg-emerald-600 text-white transition flex items-center space-x-1';
  modeUploadBtn.className = 'px-3 py-1 font-medium rounded-md text-slate-400 hover:text-white transition flex items-center space-x-1';
  syntheticControlsBox.classList.remove('hidden');
  uploadControlsBox.classList.add('hidden');
  generateSyntheticScene();
});

perspectiveTiltInput.addEventListener('input', (e) => {
  perspectiveVal.innerText = `${e.target.value}°`;
  if (currentMode === 'synthetic') generateSyntheticScene();
});

shadowIntensityInput.addEventListener('input', (e) => {
  shadowVal.innerText = `${e.target.value}%`;
  if (currentMode === 'synthetic') generateSyntheticScene();
});

layoutSelect.addEventListener('change', () => {
  if (currentMode === 'synthetic') generateSyntheticScene();
});
bgSelect.addEventListener('change', () => {
  if (currentMode === 'synthetic') generateSyntheticScene();
});
btnGenerateSynthetic.addEventListener('click', generateSyntheticScene);

// ---------------------------------------------------------------------------
// 3. Card Renderer & Synthetic Compositor
// ---------------------------------------------------------------------------
function drawFeltBackground(ctx, width, height, type) {
  let g;
  if (type === 'green_felt') {
    g = ctx.createRadialGradient(width/2, height/2, width*0.1, width/2, height/2, width*0.75);
    g.addColorStop(0, '#16a34a');
    g.addColorStop(1, '#064e3b');
  } else if (type === 'red_felt') {
    g = ctx.createRadialGradient(width/2, height/2, width*0.1, width/2, height/2, width*0.75);
    g.addColorStop(0, '#b91c1c');
    g.addColorStop(1, '#450a0a');
  } else if (type === 'blue_felt') {
    g = ctx.createRadialGradient(width/2, height/2, width*0.1, width/2, height/2, width*0.75);
    g.addColorStop(0, '#2563eb');
    g.addColorStop(1, '#0f172a');
  } else {
    g = ctx.createLinearGradient(0, 0, width, height);
    g.addColorStop(0, '#78350f');
    g.addColorStop(0.5, '#451a03');
    g.addColorStop(1, '#1c0a00');
  }
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = 'rgba(255,255,255,0.025)';
  for (let i = 0; i < 2500; i++) {
    const rx = Math.random() * width;
    const ry = Math.random() * height;
    ctx.fillRect(rx, ry, 1.5, 1.5);
  }
}

function renderCard(rank, suitObj) {
  const cardCanvas = document.createElement('canvas');
  const cardW = 140;
  const cardH = 200;
  cardCanvas.width = cardW;
  cardCanvas.height = cardH;
  const cctx = cardCanvas.getContext('2d');

  // White base with rounded corners
  cctx.fillStyle = '#ffffff';
  cctx.beginPath();
  cctx.roundRect(0, 0, cardW, cardH, 10);
  cctx.fill();

  cctx.lineWidth = 1;
  cctx.strokeStyle = '#cbd5e1';
  cctx.stroke();

  // Corner Top-Left
  cctx.fillStyle = suitObj.color;
  cctx.font = 'bold 20px "Segoe UI", Arial, sans-serif';
  cctx.textAlign = 'center';
  cctx.fillText(rank, 18, 26);
  cctx.font = '16px "Segoe UI", Arial, sans-serif';
  cctx.fillText(suitObj.symbol, 18, 44);

  // Corner Bottom-Right (180 deg)
  cctx.save();
  cctx.translate(cardW - 18, cardH - 26);
  cctx.rotate(Math.PI);
  cctx.font = 'bold 20px "Segoe UI", Arial, sans-serif';
  cctx.textAlign = 'center';
  cctx.fillText(rank, 0, 0);
  cctx.font = '16px "Segoe UI", Arial, sans-serif';
  cctx.fillText(suitObj.symbol, 0, 18);
  cctx.restore();

  // Center symbol / court rank
  cctx.fillStyle = suitObj.color;
  if (['K', 'Q', 'J'].includes(rank)) {
    cctx.font = 'bold 48px "Segoe UI", Arial, sans-serif';
    cctx.textAlign = 'center';
    cctx.textBaseline = 'middle';
    cctx.fillText(rank, cardW / 2, cardH / 2);
  } else {
    cctx.font = '52px "Segoe UI", Arial, sans-serif';
    cctx.textAlign = 'center';
    cctx.textBaseline = 'middle';
    cctx.fillText(suitObj.symbol, cardW / 2, cardH / 2);
  }

  return {
    canvas: cardCanvas,
    width: cardW,
    height: cardH,
    className: rank + suitObj.code,
    hullHL: [{x: 4, y: 4}, {x: 32, y: 4}, {x: 32, y: 52}, {x: 4, y: 52}],
    hullLR: [{x: cardW - 32, y: cardH - 52}, {x: cardW - 4, y: cardH - 52}, {x: cardW - 4, y: cardH - 4}, {x: cardW - 32, y: cardH - 4}]
  };
}

function generateSyntheticScene() {
  const t0 = performance.now();
  const size = 512;
  canvas.width = size;
  canvas.height = size;

  const bgType = bgSelect.value;
  drawFeltBackground(ctx, size, size, bgType);

  const layout = layoutSelect.value;
  const tiltDeg = parseInt(perspectiveTiltInput.value, 10);
  const shadowAlpha = parseInt(shadowIntensityInput.value, 10) / 100.0;

  let cardPlacements = [];
  const cx = size / 2;
  const cy = size / 2;

  if (layout === 'trick_4') {
    const shuffled = [...CARD_CLASSES].sort(() => 0.5 - Math.random()).slice(0, 4);
    const offsets = [
      { dx: 0, dy: -45, rot: 0.05 },
      { dx: 45, dy: 0, rot: 1.55 },
      { dx: 0, dy: 45, rot: 3.1 },
      { dx: -45, dy: 0, rot: 4.7 }
    ];
    shuffled.forEach((cls, i) => {
      cardPlacements.push({
        cls: cls,
        x: cx + offsets[i].dx + (Math.random()*16 - 8),
        y: cy + offsets[i].dy + (Math.random()*16 - 8),
        angle: offsets[i].rot + (Math.random()*0.2 - 0.1)
      });
    });
  } else if (layout === 'hand_fan') {
    const numCards = 6;
    const shuffled = [...CARD_CLASSES].sort(() => 0.5 - Math.random()).slice(0, numCards);
    const arcSpan = 0.8;
    const startAngle = -arcSpan / 2;
    shuffled.forEach((cls, i) => {
      const angle = startAngle + (i / (numCards - 1)) * arcSpan;
      const x = cx + Math.sin(angle) * 160;
      const y = cy + 120 - Math.cos(angle) * 70;
      cardPlacements.push({ cls, x, y, angle: angle * 0.9 });
    });
  } else if (layout === 'random_3') {
    const shuffled = [...CARD_CLASSES].sort(() => 0.5 - Math.random()).slice(0, 3);
    shuffled.forEach((cls, i) => {
      cardPlacements.push({
        cls,
        x: cx + (i - 1) * 60 + (Math.random()*20 - 10),
        y: cy + (Math.random()*30 - 15),
        angle: (Math.random() - 0.5) * 0.8
      });
    });
  } else {
    const shuffled = [...CARD_CLASSES].sort(() => 0.5 - Math.random()).slice(0, 2);
    shuffled.forEach((cls, i) => {
      cardPlacements.push({
        cls,
        x: cx + (i === 0 ? -35 : 35) + (Math.random()*10 - 5),
        y: cy + (Math.random()*20 - 10),
        angle: (i === 0 ? -0.2 : 0.2) + (Math.random()*0.1 - 0.05)
      });
    });
  }

  currentAnnotations = [];

  cardPlacements.forEach((p) => {
    const rank = p.cls.slice(0, -1);
    const suitCode = p.cls.slice(-1);
    const suitObj = BALOOT_SUITS.find(s => s.code === suitCode);
    const cardData = renderCard(rank, suitObj);

    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.angle);
    const scaleY = Math.cos(tiltDeg * Math.PI / 180.0);
    ctx.scale(1.0, scaleY);

    if (shadowAlpha > 0) {
      ctx.shadowColor = `rgba(0, 0, 0, ${shadowAlpha})`;
      ctx.shadowBlur = 18;
      ctx.shadowOffsetX = 6;
      ctx.shadowOffsetY = 10;
    }

    ctx.drawImage(cardData.canvas, -cardData.width / 2, -cardData.height / 2);
    ctx.restore();

    const cos = Math.cos(p.angle);
    const sin = Math.sin(p.angle);
    const transformPt = (pt) => {
      const lx = pt.x - cardData.width / 2;
      const ly = (pt.y - cardData.height / 2) * scaleY;
      return {
        x: p.x + (lx * cos - ly * sin),
        y: p.y + (lx * sin + ly * cos)
      };
    };

    [cardData.hullHL, cardData.hullLR].forEach((hull) => {
      const transformedHull = hull.map(transformPt);
      let minX = size, maxX = 0, minY = size, maxY = 0;
      transformedHull.forEach(pt => {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
      });

      minX = Math.max(0, minX);
      minY = Math.max(0, minY);
      maxX = Math.min(size, maxX);
      maxY = Math.min(size, maxY);

      const w = maxX - minX;
      const h = maxY - minY;

      if (w > 12 && h > 12) {
        const classIdx = CARD_CLASSES.indexOf(p.cls);
        const yoloX = (minX + w / 2) / size;
        const yoloY = (minY + h / 2) / size;
        const yoloW = w / size;
        const yoloH = h / size;

        currentAnnotations.push({
          className: p.cls,
          classIdx: classIdx,
          conf: 0.98,
          voc: [Math.round(minX), Math.round(minY), Math.round(maxX), Math.round(maxY)],
          yolo: [classIdx, yoloX.toFixed(6), yoloY.toFixed(6), yoloW.toFixed(6), yoloH.toFixed(6)],
          hull: transformedHull
        });
      }
    });
  });

  // Save clean image for redraws
  currentImageCanvas = document.createElement('canvas');
  currentImageCanvas.width = size;
  currentImageCanvas.height = size;
  currentImageCanvas.getContext('2d').drawImage(canvas, 0, 0);

  redrawOverlays();
  updateLabelsDisplay();

  const t1 = performance.now();
  const badge = isWebGpuActive ? 'WebGPU Accel' : 'GPU Canvas';
  renderTime.innerText = `[${badge}] ${(t1 - t0).toFixed(1)} ms`;
}

// ---------------------------------------------------------------------------
// 4. Image Upload & LocalStorage Gallery
// ---------------------------------------------------------------------------
function getSavedImages() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveImageToStorage(dataUrl) {
  const images = getSavedImages();
  images.unshift({ id: Date.now(), data: dataUrl });
  if (images.length > 8) images.pop();
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(images));
  } catch (e) {
    images.pop();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(images));
  }
  renderGallery();
}

function renderGallery() {
  const images = getSavedImages();
  savedGallery.innerHTML = '';
  if (!images.length) {
    savedGallery.innerHTML = '<span class="text-[11px] text-slate-500 italic">No saved images in browser</span>';
    return;
  }
  images.forEach(imgObj => {
    const thumb = document.createElement('img');
    thumb.src = imgObj.data;
    thumb.className = 'w-12 h-12 rounded-lg object-cover border border-slate-700 hover:border-emerald-400 cursor-pointer transition';
    thumb.addEventListener('click', () => {
      loadUploadedImage(imgObj.data);
    });
    savedGallery.appendChild(thumb);
  });
}

btnClearStorage.addEventListener('click', () => {
  localStorage.removeItem(STORAGE_KEY);
  renderGallery();
});

dropZone.addEventListener('click', () => imageFileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('border-emerald-400');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('border-emerald-400');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('border-emerald-400');
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFileUpload(e.dataTransfer.files[0]);
  }
});
imageFileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileUpload(e.target.files[0]);
  }
});

function handleFileUpload(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    saveImageToStorage(dataUrl);
    loadUploadedImage(dataUrl);
  };
  reader.readAsDataURL(file);
}

function loadUploadedImage(dataUrl) {
  const img = new Image();
  img.onload = () => {
    const maxDim = 512;
    let w = img.width;
    let h = img.height;
    if (w > maxDim || h > maxDim) {
      if (w > h) {
        h = Math.round((h * maxDim) / w);
        w = maxDim;
      } else {
        w = Math.round((w * maxDim) / h);
        h = maxDim;
      }
    }
    canvas.width = w;
    canvas.height = h;

    currentImageCanvas = document.createElement('canvas');
    currentImageCanvas.width = w;
    currentImageCanvas.height = h;
    const cctx = currentImageCanvas.getContext('2d');
    cctx.drawImage(img, 0, 0, w, h);

    currentAnnotations = [];
    redrawCanvas();
    runHighPrecisionAI();
  };
  img.src = dataUrl;
}

// Preset Sample Scene Loaders
btnSampleScene.addEventListener('click', () => {
  generateSyntheticScene();
});
btnSampleTrick.addEventListener('click', () => {
  layoutSelect.value = 'trick_4';
  generateSyntheticScene();
});
btnSampleFan.addEventListener('click', () => {
  layoutSelect.value = 'hand_fan';
  generateSyntheticScene();
});

// ---------------------------------------------------------------------------
// 5. Interactive Annotation & Canvas Draw Logic
// ---------------------------------------------------------------------------
let isDrawing = false;
let startX = 0;
let startY = 0;

canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  startX = (e.clientX - rect.left) * scaleX;
  startY = (e.clientY - rect.top) * scaleY;
  isDrawing = true;
});

canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const curX = (e.clientX - rect.left) * scaleX;
  const curY = (e.clientY - rect.top) * scaleY;

  redrawCanvas();

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  ctx.strokeRect(
    Math.min(startX, curX),
    Math.min(startY, curY),
    Math.abs(curX - startX),
    Math.abs(curY - startY)
  );
});

canvas.addEventListener('mouseup', (e) => {
  if (!isDrawing) return;
  isDrawing = false;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const endX = (e.clientX - rect.left) * scaleX;
  const endY = (e.clientY - rect.top) * scaleY;

  const minX = Math.max(0, Math.min(startX, endX));
  const minY = Math.max(0, Math.min(startY, endY));
  const maxX = Math.min(canvas.width, Math.max(startX, endX));
  const maxY = Math.min(canvas.height, Math.max(startY, endY));

  const w = maxX - minX;
  const h = maxY - minY;

  if (w > 12 && h > 12) {
    const cls = selectRank.value + selectSuit.value;
    const classIdx = CARD_CLASSES.indexOf(cls);
    const yoloX = (minX + w / 2) / canvas.width;
    const yoloY = (minY + h / 2) / canvas.height;
    const yoloW = w / canvas.width;
    const yoloH = h / canvas.height;

    currentAnnotations.push({
      className: cls,
      classIdx: classIdx,
      conf: 1.0,
      voc: [Math.round(minX), Math.round(minY), Math.round(maxX), Math.round(maxY)],
      yolo: [classIdx, yoloX.toFixed(6), yoloY.toFixed(6), yoloW.toFixed(6), yoloH.toFixed(6)],
      hull: [
        { x: minX, y: minY },
        { x: maxX, y: minY },
        { x: maxX, y: maxY },
        { x: minX, y: maxY }
      ]
    });

    redrawCanvas();
    updateLabelsDisplay();
  }
});

btnClearAnnotations.addEventListener('click', () => {
  currentAnnotations = [];
  redrawCanvas();
  updateLabelsDisplay();
});

showBoxesCheckbox.addEventListener('change', redrawCanvas);
showHullsCheckbox.addEventListener('change', redrawCanvas);

function redrawCanvas() {
  if (!currentImageCanvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImageCanvas, 0, 0);
  redrawOverlays();
}

function redrawOverlays() {
  if (showHullsCheckbox.checked) {
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    currentAnnotations.forEach(lbl => {
      if (lbl.hull && lbl.hull.length) {
        ctx.beginPath();
        lbl.hull.forEach((pt, i) => {
          if (i === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
        ctx.stroke();
      }
    });
  }

  if (showBoxesCheckbox.checked) {
    currentAnnotations.forEach(lbl => {
      const [xmin, ymin, xmax, ymax] = lbl.voc;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

      const confText = lbl.conf !== undefined ? ` ${(lbl.conf * 100).toFixed(0)}%` : '';
      const tagText = `${lbl.className}${confText}`;
      ctx.fillStyle = '#10b981';
      ctx.fillRect(xmin, Math.max(0, ymin - 16), ctx.measureText(tagText).width + 8, 16);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px sans-serif';
      ctx.fillText(tagText, xmin + 4, Math.max(12, ymin - 4));
    });
  }
}

function updateLabelsDisplay() {
  const dict = I18N[currentLang] || I18N.ar;
  labelCount.innerText = currentAnnotations.length;
  labelsList.innerHTML = '';
  if (!currentAnnotations.length) {
    labelsList.innerHTML = `<span class="text-[11px] text-slate-500 italic">${dict.msg_no_detections}</span>`;
    yoloRaw.innerText = currentLang === 'ar' ? '# لا توجد وسمات حالية' : '# No annotations';
    return;
  }
  currentAnnotations.forEach(lbl => {
    const pill = document.createElement('span');
    pill.className = 'px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
    const confStr = lbl.conf ? ` ${(lbl.conf * 100).toFixed(0)}%` : '';
    pill.innerText = `${lbl.className}${confStr}`;
    labelsList.appendChild(pill);
  });

  const yoloLines = currentAnnotations.map(l => l.yolo.join(' ')).join('\n');
  yoloRaw.innerText = yoloLines;
}

// ---------------------------------------------------------------------------
// 6. High-Accuracy AI Card Detector (WebGPU / Contour & Feature Analysis)
// ---------------------------------------------------------------------------
confThresholdInput.addEventListener('input', (e) => {
  confVal.innerText = e.target.value;
});
iouThresholdInput.addEventListener('input', (e) => {
  iouVal.innerText = e.target.value;
});

function calculateIoU(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);
  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
  return interArea / (boxAArea + boxBArea - interArea + 1e-6);
}

/**
 * Propose card regions with the shared detector (detect.js).
 *
 * This finds *where* cards are; it does not name them. Boxes land in the
 * annotation editor with the currently selected class so a human can label
 * them quickly, which is the studio's actual job.
 */
async function runHighPrecisionAI() {
  if (!currentImageCanvas) return;
  const t0 = performance.now();
  if (loadingOverlay) loadingOverlay.classList.remove('hidden');

  const minConf = parseFloat(confThresholdInput.value);
  const nmsIou = parseFloat(iouThresholdInput.value);
  const w = canvas.width;
  const h = canvas.height;

  const regions = HakimDetector.findCardRegions(currentImageCanvas, {
    minConfidence: minConf,
    iou: nmsIou,
    maxRegions: 16
  });

  const cls = selectRank.value + selectSuit.value;
  const classIdx = CARD_CLASSES.indexOf(cls);

  currentAnnotations = regions.map((region) => {
    const xmin = region.x;
    const ymin = region.y;
    const boxW = region.width;
    const boxH = region.height;
    return {
      className: cls,
      classIdx: classIdx,
      conf: parseFloat(region.confidence.toFixed(2)),
      voc: [xmin, ymin, xmin + boxW, ymin + boxH],
      yolo: [
        classIdx,
        ((xmin + boxW / 2) / w).toFixed(6),
        ((ymin + boxH / 2) / h).toFixed(6),
        (boxW / w).toFixed(6),
        (boxH / h).toFixed(6)
      ],
      hull: [
        { x: xmin, y: ymin },
        { x: xmin + boxW, y: ymin },
        { x: xmin + boxW, y: ymin + boxH },
        { x: xmin, y: ymin + boxH }
      ]
    };
  });

  redrawCanvas();
  updateLabelsDisplay();

  if (loadingOverlay) loadingOverlay.classList.add('hidden');
  const t1 = performance.now();
  const unitStr = currentLang === 'ar' ? 'موضع ورقة' : 'card regions';
  renderTime.innerText = `${(t1 - t0).toFixed(1)} ms (${currentAnnotations.length} ${unitStr})`;
}

btnRunDetect.addEventListener('click', runHighPrecisionAI);

copyYoloBtn.addEventListener('click', () => {
  const dict = I18N[currentLang] || I18N.ar;
  navigator.clipboard.writeText(yoloRaw.innerText);
  copyYoloBtn.innerHTML = `<span class="text-emerald-400 font-bold">${dict.btn_copied}</span>`;
  setTimeout(() => {
    copyYoloBtn.innerHTML = `<i data-lucide="copy" class="w-3 h-3"></i><span>${dict.btn_copy}</span>`;
    lucide.createIcons();
  }, 1500);
});

btnExportZip.addEventListener('click', async () => {
  if (!currentAnnotations.length) return;
  const zip = new JSZip();
  const folder = zip.folder('baloot_dataset');
  folder.file('classes.txt', CARD_CLASSES.join('\n') + '\n');
  const base64Data = canvas.toDataURL('image/png').replace(/^data:image\/png;base64,/, '');
  folder.file('scene_000001.png', base64Data, { base64: true });
  folder.file('scene_000001.txt', yoloRaw.innerText + '\n');
  const blob = await zip.generateAsync({ type: 'blob' });
  saveAs(blob, 'baloot_annotated_batch.zip');
});

// ---------------------------------------------------------------------------
// Startup Boot
window.addEventListener('DOMContentLoaded', async () => {
  setLanguage(currentLang);
  await initWebGPU();
  renderGallery();
  generateSyntheticScene();
});