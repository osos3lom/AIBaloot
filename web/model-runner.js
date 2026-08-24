/**
 * Hakim — In-Browser YOLO11 Card Detector with WebGPU and WASM fallback.
 *
 * Preprocessing: Stretch resize to 416x416 (no letterboxing).
 * Output decoding: 1x37x3549 tensor (4 bbox coords + 33 classes: 32 Baloot cards + 'other').
 * Coordinates are un-stretched back to original image dimensions.
 * Class 'other' is mapped to `card: null` (rendered in UI as '؟').
 */

var HakimModelRunner = (function () {
  'use strict';

  var DEFAULT_IMGSZ = 704;
  var NUM_CLASSES = 33;
  var IOU_THRESHOLD = 0.45;
  var DEFAULT_CONFIDENCE = 0.35;

  var CLASS_NAMES = [
    'Ah', 'Kh', 'Qh', 'Jh', '10h', '9h', '8h', '7h',
    'Ad', 'Kd', 'Qd', 'Jd', '10d', '9d', '8d', '7d',
    'Ac', 'Kc', 'Qc', 'Jc', '10c', '9c', '8c', '7c',
    'As', 'Ks', 'Qs', 'Js', '10s', '9s', '8s', '7s',
    'other'
  ];

  // Captured during initial script evaluation, while document.currentScript is still set.
  var SCRIPT_BASE = (typeof document !== 'undefined' && document.currentScript)
    ? new URL('./', document.currentScript.src).href
    : new URL('./', location.href).href;

  var state = {
    session: null,
    backend: 'none', // 'webgpu' | 'wasm' | 'none'
    modelVariant: '', // 'fp16' | 'int8'
    imgsz: DEFAULT_IMGSZ,
    loading: false,
    loaded: false,
    lastLatencyMs: 0,
    error: null,
    loadPromise: null
  };

  /** Dynamically load the self-hosted onnxruntime script if not present. */
  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      if (typeof ort !== 'undefined') {
        return resolve();
      }
      var existing = document.querySelector('script[src="' + src + '"]');
      if (existing) {
        existing.addEventListener('load', resolve);
        existing.addEventListener('error', reject);
        return;
      }
      var script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = resolve;
      script.onerror = function () {
        reject(new Error('Failed to load script: ' + src));
      };
      document.head.appendChild(script);
    });
  }

  /** Calculate Intersection over Union between two boxes. */
  function intersectionOverUnion(a, b) {
    var x1 = Math.max(a.x, b.x);
    var y1 = Math.max(a.y, b.y);
    var x2 = Math.min(a.x + a.width, b.x + b.width);
    var y2 = Math.min(a.y + a.height, b.y + b.height);
    var overlap = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (!overlap) return 0;
    return overlap / (a.width * a.height + b.width * b.height - overlap);
  }

  /** Non-Maximum Suppression. */
  function nms(boxes, iouLimit) {
    boxes.sort(function (a, b) { return b.confidence - a.confidence; });
    var kept = [];
    for (var i = 0; i < boxes.length; i++) {
      var candidate = boxes[i];
      var overlap = false;
      for (var j = 0; j < kept.length; j++) {
        if (intersectionOverUnion(candidate, kept[j]) > iouLimit) {
          overlap = true;
          break;
        }
      }
      if (!overlap) {
        kept.push(candidate);
      }
    }
    // Left-to-right sorting for UI presentation
    return kept.sort(function (a, b) { return a.x - b.x; });
  }

  /** Initialize the ONNX runtime and load the model session. */
  function initModel() {
    if (state.loadPromise) {
      return state.loadPromise;
    }

    state.loading = true;
    state.loadPromise = loadScript('./runtime/ort.webgpu.min.js')
      .then(function () {
        if (typeof ort === 'undefined') {
          throw new Error('ONNX Runtime Web is not available');
        }

        // Must be an absolute URL: ORT concatenates this with the .mjs filename and
        // imports the result, and a bare specifier is not a valid ES module path.
        if (ort.env && ort.env.wasm) {
          ort.env.wasm.wasmPaths = new URL('runtime/', SCRIPT_BASE).href;
          ort.env.wasm.numThreads = 1; // GitHub Pages cannot send COOP/COEP, so no SharedArrayBuffer
        }

        var hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;

        // Try WebGPU first if supported
        if (hasWebGPU) {
          return tryLoadSession('webgpu', './models/baloot-v1.fp16.onnx', 'fp16')
            .catch(function (webgpuError) {
              console.warn('WebGPU init failed, falling back to WASM:', webgpuError);
              return tryLoadWasm();
            });
        } else {
          return tryLoadWasm();
        }
      })
      .then(function (sessionInfo) {
        state.session = sessionInfo.session;
        state.backend = sessionInfo.backend;
        state.modelVariant = sessionInfo.variant;
        state.loaded = true;
        state.loading = false;

        // Warm up session with dummy tensor to pre-compile WebGPU shaders
        return warmupSession(state.session).then(function () {
          console.info('Hakim card detector initialized:', state.backend, state.modelVariant);
          return state;
        });
      })
      .catch(function (error) {
        state.loading = false;
        state.error = error;
        console.error('Failed to initialize Hakim card detector:', error);
        throw error;
      });

    return state.loadPromise;
  }

  function tryLoadWasm() {
    // Prefer int8 on WASM if available, fallback to fp16
    return tryLoadSession('wasm', './models/baloot-v1.int8.onnx', 'int8')
      .catch(function () {
        return tryLoadSession('wasm', './models/baloot-v1.fp16.onnx', 'fp16');
      });
  }

  function tryLoadSession(backend, modelPath, variant) {
    var options = {
      executionProviders: [backend],
      graphOptimizationLevel: 'all'
    };
    return ort.InferenceSession.create(modelPath, options).then(function (session) {
      return { session: session, backend: backend, variant: variant };
    });
  }

  function warmupSession(session) {
    try {
      var currentImgsz = state.imgsz || DEFAULT_IMGSZ;
      var dummyData = new Float32Array(3 * currentImgsz * currentImgsz);
      var dummyTensor = new ort.Tensor('float32', dummyData, [1, 3, currentImgsz, currentImgsz]);
      var inputName = session.inputNames[0] || 'images';
      var feeds = {};
      feeds[inputName] = dummyTensor;
      return session.run(feeds).then(function () {});
    } catch (e) {
      return Promise.resolve();
    }
  }

  /**
   * Preprocess source canvas/image to Float32 tensor (RGB, normalized [0, 1]).
   */
  function preprocess(source) {
    var currentImgsz = state.imgsz || DEFAULT_IMGSZ;
    var offscreen = document.createElement('canvas');
    offscreen.width = currentImgsz;
    offscreen.height = currentImgsz;
    var ctx = offscreen.getContext('2d', { willReadFrequently: true });
    
    // Stretch resize (aspect ratio not preserved)
    ctx.drawImage(source, 0, 0, currentImgsz, currentImgsz);
    var imageData = ctx.getImageData(0, 0, currentImgsz, currentImgsz);
    var data = imageData.data;
    var floatData = new Float32Array(3 * currentImgsz * currentImgsz);
    var numPixels = currentImgsz * currentImgsz;

    for (var i = 0; i < numPixels; i++) {
      floatData[i] = data[i * 4] / 255.0;                   // R
      floatData[numPixels + i] = data[i * 4 + 1] / 255.0;   // G
      floatData[2 * numPixels + i] = data[i * 4 + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', floatData, [1, 3, currentImgsz, currentImgsz]);
  }

  /**
   * Decode YOLO11 1x37xN tensor into candidate bounding boxes.
   */
  function decodeOutput(tensorData, dims, sourceWidth, sourceHeight, minConf) {
    var candidates = [];
    var currentImgsz = state.imgsz || DEFAULT_IMGSZ;
    var scaleX = sourceWidth / currentImgsz;
    var scaleY = sourceHeight / currentImgsz;

    var numAnchors = (dims && dims.length >= 3) ? dims[2] : Math.floor(tensorData.length / (4 + NUM_CLASSES));

    for (var a = 0; a < numAnchors; a++) {
      var maxScore = -1;
      var maxClass = -1;

      // Class probabilities start at row 4
      for (var c = 0; c < NUM_CLASSES; c++) {
        var score = tensorData[(4 + c) * numAnchors + a];
        if (score > maxScore) {
          maxScore = score;
          maxClass = c;
        }
      }

      if (maxScore >= minConf) {
        var cx = tensorData[0 * numAnchors + a];
        var cy = tensorData[1 * numAnchors + a];
        var w = tensorData[2 * numAnchors + a];
        var h = tensorData[3 * numAnchors + a];

        var x1 = cx - w / 2;
        var y1 = cy - h / 2;

        var origX = Math.max(0, x1 * scaleX);
        var origY = Math.max(0, y1 * scaleY);
        var origW = Math.min(sourceWidth - origX, w * scaleX);
        var origH = Math.min(sourceHeight - origY, h * scaleY);

        var className = CLASS_NAMES[maxClass] || 'other';
        var card = (className === 'other' || maxClass === 32) ? null : className;

        candidates.push({
          x: origX,
          y: origY,
          width: origW,
          height: origH,
          confidence: maxScore,
          card: card,
          className: className
        });
      }
    }

    return dedupeByCard(nms(candidates, IOU_THRESHOLD));
  }

  /**
   * Labels mark corner indices, so one card yields up to two boxes (top-left and
   * bottom-right) that never overlap enough for NMS to merge. A Baloot deck holds exactly
   * one of each card, so a repeated label is always the same physical card — keep the
   * most confident box. 'other' is exempt: those are distinct unidentified cards.
   */
  function dedupeByCard(boxes) {
    var bestByCard = Object.create(null);
    var kept = [];

    boxes.forEach(function (box) {
      if (!box.card) {
        kept.push(box);
        return;
      }
      var incumbent = bestByCard[box.card];
      if (!incumbent) {
        bestByCard[box.card] = box;
        kept.push(box);
      } else if (box.confidence > incumbent.confidence) {
        kept[kept.indexOf(incumbent)] = box;
        bestByCard[box.card] = box;
      }
    });

    return kept.sort(function (a, b) { return a.x - b.x; });
  }

  /**
   * Run card detection using trained YOLO WebGPU/WASM model.
   *
   * @param {HTMLCanvasElement|HTMLImageElement} source
   * @param {{minConfidence?: number}} [options]
   * @returns {Promise<{regions: Array, labelled: boolean, elapsedMs: number, backend: string, modelVariant: string}>}
   */
  function detectCards(source, options) {
    var startedAt = (typeof performance !== 'undefined' ? performance : Date).now();
    var opts = options || {};
    var minConf = typeof opts.minConfidence === 'number' ? opts.minConfidence : DEFAULT_CONFIDENCE;

    return initModel().then(function () {
      var inputTensor = preprocess(source);
      var inputName = state.session.inputNames[0] || 'images';
      var feeds = {};
      feeds[inputName] = inputTensor;

      return state.session.run(feeds).then(function (results) {
        var outputName = state.session.outputNames[0];
        var outputTensor = results[outputName];
        var regions = decodeOutput(outputTensor.data, outputTensor.dims, source.width, source.height, minConf);
        var elapsedMs = (typeof performance !== 'undefined' ? performance : Date).now() - startedAt;
        state.lastLatencyMs = elapsedMs;

        return {
          regions: regions,
          labelled: true,
          elapsedMs: elapsedMs,
          backend: state.backend,
          modelVariant: state.modelVariant
        };
      });
    });
  }

  // Register with HakimDetector if available
  if (typeof HakimDetector !== 'undefined' && HakimDetector.registerDetector) {
    HakimDetector.registerDetector(detectCards);
  }

  return {
    initModel: initModel,
    detectCards: detectCards,
    getStatus: function () {
      return {
        loaded: state.loaded,
        loading: state.loading,
        backend: state.backend,
        modelVariant: state.modelVariant,
        lastLatencyMs: state.lastLatencyMs,
        error: state.error ? state.error.message : null
      };
    },
    CLASS_NAMES: CLASS_NAMES
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HakimModelRunner;
}
