/**
 * Hakim — In-Browser YOLO11 Card Detector with WebGPU and WASM fallback.
 *
 * Preprocessing: stretch resize to 704x704 (no letterboxing), matching training.
 * Output decoding: 1x37x10164 tensor (4 bbox coords + 33 classes: 32 Baloot cards + 'other').
 * Coordinates are un-stretched back to original image dimensions.
 * Class 'other' is mapped to `card: null` (rendered in UI as '؟').
 *
 * Two preprocessing paths share one resize:
 *
 * - On WebGPU the resized frame is uploaded straight into a GPU texture and a compute
 *   shader writes planar normalised RGB into a storage buffer that becomes the input
 *   tensor. Nothing crosses back to the CPU, which is the entire point of the backend.
 * - Everywhere else the frame is read back once and converted in a tight loop.
 *
 * Both paths allocate their canvas, buffers, and tensor once and reuse them for every
 * frame. The previous version allocated a canvas and a ~6 MB Float32Array per detection,
 * which on a phone cost more than the inference it was feeding.
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
    loadPromise: null,
    preprocess: 'cpu' // 'gpu' | 'cpu'
  };

  // One scratch canvas for the stretch resize, reused across frames.
  // `willReadFrequently` is set only for the CPU path: it hints at a software-backed
  // canvas, which speeds up getImageData but slows the GPU texture upload.
  var scratch = { canvas: null, ctx: null, size: 0, readback: null };

  // CPU path: the tensor wraps this array, and ORT copies out of it on every run,
  // so both survive from frame to frame.
  var cpuTensor = { data: null, tensor: null, size: 0 };

  // GPU path: texture, storage buffer, and the tensor viewing that buffer.
  // `status` is 'unknown' until first use, then 'ready' or 'unavailable' (sticky).
  var gpu = {
    device: null,
    pipeline: null,
    texture: null,
    buffer: null,
    bindGroup: null,
    tensor: null,
    size: 0,
    status: 'unknown'
  };

  // Reads the resized frame and writes planar float RGB: the NCHW layout YOLO expects.
  // rgba8unorm texels arrive already in [0, 1], so no division is needed.
  var PREPROCESS_WGSL = [
    '@group(0) @binding(0) var src : texture_2d<f32>;',
    '@group(0) @binding(1) var<storage, read_write> dst : array<f32>;',
    '',
    '@compute @workgroup_size(8, 8, 1)',
    'fn main(@builtin(global_invocation_id) gid : vec3<u32>) {',
    '  let dims = textureDimensions(src);',
    '  if (gid.x >= dims.x || gid.y >= dims.y) { return; }',
    '  let texel = textureLoad(src, vec2<i32>(i32(gid.x), i32(gid.y)), 0);',
    '  let plane = dims.x * dims.y;',
    '  let idx = gid.y * dims.x + gid.x;',
    '  dst[idx] = texel.r;',
    '  dst[plane + idx] = texel.g;',
    '  dst[plane * 2u + idx] = texel.b;',
    '}'
  ].join('\n');

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
          return tryLoadSession('webgpu', './models/baloot-fan-v2.fp16.onnx', 'fp16')
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
    return tryLoadSession('wasm', './models/baloot-fan-v2.int8.onnx', 'int8')
      .catch(function () {
        return tryLoadSession('wasm', './models/baloot-fan-v2.fp16.onnx', 'fp16');
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

  /**
   * Compile WebGPU shaders and allocate every reusable buffer before the first real
   * frame, so the opening detection is not the one that pays for all of it.
   */
  function warmupSession(session) {
    try {
      var blank = document.createElement('canvas');
      blank.width = 8;
      blank.height = 8;
      var feeds = {};
      feeds[session.inputNames[0] || 'images'] = preprocess(blank);
      return session.run(feeds).then(function () {});
    } catch (e) {
      return Promise.resolve();
    }
  }

  /**
   * Allocate (or resize) the shared scratch canvas used for the stretch resize.
   *
   * Deliberately a detached <canvas> rather than an OffscreenCanvas: the two draw
   * identically, but OffscreenCanvas.getImageData measured ~3x slower in Chrome, which
   * would penalise exactly the CPU path that can least afford it. `alpha: false` skips
   * the premultiply round-trip and is pixel-identical here for opaque photos.
   */
  function ensureScratch(size, readback) {
    if (scratch.canvas && scratch.size === size && scratch.readback === readback) {
      return;
    }
    var canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    scratch.canvas = canvas;
    scratch.ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: readback });
    scratch.size = size;
    scratch.readback = readback;
  }

  function releaseGpuBuffers() {
    // The pipeline is size-independent, so it is deliberately kept.
    if (gpu.texture) { try { gpu.texture.destroy(); } catch (e) { /* already gone */ } }
    if (gpu.buffer) { try { gpu.buffer.destroy(); } catch (e) { /* already gone */ } }
    gpu.texture = null;
    gpu.buffer = null;
    gpu.bindGroup = null;
    gpu.tensor = null;
  }

  /**
   * Build the compute pipeline and its buffers on the device ORT is already using.
   * Returns false - permanently - if anything is missing, so the CPU path takes over
   * without retrying the same failure on every frame.
   */
  function ensureGpuPipeline(size) {
    if (gpu.status === 'unavailable') return false;
    if (gpu.status === 'ready' && gpu.size === size) return true;

    try {
      var device = ort.env && ort.env.webgpu && ort.env.webgpu.device;
      if (!device || typeof ort.Tensor.fromGpuBuffer !== 'function') {
        gpu.status = 'unavailable';
        return false;
      }

      releaseGpuBuffers();
      gpu.device = device;

      if (!gpu.pipeline) {
        gpu.pipeline = device.createComputePipeline({
          layout: 'auto',
          compute: {
            module: device.createShaderModule({ code: PREPROCESS_WGSL }),
            entryPoint: 'main'
          }
        });
      }

      // RENDER_ATTACHMENT is required by copyExternalImageToTexture.
      gpu.texture = device.createTexture({
        size: [size, size, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.RENDER_ATTACHMENT
      });

      gpu.buffer = device.createBuffer({
        size: 3 * size * size * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      });

      gpu.bindGroup = device.createBindGroup({
        layout: gpu.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: gpu.texture.createView() },
          { binding: 1, resource: { buffer: gpu.buffer } }
        ]
      });

      gpu.tensor = ort.Tensor.fromGpuBuffer(gpu.buffer, {
        dataType: 'float32',
        dims: [1, 3, size, size]
      });

      gpu.size = size;
      gpu.status = 'ready';
      return true;
    } catch (error) {
      console.warn('WebGPU preprocessing unavailable, falling back to CPU:', error);
      gpu.status = 'unavailable';
      releaseGpuBuffers();
      return false;
    }
  }

  /** Upload the resized frame and run the conversion shader. Never touches the CPU. */
  function preprocessOnGpu(size) {
    var device = gpu.device;

    device.queue.copyExternalImageToTexture(
      { source: scratch.canvas, flipY: false },
      { texture: gpu.texture, premultipliedAlpha: false },
      [size, size, 1]
    );

    var encoder = device.createCommandEncoder();
    var pass = encoder.beginComputePass();
    pass.setPipeline(gpu.pipeline);
    pass.setBindGroup(0, gpu.bindGroup);
    var groups = Math.ceil(size / 8);
    pass.dispatchWorkgroups(groups, groups, 1);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Queued work is ordered, so ORT's own submission sees the finished buffer.
    return gpu.tensor;
  }

  /** Read the resized frame back once and interleave it into planar RGB. */
  function preprocessOnCpu(size) {
    var pixels = size * size;

    if (!cpuTensor.data || cpuTensor.size !== size) {
      cpuTensor.data = new Float32Array(3 * pixels);
      cpuTensor.tensor = new ort.Tensor('float32', cpuTensor.data, [1, 3, size, size]);
      cpuTensor.size = size;
    }

    var rgba = scratch.ctx.getImageData(0, 0, size, size).data;
    var out = cpuTensor.data;
    var greenPlane = pixels;
    var bluePlane = pixels * 2;

    // `read` walks the RGBA source by stride so the loop body has no multiplications.
    for (var i = 0, read = 0; i < pixels; i++, read += 4) {
      out[i] = rgba[read] / 255;
      out[greenPlane + i] = rgba[read + 1] / 255;
      out[bluePlane + i] = rgba[read + 2] / 255;
    }

    return cpuTensor.tensor;
  }

  /**
   * Resize the source to the model's input square and hand back an input tensor.
   * The returned tensor is shared across calls - consume it before the next frame.
   */
  function preprocess(source) {
    var size = state.imgsz || DEFAULT_IMGSZ;
    var useGpu = state.backend === 'webgpu' && ensureGpuPipeline(size);
    state.preprocess = useGpu ? 'gpu' : 'cpu';

    ensureScratch(size, !useGpu);
    // Stretch resize: the aspect ratio is deliberately not preserved, matching training.
    scratch.ctx.drawImage(source, 0, 0, size, size);

    return useGpu ? preprocessOnGpu(size) : preprocessOnCpu(size);
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
          preprocess: state.preprocess,
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
        preprocess: state.preprocess,
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
