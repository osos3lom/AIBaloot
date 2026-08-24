/**
 * Hakim — card detection adapter.
 *
 * Two layers, deliberately separated so the UI never has to pretend:
 *
 *   1. `findCardRegions()`  — always available. Deterministic image analysis
 *      that finds bright, card-shaped regions in the photo. It says *where*
 *      cards are, never *which* cards they are.
 *   2. `registerClassifier()` — optional. Hand the detector a trained model
 *      (see the contract below) and regions get rank/suit labels with real
 *      confidences. Until one is registered the app asks the player to
 *      identify each region, rather than inventing a label.
 *
 * Classifier contract:
 *   async classify(imageData, region) -> { card: "Ah", confidence: 0.93 } | null
 *   where `card` is `<rank><suit>` from the 32-card Baloot deck.
 */

var HakimDetector = (function () {
  'use strict';

  /** Longest edge the analysis grid is downscaled to, in pixels. */
  var ANALYSIS_MAX_EDGE = 360;
  /** Luminance above which a pixel is treated as card-face white. */
  var FACE_LUMINANCE = 150;
  /** Region area bounds as a fraction of the analysed image. */
  var MIN_AREA_FRACTION = 0.004;
  var MAX_AREA_FRACTION = 0.6;
  /** Plausible width:height ratios for a card at any rotation. */
  var MIN_ASPECT = 0.25;
  var MAX_ASPECT = 4.0;
  /** Width:height of an upright card (57mm x 87mm). */
  var CARD_ASPECT = 57 / 87;
  /** A blob this much wider than one upright card is probably a fan. */
  var FAN_WIDTH_FACTOR = 1.2;
  /** A column with less face than this, running the blob's full height, is a seam. */
  var SEAM_FILL = 0.35;
  /** Narrowest sliver of an overlapped card worth keeping, as a share of a card. */
  var MIN_SLIVER_FACTOR = 0.18;

  var classifier = null;
  var detector = null;

  function registerClassifier(fn) {
    classifier = typeof fn === 'function' ? fn : null;
  }

  function hasClassifier() {
    return classifier !== null;
  }

  function registerDetector(fn) {
    detector = typeof fn === 'function' ? fn : null;
  }

  function hasDetector() {
    return detector !== null;
  }

  /** Downscale into an offscreen canvas so analysis cost is bounded. */
  function toAnalysisCanvas(source) {
    var scale = Math.min(1, ANALYSIS_MAX_EDGE / Math.max(source.width, source.height));
    var width = Math.max(1, Math.round(source.width * scale));
    var height = Math.max(1, Math.round(source.height * scale));
    var small = document.createElement('canvas');
    small.width = width;
    small.height = height;
    small.getContext('2d').drawImage(source, 0, 0, width, height);
    return { canvas: small, scale: scale };
  }

  /** Binary mask of pixels bright enough to be a card face. */
  function buildFaceMask(imageData) {
    var data = imageData.data;
    var mask = new Uint8Array(imageData.width * imageData.height);
    for (var i = 0, p = 0; i < data.length; i += 4, p += 1) {
      var luminance = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      var maxChannel = Math.max(data[i], data[i + 1], data[i + 2]);
      var minChannel = Math.min(data[i], data[i + 1], data[i + 2]);
      // Card faces are bright and close to neutral; felt is bright-ish but saturated.
      var isNeutral = maxChannel - minChannel < 70;
      mask[p] = luminance > FACE_LUMINANCE && isNeutral ? 1 : 0;
    }
    return mask;
  }

  /** Iterative flood fill — recursion would blow the stack on large regions. */
  function collectComponent(mask, visited, width, height, startIndex) {
    var stack = [startIndex];
    var minX = width;
    var minY = height;
    var maxX = 0;
    var maxY = 0;
    var area = 0;

    visited[startIndex] = 1;
    while (stack.length) {
      var index = stack.pop();
      var x = index % width;
      var y = (index - x) / width;
      area += 1;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;

      if (x > 0) pushIfFace(index - 1);
      if (x < width - 1) pushIfFace(index + 1);
      if (y > 0) pushIfFace(index - width);
      if (y < height - 1) pushIfFace(index + width);
    }

    function pushIfFace(neighbour) {
      if (!visited[neighbour] && mask[neighbour]) {
        visited[neighbour] = 1;
        stack.push(neighbour);
      }
    }

    return { minX: minX, minY: minY, maxX: maxX, maxY: maxY, area: area };
  }

  /**
   * Confidence that a component really is a card: how completely it fills its
   * own bounding box (cards are convex rectangles) weighted by how card-like
   * that box's aspect ratio is.
   */
  function regionConfidence(component) {
    var boxWidth = component.maxX - component.minX + 1;
    var boxHeight = component.maxY - component.minY + 1;
    var fill = component.area / (boxWidth * boxHeight);
    var aspect = boxWidth / boxHeight;
    var cardAspect = 57 / 87; // physical Baloot card, portrait
    var ratio = aspect < 1 ? aspect : 1 / aspect;
    var aspectScore = 1 - Math.min(1, Math.abs(ratio - cardAspect) / cardAspect);
    return Math.max(0, Math.min(0.99, 0.45 * aspectScore + 0.55 * fill));
  }

  /**
   * Cards held in a fan overlap, so they flood-fill into one blob. Their
   * borders still show as dark columns running the blob's whole height, which
   * pip clusters and rank glyphs never do — so split on those columns.
   *
   * @returns {Array<{minX, minY, maxX, maxY, area}>} one component per card
   */
  function splitFannedComponent(mask, width, component) {
    var boxWidth = component.maxX - component.minX + 1;
    var boxHeight = component.maxY - component.minY + 1;
    var cardWidth = boxHeight * CARD_ASPECT;
    if (boxWidth < cardWidth * FAN_WIDTH_FACTOR) return [component];

    var profile = new Float32Array(boxWidth);
    for (var x = 0; x < boxWidth; x += 1) {
      var filled = 0;
      for (var y = 0; y < boxHeight; y += 1) {
        if (mask[(component.minY + y) * width + component.minX + x]) filled += 1;
      }
      profile[x] = filled / boxHeight;
    }

    // Collapse runs of seam columns to their midpoint.
    var cuts = [];
    var runStart = -1;
    for (var i = 0; i < boxWidth; i += 1) {
      if (profile[i] < SEAM_FILL) {
        if (runStart === -1) runStart = i;
      } else if (runStart !== -1) {
        cuts.push(Math.round((runStart + i - 1) / 2));
        runStart = -1;
      }
    }
    if (runStart !== -1) cuts.push(Math.round((runStart + boxWidth - 1) / 2));

    var minSliver = Math.max(4, cardWidth * MIN_SLIVER_FACTOR);
    var bounds = [0].concat(cuts, [boxWidth - 1]);
    var pieces = [];
    for (var b = 0; b < bounds.length - 1; b += 1) {
      var from = bounds[b];
      var to = bounds[b + 1];
      if (to - from < minSliver) continue;
      var area = 0;
      for (var px = from; px <= to; px += 1) {
        for (var py = 0; py < boxHeight; py += 1) {
          if (mask[(component.minY + py) * width + component.minX + px]) area += 1;
        }
      }
      if (!area) continue;
      pieces.push({
        minX: component.minX + from,
        minY: component.minY,
        maxX: component.minX + to,
        maxY: component.maxY,
        area: area
      });
    }

    return pieces.length > 1 ? pieces : [component];
  }

  function intersectionOverUnion(a, b) {
    var x1 = Math.max(a.x, b.x);
    var y1 = Math.max(a.y, b.y);
    var x2 = Math.min(a.x + a.width, b.x + b.width);
    var y2 = Math.min(a.y + a.height, b.y + b.height);
    var overlap = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (!overlap) return 0;
    return overlap / (a.width * a.height + b.width * b.height - overlap);
  }

  /**
   * Find card-shaped regions in an image.
   *
   * @param {HTMLCanvasElement|HTMLImageElement} source
   * @param {{minConfidence?: number, maxRegions?: number, iou?: number}} [options]
   * @returns {Array<{x: number, y: number, width: number, height: number,
   *   confidence: number, card: null}>} boxes in source-image pixels
   */
  function findCardRegions(source, options) {
    var opts = options || {};
    var minConfidence = typeof opts.minConfidence === 'number' ? opts.minConfidence : 0.35;
    var maxRegions = opts.maxRegions || 12;
    var iouLimit = typeof opts.iou === 'number' ? opts.iou : 0.3;

    var analysis = toAnalysisCanvas(source);
    var width = analysis.canvas.width;
    var height = analysis.canvas.height;
    var imageData = analysis.canvas.getContext('2d').getImageData(0, 0, width, height);
    var mask = buildFaceMask(imageData);
    var visited = new Uint8Array(width * height);

    var totalArea = width * height;
    var candidates = [];

    for (var index = 0; index < mask.length; index += 1) {
      if (!mask[index] || visited[index]) continue;
      var blob = collectComponent(mask, visited, width, height, index);
      var blobFraction = blob.area / totalArea;
      if (blobFraction < MIN_AREA_FRACTION || blobFraction > MAX_AREA_FRACTION) continue;

      splitFannedComponent(mask, width, blob).forEach(function (component) {
        var boxWidth = component.maxX - component.minX + 1;
        var boxHeight = component.maxY - component.minY + 1;
        var aspect = boxWidth / boxHeight;
        if (aspect < MIN_ASPECT || aspect > MAX_ASPECT) return;

        var confidence = regionConfidence(component);
        if (confidence < minConfidence) return;

        candidates.push({
          x: component.minX / analysis.scale,
          y: component.minY / analysis.scale,
          width: boxWidth / analysis.scale,
          height: boxHeight / analysis.scale,
          confidence: confidence,
          card: null
        });
      });
    }

    candidates.sort(function (a, b) { return b.confidence - a.confidence; });

    var kept = [];
    candidates.forEach(function (candidate) {
      if (kept.length >= maxRegions) return;
      var overlaps = kept.some(function (existing) {
        return intersectionOverUnion(candidate, existing) > iouLimit;
      });
      if (!overlaps) kept.push(candidate);
    });

    // Left-to-right reads naturally against the photo.
    return kept.sort(function (a, b) { return a.x - b.x; });
  }

  /**
   * Full pass: find regions, then label them if a classifier is registered.
   *
   * @returns {Promise<{regions: Array, labelled: boolean, elapsedMs: number}>}
   */
  function runFallbackDetection(source, options, startedAt) {
    var regions = findCardRegions(source, options);

    if (!classifier) {
      return Promise.resolve({
        regions: regions,
        labelled: false,
        elapsedMs: (typeof performance !== 'undefined' ? performance : Date).now() - startedAt
      });
    }

    var context = source.getContext ? source.getContext('2d') : null;
    var work = regions.map(function (region) {
      var patch = null;
      if (context) {
        patch = context.getImageData(
          Math.max(0, Math.round(region.x)),
          Math.max(0, Math.round(region.y)),
          Math.max(1, Math.round(region.width)),
          Math.max(1, Math.round(region.height))
        );
      }
      return Promise.resolve(classifier(patch, region))
        .then(function (result) {
          if (result && result.card) {
            region.card = result.card;
            region.confidence = typeof result.confidence === 'number'
              ? result.confidence
              : region.confidence;
          }
          return region;
        })
        .catch(function () { return region; });
    });

    return Promise.all(work).then(function (labelled) {
      return {
        regions: labelled,
        labelled: true,
        elapsedMs: (typeof performance !== 'undefined' ? performance : Date).now() - startedAt
      };
    });
  }

  /**
   * Full pass: run registered detector (e.g. YOLO WebGPU), falling back to heuristic
   * segmentation if absent or on error.
   *
   * @returns {Promise<{regions: Array, labelled: boolean, elapsedMs: number}>}
   */
  function detect(source, options) {
    var startedAt = (typeof performance !== 'undefined' ? performance : Date).now();

    if (detector) {
      return Promise.resolve(detector(source, options))
        .then(function (result) {
          if (result && Array.isArray(result.regions)) {
            if (typeof result.elapsedMs !== 'number') {
              result.elapsedMs = (typeof performance !== 'undefined' ? performance : Date).now() - startedAt;
            }
            if (typeof result.labelled !== 'boolean') {
              result.labelled = true;
            }
            return result;
          }
          return runFallbackDetection(source, options, startedAt);
        })
        .catch(function (err) {
          if (typeof console !== 'undefined' && console.warn) {
            console.warn('Registered detector failed, falling back to heuristic regions:', err);
          }
          return runFallbackDetection(source, options, startedAt);
        });
    }

    return runFallbackDetection(source, options, startedAt);
  }

  return {
    registerClassifier: registerClassifier,
    hasClassifier: hasClassifier,
    registerDetector: registerDetector,
    hasDetector: hasDetector,
    findCardRegions: findCardRegions,
    detect: detect
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HakimDetector;
}

