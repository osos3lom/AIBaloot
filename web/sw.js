/**
 * Hakim — Service Worker for offline asset caching.
 *
 * Two strategies, because the assets have opposite update characteristics:
 *
 * - The runtime and model blobs are large, immutable, and versioned by filename. Serving
 *   them cache-first is the whole point of the worker — a repeat visit skips ~7 MB.
 * - The app scripts are small and change on every deploy. Cache-first would freeze them at
 *   whatever version installed first, so a shipped bugfix would never reach anyone who had
 *   already loaded the page. They go network-first, falling back to cache only offline.
 *
 * Bump CACHE_VERSION on release; `activate` deletes every cache that does not match.
 */

var CACHE_VERSION = 'v13';
var CACHE_NAME = 'hakim-' + CACHE_VERSION;

// Immutable, content-versioned by filename. Precached and served cache-first.
var IMMUTABLE_ASSETS = [
  'runtime/ort.webgpu.min.js',
  'runtime/ort-wasm-simd-threaded.jsep.mjs',
  'runtime/ort-wasm-simd-threaded.jsep.wasm',
  'models/model.json',
  'models/baloot-fan-v2.fp16.onnx',
  'models/baloot-fan-v2.int8.onnx'
];

// Changes on every deploy. Precached for offline use, but served network-first.
var APP_ASSETS = [
  './',
  'index.html',
  'manifest.webmanifest',
  'icons/favicon.ico',
  'icons/apple-touch-icon.png',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/icon-maskable-512.png',
  'i18n.js',
  'icons.js',
  'ui-kit.js',
  'scoring.js',
  'match.js',
  'detect.js',
  'model-runner.js',
  'scan-ui.js',
  'app.js'
];

function isImmutable(url) {
  return /\/(runtime|models|vid)\//.test(url.pathname);
}

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(function (cache) {
      // Added individually: cache.addAll rejects the whole batch if any single request
      // fails, which would leave the worker with nothing cached at all.
      return Promise.all(
        APP_ASSETS.concat(IMMUTABLE_ASSETS).map(function (asset) {
          return cache.add(asset).catch(function (err) {
            console.warn('Precache skipped', asset, err);
          });
        })
      );
    }).then(function () {
      return self.skipWaiting();
    })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(
        keys.map(function (key) {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(function () {
      return self.clients.claim();
    })
  );
});

function cacheFirst(request) {
  return caches.match(request).then(function (cached) {
    if (cached) return cached;
    return fetch(request).then(function (response) {
      if (response && response.status === 200 && response.type === 'basic') {
        var copy = response.clone();
        caches.open(CACHE_NAME).then(function (cache) { cache.put(request, copy); });
      }
      return response;
    });
  });
}

function networkFirst(request) {
  return fetch(request).then(function (response) {
    if (response && response.status === 200 && response.type === 'basic') {
      var copy = response.clone();
      caches.open(CACHE_NAME).then(function (cache) { cache.put(request, copy); });
    }
    return response;
  }).catch(function () {
    return caches.match(request).then(function (cached) {
      // A navigation that misses the cache still needs a document to render.
      return cached || (request.mode === 'navigate' ? caches.match('index.html') : undefined);
    });
  });
}

self.addEventListener('fetch', function (event) {
  var request = event.request;
  if (request.method !== 'GET') return;

  var url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(isImmutable(url) ? cacheFirst(request) : networkFirst(request));
});
