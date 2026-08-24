/**
 * Service worker routing tests.
 *
 * sw.js cannot be imported normally: it references ServiceWorkerGlobalScope APIs at load
 * time. It is evaluated in a vm context with stubs, which also exposes its module-scope
 * functions for direct assertion.
 */

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadServiceWorker() {
  const listeners = {};
  const context = {
    self: {
      addEventListener: (type, fn) => { listeners[type] = fn; },
      location: { origin: 'https://example.github.io' },
      skipWaiting: () => Promise.resolve(),
      clients: { claim: () => Promise.resolve() }
    },
    caches: { open: () => Promise.resolve({}), keys: () => Promise.resolve([]) },
    fetch: () => Promise.resolve(),
    console,
    URL
  };
  vm.createContext(context);
  vm.runInContext(
    fs.readFileSync(path.join(__dirname, 'sw.js'), 'utf8'),
    context
  );
  return { context, listeners };
}

test('registers install, activate, and fetch handlers', () => {
  const { listeners } = loadServiceWorker();
  assert.deepStrictEqual(
    Object.keys(listeners).sort(),
    ['activate', 'fetch', 'install']
  );
});

test('routes versioned runtime and model blobs to the cache-first branch', () => {
  const { context } = loadServiceWorker();
  for (const p of [
    '/runtime/ort.webgpu.min.js',
    '/runtime/ort-wasm-simd-threaded.jsep.wasm',
    '/models/baloot-v1.int8.onnx',
    '/AIBaloot/models/model.json'
  ]) {
    assert.strictEqual(context.isImmutable(new URL('https://x' + p)), true, p);
  }
});

test('routes app code to the network-first branch so deploys reach clients', () => {
  const { context } = loadServiceWorker();
  for (const p of ['/', '/index.html', '/app.js', '/model-runner.js', '/scoring.js']) {
    assert.strictEqual(context.isImmutable(new URL('https://x' + p)), false, p);
  }
});

test('precaches every asset the two runtime paths actually fetch', () => {
  const { context } = loadServiceWorker();
  const all = context.APP_ASSETS.concat(context.IMMUTABLE_ASSETS);
  for (const required of [
    'index.html',
    'model-runner.js',
    'runtime/ort.webgpu.min.js',
    'runtime/ort-wasm-simd-threaded.jsep.mjs',
    'runtime/ort-wasm-simd-threaded.jsep.wasm',
    'models/baloot-v1.fp16.onnx',
    'models/baloot-v1.int8.onnx'
  ]) {
    assert.ok(all.includes(required), 'missing from precache list: ' + required);
  }
});

test('precache list has no entry that is absent from disk', () => {
  const { context } = loadServiceWorker();
  for (const asset of context.APP_ASSETS.concat(context.IMMUTABLE_ASSETS)) {
    if (asset === './') continue;
    assert.ok(
      fs.existsSync(path.join(__dirname, asset)),
      'precached asset does not exist: ' + asset
    );
  }
});

test('cache name is versioned so activate can evict prior releases', () => {
  const { context } = loadServiceWorker();
  assert.match(context.CACHE_NAME, /^hakim-v\d+$/);
});
