/**
 * Hakim — hand-value app controller.
 *
 * Flow: photograph the hand -> confirm the cards -> read the value.
 * Scoring lives in scoring.js, detection in detect.js, copy in i18n.js.
 */
(function () {
  'use strict';

  var SUIT_META = {
    h: { symbol: '♥', color: 'red' },
    d: { symbol: '♦', color: 'red' },
    c: { symbol: '♣', color: 'black' },
    s: { symbol: '♠', color: 'black' }
  };

  /** Longest edge kept when a photo is loaded, in pixels. */
  var MAX_IMAGE_EDGE = 1280;

  var state = {
    lang: 'ar',
    sourceCanvas: null,
    hand: [],
    nextId: 1,
    mode: 'sun',
    trump: null,
    editingId: null
  };

  var el = {};

  function $(id) { return document.getElementById(id); }

  function cacheDom() {
    [
      'lang-toggle', 'dropzone', 'btn-camera', 'btn-upload', 'btn-skip-photo',
      'file-camera', 'file-upload', 'capture-empty', 'capture-preview',
      'photo-canvas', 'detect-overlay', 'detect-summary', 'btn-redetect', 'btn-retake',
      'step-hand', 'hand-strip', 'hand-count', 'hand-notices', 'btn-clear-hand',
      'step-score', 'mode-sun', 'mode-hokum', 'trump-picker', 'trump-buttons',
      'score-total', 'score-cards', 'score-meter', 'score-share', 'score-projects-total',
      'projects-row', 'breakdown-body', 'suggestion', 'btn-copy',
      'sticky-label', 'sticky-total', 'sticky-action',
      'card-picker', 'picker-grid', 'picker-remove'
    ].forEach(function (id) {
      el[id] = $(id);
    });
  }

  var t = function (key, vars) { return HakimI18N.t(key, vars); };

  // ---- Card rendering -------------------------------------------------

  /**
   * `regionIndex` ties the card back to the numbered box drawn on the photo,
   * which matters because the strip reads right-to-left in Arabic while the
   * boxes are numbered left-to-right across the image.
   */
  function cardFaceMarkup(card, regionIndex) {
    var badge = regionIndex
      ? '<span class="card-face__badge" aria-hidden="true">' + regionIndex + '</span>'
      : '';
    if (!card) {
      return '<span class="card-face card-face--unknown">' + badge +
        '<span class="card-face__pip">؟</span></span>';
    }
    var parsed = HakimScoring.parseCard(card);
    var meta = SUIT_META[parsed.suit];
    var isTrump = state.mode === 'hokum' && parsed.suit === state.trump;
    var classes = 'card-face' + (meta.color === 'red' ? ' card-face--red' : '') + (isTrump ? ' card-face--trump' : '');
    return '<span class="' + classes + '">' + badge +
      '<span class="card-face__rank">' + parsed.rank + '</span>' +
      '<span class="card-face__pip">' + meta.symbol + '</span>' +
      '<span class="card-face__corner">' + meta.symbol + '</span>' +
      '</span>';
  }

  function cardLabel(card) {
    if (!card) return t('unknown_card');
    var parsed = HakimScoring.parseCard(card);
    return parsed.rank + ' ' + t('suit_' + parsed.suit);
  }

  // ---- Hand state -----------------------------------------------------

  function addCard(card, confidence, regionIndex) {
    state.hand.push({
      id: state.nextId++,
      card: card || null,
      confidence: typeof confidence === 'number' ? confidence : null,
      regionIndex: typeof regionIndex === 'number' ? regionIndex : null
    });
  }

  function heldCards() {
    return state.hand
      .map(function (entry) { return entry.card; })
      .filter(function (card) { return card !== null; });
  }

  function findEntry(id) {
    return state.hand.filter(function (entry) { return entry.id === id; })[0] || null;
  }

  // ---- Photo capture --------------------------------------------------

  function loadImageFile(file) {
    if (!file || !/^image\//.test(file.type)) return;
    var reader = new FileReader();
    reader.onload = function (event) {
      var image = new Image();
      image.onload = function () { adoptImage(image); };
      image.onerror = function () { showDetectMessage(t('detect_error'), 'error'); };
      image.src = event.target.result;
    };
    reader.onerror = function () { showDetectMessage(t('detect_error'), 'error'); };
    reader.readAsDataURL(file);
  }

  function adoptImage(image) {
    var scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(image.width, image.height));
    var canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(image.width * scale));
    canvas.height = Math.max(1, Math.round(image.height * scale));
    canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);

    state.sourceCanvas = canvas;
    el['capture-empty'].classList.add('hidden');
    el['capture-preview'].classList.remove('hidden');
    runDetection();
  }

  function drawPhoto(regions) {
    var source = state.sourceCanvas;
    if (!source) return;
    var canvas = el['photo-canvas'];
    canvas.width = source.width;
    canvas.height = source.height;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(source, 0, 0);

    (regions || []).forEach(function (region, index) {
      var known = region.card !== null && region.card !== undefined;
      ctx.lineWidth = Math.max(2, source.width / 320);
      ctx.strokeStyle = known ? '#e3b862' : 'rgba(227, 184, 98, 0.75)';
      if (!known) ctx.setLineDash([10, 8]);
      ctx.strokeRect(region.x, region.y, region.width, region.height);
      ctx.setLineDash([]);

      var badge = String(index + 1);
      var size = Math.max(18, source.width / 34);
      ctx.font = '700 ' + size + 'px Tajawal, system-ui, sans-serif';
      ctx.fillStyle = '#e3b862';
      ctx.fillRect(region.x, region.y - size * 1.25, size * 1.3, size * 1.25);
      ctx.fillStyle = '#062b21';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';
      ctx.fillText(badge, region.x + size * 0.65, region.y - size * 0.6);
    });
  }

  function showDetectMessage(message, tone) {
    el['detect-summary'].textContent = message;
    el['detect-summary'].className = 'text-xs ' +
      (tone === 'error' ? 'text-[var(--rose-400)]' : 'text-white/60');
  }

  function runDetection() {
    if (!state.sourceCanvas) return;
    el['detect-overlay'].classList.remove('hidden');
    el['detect-overlay'].classList.add('grid');

    // Yield a frame so the overlay paints before the synchronous scan starts.
    requestAnimationFrame(function () {
      HakimDetector.detect(state.sourceCanvas)
        .then(function (result) {
          state.hand = [];
          result.regions.forEach(function (region, index) {
            addCard(region.card, region.confidence, index + 1);
          });
          drawPhoto(result.regions);

          if (!result.regions.length) {
            showDetectMessage(t('detect_none'), 'error');
          } else {
            var found = t('detect_found', {
              count: result.regions.length,
              ms: Math.round(result.elapsedMs)
            });
            showDetectMessage(result.labelled ? found : found + ' ' + t('detect_needs_naming'));
          }

          var diagEl = document.getElementById('detect-diagnostics');
          if (result.backend && result.backend !== 'none') {
            var backendName = result.backend === 'webgpu' ? 'WebGPU' : 'WASM';
            var variantName = (result.modelVariant || 'fp16').toUpperCase();
            if (diagEl) {
              diagEl.textContent = t('detect_diagnostics', {
                backend: backendName,
                variant: variantName,
                ms: Math.round(result.elapsedMs)
              });
              diagEl.classList.remove('hidden');
            }
          } else if (diagEl) {
            diagEl.classList.add('hidden');
          }

          render();
        })
        .catch(function () {
          showDetectMessage(t('detect_error'), 'error');
        })
        .then(function () {
          el['detect-overlay'].classList.add('hidden');
          el['detect-overlay'].classList.remove('grid');
        });
    });
  }

  // ---- Hand rendering -------------------------------------------------

  function renderHand() {
    var strip = el['hand-strip'];
    strip.innerHTML = '';

    if (!state.hand.length) {
      var empty = document.createElement('p');
      empty.className = 'text-sm text-white/50';
      empty.textContent = t('empty_hand');
      strip.appendChild(empty);
    }

    state.hand.forEach(function (entry) {
      var slot = document.createElement('button');
      slot.type = 'button';
      slot.className = 'card-slot';
      slot.setAttribute('aria-label', t('edit_card_label', { card: cardLabel(entry.card) }));
      slot.innerHTML = cardFaceMarkup(entry.card, entry.regionIndex) +
        '<span class="card-slot__points">' + entryPointsLabel(entry) + '</span>';
      slot.addEventListener('click', function () { openPicker(entry.id); });
      strip.appendChild(slot);
    });

    var add = document.createElement('button');
    add.type = 'button';
    add.className = 'card-slot';
    add.setAttribute('aria-label', t('add_card'));
    add.innerHTML = '<span class="add-slot" aria-hidden="true">+</span>' +
      '<span class="card-slot__points">' + t('add_card') + '</span>';
    add.addEventListener('click', function () { openPicker(null); });
    strip.appendChild(add);

    el['hand-count'].textContent = t('hand_count', {
      count: state.hand.length,
      expected: HakimScoring.HAND_SIZE
    });
  }

  function entryPointsLabel(entry) {
    if (!entry.card) return t('tap_to_identify');
    return '+' + HakimScoring.cardPoints(entry.card, state.mode, state.trump);
  }

  function renderNotices(result) {
    var box = el['hand-notices'];
    box.innerHTML = '';

    var unknown = state.hand.filter(function (entry) { return !entry.card; }).length;
    var notices = [];

    if (unknown) notices.push({ tone: 'warn', text: t('notice_unknown', { count: unknown }) });
    result.warnings.forEach(function (warning) {
      if (warning.code === 'duplicate') {
        notices.push({ tone: 'error', text: t('notice_duplicate', { card: cardLabel(warning.card) }) });
      } else if (warning.code === 'hand_size' && warning.count > 0) {
        notices.push({ tone: 'warn', text: t('notice_hand_size', { count: warning.count }) });
      } else if (warning.code === 'no_trump') {
        notices.push({ tone: 'info', text: t('notice_no_trump') });
      }
    });

    notices.forEach(function (notice) {
      var node = document.createElement('p');
      node.className = 'notice notice--' + notice.tone;
      node.textContent = notice.text;
      box.appendChild(node);
    });
  }

  // ---- Score rendering ------------------------------------------------

  function renderScore(result) {
    el['score-total'].textContent = result.total;
    el['score-cards'].textContent = result.cardPoints;
    el['score-projects-total'].textContent = result.projectPoints;
    el['score-meter'].style.width = Math.round(result.share * 100) + '%';
    el['score-share'].textContent = t('share_text', {
      points: result.cardPoints,
      total: result.deckTotal
    });

    var projects = el['projects-row'];
    projects.innerHTML = '';
    result.projects.forEach(function (project) {
      var badge = document.createElement('span');
      badge.className = 'project-badge';
      var suit = project.suit ? ' ' + SUIT_META[project.suit].symbol : '';
      badge.innerHTML = '<span>' + t('project_' + project.id) + suit + '</span>' +
        '<span class="project-badge__value">+' + project.value + '</span>';
      projects.appendChild(badge);
    });

    var body = el['breakdown-body'];
    body.innerHTML = '';
    result.breakdown.forEach(function (entry) {
      var row = document.createElement('tr');
      var note = entry.isTrump ? t('note_trump') : '';
      row.innerHTML =
        '<td class="py-1.5">' + entry.rank + ' ' + SUIT_META[entry.suit].symbol + '</td>' +
        '<td class="py-1.5 text-xs text-white/50">' + note + '</td>' +
        '<td class="py-1.5 text-end font-bold tabular-nums">' + entry.points + '</td>';
      body.appendChild(row);
    });

    renderSuggestion(result);

    el['sticky-total'].textContent = result.total;
    el['sticky-label'].textContent = state.mode === 'hokum'
      ? t('sticky_label_hokum', { suit: state.trump ? SUIT_META[state.trump].symbol : '—' })
      : t('sticky_label_sun');
  }

  function declarationLabel(option) {
    if (option.mode === 'sun') return t('mode_sun');
    return t('mode_hokum') + ' ' + SUIT_META[option.trump].symbol;
  }

  function renderSuggestion(result) {
    var box = el['suggestion'];
    box.innerHTML = '';
    var cards = heldCards();
    if (cards.length < 2) return;

    var best = HakimScoring.compareModes(cards)[0];
    // A different declaration worth the same is not worth nagging about.
    var isCurrent = best.total <= result.total;
    var node = document.createElement('p');
    node.className = 'notice notice--info';
    node.textContent = isCurrent
      ? t('suggestion_current')
      : t('suggestion_best', { label: declarationLabel(best), total: best.total });
    box.appendChild(node);
  }

  // ---- Card picker ----------------------------------------------------

  function buildPickerGrid() {
    var grid = el['picker-grid'];
    grid.innerHTML = '';
    HakimScoring.SUITS.forEach(function (suit) {
      var row = document.createElement('div');
      row.className = 'grid grid-cols-[auto_repeat(8,1fr)] items-center gap-1.5';

      var label = document.createElement('span');
      label.className = 'w-6 text-center text-lg';
      label.style.color = SUIT_META[suit].color === 'red' ? 'var(--rose-400)' : 'var(--cream-50)';
      label.textContent = SUIT_META[suit].symbol;
      label.setAttribute('aria-hidden', 'true');
      row.appendChild(label);

      HakimScoring.RANKS.forEach(function (rank) {
        var card = rank + suit;
        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'picker-cell';
        button.dataset.card = card;
        button.dataset.color = SUIT_META[suit].color;
        button.textContent = rank;
        button.setAttribute('aria-label', rank + ' ' + t('suit_' + suit));
        button.addEventListener('click', function () { choosePickerCard(card); });
        row.appendChild(button);
      });

      grid.appendChild(row);
    });
  }

  function openPicker(entryId) {
    state.editingId = entryId;
    var taken = {};
    state.hand.forEach(function (entry) {
      if (entry.card && entry.id !== entryId) taken[entry.card] = true;
    });

    el['picker-grid'].querySelectorAll('.picker-cell').forEach(function (button) {
      button.disabled = Boolean(taken[button.dataset.card]);
    });
    el['picker-remove'].classList.toggle('hidden', entryId === null);
    el['card-picker'].showModal();
  }

  function choosePickerCard(card) {
    var entry = state.editingId === null ? null : findEntry(state.editingId);
    if (entry) {
      entry.card = card;
      entry.confidence = null;
    } else {
      addCard(card, null);
    }
    el['card-picker'].close();
    render();
  }

  function removeEditingCard() {
    if (state.editingId === null) return;
    state.hand = state.hand.filter(function (entry) { return entry.id !== state.editingId; });
    el['card-picker'].close();
    render();
  }

  // ---- Mode and trump -------------------------------------------------

  function buildTrumpButtons() {
    var box = el['trump-buttons'];
    box.innerHTML = '';
    HakimScoring.SUITS.forEach(function (suit) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'suit-btn';
      button.dataset.suit = suit;
      button.dataset.color = SUIT_META[suit].color;
      button.setAttribute('aria-pressed', String(state.trump === suit));
      button.innerHTML = '<span aria-hidden="true">' + SUIT_META[suit].symbol + '</span>' +
        '<span class="sr-only">' + t('suit_' + suit) + '</span>';
      button.addEventListener('click', function () {
        state.trump = suit;
        render();
      });
      box.appendChild(button);
    });
  }

  function setMode(mode) {
    state.mode = mode;
    if (mode === 'sun') state.trump = null;
    render();
  }

  function renderModeControls() {
    el['mode-sun'].setAttribute('aria-pressed', String(state.mode === 'sun'));
    el['mode-hokum'].setAttribute('aria-pressed', String(state.mode === 'hokum'));
    el['trump-picker'].classList.toggle('hidden', state.mode !== 'hokum');
    el['trump-buttons'].querySelectorAll('.suit-btn').forEach(function (button) {
      button.setAttribute('aria-pressed', String(state.trump === button.dataset.suit));
    });
  }

  // ---- Render ---------------------------------------------------------

  function render() {
    var result = HakimScoring.scoreHand(heldCards(), { mode: state.mode, trump: state.trump });
    renderModeControls();
    renderHand();
    renderNotices(result);
    renderScore(result);

    el['step-hand'].classList.toggle('panel--muted', state.hand.length === 0);
    el['step-score'].classList.toggle('panel--muted', heldCards().length === 0);
  }

  // ---- Language -------------------------------------------------------

  function setLanguage(lang) {
    state.lang = HakimI18N.setLanguage(lang);
    var dict = HakimI18N.STRINGS[state.lang];
    document.documentElement.lang = state.lang;
    document.documentElement.dir = dict.dir;
    document.title = t('page_title');
    HakimI18N.applyToDocument();
    el['lang-toggle'].textContent = t('lang_switch_label');

    var digits = state.lang === 'ar' ? ['١', '٢', '٣'] : ['1', '2', '3'];
    document.querySelectorAll('.step-index').forEach(function (node, index) {
      node.textContent = digits[index] || String(index + 1);
    });

    buildPickerGrid();
    buildTrumpButtons();
    render();
  }

  // ---- Wiring ---------------------------------------------------------

  function bindCapture() {
    el['btn-camera'].addEventListener('click', function () { el['file-camera'].click(); });
    el['btn-upload'].addEventListener('click', function () { el['file-upload'].click(); });

    [el['file-camera'], el['file-upload']].forEach(function (input) {
      input.addEventListener('change', function (event) {
        loadImageFile(event.target.files && event.target.files[0]);
        event.target.value = '';
      });
    });

    var zone = el['dropzone'];
    ['dragenter', 'dragover'].forEach(function (type) {
      zone.addEventListener(type, function (event) {
        event.preventDefault();
        zone.dataset.active = 'true';
      });
    });
    ['dragleave', 'drop'].forEach(function (type) {
      zone.addEventListener(type, function (event) {
        event.preventDefault();
        zone.dataset.active = 'false';
      });
    });
    zone.addEventListener('drop', function (event) {
      var files = event.dataTransfer && event.dataTransfer.files;
      loadImageFile(files && files[0]);
    });

    el['btn-skip-photo'].addEventListener('click', function () {
      openPicker(null);
    });

    el['btn-redetect'].addEventListener('click', runDetection);
    el['btn-retake'].addEventListener('click', function () {
      state.sourceCanvas = null;
      el['capture-preview'].classList.add('hidden');
      el['capture-empty'].classList.remove('hidden');
      showDetectMessage('');
    });
  }

  function bindHand() {
    el['btn-clear-hand'].addEventListener('click', function () {
      state.hand = [];
      render();
    });
    el['picker-remove'].addEventListener('click', removeEditingCard);
    el['card-picker'].addEventListener('close', function () { state.editingId = null; });
  }

  function bindScore() {
    el['mode-sun'].addEventListener('click', function () { setMode('sun'); });
    el['mode-hokum'].addEventListener('click', function () { setMode('hokum'); });

    el['btn-copy'].addEventListener('click', function () {
      var result = HakimScoring.scoreHand(heldCards(), { mode: state.mode, trump: state.trump });
      var lines = [
        t('app_name') + ' — ' + (state.mode === 'sun'
          ? t('mode_sun')
          : t('mode_hokum') + ' ' + (state.trump ? SUIT_META[state.trump].symbol : '')),
        heldCards().join(' '),
        t('lbl_card_points') + ': ' + result.cardPoints,
        t('lbl_project_points') + ': ' + result.projectPoints,
        t('lbl_total') + ': ' + result.total
      ];
      var text = lines.join('\n');
      var done = function () {
        var original = el['btn-copy'].textContent;
        el['btn-copy'].textContent = t('copied');
        setTimeout(function () { el['btn-copy'].textContent = original; }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      }
    });

    el['sticky-action'].addEventListener('click', function () {
      var unknown = state.hand.filter(function (entry) { return !entry.card; })[0];
      el['step-hand'].scrollIntoView({ behavior: 'smooth', block: 'center' });
      openPicker(unknown ? unknown.id : null);
    });
  }

  function init() {
    cacheDom();
    bindCapture();
    bindHand();
    bindScore();

    el['lang-toggle'].addEventListener('click', function () {
      setLanguage(state.lang === 'ar' ? 'en' : 'ar');
    });

    setLanguage('ar');

    // Register Service Worker for offline caching
    if (
      'serviceWorker' in navigator &&
      (window.location.protocol === 'https:' ||
        window.location.hostname === 'localhost' ||
        window.location.hostname === '127.0.0.1')
    ) {
      navigator.serviceWorker.register('sw.js').catch(function (err) {
        console.warn('Hakim SW registration skipped/failed:', err);
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
