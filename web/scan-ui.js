/**
 * Hakim — the scan experience.
 *
 * Everything between "player taps the orb" and "a hand value is ready":
 * photograph, detect, review the cards, adjust the declaration, confirm.
 *
 *   aim → analyse (three named stages) → verify → confirm
 *
 * The value itself always comes from `HakimScoring.scoreHand()`. This module
 * never computes points; it collects the cards, shows what the engine makes of
 * them, and hands the engine's number back through `onConfirm`. Which is what
 * lets the detector be swapped or improved without touching the experience.
 */

var HakimScanUI = (function () {
  'use strict';

  var SUIT_META = {
    h: { symbol: '♥', color: 'red' },
    d: { symbol: '♦', color: 'red' },
    c: { symbol: '♣', color: 'black' },
    s: { symbol: '♠', color: 'black' }
  };

  /** Longest edge kept when a photo is loaded, in pixels. */
  var MAX_IMAGE_EDGE = 1280;

  /** The three stages shown while a photo is being read. */
  var SCAN_STEPS = ['scan_step_detect', 'scan_step_read', 'scan_step_calc'];

  var el = {};
  var config = {
    teams: ['us', 'them'],
    teamName: function (team) { return team; },
    onConfirm: function () {},
    onClose: function () {}
  };

  var state = {
    sourceCanvas: null,
    hand: [],
    nextCardId: 1,
    mode: 'sun',
    trump: null,
    editingId: null,
    assignTo: 'us'
  };

  var t = function (key, vars) { return HakimI18N.t(key, vars); };
  var $ = function (id) { return document.getElementById(id); };

  // ---- Hand state -------------------------------------------------------

  function addCard(card, confidence, regionIndex) {
    state.hand.push({
      id: state.nextCardId++,
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

  function cardLabel(card) {
    if (!card) return t('unknown_card');
    var parsed = HakimScoring.parseCard(card);
    return parsed.rank + ' ' + t('suit_' + parsed.suit);
  }

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
    var classes = 'card-face' +
      (meta.color === 'red' ? ' card-face--red' : '') +
      (isTrump ? ' card-face--trump' : '');
    return '<span class="' + classes + '">' + badge +
      '<span class="card-face__rank">' + parsed.rank + '</span>' +
      '<span class="card-face__pip">' + meta.symbol + '</span></span>';
  }

  // ---- Photo intake -------------------------------------------------------

  function loadImageFile(file) {
    if (!file || !/^image\//.test(file.type)) return;
    var reader = new FileReader();
    reader.onload = function (event) {
      var image = new Image();
      image.onload = function () { adoptImage(image); };
      image.onerror = function () { showMessage(t('detect_error'), 'error'); };
      image.src = event.target.result;
    };
    reader.onerror = function () { showMessage(t('detect_error'), 'error'); };
    reader.readAsDataURL(file);
  }

  function adoptImage(image) {
    var scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(image.width, image.height));
    var canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(image.width * scale));
    canvas.height = Math.max(1, Math.round(image.height * scale));
    canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);

    state.sourceCanvas = canvas;
    el['stage-aim'].classList.add('hidden');
    el['stage-photo'].classList.remove('hidden');
    runDetection();
  }

  function showAimStage() {
    state.sourceCanvas = null;
    el['stage-photo'].classList.add('hidden');
    el['stage-aim'].classList.remove('hidden');
    showMessage('');
    el['scan-diagnostics'].classList.add('hidden');
  }

  function showMessage(message, tone) {
    el['scan-message'].textContent = message;
    el['scan-message'].className = 'scan-message' + (tone === 'error' ? ' is-error' : '');
  }

  // ---- Stages --------------------------------------------------------------

  var stepTimers = [];

  /**
   * The stages are honest about the pipeline — regions, then labels, then the
   * scoring call — without naming a single piece of technology at the player.
   */
  function renderSteps(activeIndex) {
    var host = el['scan-steps'];
    host.innerHTML = '';
    SCAN_STEPS.forEach(function (key, index) {
      var stepState = index < activeIndex ? 'done' : (index === activeIndex ? 'active' : 'idle');
      var row = document.createElement('p');
      row.className = 'scan-step';
      row.setAttribute('data-state', stepState);
      row.innerHTML = '<span class="scan-step__dot" aria-hidden="true"></span><span>' + t(key) + '</span>';
      host.appendChild(row);
    });
  }

  function startStages() {
    stopStages();
    renderSteps(0);
    el['scan-overlay'].classList.remove('hidden');
    // The real work is one async call, so the middle stages are paced rather
    // than driven by it: they advance, then wait for the result.
    stepTimers.push(setTimeout(function () { renderSteps(1); }, 420));
    stepTimers.push(setTimeout(function () { renderSteps(2); }, 900));
  }

  function stopStages() {
    stepTimers.forEach(clearTimeout);
    stepTimers = [];
    el['scan-overlay'].classList.add('hidden');
  }

  // ---- Detection -------------------------------------------------------------

  function runDetection() {
    if (!state.sourceCanvas) return;
    startStages();

    HakimUI.afterPaint(function () {
      HakimDetector.detect(state.sourceCanvas)
        .then(function (result) {
          state.hand = [];
          result.regions.forEach(function (region, index) {
            addCard(region.card, region.confidence, index + 1);
          });
          drawPhoto(result.regions);
          reportDetection(result);
          renderVerify();
        })
        .catch(function () {
          showMessage(t('detect_error'), 'error');
        })
        .then(stopStages);
    });
  }

  function reportDetection(result) {
    if (!result.regions.length) {
      showMessage(t('detect_none'), 'error');
    } else {
      showMessage(result.labelled ? '' : t('detect_needs_naming'));
    }

    var diagnostics = el['scan-diagnostics'];
    if (result.backend && result.backend !== 'none') {
      diagnostics.textContent = t('detect_diagnostics', {
        backend: result.backend === 'webgpu' ? 'WebGPU' : 'WASM',
        variant: (result.modelVariant || 'fp16').toUpperCase(),
        ms: Math.round(result.elapsedMs)
      });
      diagnostics.classList.remove('hidden');
    } else {
      diagnostics.classList.add('hidden');
    }
  }

  /** Draw the photo with a numbered box per detected region. */
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
      ctx.strokeStyle = known ? '#d9a83c' : 'rgba(217, 168, 60, 0.65)';
      if (!known) ctx.setLineDash([10, 8]);
      ctx.strokeRect(region.x, region.y, region.width, region.height);
      ctx.setLineDash([]);

      // The badge ties the box to its card in the strip, which matters because
      // the strip reads right-to-left while the boxes run left-to-right.
      var size = Math.max(18, source.width / 34);
      ctx.font = '700 ' + size + 'px Tajawal, system-ui, sans-serif';
      ctx.fillStyle = '#d9a83c';
      ctx.fillRect(region.x, region.y - size * 1.25, size * 1.3, size * 1.25);
      ctx.fillStyle = '#131518';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';
      ctx.fillText(String(index + 1), region.x + size * 0.65, region.y - size * 0.6);
    });
  }

  // ---- Verification sheet -------------------------------------------------------

  /**
   * Everything the player needs to trust the number: the cards as read, what
   * each is worth, the projects found, and the total — in that order, so "is
   * this right?" is answered before it is asked.
   */
  function renderVerify() {
    var result = HakimScoring.scoreHand(heldCards(), { mode: state.mode, trump: state.trump });

    renderCards();
    renderProjects(result);
    renderNotices(result);

    el['verify-total'].textContent = result.total;
    el['mode-sun'].setAttribute('aria-pressed', String(state.mode === 'sun'));
    el['mode-hokum'].setAttribute('aria-pressed', String(state.mode === 'hokum'));
    el['trump-buttons'].classList.toggle('hidden', state.mode !== 'hokum');
    el['trump-buttons'].querySelectorAll('.suit-btn').forEach(function (button) {
      button.setAttribute('aria-pressed', String(state.trump === button.dataset.suit));
    });

    config.teams.forEach(function (team) {
      var button = $('assign-' + team);
      button.textContent = config.teamName(team);
      button.setAttribute('aria-pressed', String(state.assignTo === team));
    });
  }

  function renderCards() {
    var row = el['verify-cards'];
    row.innerHTML = '';

    state.hand.forEach(function (entry) {
      var slot = document.createElement('button');
      slot.type = 'button';
      slot.className = 'card-slot';
      slot.setAttribute('aria-label', t('aria_edit_card', { card: cardLabel(entry.card) }));
      var points = entry.card
        ? '+' + HakimScoring.cardPoints(entry.card, state.mode, state.trump)
        : t('tap_to_identify');
      slot.innerHTML = cardFaceMarkup(entry.card, entry.regionIndex) +
        '<span class="card-slot__points ltr">' + points + '</span>';
      slot.addEventListener('click', function () { openPicker(entry.id); });
      row.appendChild(slot);
    });

    var add = document.createElement('button');
    add.type = 'button';
    add.className = 'card-slot';
    add.setAttribute('aria-label', t('aria_add_card'));
    add.title = t('aria_add_card');
    add.innerHTML = '<span class="add-face">' + HakimIcons.svg('plus', { size: 20 }) + '</span>' +
      '<span class="card-slot__points">' + t('hand_count', {
        count: state.hand.length, expected: HakimScoring.HAND_SIZE
      }) + '</span>';
    add.addEventListener('click', function () { openPicker(null); });
    row.appendChild(add);
  }

  function renderProjects(result) {
    var host = el['verify-projects'];
    host.innerHTML = '';
    result.projects.forEach(function (project) {
      var badge = document.createElement('span');
      badge.className = 'project-badge';
      var suit = project.suit ? ' ' + SUIT_META[project.suit].symbol : '';
      badge.innerHTML = '<span>' + t('project_' + project.id) + suit + '</span>' +
        '<span class="num ltr">+' + project.value + '</span>';
      host.appendChild(badge);
    });
  }

  function renderNotices(result) {
    var host = el['verify-notices'];
    host.innerHTML = '';

    var notices = [];
    var unknown = state.hand.filter(function (entry) { return !entry.card; }).length;
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
      host.appendChild(node);
    });
  }

  /** Confirm: hand the engine's total to the caller as one finished round. */
  function confirmHand() {
    var result = HakimScoring.scoreHand(heldCards(), { mode: state.mode, trump: state.trump });
    if (!result.total) {
      HakimUI.toast(t('toast_empty_round'));
      HakimUI.buzz([10, 40, 10]);
      return;
    }

    HakimUI.close('sheet-verify');
    HakimUI.buzz([12, 30, 18]);
    config.onConfirm({
      team: state.assignTo,
      points: result.total,
      meta: {
        mode: result.mode,
        trump: result.trump,
        cards: heldCards(),
        cardPoints: result.cardPoints,
        projectPoints: result.projectPoints
      }
    });
  }

  // ---- Card picker ---------------------------------------------------------------

  function buildPickerGrid() {
    var grid = el['picker-grid'];
    grid.innerHTML = '';
    HakimScoring.SUITS.forEach(function (suit) {
      var row = document.createElement('div');
      row.className = 'picker-row';

      var label = document.createElement('span');
      label.className = 'picker-suit';
      label.style.color = SUIT_META[suit].color === 'red' ? 'var(--them-deep)' : 'var(--ink)';
      label.textContent = SUIT_META[suit].symbol;
      label.setAttribute('aria-hidden', 'true');
      row.appendChild(label);

      HakimScoring.RANKS.forEach(function (rank) {
        var card = rank + suit;
        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'picker-cell num';
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

  var externalPickerCallback = null;

  function openPicker(entryId) {
    externalPickerCallback = null;
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

  function openPickerExternal(options) {
    var opts = options || {};
    externalPickerCallback = opts;
    state.editingId = null;
    var taken = opts.takenCards || {};

    el['picker-grid'].querySelectorAll('.picker-cell').forEach(function (button) {
      button.disabled = Boolean(taken[button.dataset.card]);
    });
    el['picker-remove'].classList.toggle('hidden', !opts.allowRemove);
    el['card-picker'].showModal();
  }

  function choosePickerCard(card) {
    if (externalPickerCallback && typeof externalPickerCallback.onChoose === 'function') {
      var cb = externalPickerCallback.onChoose;
      externalPickerCallback = null;
      el['card-picker'].close();
      cb(card);
      return;
    }
    var entry = state.editingId === null ? null : findEntry(state.editingId);
    if (entry) {
      entry.card = card;
      entry.confidence = null;
    } else {
      addCard(card, null);
    }
    el['card-picker'].close();
    renderVerify();
  }

  function removeEditingCard() {
    if (externalPickerCallback && typeof externalPickerCallback.onRemove === 'function') {
      var cb = externalPickerCallback.onRemove;
      externalPickerCallback = null;
      el['card-picker'].close();
      cb();
      return;
    }
    if (state.editingId === null) return;
    state.hand = state.hand.filter(function (entry) { return entry.id !== state.editingId; });
    el['card-picker'].close();
    renderVerify();
  }

  // ---- Declaration controls ----------------------------------------------------------

  function buildTrumpButtons() {
    var host = el['trump-buttons'];
    host.innerHTML = '';
    HakimScoring.SUITS.forEach(function (suit) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'suit-btn';
      button.dataset.suit = suit;
      button.dataset.color = SUIT_META[suit].color;
      button.setAttribute('aria-pressed', String(state.trump === suit));
      button.setAttribute('aria-label', t('suit_' + suit));
      button.title = t('suit_' + suit);
      button.innerHTML = '<span aria-hidden="true">' + SUIT_META[suit].symbol + '</span>';
      button.addEventListener('click', function () {
        state.trump = suit;
        renderVerify();
      });
      host.appendChild(button);
    });
  }

  function setMode(mode) {
    state.mode = mode;
    if (mode === 'sun') state.trump = null;
    renderVerify();
  }

  // ---- Wiring ---------------------------------------------------------------------------

  var customScanCallback = null;

  function openScanner(options) {
    state.hand = [];
    var opts = options || {};
    if (typeof opts === 'function') {
      customScanCallback = opts;
    } else if (opts && typeof opts.onResult === 'function') {
      customScanCallback = opts.onResult;
      if (opts.mode) state.mode = opts.mode;
      if (opts.trump) state.trump = opts.trump;
      if (opts.team) state.assignTo = opts.team;
    } else {
      customScanCallback = null;
    }
    showAimStage();
    // The scanner is the app's only full-bleed dark surface, so the system
    // chrome follows it in and back out again.
    HakimUI.useDarkChrome();
    HakimUI.open('scanner', el['btn-capture']);
  }

  function closeScanner() {
    customScanCallback = null;
    HakimUI.close('scanner');
    HakimUI.usePaperChrome();
    if (typeof config.onClose === 'function') config.onClose();
  }

  function openVerify() {
    renderVerify();
    HakimUI.open('sheet-verify');
  }

  function bindCapture() {
    el['btn-capture'].addEventListener('click', function () { el['file-camera'].click(); });
    el['btn-gallery'].addEventListener('click', function () { el['file-upload'].click(); });
    [el['file-camera'], el['file-upload']].forEach(function (input) {
      input.addEventListener('change', function (event) {
        loadImageFile(event.target.files && event.target.files[0]);
        event.target.value = '';
      });
    });

    el['btn-retake'].addEventListener('click', showAimStage);
    el['btn-redetect'].addEventListener('click', runDetection);
    el['btn-close-scanner'].addEventListener('click', closeScanner);

    el['btn-scan-confirm'].addEventListener('click', function () {
      if (customScanCallback) {
        var cb = customScanCallback;
        customScanCallback = null;
        var cards = heldCards();
        var scoreResult = HakimScoring.scoreHand(cards, { mode: state.mode, trump: state.trump });
        closeScanner();
        cb({
          cards: cards,
          mode: state.mode,
          trump: state.trump,
          team: state.assignTo,
          result: scoreResult
        });
        return;
      }
      closeScanner();
      openVerify();
    });

    // Enter the hand by hand: same verification sheet, no photo.
    el['btn-scan-manual'].addEventListener('click', function () {
      if (customScanCallback) {
        var cb = customScanCallback;
        customScanCallback = null;
        closeScanner();
        cb({
          cards: [],
          mode: state.mode,
          trump: state.trump,
          team: state.assignTo,
          result: HakimScoring.scoreHand([], { mode: state.mode, trump: state.trump }),
          manual: true
        });
        return;
      }
      closeScanner();
      openVerify();
      openPicker(null);
    });
  }

  function bindDropzone() {
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

    // A file dropped anywhere on the page is a request to scan it.
    ['dragover', 'drop'].forEach(function (type) {
      document.addEventListener(type, function (event) { event.preventDefault(); });
    });
    document.addEventListener('drop', function (event) {
      var files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length || HakimUI.isOpen('scanner')) return;
      openScanner();
      loadImageFile(files[0]);
    });
  }

  function bindVerify() {
    el['mode-sun'].addEventListener('click', function () { setMode('sun'); });
    el['mode-hokum'].addEventListener('click', function () { setMode('hokum'); });

    config.teams.forEach(function (team) {
      $('assign-' + team).addEventListener('click', function () {
        state.assignTo = team;
        HakimUI.buzz(6);
        renderVerify();
      });
    });

    el['btn-verify-confirm'].addEventListener('click', confirmHand);
    el['btn-verify-edit'].addEventListener('click', function () {
      var unknown = state.hand.filter(function (entry) { return !entry.card; })[0];
      openPicker(unknown ? unknown.id : null);
    });

    el['picker-close'].addEventListener('click', function () { el['card-picker'].close(); });
    el['picker-remove'].addEventListener('click', removeEditingCard);
    el['card-picker'].addEventListener('close', function () { state.editingId = null; });
  }

  function cacheDom() {
    [
      'scanner', 'btn-close-scanner', 'btn-scan-manual', 'stage-aim', 'stage-photo',
      'dropzone', 'btn-capture', 'btn-gallery', 'file-camera', 'file-upload',
      'photo-canvas', 'scan-overlay', 'scan-steps', 'scan-message', 'scan-diagnostics',
      'btn-retake', 'btn-redetect', 'btn-scan-confirm',
      'verify-cards', 'verify-projects', 'verify-notices', 'verify-total',
      'mode-sun', 'mode-hokum', 'trump-buttons',
      'btn-verify-confirm', 'btn-verify-edit',
      'card-picker', 'picker-grid', 'picker-remove', 'picker-close'
    ].forEach(function (id) { el[id] = $(id); });
  }

  /**
   * @param {{teams: string[], teamName: function, onConfirm: function}} options
   *        `onConfirm` receives `{team, points, meta}` — an already-scored hand.
   */
  function init(options) {
    config = Object.assign({}, config, options || {});
    cacheDom();
    bindCapture();
    bindDropzone();
    bindVerify();
    relocalise();
  }

  /** Rebuild the language-dependent controls after a locale switch. */
  function relocalise() {
    buildPickerGrid();
    buildTrumpButtons();
    renderVerify();
  }

  return {
    init: init,
    open: openScanner,
    close: closeScanner,
    openVerify: openVerify,
    relocalise: relocalise,
    getHand: heldCards,
    openPickerExternal: openPickerExternal,
    cardFaceMarkup: cardFaceMarkup,
    SUIT_META: SUIT_META
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HakimScanUI;
}
