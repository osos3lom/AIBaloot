/**
 * Hakim — app shell.
 *
 * Two ways to score a round, one place they land:
 *
 *   tap a score → keypad → احسب      ─┐
 *                                     ├─→ HakimMatch.addRound()
 *   orb → photo → verify → confirm   ─┘
 *
 * Division of labour, which this file is careful not to blur:
 *   scoring.js — every Baloot formula. Never modified, never duplicated here.
 *   match.js   — teams, rounds, running totals, undo, persistence.
 *   scan-ui.js — the camera-to-hand-value experience.
 *   ui-kit.js  — overlays, toasts, haptics, number animation.
 *   this file  — views, the table, the keypad, history, settings.
 *
 * Neither entry route invents a point value: the scanner reports what
 * `HakimScoring.scoreHand()` returned, and the keypad reports what the player
 * typed. This file only decides which team the number belongs to.
 */
(function () {
  'use strict';

  /** Common round values, offered as one-tap chips above the keypad. */
  var PROJECT_CONFIG = {
    sra: { id: 'sra', max: 2, modes: ['sun', 'hokum'] },
    fifty: { id: 'fifty', max: 2, modes: ['sun', 'hokum'] },
    hundred: { id: 'hundred', max: 2, modes: ['sun', 'hokum'] },
    fourHundred: { id: 'fourHundred', max: 1, modes: ['sun'] },
    baloot: { id: 'baloot', max: 1, modes: ['hokum'] }
  };

  function getProjectValue(projId, mode) {
    if (mode === 'sun') {
      if (projId === 'sra') return 4;
      if (projId === 'fifty') return 10;
      if (projId === 'hundred') return 20;
      if (projId === 'fourHundred') return 40;
      return 0;
    } else {
      if (projId === 'sra') return 2;
      if (projId === 'fifty') return 5;
      if (projId === 'hundred') return 10;
      if (projId === 'baloot') return 2;
      return 0;
    }
  }

  var HINT_SEEN_KEY = 'hakim.hint.ai.v1';
  var VIEWS = ['home', 'game', 'history', 'settings'];

  var state = {
    lang: 'ar',
    view: 'game',

    /** Points typed but not yet committed as a round. */
    pending: { us: null, them: null },

    /** Dual-team Calculator session: active team, mode, trump, base scores, projects, cards, and source */
    calc: {
      activeTeam: 'us',
      mode: 'sun',
      trump: 'h',
      base: { us: 13, them: 13 },
      buffer: '',
      projects: {
        us: { sra: 0, fifty: 0, hundred: 0, fourHundred: 0, baloot: 0 },
        them: { sra: 0, fifty: 0, hundred: 0, fourHundred: 0, baloot: 0 }
      },
      cards: [],
      source: 'manual'
    },

    /** Last score painted per team, so a change can be animated from it. */
    painted: { us: 0, them: 0 },
    winDismissedAt: null
  };

  var el = {};
  var t = function (key, vars) { return HakimI18N.t(key, vars); };
  var $ = function (id) { return document.getElementById(id); };

  /** Display name of a team: whatever the player set, else لنا / لهم. */
  function teamName(team) {
    var names = HakimMatch.getState().names;
    return names[team] || t(team === 'us' ? 'team_us' : 'team_them');
  }

  // ---- Score Calculation Engine in Calculator (Auto-balanced 26 / 16 scores) -

  function getRoundBaseConstant(mode) {
    return mode === 'hokum' ? 16 : 26;
  }

  function getTeamProjectsSum(team) {
    var sum = 0;
    Object.keys(PROJECT_CONFIG).forEach(function (projId) {
      var cfg = PROJECT_CONFIG[projId];
      if (cfg.modes.indexOf(state.calc.mode) !== -1) {
        var count = (state.calc.projects[team] && state.calc.projects[team][projId]) || 0;
        sum += count * getProjectValue(projId, state.calc.mode);
      }
    });
    return sum;
  }

  function computeDualCalcScore() {
    var constant = getRoundBaseConstant(state.calc.mode);
    var active = state.calc.activeTeam;
    var other = active === 'us' ? 'them' : 'us';

    var activeBase = 0;
    if (state.calc.cards.length > 0) {
      var scoreRes = HakimScoring.scoreHand(state.calc.cards, {
        mode: state.calc.mode,
        trump: state.calc.trump
      });
      // Convert raw deck card points (120/152) into standard 26/16 score scale
      if (state.calc.mode === 'sun') {
        activeBase = Math.min(26, Math.round(scoreRes.cardPoints / 5));
      } else {
        activeBase = Math.min(16, Math.round(scoreRes.cardPoints / 10));
      }
    } else {
      activeBase = state.calc.base[active] !== undefined ? state.calc.base[active] : Math.round(constant / 2);
    }

    // Automatically compute other team's base score to keep the sum constant without projects
    var otherBase = Math.max(0, constant - activeBase);

    state.calc.base[active] = activeBase;
    state.calc.base[other] = otherBase;

    var projectTotals = {
      us: getTeamProjectsSum('us'),
      them: getTeamProjectsSum('them')
    };

    return {
      constant: constant,
      base: { us: state.calc.base.us, them: state.calc.base.them },
      projects: projectTotals,
      total: {
        us: state.calc.base.us + projectTotals.us,
        them: state.calc.base.them + projectTotals.them
      }
    };
  }

  // ---- Views -------------------------------------------------------------

  function setView(view) {
    if (VIEWS.indexOf(view) === -1) return;
    state.view = view;

    VIEWS.forEach(function (name) {
      var panel = $('view-' + name);
      var tab = $('tab-' + name);
      if (panel) panel.hidden = name !== view;
      if (tab) tab.setAttribute('aria-selected', String(name === view));
    });

    // The stage video and home's glass surfaces key off this attribute.
    document.body.setAttribute('data-view', view);

    window.scrollTo({ top: 0, behavior: 'auto' });
    syncStageVideo();
    render();
  }

  // ---- Home stage video -----------------------------------------------------

  /** Every `.stage__video`, each tagged with the view it belongs to. */
  var stageVideos = [];

  /**
   * Decide whether this device should download the clips at all.
   *
   * They are decoration, and decoration does not get to override a stated
   * preference or spend someone's metered data. In either case the elements
   * keep their posters, so those views still have a picture behind them — just
   * a still one — and nothing downstream has to care which happened.
   *
   * The source lives on `data-src` rather than `src` precisely so that this
   * decision happens before a single byte is requested.
   */
  function initStageVideo() {
    stageVideos = Array.prototype.slice.call(document.querySelectorAll('.stage__video'));
    if (!stageVideos.length) return;

    var connection = navigator.connection || navigator.webkitConnection || {};
    if (HakimUI.prefersReducedMotion() || connection.saveData) return;

    stageVideos.forEach(function (video) {
      var source = video.getAttribute('data-src');
      if (!source) return;
      video.src = source;
      // A failed load is not worth a message: the poster and the paper behind
      // it are already a complete background.
      video.addEventListener('error', function () { video.removeAttribute('src'); });
    });
  }

  /**
   * Exactly one clip runs at a time — the one whose view is on screen, and only
   * while there is something to see. Everything else is paused, so a background
   * view never keeps a decoder alive.
   */
  function syncStageVideo() {
    var onScreen = !document.hidden && !HakimUI.isOpen('scanner');

    stageVideos.forEach(function (video) {
      if (!video.getAttribute('src')) return;
      var mine = video.getAttribute('data-stage') === state.view;

      if (!mine || !onScreen) {
        video.pause();
        return;
      }
      var attempt = video.play();
      // Autoplay can still be refused (a policy this app cannot argue with);
      // the poster is the fallback and needs no handling.
      if (attempt && attempt.catch) attempt.catch(function () {});
    });
  }

  // ---- Score Calculator & Keypad -------------------------------------------

  function buildKeypad() {
    var chips = el['keypad-chips'];
    if (!chips) return;
    chips.innerHTML = '';
    var list = state.calc.mode === 'sun'
      ? [13, 26, '+2', '+4', '+10', '+20']
      : [8, 16, '+2', '+5', '+10'];

    list.forEach(function (points) {
      var chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip num ltr';
      chip.textContent = points;
      chip.addEventListener('click', function () {
        HakimUI.buzz(6);
        var active = state.calc.activeTeam;
        var constant = getRoundBaseConstant(state.calc.mode);
        if (typeof points === 'string' && points.indexOf('+') === 0) {
          var addVal = Number(points.slice(1)) || 0;
          var current = state.calc.base[active] || 0;
          state.calc.base[active] = Math.min(constant, current + addVal);
        } else {
          state.calc.base[active] = Math.min(constant, Number(points));
          state.calc.cards = [];
        }
        var other = active === 'us' ? 'them' : 'us';
        state.calc.base[other] = Math.max(0, constant - state.calc.base[active]);
        state.calc.buffer = String(state.calc.base[active]);
        renderCalculator();
      });
      chips.appendChild(chip);
    });

    buildCalcTrumpButtons();
  }

  function buildCalcTrumpButtons() {
    var host = el['calc-trump-buttons'];
    if (!host) return;
    host.innerHTML = '';
    HakimScoring.SUITS.forEach(function (suit) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'suit-btn';
      button.dataset.suit = suit;
      button.dataset.color = HakimScanUI.SUIT_META[suit].color;
      button.setAttribute('aria-pressed', String(state.calc.trump === suit));
      button.setAttribute('aria-label', t('suit_' + suit));
      button.title = t('suit_' + suit);
      button.innerHTML = '<span aria-hidden="true">' + HakimScanUI.SUIT_META[suit].symbol + '</span>';
      button.addEventListener('click', function () {
        setCalcTrump(suit);
      });
      host.appendChild(button);
    });
  }

  function selectActiveCalcTeam(team) {
    if (state.calc.activeTeam === team) return;
    state.calc.activeTeam = team;
    state.calc.buffer = state.calc.base[team] > 0 ? String(state.calc.base[team]) : '';
    state.calc.cards = [];
    HakimUI.buzz(6);
    renderCalculator();
    var input = el['calc-input-' + team];
    if (input) input.focus();
  }

  function toggleProject(projId) {
    var cfg = PROJECT_CONFIG[projId];
    if (!cfg) return;
    if (cfg.modes.indexOf(state.calc.mode) === -1) return;

    var activeTeam = state.calc.activeTeam;
    var current = (state.calc.projects[activeTeam] && state.calc.projects[activeTeam][projId]) || 0;
    var next = current + 1;
    if (next > cfg.max) next = 0;

    if (!state.calc.projects[activeTeam]) {
      state.calc.projects[activeTeam] = { sra: 0, fifty: 0, hundred: 0, fourHundred: 0, baloot: 0 };
    }
    state.calc.projects[activeTeam][projId] = next;
    HakimUI.buzz(6);
    renderCalculator();
  }

  function clearCalcProjects() {
    var activeTeam = state.calc.activeTeam;
    if (state.calc.projects[activeTeam]) {
      Object.keys(state.calc.projects[activeTeam]).forEach(function (key) {
        state.calc.projects[activeTeam][key] = 0;
      });
    }
    HakimUI.buzz(6);
    renderCalculator();
  }

  function setCalcMode(mode) {
    state.calc.mode = mode;
    var constant = getRoundBaseConstant(mode);
    state.calc.base.us = Math.round(constant / 2);
    state.calc.base.them = constant - state.calc.base.us;
    state.calc.buffer = String(state.calc.base[state.calc.activeTeam]);

    if (mode === 'hokum' && !state.calc.trump) {
      state.calc.trump = 'h';
    }
    // Clean up projects not allowed in new mode for both teams
    ['us', 'them'].forEach(function (team) {
      if (state.calc.projects[team]) {
        if (mode === 'sun') {
          state.calc.projects[team].baloot = 0;
        } else {
          state.calc.projects[team].fourHundred = 0;
        }
      }
    });
    HakimUI.buzz(6);
    buildKeypad();
    renderCalculator();
  }

  function setCalcTrump(suit) {
    state.calc.trump = suit;
    HakimUI.buzz(6);
    renderCalculator();
  }

  function openKeypad(team) {
    state.calc.activeTeam = team || 'us';
    var constant = getRoundBaseConstant(state.calc.mode);

    state.calc.base.us = Math.round(constant / 2);
    state.calc.base.them = constant - state.calc.base.us;

    state.calc.cards = [];
    state.calc.source = 'manual';

    if (el['calc-input-us']) el['calc-input-us'].value = '';
    if (el['calc-input-them']) el['calc-input-them'].value = '';

    var panel = $('sheet-keypad').querySelector('.sheet__panel');
    if (panel) {
      panel.style.setProperty('--team-color', state.calc.activeTeam === 'us' ? 'var(--us)' : 'var(--them)');
    }

    buildKeypad();
    renderCalculator();
    HakimUI.open('sheet-keypad', el['btn-calc-commit']);
    render();
  }

  function commitKeypad(event) {
    if (event && event.preventDefault) event.preventDefault();
    if (document.activeElement && document.activeElement.blur) {
      document.activeElement.blur();
    }

    var usVal = el['calc-input-us'] ? parseInt(el['calc-input-us'].value, 10) : NaN;
    var themVal = el['calc-input-them'] ? parseInt(el['calc-input-them'].value, 10) : NaN;

    var usProj = getTeamProjectsSum('us');
    var themProj = getTeamProjectsSum('them');

    var usTotal = !isNaN(usVal) ? usVal : (state.calc.base.us + usProj);
    var themTotal = !isNaN(themVal) ? themVal : (state.calc.base.them + themProj);

    if (!usTotal && !themTotal) {
      HakimUI.toast(t('toast_empty_round'));
      HakimUI.buzz([10, 40, 10]);
      return;
    }

    var round = HakimMatch.addRound({
      us: usTotal,
      them: themTotal,
      source: state.calc.source,
      meta: {
        mode: state.calc.mode,
        trump: state.calc.mode === 'hokum' ? state.calc.trump : null,
        base: {
          us: Math.max(0, usTotal - usProj),
          them: Math.max(0, themTotal - themProj)
        },
        projects: {
          us: usProj,
          them: themProj
        }
      }
    });

    clearPending();
    HakimUI.close('sheet-keypad');
    HakimUI.buzz([12, 30, 18]);
    if (round) HakimUI.toast(t('toast_round_added'));
    render();
  }

  function startCalcCamera() {
    HakimUI.close('sheet-keypad');
    HakimScanUI.open({
      mode: state.calc.mode,
      trump: state.calc.trump,
      team: state.calc.activeTeam,
      onResult: handleScanResult
    });
  }

  function handleScanResult(res) {
    if (!res) return;
    state.calc.cards = res.cards || [];
    state.calc.mode = res.mode || state.calc.mode;
    state.calc.trump = res.trump || state.calc.trump;
    if (res.team) state.calc.activeTeam = res.team;
    state.calc.source = 'scan';
    state.calc.buffer = '';

    var activeTeam = state.calc.activeTeam;
    if (res.result && res.result.projects) {
      clearCalcProjects();
      res.result.projects.forEach(function (p) {
        if (state.calc.projects[activeTeam] && state.calc.projects[activeTeam][p.id] !== undefined) {
          state.calc.projects[activeTeam][p.id] = (state.calc.projects[activeTeam][p.id] || 0) + 1;
        }
      });
    }

    HakimUI.open('sheet-keypad', el['btn-calc-commit']);
    buildKeypad();
    renderCalculator();
    render();
  }

  function renderCalcCards() {
    var container = el['calc-cards-container'];
    var strip = el['calc-cards-strip'];
    if (!container || !strip) return;

    if (!state.calc.cards.length) {
      container.classList.add('hidden');
      strip.innerHTML = '';
      return;
    }

    container.classList.remove('hidden');
    strip.innerHTML = '';

    state.calc.cards.forEach(function (card, index) {
      var slot = document.createElement('button');
      slot.type = 'button';
      slot.className = 'card-slot';
      var pts = HakimScoring.cardPoints(card, state.calc.mode, state.calc.trump);
      slot.innerHTML = HakimScanUI.cardFaceMarkup(card, index + 1) +
        '<span class="card-slot__points ltr">+' + pts + '</span>';
      slot.addEventListener('click', function () {
        editCalcCard(index);
      });
      strip.appendChild(slot);
    });

    var addSlot = document.createElement('button');
    addSlot.type = 'button';
    addSlot.className = 'card-slot';
    addSlot.innerHTML = '<span class="add-face">' + HakimIcons.svg('plus', { size: 18 }) + '</span>' +
      '<span class="card-slot__points">' + t('calc_add_card') + '</span>';
    addSlot.addEventListener('click', addCalcCard);
    strip.appendChild(addSlot);
  }

  function editCalcCard(index) {
    var current = state.calc.cards[index];
    var taken = {};
    state.calc.cards.forEach(function (c, i) {
      if (i !== index) taken[c] = true;
    });

    HakimScanUI.openPickerExternal({
      currentCard: current,
      takenCards: taken,
      allowRemove: true,
      onChoose: function (newCard) {
        state.calc.cards[index] = newCard;
        renderCalculator();
      },
      onRemove: function () {
        state.calc.cards.splice(index, 1);
        renderCalculator();
      }
    });
  }

  function addCalcCard() {
    var taken = {};
    state.calc.cards.forEach(function (c) { taken[c] = true; });

    HakimScanUI.openPickerExternal({
      currentCard: null,
      takenCards: taken,
      allowRemove: false,
      onChoose: function (newCard) {
        state.calc.cards.push(newCard);
        renderCalculator();
      }
    });
  }

  function renderCalculator() {
    var scoreData = computeDualCalcScore();
    var activeTeam = state.calc.activeTeam;

    // Update Team cards displays and inputs
    if (el['calc-name-us']) el['calc-name-us'].textContent = teamName('us');
    if (el['calc-name-them']) el['calc-name-them'].textContent = teamName('them');

    if (el['calc-input-us'] && document.activeElement !== el['calc-input-us']) {
      el['calc-input-us'].value = scoreData.total.us;
    }
    if (el['calc-input-them'] && document.activeElement !== el['calc-input-them']) {
      el['calc-input-them'].value = scoreData.total.them;
    }

    if (el['calc-base-us']) el['calc-base-us'].textContent = scoreData.base.us;
    if (el['calc-base-them']) el['calc-base-them'].textContent = scoreData.base.them;
    if (el['calc-proj-us']) el['calc-proj-us'].textContent = '+' + scoreData.projects.us;
    if (el['calc-proj-them']) el['calc-proj-them'].textContent = '+' + scoreData.projects.them;

    if (el['card-calc-us']) {
      el['card-calc-us'].setAttribute('data-active', String(activeTeam === 'us'));
      el['card-calc-us'].setAttribute('aria-pressed', String(activeTeam === 'us'));
    }
    if (el['card-calc-them']) {
      el['card-calc-them'].setAttribute('data-active', String(activeTeam === 'them'));
      el['card-calc-them'].setAttribute('aria-pressed', String(activeTeam === 'them'));
    }

    if (el['calc-active-team-label']) {
      el['calc-active-team-label'].textContent = teamName(activeTeam);
      el['calc-active-team-label'].style.color = activeTeam === 'us' ? 'var(--us)' : 'var(--them)';
    }

    var panel = $('sheet-keypad').querySelector('.sheet__panel');
    if (panel) {
      panel.style.setProperty('--team-color', activeTeam === 'us' ? 'var(--us)' : 'var(--them)');
    }

    // Mode selector
    var isSun = state.calc.mode === 'sun';
    if (el['calc-mode-sun']) el['calc-mode-sun'].setAttribute('aria-pressed', String(isSun));
    if (el['calc-mode-hokum']) el['calc-mode-hokum'].setAttribute('aria-pressed', String(!isSun));

    // Trump selector bar
    if (el['calc-trump-bar']) el['calc-trump-bar'].classList.toggle('hidden', isSun);
    if (!isSun && el['calc-trump-buttons']) {
      el['calc-trump-buttons'].querySelectorAll('.suit-btn').forEach(function (btn) {
        btn.setAttribute('aria-pressed', String(btn.dataset.suit === state.calc.trump));
      });
    }

    // Update project buttons labels/values according to mode
    if (el['val-proj-sra']) el['val-proj-sra'].textContent = '+' + getProjectValue('sra', state.calc.mode);
    if (el['val-proj-fifty']) el['val-proj-fifty'].textContent = '+' + getProjectValue('fifty', state.calc.mode);
    if (el['val-proj-hundred']) el['val-proj-hundred'].textContent = '+' + getProjectValue('hundred', state.calc.mode);
    if (el['val-proj-fourHundred']) el['val-proj-fourHundred'].textContent = '+' + getProjectValue('fourHundred', state.calc.mode);
    if (el['val-proj-baloot']) el['val-proj-baloot'].textContent = '+' + getProjectValue('baloot', state.calc.mode);

    // Projects buttons for the active team
    var hasActiveProjects = false;
    var activeProjects = state.calc.projects[activeTeam] || {};
    Object.keys(PROJECT_CONFIG).forEach(function (projId) {
      var cfg = PROJECT_CONFIG[projId];
      var btn = el['proj-' + projId];
      if (!btn) return;
      var allowed = cfg.modes.indexOf(state.calc.mode) !== -1;
      btn.disabled = !allowed;
      var count = allowed ? (activeProjects[projId] || 0) : 0;
      btn.setAttribute('data-active', String(count > 0));
      if (count > 0) hasActiveProjects = true;

      // Add or remove count badge
      var existingBadge = btn.querySelector('.calc-proj-badge');
      if (count > 1) {
        if (!existingBadge) {
          var badge = document.createElement('span');
          badge.className = 'calc-proj-badge num ltr';
          badge.textContent = '×' + count;
          btn.appendChild(badge);
        } else {
          existingBadge.textContent = '×' + count;
        }
      } else if (existingBadge) {
        existingBadge.remove();
      }
    });

    if (el['btn-calc-clear-projects']) {
      el['btn-calc-clear-projects'].classList.toggle('hidden', !hasActiveProjects);
    }

    // Cards strip
    renderCalcCards();
  }

  // ---- Recording a round ------------------------------------------------------

  function hasPending() {
    return state.pending.us !== null || state.pending.them !== null;
  }

  function clearPending() {
    state.pending = { us: null, them: null };
  }

  /** The احسب button: commit whatever is pending as one round. */
  function calculate() {
    if (!hasPending()) {
      HakimUI.toast(t('toast_empty_round'));
      HakimUI.buzz([10, 40, 10]);
      return;
    }

    var button = el['btn-calculate'];
    button.classList.add('is-firing');
    setTimeout(function () { button.classList.remove('is-firing'); }, 440);
    HakimUI.buzz([12, 30, 18]);

    var round = HakimMatch.addRound({
      us: state.pending.us || 0,
      them: state.pending.them || 0,
      source: 'manual'
    });

    clearPending();
    if (round) HakimUI.toast(t('toast_round_added'));
    render();
  }

  function undoRound() {
    // A pending entry is the cheaper thing to undo, so it goes first.
    if (hasPending()) {
      clearPending();
      HakimUI.buzz(8);
      render();
      return;
    }
    var removed = HakimMatch.undo();
    HakimUI.buzz(removed ? 8 : [10, 40, 10]);
    HakimUI.toast(t(removed ? 'toast_undone' : 'toast_nothing_undo'));
    render();
  }

  function startNewRound() {
    clearPending();
    setView('game');
    openKeypad('us');
  }

  /** A hand the scanner already scored, arriving as a finished round. */
  function recordScannedHand(payload) {
    if (!payload) return;
    var mode = (payload.meta && payload.meta.mode) || 'sun';
    var constant = getRoundBaseConstant(mode);
    var targetTeam = payload.team || 'us';
    var otherTeam = targetTeam === 'us' ? 'them' : 'us';

    var rawPts = payload.points || 0;
    var activeBase = mode === 'sun'
      ? Math.min(26, Math.round(rawPts / 5))
      : Math.min(16, Math.round(rawPts / 10));

    var otherBase = Math.max(0, constant - activeBase);

    var roundPoints = { us: 0, them: 0 };
    roundPoints[targetTeam] = activeBase;
    roundPoints[otherTeam] = otherBase;

    var projSum = 0;
    if (payload.meta && payload.meta.projects) {
      if (Array.isArray(payload.meta.projects)) {
        payload.meta.projects.forEach(function (p) {
          projSum += getProjectValue(p.id, mode);
        });
      }
    }
    roundPoints[targetTeam] += projSum;

    HakimMatch.addRound({
      us: roundPoints.us,
      them: roundPoints.them,
      source: 'scan',
      meta: payload.meta
    });

    HakimUI.toast(t('toast_round_added'));
    setView('game');
  }

  // ---- Rendering: the table ------------------------------------------------------

  function renderTable() {
    var totals = HakimMatch.totals();
    var leader = HakimMatch.leader();
    var champion = HakimMatch.winner();

    HakimMatch.TEAMS.forEach(function (team) {
      renderScoreCard(team, totals[team], leader === team, champion === team);
    });

    el['btn-undo'].disabled = !hasPending() && !HakimMatch.getState().rounds.length;

    var target = HakimMatch.getState().target;
    el['target-text'].textContent = t('target_label', { target: target });
    el['home-target'].textContent = t('target_label', { target: target });
  }

  function renderScoreCard(team, total, isLeading, hasWon) {
    var card = $('card-' + team);
    var value = $('score-' + team);
    var pendingNode = $('pending-' + team);

    $('name-' + team).textContent = teamName(team);
    card.setAttribute('aria-label', t('aria_points_for', { team: teamName(team) }));
    card.setAttribute('data-leading', String(isLeading));
    card.setAttribute('data-active', String(state.keypad.team === team));

    if (state.painted[team] !== total) {
      HakimUI.animateNumber(value, state.painted[team], total);
      if (total > state.painted[team]) HakimUI.bump(value);
      state.painted[team] = total;
    } else {
      value.textContent = total;
    }

    var pending = state.pending[team];
    pendingNode.textContent = pending === null ? t('pending_zero') : '+' + pending;
    pendingNode.setAttribute('data-filled', String(pending !== null));

    $('trophy-' + team).classList.toggle('hidden', !hasWon);
  }

  // ---- Rendering: rounds ------------------------------------------------------------

  /**
   * One round as a timeline row: the two deltas facing each other across a
   * node, with the running total underneath in a quieter weight. A round
   * calculated from a photo carries a small sparkle, so the source stays
   * visible long after the photo is gone.
   */
  function roundRow(round) {
    var row = document.createElement('div');
    row.className = 'round-row';

    function side(team) {
      var points = round[team];
      var total = team === 'us' ? round.usTotal : round.themTotal;
      return '<span class="round-side" data-team="' + team + '">' +
        '<span class="round-delta num ltr" data-zero="' + String(points === 0) + '">' +
        (points === 0 ? '—' : '+' + points) + '</span>' +
        '<span class="round-total num">' + total + '</span>' +
        '</span>';
    }

    var badge = round.source === 'scan'
      ? '<span class="round-ai" title="' + t('history_ai_badge') + '">' +
        HakimIcons.svg('sparkles', { size: 10 }) +
        '<span class="sr-only">' + t('history_ai_badge') + '</span></span>'
      : '<span class="round-dot" aria-hidden="true"></span>';

    row.innerHTML = side('us') +
      '<span class="round-node">' + badge +
      '<span class="round-index num">' + round.index + '</span></span>' +
      side('them');
    return row;
  }

  function fillRounds(host, rounds, limit) {
    host.innerHTML = '';
    if (!rounds.length) {
      var empty = document.createElement('p');
      empty.className = 'mini-empty';
      empty.textContent = t('history_empty');
      host.appendChild(empty);
      return;
    }
    // Newest first; the table only needs the last few at a glance.
    var slice = limit ? rounds.slice(-limit) : rounds.slice();
    slice.reverse().forEach(function (round) { host.appendChild(roundRow(round)); });
  }

  function renderRounds() {
    var rounds = HakimMatch.timeline();
    fillRounds(el['mini-rounds'], rounds, 3);
    fillRounds(el['history-rounds'], rounds, 0);

    var totals = HakimMatch.totals();
    el['hist-name-us'].textContent = teamName('us');
    el['hist-name-them'].textContent = teamName('them');
    el['hist-score-us'].textContent = totals.us;
    el['hist-score-them'].textContent = totals.them;
    el['hist-count'].textContent = rounds.length;
  }

  // ---- Rendering: statistics ---------------------------------------------------------

  function statTile(label, usValue, themValue) {
    var tile = document.createElement('div');
    tile.className = 'stat';
    var body = themValue === undefined
      ? '<span class="stat__value"><span class="num">' + usValue + '</span></span>'
      : '<span class="stat__value">' +
        '<span class="stat__us num">' + usValue + '</span>' +
        '<span class="stat__sep" aria-hidden="true">/</span>' +
        '<span class="stat__them num">' + themValue + '</span></span>';
    tile.innerHTML = '<span class="stat__label">' + label + '</span>' + body;
    return tile;
  }

  function renderStats() {
    var stats = HakimMatch.stats();
    var tiles = [
      statTile(t('stat_rounds'), stats.rounds),
      statTile(t('stat_rounds_won'), stats.roundsWon.us, stats.roundsWon.them),
      statTile(t('stat_best'), stats.best.us, stats.best.them),
      statTile(t('stat_average'), stats.average.us, stats.average.them),
      statTile(t('stat_remaining'), stats.remaining.us, stats.remaining.them),
      statTile(t('stat_games'), stats.games.us, stats.games.them),
      statTile(t('stat_scanned'), stats.scanned)
    ];

    [el['history-stats'], el['stats-grid']].forEach(function (host) {
      host.innerHTML = '';
      tiles.forEach(function (tile) { host.appendChild(tile.cloneNode(true)); });
    });
  }

  // ---- Rendering: home -----------------------------------------------------------------

  function renderHome() {
    var totals = HakimMatch.totals();
    var stats = HakimMatch.stats();

    el['home-us'].textContent = totals.us;
    el['home-them'].textContent = totals.them;
    el['home-games-us'].textContent = stats.games.us;
    el['home-games-them'].textContent = stats.games.them;
    el['home-rounds'].textContent = stats.rounds;
    el['home-note'].textContent = homeNote(stats.rounds);
    el['home-play-label'].textContent = t(stats.rounds ? 'home_resume' : 'home_start');

    var idle = !stats.rounds || Boolean(HakimMatch.winner());
    el['status-dot'].setAttribute('data-idle', String(idle));
    el['status-text'].textContent = idle ? t('app_tagline') : t('game_live');
  }

  function homeNote(rounds) {
    var champion = HakimMatch.winner();
    if (champion) return t('win_title', { team: teamName(champion) });
    if (!rounds) return t('home_no_game');
    var leader = HakimMatch.leader();
    return leader ? teamName(leader) + ' — ' + t('leading') : t('tied');
  }

  // ---- Win state ----------------------------------------------------------------------------

  function winSignature() {
    var champion = HakimMatch.winner();
    if (!champion) return null;
    var totals = HakimMatch.totals();
    return champion + ':' + totals.us + ':' + totals.them;
  }

  function renderWin() {
    var signature = winSignature();
    if (!signature) {
      state.winDismissedAt = null;
      if (HakimUI.isOpen('win')) HakimUI.close('win');
      return;
    }

    // Dismissing must not make it reappear on the next render, but a later
    // round that changes the score is worth announcing again.
    if (state.winDismissedAt === signature || HakimUI.isOpen('win')) return;

    var totals = HakimMatch.totals();
    el['win-title'].textContent = t('win_title', { team: teamName(HakimMatch.winner()) });
    el['win-score'].textContent = t('win_score', { us: totals.us, them: totals.them });
    HakimUI.open('win', $('btn-win-new'));
    HakimUI.buzz([18, 60, 18, 60, 30]);
  }

  function dismissWin() {
    state.winDismissedAt = winSignature();
    HakimUI.close('win');
  }

  function startFreshGame() {
    HakimMatch.newGame();
    clearPending();
    state.painted = { us: 0, them: 0 };
    state.winDismissedAt = null;
    HakimUI.toast(t('toast_new_game'));
    setView('game');
  }

  // ---- Settings ------------------------------------------------------------------------------

  function buildTargetSegment() {
    var host = el['target-seg'];
    host.innerHTML = '';
    HakimMatch.TARGET_CHOICES.forEach(function (target) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'num';
      button.textContent = target;
      button.setAttribute('aria-pressed', String(HakimMatch.getState().target === target));
      button.addEventListener('click', function () {
        HakimMatch.setTarget(target);
        HakimUI.buzz(6);
      });
      host.appendChild(button);
    });
  }

  function renderSettings() {
    var stored = HakimMatch.getState();
    // Never overwrite the field the player is typing into.
    if (document.activeElement !== el['input-name-us']) {
      el['input-name-us'].value = stored.names.us || '';
    }
    if (document.activeElement !== el['input-name-them']) {
      el['input-name-them'].value = stored.names.them || '';
    }
    el['input-name-us'].placeholder = t('team_us');
    el['input-name-them'].placeholder = t('team_them');

    el['target-seg'].querySelectorAll('button').forEach(function (button) {
      button.setAttribute('aria-pressed', String(Number(button.textContent) === stored.target));
    });
  }

  // ---- Master render ----------------------------------------------------------------------------

  function render() {
    renderTable();
    renderRounds();
    renderStats();
    renderHome();
    renderSettings();
    renderWin();
  }

  // ---- Language ------------------------------------------------------------------------------------

  function setLanguage(lang) {
    state.lang = HakimI18N.setLanguage(lang);
    var dict = HakimI18N.STRINGS[state.lang];
    document.documentElement.lang = state.lang;
    document.documentElement.dir = dict.dir;
    document.title = t('page_title');
    HakimI18N.applyToDocument();

    buildKeypad();
    buildTargetSegment();
    HakimScanUI.relocalise();
    render();
  }

  // ---- First-run hint -------------------------------------------------------------------------------

  /**
   * The orb is the one control whose meaning cannot be inferred from its icon
   * alone, so it gets exactly one sentence, exactly once.
   */
  function maybeShowOrbHint() {
    if (HakimUI.readFlag(HINT_SEEN_KEY)) return;

    var hint = document.createElement('p');
    hint.className = 'orb-hint';
    hint.setAttribute('role', 'status');
    hint.textContent = t('home_ai_hint');
    document.body.appendChild(hint);

    var dismiss = function () {
      hint.remove();
      HakimUI.writeFlag(HINT_SEEN_KEY, '1');
      document.removeEventListener('pointerdown', dismiss);
    };
    setTimeout(function () { document.addEventListener('pointerdown', dismiss); }, 400);
    setTimeout(dismiss, 6000);
  }

  // ---- Wiring ---------------------------------------------------------------------------------------

  function bindNavigation() {
    VIEWS.forEach(function (view) {
      $('tab-' + view).addEventListener('click', function () {
        HakimUI.buzz(5);
        setView(view);
      });
    });

    el['btn-settings-top'].addEventListener('click', function () { setView('settings'); });
    el['btn-see-all'].addEventListener('click', function () { setView('history'); });
    el['btn-stats'].addEventListener('click', function () { HakimUI.open('sheet-stats'); });
    el['btn-lang'].addEventListener('click', function () {
      setLanguage(state.lang === 'ar' ? 'en' : 'ar');
    });
  }

  function bindCalculator() {
    if (el['btn-calc-camera']) {
      el['btn-calc-camera'].addEventListener('click', startCalcCamera);
    }
    if (el['card-calc-us']) {
      el['card-calc-us'].addEventListener('click', function () { selectActiveCalcTeam('us'); });
    }
    if (el['card-calc-them']) {
      el['card-calc-them'].addEventListener('click', function () { selectActiveCalcTeam('them'); });
    }

    function handleTeamInput(team, event) {
      state.calc.activeTeam = team;
      var other = team === 'us' ? 'them' : 'us';
      var constant = getRoundBaseConstant(state.calc.mode);
      var typedVal = parseInt(event.target.value, 10);
      if (isNaN(typedVal)) typedVal = 0;

      // Project total for this team
      var projSum = 0;
      Object.keys(PROJECT_CONFIG).forEach(function (projId) {
        var count = (state.calc.projects[team] && state.calc.projects[team][projId]) || 0;
        projSum += count * getProjectValue(projId, state.calc.mode);
      });

      var baseVal = Math.max(0, typedVal - projSum);
      var otherBase = Math.max(0, constant - baseVal);

      state.calc.base[team] = baseVal;
      state.calc.base[other] = otherBase;
      state.calc.buffer = String(baseVal);
      state.calc.cards = [];

      renderCalculator();
    }

    if (el['calc-input-us']) {
      el['calc-input-us'].addEventListener('input', function (e) { handleTeamInput('us', e); });
      el['calc-input-us'].addEventListener('focus', function () { selectActiveCalcTeam('us'); });
    }
    if (el['calc-input-them']) {
      el['calc-input-them'].addEventListener('input', function (e) { handleTeamInput('them', e); });
      el['calc-input-them'].addEventListener('focus', function () { selectActiveCalcTeam('them'); });
    }

    if (el['calc-mode-sun']) {
      el['calc-mode-sun'].addEventListener('click', function () { setCalcMode('sun'); });
    }
    if (el['calc-mode-hokum']) {
      el['calc-mode-hokum'].addEventListener('click', function () { setCalcMode('hokum'); });
    }
    if (el['btn-calc-clear-cards']) {
      el['btn-calc-clear-cards'].addEventListener('click', function () {
        state.calc.cards = [];
        renderCalculator();
      });
    }
    if (el['btn-calc-clear-projects']) {
      el['btn-calc-clear-projects'].addEventListener('click', clearCalcProjects);
    }
    if (el['btn-calc-commit']) {
      el['btn-calc-commit'].addEventListener('click', commitKeypad);
      el['btn-calc-commit'].addEventListener('touchend', function (e) {
        commitKeypad(e);
      });
    }

    ['sra', 'fifty', 'hundred', 'fourHundred', 'baloot'].forEach(function (projId) {
      var btn = el['proj-' + projId];
      if (btn) {
        btn.addEventListener('click', function () {
          toggleProject(projId);
        });
      }
    });
  }

  function bindTable() {
    HakimMatch.TEAMS.forEach(function (team) {
      $('card-' + team).addEventListener('click', function () { openKeypad(team); });
    });
    el['btn-calculate'].addEventListener('click', calculate);
    el['btn-undo'].addEventListener('click', undoRound);
    el['btn-new-round'].addEventListener('click', startNewRound);

    var openScanner = function () {
      HakimScanUI.open();
      syncStageVideo();
    };
    el['btn-scan'].addEventListener('click', openScanner);
    el['btn-home-scan'].addEventListener('click', openScanner);
    el['btn-home-play'].addEventListener('click', function () { setView('game'); });
  }

  function bindSettings() {
    [['input-name-us', 'us'], ['input-name-them', 'them']].forEach(function (pair) {
      el[pair[0]].addEventListener('input', function (event) {
        HakimMatch.setTeamName(pair[1], event.target.value);
      });
    });

    el['btn-new-game'].addEventListener('click', startFreshGame);

    el['btn-reset'].addEventListener('click', function () {
      // Destructive and not undoable, so it asks — the only confirm in the app.
      if (!window.confirm(t('set_reset') + ' — ' + t('set_reset_hint'))) return;
      HakimMatch.resetAll();
      clearPending();
      state.painted = { us: 0, them: 0 };
      state.winDismissedAt = null;
      HakimUI.toast(t('toast_reset'));
      setView('game');
    });

    el['btn-win-dismiss'].addEventListener('click', dismissWin);
    el['btn-win-new'].addEventListener('click', function () {
      dismissWin();
      startFreshGame();
    });
  }

  function bindOverlayDismissal() {
    document.querySelectorAll('[data-close]').forEach(function (node) {
      node.addEventListener('click', function () {
        HakimUI.close('sheet-' + node.getAttribute('data-close'));
      });
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Tab') {
        HakimUI.trapTab(event);
        return;
      }
      if (event.key !== 'Escape') return;

      var top = ['sheet-keypad', 'sheet-verify', 'sheet-stats', 'scanner']
        .filter(HakimUI.isOpen)[0];
      if (top) {
        event.preventDefault();
        // The scanner owns extra teardown (it restores the system chrome), so
        // it closes through its own module rather than the generic path.
        if (top === 'scanner') HakimScanUI.close();
        else HakimUI.close(top);
      } else if (HakimUI.isOpen('win')) {
        event.preventDefault();
        dismissWin();
      }
    });
  }

  function cacheDom() {
    [
      'status-dot', 'status-text', 'btn-lang', 'btn-settings-top',
      'home-target', 'home-us', 'home-them', 'home-note', 'home-play-label',
      'home-games-us', 'home-games-them', 'home-rounds', 'btn-home-scan', 'btn-home-play',
      'target-text', 'btn-calculate', 'btn-undo', 'btn-new-round', 'btn-scan',
      'mini-rounds', 'btn-see-all',
      'history-rounds', 'history-stats', 'hist-name-us', 'hist-name-them',
      'hist-score-us', 'hist-score-them', 'hist-count', 'btn-stats', 'stats-grid',
      'input-name-us', 'input-name-them', 'target-seg', 'btn-new-game', 'btn-reset',
      'keypad-title', 'keypad-chips',
      'btn-calc-camera', 'calc-mode-sun', 'calc-mode-hokum', 'calc-trump-bar', 'calc-trump-buttons',
      'calc-dual-cards', 'card-calc-us', 'card-calc-them',
      'calc-name-us', 'calc-name-them', 'calc-input-us', 'calc-input-them',
      'calc-base-us', 'calc-base-them', 'calc-proj-us', 'calc-proj-them',
      'calc-active-team-label',
      'calc-cards-container', 'calc-cards-strip', 'btn-calc-clear-cards',
      'calc-projects-grid', 'proj-sra', 'proj-fifty', 'proj-hundred', 'proj-fourHundred', 'proj-baloot',
      'val-proj-sra', 'val-proj-fifty', 'val-proj-hundred', 'val-proj-fourHundred', 'val-proj-baloot',
      'btn-calc-clear-projects', 'btn-calc-commit',
      'win-title', 'win-score', 'btn-win-new', 'btn-win-dismiss'
    ].forEach(function (id) { el[id] = $(id); });
  }

  function registerServiceWorker() {
    var allowed = window.location.protocol === 'https:' ||
      window.location.hostname === 'localhost' ||
      window.location.hostname === '127.0.0.1';
    if (!('serviceWorker' in navigator) || !allowed) return;
    navigator.serviceWorker.register('sw.js').catch(function (err) {
      console.warn('Hakim SW registration skipped/failed:', err);
    });
  }

  function init() {
    cacheDom();
    HakimIcons.paint();
    HakimUI.setFallbackFocus('btn-scan');
    initStageVideo();

    // Backgrounding the app should not leave a video decoding behind it.
    document.addEventListener('visibilitychange', syncStageVideo);

    HakimMatch.load();
    var totals = HakimMatch.totals();
    state.painted = { us: totals.us, them: totals.them };
    // A game already decided when the app opens was decided in a past session:
    // announce it in the score, not with a modal over the player's first tap.
    state.winDismissedAt = winSignature();

    HakimScanUI.init({
      teams: HakimMatch.TEAMS,
      teamName: teamName,
      onConfirm: recordScannedHand,
      onClose: syncStageVideo
    });

    bindNavigation();
    bindTable();
    bindCalculator();
    bindSettings();
    bindOverlayDismissal();

    HakimMatch.subscribe(render);
    setLanguage('ar');
    setView(HakimMatch.getState().rounds.length ? 'game' : 'home');
    maybeShowOrbHint();
    registerServiceWorker();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
