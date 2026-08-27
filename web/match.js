/**
 * Hakim — match state (لنا / لهم).
 *
 * WHAT THIS IS NOT: a scoring engine. `scoring.js` owns every Baloot formula —
 * card values, projects, sun/hokum — and this file never duplicates or second-
 * guesses it. What lives here is the bookkeeping layer the table needs and the
 * hand engine has no opinion about: two teams, a list of rounds, running
 * totals, undo, and persistence.
 *
 * Round points arrive already decided from one of two places:
 *   - the scanner path, where `HakimScoring.scoreHand()` produced the value and
 *     the player confirmed it, or
 *   - the manual path, where the player typed the number.
 * Either way this module stores an integer. It computes no points of its own.
 *
 * Totals are derived from the round list on every read rather than accumulated,
 * so undo is a pop and can never drift out of sync with the history the player
 * is looking at.
 */

var HakimMatch = (function () {
  'use strict';

  var STORAGE_KEY = 'hakim.match.v1';

  /** Team ids, in RTL reading order: لنا sits first (right), لهم second. */
  var TEAMS = ['us', 'them'];

  /**
   * Where a game is played to. Houses differ — 152 is the common default, so it
   * is a setting rather than a constant baked into the logic.
   */
  var DEFAULT_TARGET = 152;

  var TARGET_CHOICES = [100, 121, 152, 200];

  /** Guards against a fat-fingered keypad entry becoming a permanent round. */
  var MAX_ROUND_POINTS = 9999;

  function emptyState() {
    return {
      names: { us: null, them: null },
      target: DEFAULT_TARGET,
      rounds: [],
      games: { us: 0, them: 0 },
      nextId: 1,
      startedAt: Date.now()
    };
  }

  var state = emptyState();
  var listeners = [];

  // ---- Pure helpers ---------------------------------------------------

  /** Clamp to a whole, non-negative, sane number of points. */
  function normalisePoints(value) {
    var number = Math.round(Number(value));
    if (!isFinite(number) || number < 0) return 0;
    return Math.min(number, MAX_ROUND_POINTS);
  }

  /**
   * Running totals after every round, plus the final pair.
   * @param {Array} rounds
   */
  function tally(rounds) {
    var us = 0;
    var them = 0;
    var running = (rounds || []).map(function (round) {
      us += round.us;
      them += round.them;
      return { id: round.id, us: us, them: them };
    });
    return { us: us, them: them, running: running };
  }

  function totals() {
    var result = tally(state.rounds);
    return { us: result.us, them: result.them };
  }

  /** Rounds decorated with the running total as it stood after each one. */
  function timeline() {
    var running = tally(state.rounds).running;
    return state.rounds.map(function (round, index) {
      return {
        id: round.id,
        index: index + 1,
        us: round.us,
        them: round.them,
        usTotal: running[index].us,
        themTotal: running[index].them,
        source: round.source,
        meta: round.meta,
        at: round.at
      };
    });
  }

  /** The team that has crossed the target, or null while the game is live. */
  function winner() {
    var score = totals();
    if (score.us < state.target && score.them < state.target) return null;
    if (score.us === score.them) return null;
    return score.us > score.them ? 'us' : 'them';
  }

  function isOver() {
    return winner() !== null;
  }

  /** Who is ahead right now — drives which score card reads as leading. */
  function leader() {
    var score = totals();
    if (score.us === score.them) return null;
    return score.us > score.them ? 'us' : 'them';
  }

  /** Headline numbers for the statistics sheet. */
  function stats() {
    var score = totals();
    var rounds = state.rounds;
    var best = { us: 0, them: 0 };
    var won = { us: 0, them: 0 };
    var scanned = 0;

    rounds.forEach(function (round) {
      if (round.us > best.us) best.us = round.us;
      if (round.them > best.them) best.them = round.them;
      if (round.us > round.them) won.us += 1;
      else if (round.them > round.us) won.them += 1;
      if (round.source === 'scan') scanned += 1;
    });

    return {
      rounds: rounds.length,
      totals: score,
      best: best,
      roundsWon: won,
      scanned: scanned,
      average: {
        us: rounds.length ? Math.round(score.us / rounds.length) : 0,
        them: rounds.length ? Math.round(score.them / rounds.length) : 0
      },
      games: { us: state.games.us, them: state.games.them },
      target: state.target,
      remaining: {
        us: Math.max(0, state.target - score.us),
        them: Math.max(0, state.target - score.them)
      }
    };
  }

  // ---- Persistence ----------------------------------------------------

  /**
   * Storage is best-effort: Safari private mode throws on write, and a stored
   * blob from an older build may not parse. Neither is worth breaking a live
   * game over, so both failure paths fall through to whatever is in memory.
   */
  function save() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (err) {
      /* storage unavailable — the game continues in memory */
    }
  }

  function load() {
    var raw = null;
    try {
      raw = window.localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      return false;
    }
    if (!raw) return false;

    var parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      return false;
    }
    if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.rounds)) return false;

    var restored = emptyState();
    restored.names = {
      us: (parsed.names && typeof parsed.names.us === 'string') ? parsed.names.us : null,
      them: (parsed.names && typeof parsed.names.them === 'string') ? parsed.names.them : null
    };
    restored.target = TARGET_CHOICES.indexOf(parsed.target) === -1 ? DEFAULT_TARGET : parsed.target;
    restored.games = {
      us: normalisePoints(parsed.games && parsed.games.us),
      them: normalisePoints(parsed.games && parsed.games.them)
    };
    restored.startedAt = typeof parsed.startedAt === 'number' ? parsed.startedAt : Date.now();

    var maxId = 0;
    restored.rounds = parsed.rounds.filter(function (round) {
      return round && typeof round === 'object';
    }).map(function (round) {
      var id = typeof round.id === 'number' ? round.id : ++maxId;
      if (id > maxId) maxId = id;
      return {
        id: id,
        us: normalisePoints(round.us),
        them: normalisePoints(round.them),
        source: round.source === 'scan' ? 'scan' : 'manual',
        meta: round.meta && typeof round.meta === 'object' ? round.meta : null,
        at: typeof round.at === 'number' ? round.at : Date.now()
      };
    });
    restored.nextId = maxId + 1;

    state = restored;
    return true;
  }

  // ---- Mutations ------------------------------------------------------

  function notify() {
    save();
    listeners.forEach(function (listener) {
      try {
        listener(state);
      } catch (err) {
        if (typeof console !== 'undefined' && console.warn) {
          console.warn('Hakim match listener failed:', err);
        }
      }
    });
  }

  function subscribe(listener) {
    if (typeof listener !== 'function') return function () {};
    listeners.push(listener);
    return function () {
      listeners = listeners.filter(function (entry) { return entry !== listener; });
    };
  }

  /**
   * Record a round.
   *
   * @param {{us?: number, them?: number, source?: 'manual'|'scan', meta?: object}} input
   * @returns {object|null} the stored round, or null when both sides are zero
   */
  function addRound(input) {
    var payload = input || {};
    var round = {
      id: state.nextId,
      us: normalisePoints(payload.us),
      them: normalisePoints(payload.them),
      source: payload.source === 'scan' ? 'scan' : 'manual',
      meta: payload.meta || null,
      at: Date.now()
    };

    // A round where nobody scored is a mis-tap, not a record worth keeping.
    if (!round.us && !round.them) return null;

    var wasOver = isOver();
    state = Object.assign({}, state, {
      rounds: state.rounds.concat([round]),
      nextId: state.nextId + 1
    });

    // Credit the game the moment it is decided, and only once.
    var champion = winner();
    if (!wasOver && champion) {
      var games = Object.assign({}, state.games);
      games[champion] += 1;
      state = Object.assign({}, state, { games: games });
    }

    notify();
    return round;
  }

  /** Remove the most recent round. Returns it, or null when there is none. */
  function undo() {
    if (!state.rounds.length) return null;

    var wasOver = isOver();
    var removed = state.rounds[state.rounds.length - 1];
    state = Object.assign({}, state, {
      rounds: state.rounds.slice(0, -1)
    });

    // Undoing the round that ended the game takes the game credit back with it.
    if (wasOver && !isOver()) {
      var champion = null;
      var before = tally(state.rounds.concat([removed]));
      if (before.us >= state.target || before.them >= state.target) {
        champion = before.us > before.them ? 'us' : 'them';
      }
      if (champion && state.games[champion] > 0) {
        var games = Object.assign({}, state.games);
        games[champion] -= 1;
        state = Object.assign({}, state, { games: games });
      }
    }

    notify();
    return removed;
  }

  /** Clear the board for a new game, keeping names, target, and games won. */
  function newGame() {
    state = Object.assign({}, state, {
      rounds: [],
      nextId: 1,
      startedAt: Date.now()
    });
    notify();
  }

  /** Wipe everything, including names and the games-won tally. */
  function resetAll() {
    state = emptyState();
    notify();
  }

  function setTeamName(team, name) {
    if (TEAMS.indexOf(team) === -1) return;
    var trimmed = typeof name === 'string' ? name.trim() : '';
    var names = Object.assign({}, state.names);
    names[team] = trimmed ? trimmed.slice(0, 24) : null;
    state = Object.assign({}, state, { names: names });
    notify();
  }

  function setTarget(value) {
    var target = Number(value);
    if (TARGET_CHOICES.indexOf(target) === -1) return;
    state = Object.assign({}, state, { target: target });
    notify();
  }

  function getState() {
    return {
      names: Object.assign({}, state.names),
      target: state.target,
      rounds: state.rounds.slice(),
      games: Object.assign({}, state.games),
      startedAt: state.startedAt
    };
  }

  return {
    TEAMS: TEAMS,
    DEFAULT_TARGET: DEFAULT_TARGET,
    TARGET_CHOICES: TARGET_CHOICES,
    MAX_ROUND_POINTS: MAX_ROUND_POINTS,
    STORAGE_KEY: STORAGE_KEY,

    normalisePoints: normalisePoints,
    tally: tally,

    getState: getState,
    subscribe: subscribe,
    load: load,

    totals: totals,
    timeline: timeline,
    stats: stats,
    winner: winner,
    isOver: isOver,
    leader: leader,

    addRound: addRound,
    undo: undo,
    newGame: newGame,
    resetAll: resetAll,
    setTeamName: setTeamName,
    setTarget: setTarget
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HakimMatch;
}
