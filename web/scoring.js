/**
 * Hakim — Baloot hand scoring engine.
 *
 * Pure functions only: no DOM and no globals beyond the exported namespace, so
 * the same file backs both the browser app and `node --test web/scoring.test.js`.
 *
 * A card is the string `<rank><suit>` used by the detector classes, e.g. "Ah",
 * "10s", "Jd" — ranks A K Q J 10 9 8 7, suits h d c s.
 */

var HakimScoring = (function () {
  'use strict';

  /** Ranks in Baloot trick order (strongest first in Sun). */
  var RANKS = ['A', 'K', 'Q', 'J', '10', '9', '8', '7'];
  var SUITS = ['h', 'd', 'c', 's'];

  /** Sequence order for projects — same as RANKS, A is high, no wrap-around. */
  var SEQUENCE = RANKS;

  /** Card point values when the hand is played as Sun (صن). */
  var SUN_POINTS = { A: 11, '10': 10, K: 4, Q: 3, J: 2, '9': 0, '8': 0, '7': 0 };

  /** Card point values for the trump suit in Hokum (حكم). */
  var TRUMP_POINTS = { J: 20, '9': 14, A: 11, '10': 10, K: 4, Q: 3, '8': 0, '7': 0 };

  /** Card points available in a full deck, per mode (before the last-trick 10). */
  var DECK_TOTAL = { sun: 120, hokum: 152 };

  /**
   * Project (مشروع) values. House rules vary, so this table is the single place
   * to adjust them.
   */
  var PROJECTS = {
    sra: { id: 'sra', value: 20, runLength: 3 },
    fifty: { id: 'fifty', value: 50, runLength: 4 },
    hundred: { id: 'hundred', value: 100, runLength: 5 },
    fourHundred: { id: 'fourHundred', value: 400 },
    baloot: { id: 'baloot', value: 20 }
  };

  var HAND_SIZE = 8;

  function parseCard(card) {
    if (typeof card !== 'string') return null;
    var suit = card.slice(-1);
    var rank = card.slice(0, -1);
    if (SUITS.indexOf(suit) === -1) return null;
    if (RANKS.indexOf(rank) === -1) return null;
    return { rank: rank, suit: suit, id: rank + suit };
  }

  /**
   * Point value of a single card.
   * @param {string} card e.g. "Ah"
   * @param {'sun'|'hokum'} mode
   * @param {string|null} trump suit letter, required when mode is "hokum"
   */
  function cardPoints(card, mode, trump) {
    var parsed = parseCard(card);
    if (!parsed) return 0;
    var isTrump = mode === 'hokum' && parsed.suit === trump;
    var table = isTrump ? TRUMP_POINTS : SUN_POINTS;
    return table[parsed.rank] || 0;
  }

  /** Split a hand into maximal same-suit runs of consecutive sequence ranks. */
  function findRuns(cards) {
    var bySuit = {};
    cards.forEach(function (card) {
      var parsed = parseCard(card);
      if (!parsed) return;
      if (!bySuit[parsed.suit]) bySuit[parsed.suit] = [];
      if (bySuit[parsed.suit].indexOf(parsed.rank) === -1) bySuit[parsed.suit].push(parsed.rank);
    });

    var runs = [];
    Object.keys(bySuit).forEach(function (suit) {
      var current = [];
      SEQUENCE.forEach(function (rank) {
        if (bySuit[suit].indexOf(rank) !== -1) {
          current.push(rank);
          return;
        }
        if (current.length) runs.push({ suit: suit, ranks: current });
        current = [];
      });
      if (current.length) runs.push({ suit: suit, ranks: current });
    });
    return runs;
  }

  /**
   * Projects held in a hand. Sequences of six or more still score 100, and a
   * card is never counted in two sequence projects.
   */
  function findProjects(cards, mode, trump) {
    var found = [];

    findRuns(cards).forEach(function (run) {
      var length = run.ranks.length;
      var spec = null;
      if (length >= PROJECTS.hundred.runLength) spec = PROJECTS.hundred;
      else if (length === PROJECTS.fifty.runLength) spec = PROJECTS.fifty;
      else if (length === PROJECTS.sra.runLength) spec = PROJECTS.sra;
      if (!spec) return;
      found.push({
        id: spec.id,
        value: spec.value,
        suit: run.suit,
        cards: run.ranks.map(function (rank) { return rank + run.suit; })
      });
    });

    var aces = cards.filter(function (card) {
      var parsed = parseCard(card);
      return parsed && parsed.rank === 'A';
    });
    if (aces.length === 4) {
      found.push({
        id: PROJECTS.fourHundred.id,
        value: PROJECTS.fourHundred.value,
        suit: null,
        cards: aces.slice()
      });
    }

    if (mode === 'hokum' && trump) {
      var hasKing = cards.indexOf('K' + trump) !== -1;
      var hasQueen = cards.indexOf('Q' + trump) !== -1;
      if (hasKing && hasQueen) {
        found.push({
          id: PROJECTS.baloot.id,
          value: PROJECTS.baloot.value,
          suit: trump,
          cards: ['K' + trump, 'Q' + trump]
        });
      }
    }

    return found.sort(function (a, b) { return b.value - a.value; });
  }

  /** Cards that appear more than once — impossible in a real hand. */
  function findDuplicates(cards) {
    var seen = {};
    var duplicates = [];
    cards.forEach(function (card) {
      seen[card] = (seen[card] || 0) + 1;
      if (seen[card] === 2) duplicates.push(card);
    });
    return duplicates;
  }

  /**
   * Score a hand.
   * @param {string[]} cards
   * @param {{mode?: 'sun'|'hokum', trump?: string|null}} [options]
   */
  function scoreHand(cards, options) {
    var opts = options || {};
    var mode = opts.mode === 'hokum' ? 'hokum' : 'sun';
    var trump = mode === 'hokum' ? (opts.trump || null) : null;
    var list = (cards || []).filter(function (card) { return parseCard(card) !== null; });

    var breakdown = list.map(function (card) {
      var parsed = parseCard(card);
      return {
        card: card,
        rank: parsed.rank,
        suit: parsed.suit,
        isTrump: mode === 'hokum' && parsed.suit === trump,
        points: cardPoints(card, mode, trump)
      };
    }).sort(function (a, b) { return b.points - a.points; });

    var cardTotal = breakdown.reduce(function (sum, entry) { return sum + entry.points; }, 0);
    var projects = findProjects(list, mode, trump);
    var projectTotal = projects.reduce(function (sum, project) { return sum + project.value; }, 0);

    var warnings = [];
    findDuplicates(list).forEach(function (card) {
      warnings.push({ code: 'duplicate', card: card });
    });
    if (list.length !== HAND_SIZE) {
      warnings.push({ code: 'hand_size', count: list.length, expected: HAND_SIZE });
    }
    if (mode === 'hokum' && !trump) {
      warnings.push({ code: 'no_trump' });
    }

    var deckTotal = DECK_TOTAL[mode];
    return {
      mode: mode,
      trump: trump,
      breakdown: breakdown,
      cardPoints: cardTotal,
      projectPoints: projectTotal,
      total: cardTotal + projectTotal,
      deckTotal: deckTotal,
      share: deckTotal ? cardTotal / deckTotal : 0,
      projects: projects,
      warnings: warnings
    };
  }

  /**
   * What the same hand is worth as Sun and with each suit as trump, best first.
   * Drives the "declare this" suggestion.
   */
  function compareModes(cards) {
    var options = [{ mode: 'sun', trump: null }];
    SUITS.forEach(function (suit) { options.push({ mode: 'hokum', trump: suit }); });
    return options.map(function (option) {
      var result = scoreHand(cards, option);
      return {
        mode: option.mode,
        trump: option.trump,
        cardPoints: result.cardPoints,
        projectPoints: result.projectPoints,
        total: result.total,
        share: result.share
      };
    }).sort(function (a, b) { return b.total - a.total; });
  }

  return {
    RANKS: RANKS,
    SUITS: SUITS,
    HAND_SIZE: HAND_SIZE,
    SUN_POINTS: SUN_POINTS,
    TRUMP_POINTS: TRUMP_POINTS,
    DECK_TOTAL: DECK_TOTAL,
    PROJECTS: PROJECTS,
    parseCard: parseCard,
    cardPoints: cardPoints,
    findRuns: findRuns,
    findProjects: findProjects,
    findDuplicates: findDuplicates,
    scoreHand: scoreHand,
    compareModes: compareModes
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HakimScoring;
}
