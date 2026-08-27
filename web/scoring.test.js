/**
 * Scoring engine tests. Run with:  node --test web/scoring.test.js
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const scoring = require('./scoring.js');

test('sun card values follow the 120-point deck', () => {
  assert.equal(scoring.cardPoints('Ah', 'sun', null), 11);
  assert.equal(scoring.cardPoints('10h', 'sun', null), 10);
  assert.equal(scoring.cardPoints('Kh', 'sun', null), 4);
  assert.equal(scoring.cardPoints('Qh', 'sun', null), 3);
  assert.equal(scoring.cardPoints('Jh', 'sun', null), 2);
  assert.equal(scoring.cardPoints('9h', 'sun', null), 0);

  const deck = [];
  scoring.SUITS.forEach((suit) => scoring.RANKS.forEach((rank) => deck.push(rank + suit)));
  const total = deck.reduce((sum, card) => sum + scoring.cardPoints(card, 'sun', null), 0);
  assert.equal(total, scoring.DECK_TOTAL.sun);
});

test('hokum promotes the trump jack and nine only in the trump suit', () => {
  assert.equal(scoring.cardPoints('Jh', 'hokum', 'h'), 20);
  assert.equal(scoring.cardPoints('9h', 'hokum', 'h'), 14);
  assert.equal(scoring.cardPoints('Jd', 'hokum', 'h'), 2);
  assert.equal(scoring.cardPoints('9d', 'hokum', 'h'), 0);

  const deck = [];
  scoring.SUITS.forEach((suit) => scoring.RANKS.forEach((rank) => deck.push(rank + suit)));
  const total = deck.reduce((sum, card) => sum + scoring.cardPoints(card, 'hokum', 'h'), 0);
  assert.equal(total, scoring.DECK_TOTAL.hokum);
});

test('unknown card strings score zero and are dropped from the hand', () => {
  assert.equal(scoring.cardPoints('Zx', 'sun', null), 0);
  assert.equal(scoring.cardPoints('', 'sun', null), 0);
  const result = scoring.scoreHand(['Ah', 'nope', '6h'], { mode: 'sun' });
  assert.equal(result.breakdown.length, 1);
  assert.equal(result.cardPoints, 11);
});

test('runs are maximal, per suit, and never wrap around the ace', () => {
  const runs = scoring.findRuns(['Ah', 'Kh', 'Qh', '9h', '8h', 'Ad']);
  const hearts = runs.filter((run) => run.suit === 'h').map((run) => run.ranks.join(''));
  assert.deepEqual(hearts.sort(), ['98', 'AKQ'].sort());
});

test('sequence projects: 3 scores sra, 4 scores fifty, 5+ scores hundred', () => {
  const sra = scoring.findProjects(['Ah', 'Kh', 'Qh', '7s'], 'sun', null);
  assert.deepEqual(sra.map((p) => p.id), ['sra']);

  const fifty = scoring.findProjects(['Ah', 'Kh', 'Qh', 'Jh'], 'sun', null);
  assert.deepEqual(fifty.map((p) => p.id), ['fifty']);

  const hundred = scoring.findProjects(['Ah', 'Kh', 'Qh', 'Jh', '10h'], 'sun', null);
  assert.deepEqual(hundred.map((p) => p.id), ['hundred']);

  const sixLong = scoring.findProjects(['Ah', 'Kh', 'Qh', 'Jh', '10h', '9h'], 'sun', null);
  assert.deepEqual(sixLong.map((p) => p.id), ['hundred']);
});

test('a card is never counted in two sequence projects', () => {
  const projects = scoring.findProjects(['Ah', 'Kh', 'Qh', 'Jh', '10h'], 'sun', null);
  assert.equal(projects.length, 1);
  assert.equal(projects[0].cards.length, 5);
});

test('four aces score 400, three do not', () => {
  const four = scoring.findProjects(['Ah', 'Ad', 'Ac', 'As'], 'sun', null);
  assert.ok(four.some((p) => p.id === 'fourHundred' && p.value === 400));

  const three = scoring.findProjects(['Ah', 'Ad', 'Ac', '7s'], 'sun', null);
  assert.ok(!three.some((p) => p.id === 'fourHundred'));
});

test('baloot needs the trump king and queen and only exists in hokum', () => {
  const inHokum = scoring.findProjects(['Kh', 'Qh', '7s'], 'hokum', 'h');
  assert.ok(inHokum.some((p) => p.id === 'baloot' && p.value === 20));

  const offSuit = scoring.findProjects(['Kd', 'Qd', '7s'], 'hokum', 'h');
  assert.ok(!offSuit.some((p) => p.id === 'baloot'));

  const inSun = scoring.findProjects(['Kh', 'Qh', '7s'], 'sun', null);
  assert.ok(!inSun.some((p) => p.id === 'baloot'));
});

test('scoreHand adds card points and project points', () => {
  // A K Q J 10 of hearts as trump: 11 + 4 + 3 + 20 + 10 = 48, plus hundred (100)
  // and baloot (20). Remaining 7s + 8c contribute nothing.
  const result = scoring.scoreHand(['Ah', 'Kh', 'Qh', 'Jh', '10h', '7s', '8c', '9d'], {
    mode: 'hokum',
    trump: 'h'
  });
  assert.equal(result.cardPoints, 48);
  assert.equal(result.projectPoints, 120);
  assert.equal(result.total, 168);
  assert.equal(result.deckTotal, 152);
  assert.equal(result.warnings.length, 0);
});

test('scoreHand flags duplicates, wrong hand size, and missing trump', () => {
  const duplicated = scoring.scoreHand(['Ah', 'Ah'], { mode: 'sun' });
  assert.ok(duplicated.warnings.some((w) => w.code === 'duplicate' && w.card === 'Ah'));
  assert.ok(duplicated.warnings.some((w) => w.code === 'hand_size' && w.count === 2));

  const noTrump = scoring.scoreHand(['Ah'], { mode: 'hokum', trump: null });
  assert.ok(noTrump.warnings.some((w) => w.code === 'no_trump'));

  const clean = scoring.scoreHand(['Ah', 'Kh', 'Qd', 'Jc', '10s', '9h', '8d', '7c'], { mode: 'sun' });
  assert.equal(clean.warnings.length, 0);
});

test('scoreHand defaults to sun and ignores trump outside hokum', () => {
  const result = scoring.scoreHand(['Jh'], { trump: 'h' });
  assert.equal(result.mode, 'sun');
  assert.equal(result.trump, null);
  assert.equal(result.cardPoints, 2);
});

test('breakdown is sorted by points and marks trump cards', () => {
  const result = scoring.scoreHand(['7h', 'Jh', 'Ad'], { mode: 'hokum', trump: 'h' });
  assert.deepEqual(result.breakdown.map((entry) => entry.card), ['Jh', 'Ad', '7h']);
  assert.equal(result.breakdown[0].isTrump, true);
  assert.equal(result.breakdown[1].isTrump, false);
});

test('compareModes ranks declarations best first', () => {
  const options = scoring.compareModes(['Jh', '9h', 'Ah', '10h', '7s', '8c', '7d', '8d']);
  assert.equal(options.length, 5);
  assert.equal(options[0].mode, 'hokum');
  assert.equal(options[0].trump, 'h');
  for (let i = 1; i < options.length; i += 1) {
    assert.ok(options[i - 1].total >= options[i].total);
  }
});

test('an empty hand scores zero without throwing', () => {
  const result = scoring.scoreHand([], { mode: 'sun' });
  assert.equal(result.total, 0);
  assert.equal(result.share, 0);
  assert.deepEqual(result.projects, []);
});

test('project combinations correctly calculate for sun and hokum', () => {
  // Sra (20) in Sun
  const sraHand = scoring.scoreHand(['Ah', 'Kh', 'Qh', '8c', '7d'], { mode: 'sun' });
  assert.equal(sraHand.projectPoints, 20);

  // Fifty (50) in Hokum
  const fiftyHand = scoring.scoreHand(['Ah', 'Kh', 'Qh', 'Jh', '7d'], { mode: 'hokum', trump: 's' });
  assert.equal(fiftyHand.projectPoints, 50);

  // Hundred (100) in Sun
  const hundredHand = scoring.scoreHand(['Ah', 'Kh', 'Qh', 'Jh', '10h'], { mode: 'sun' });
  assert.equal(hundredHand.projectPoints, 100);

  // 400 (4 Aces) in Sun
  const fourHundredHand = scoring.scoreHand(['Ah', 'Ad', 'Ac', 'As'], { mode: 'sun' });
  assert.equal(fourHundredHand.projectPoints, 400);

  // Baloot (K + Q of trump) in Hokum
  const balootHand = scoring.scoreHand(['Kh', 'Qh', '8s'], { mode: 'hokum', trump: 'h' });
  assert.equal(balootHand.projectPoints, 20);
});

