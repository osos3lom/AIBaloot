/**
 * Match bookkeeping tests.
 *
 * The module is a singleton store, so every test starts from `resetAll()`.
 * `window` is absent under node, which exercises the storage-unavailable path
 * for free — the store must stay fully usable without it.
 */

const test = require('node:test');
const assert = require('node:assert');

const HakimMatch = require('./match.js');

function fresh() {
  HakimMatch.resetAll();
  return HakimMatch;
}

test('running totals are derived from the round list', () => {
  const match = fresh();
  match.addRound({ us: 50, them: 100 });
  match.addRound({ us: 100, them: 50 });

  assert.deepStrictEqual(match.totals(), { us: 150, them: 150 });

  const rows = match.timeline();
  assert.strictEqual(rows.length, 2);
  assert.deepStrictEqual(
    rows.map((row) => [row.us, row.them, row.usTotal, row.themTotal]),
    [[50, 100, 50, 100], [100, 50, 150, 150]]
  );
});

test('undo removes the last round and the totals follow', () => {
  const match = fresh();
  match.addRound({ us: 50, them: 0 });
  match.addRound({ us: 0, them: 90 });

  const removed = match.undo();
  assert.strictEqual(removed.them, 90);
  assert.deepStrictEqual(match.totals(), { us: 50, them: 0 });

  match.undo();
  assert.deepStrictEqual(match.totals(), { us: 0, them: 0 });
  assert.strictEqual(match.undo(), null, 'undo on an empty board is a no-op');
});

test('a round where nobody scored is rejected', () => {
  const match = fresh();
  assert.strictEqual(match.addRound({ us: 0, them: 0 }), null);
  assert.strictEqual(match.timeline().length, 0);
});

test('points are coerced to whole, non-negative, bounded numbers', () => {
  const match = fresh();
  match.addRound({ us: '75.6', them: -40 });
  const [round] = match.timeline();
  assert.strictEqual(round.us, 76);
  assert.strictEqual(round.them, 0);

  match.addRound({ us: 999999, them: 0 });
  assert.strictEqual(match.timeline()[1].us, HakimMatch.MAX_ROUND_POINTS);
});

test('crossing the target decides the game and credits it once', () => {
  const match = fresh();
  match.setTarget(152);
  assert.strictEqual(match.winner(), null);

  match.addRound({ us: 100, them: 20 });
  assert.strictEqual(match.isOver(), false);

  match.addRound({ us: 60, them: 10 });
  assert.strictEqual(match.winner(), 'us');
  assert.strictEqual(match.stats().games.us, 1);

  // Further rounds must not credit the same game a second time.
  match.addRound({ us: 10, them: 10 });
  assert.strictEqual(match.stats().games.us, 1);
});

test('a tie at or past the target is not a win', () => {
  const match = fresh();
  match.setTarget(152);
  match.addRound({ us: 160, them: 160 });
  assert.strictEqual(match.winner(), null);
  assert.strictEqual(match.leader(), null);
});

test('undoing the deciding round takes the game credit back', () => {
  const match = fresh();
  match.setTarget(152);
  match.addRound({ us: 100, them: 20 });
  match.addRound({ us: 60, them: 10 });
  assert.strictEqual(match.stats().games.us, 1);

  match.undo();
  assert.strictEqual(match.isOver(), false);
  assert.strictEqual(match.stats().games.us, 0);
});

test('newGame clears the board but keeps names, target, and games won', () => {
  const match = fresh();
  match.setTeamName('us', 'الديوانية');
  match.setTarget(121);
  match.addRound({ us: 130, them: 10 });
  assert.strictEqual(match.stats().games.us, 1);

  match.newGame();
  assert.deepStrictEqual(match.totals(), { us: 0, them: 0 });
  assert.strictEqual(match.getState().names.us, 'الديوانية');
  assert.strictEqual(match.getState().target, 121);
  assert.strictEqual(match.stats().games.us, 1);
});

test('an unknown target is ignored rather than stored', () => {
  const match = fresh();
  match.setTarget(999);
  assert.strictEqual(match.getState().target, HakimMatch.DEFAULT_TARGET);
});

test('scanned rounds are tallied separately for the AI indicator', () => {
  const match = fresh();
  match.addRound({ us: 50, them: 0, source: 'scan', meta: { mode: 'sun' } });
  match.addRound({ us: 0, them: 30 });

  const rows = match.timeline();
  assert.strictEqual(rows[0].source, 'scan');
  assert.deepStrictEqual(rows[0].meta, { mode: 'sun' });
  assert.strictEqual(rows[1].source, 'manual');
  assert.strictEqual(match.stats().scanned, 1);
});

test('subscribers are notified and a failing one cannot break the store', () => {
  const match = fresh();
  const seen = [];
  const stop = match.subscribe(() => { throw new Error('listener blew up'); });
  match.subscribe(() => seen.push(match.totals().us));

  match.addRound({ us: 20, them: 0 });
  assert.deepStrictEqual(seen, [20]);

  stop();
  match.addRound({ us: 5, them: 0 });
  assert.deepStrictEqual(seen, [20, 25]);
});

test('getState hands back copies, so callers cannot mutate the store', () => {
  const match = fresh();
  const state = match.getState();
  state.target = 999;
  assert.notStrictEqual(match.getState().target, 999);
});

test('dual team round recording preserves base constant and project additions', () => {
  const match = fresh();
  // Round 1: Sun mode, constant 130 (us base 80 + them base 50), us has 100 project
  const round = match.addRound({
    us: 180, // 80 base + 100 project
    them: 50, // 50 base + 0 project
    meta: {
      mode: 'sun',
      base: { us: 80, them: 50 },
      projects: { us: 100, them: 0 }
    }
  });

  assert.ok(round);
  assert.strictEqual(round.us, 180);
  assert.strictEqual(round.them, 50);
  assert.strictEqual(round.meta.base.us + round.meta.base.them, 130);
  assert.deepStrictEqual(match.totals(), { us: 180, them: 50 });
});

test('getState hands back copies, so callers cannot mutate the store', () => {
  const match = fresh();
  match.addRound({ us: 10, them: 0 });

  const snapshot = match.getState();
  snapshot.rounds.push({ us: 500, them: 500 });
  snapshot.names.us = 'tampered';

  assert.strictEqual(match.getState().rounds.length, 1);
  assert.strictEqual(match.getState().names.us, null);
});
