import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classify, nextLevels, backOff, BlackBox } from '../src/blackbox.js';

const mem = () => { const m = new Map(); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, String(v)) }; };
const want = { ctx: 4096, chunk: 64, maxChunk: 256, submitBudgetMs: 2000 };

test('classify: crash, clean exit, discard, GPU loss', () => {
  assert.equal(classify(undefined), 'none');
  assert.equal(classify({ stage: 'reading' }), 'killed');
  assert.equal(classify({ stage: 'reading' }, true), 'discarded');
  assert.equal(classify({ closed: true }), 'clean');
  assert.equal(classify({ lost: true, closed: true }), 'gpu-lost');
});

test('nextLevels: deaths raise a level, three good runs give one back', () => {
  assert.deepEqual(nextLevels({}, 'killed'), { mem: 1, gpu: 0, good: 0 });
  assert.deepEqual(nextLevels({ mem: 3 }, 'killed'), { mem: 3, gpu: 0, good: 0 });
  assert.deepEqual(nextLevels({ mem: 1 }, 'gpu-lost'), { mem: 1, gpu: 1, good: 0 });
  let l = { mem: 1, gpu: 1, good: 0 };
  for (let i = 0; i < 2; i++) l = nextLevels(l, 'clean', { ran: true });
  assert.deepEqual(l, { mem: 1, gpu: 1, good: 2 });
  assert.deepEqual(nextLevels(l, 'clean', { ran: true }), { mem: 1, gpu: 0, good: 0 }); // GPU level first
  assert.deepEqual(nextLevels({ mem: 1, good: 2 }, 'clean', { ran: false }), { mem: 1, gpu: 0, good: 2 }); // didn't run: no credit
});

test('backOff: chunk halves, then context; budget shrinks 30% per level, with floors', () => {
  assert.deepEqual(backOff(want, { mem: 0, gpu: 0 }), want);
  assert.deepEqual(backOff(want, { mem: 1, gpu: 0 }), { ctx: 4096, chunk: 64, maxChunk: 128, submitBudgetMs: 2000 });
  assert.deepEqual(backOff(want, { mem: 2, gpu: 0 }), { ctx: 2048, chunk: 64, maxChunk: 64, submitBudgetMs: 2000 });
  assert.deepEqual(backOff(want, { mem: 3, gpu: 0 }), { ctx: 1024, chunk: 64, maxChunk: 64, submitBudgetMs: 2000 });
  assert.equal(backOff(want, { mem: 0, gpu: 2 }).submitBudgetMs, 980);
  assert.equal(backOff({ ...want, submitBudgetMs: 400 }, { mem: 0, gpu: 3 }).submitBudgetMs, 300);
  assert.equal(backOff({ ...want, maxChunk: 32, chunk: 32 }, { mem: 2, gpu: 0 }).maxChunk, 16);
});

test('BlackBox: a killed session backs off the next one; a clean one does not', () => {
  const storage = mem();
  const a = new BlackBox('m', want, { storage, target: null });
  assert.equal(a.verdict, 'none');
  a.note({ stage: 'reading', mb: 1065, ran: true });
  // tab killed here: no close()
  const b = new BlackBox('m', want, { storage, target: null });
  assert.equal(b.verdict, 'killed');
  assert.equal(b.previous.stage, 'reading');
  assert.equal(b.settings.maxChunk, 128);
  b.note({ ran: true }, true); b.close();
  const c = new BlackBox('m', want, { storage, target: null });
  assert.equal(c.verdict, 'clean');
  assert.equal(c.settings.maxChunk, 128); // one good run isn't enough to relax
  const d = new BlackBox('m', want, { storage, target: null, apply: false, wasDiscarded: true });
  assert.deepEqual(d.settings, want); // apply: false keeps the requested settings
});
