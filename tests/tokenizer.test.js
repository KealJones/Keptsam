import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, openSync, readSync } from 'node:fs';
import { readGGUF } from '../src/gguf.js';
import { Tokenizer } from '../src/tokenizer.js';

const MODEL = process.env.TILDE_GGUF || '/home/claude/models/LFM3-1.7B-QAD-Q4_0.gguf';
const cases = JSON.parse(readFileSync(new URL('./fixtures/tokenize.json', import.meta.url)));

test('tokenizes like llama.cpp', { skip: !existsSync(MODEL) && 'model file not present' }, async () => {
  const fd = openSync(MODEL, 'r');
  const g = await readGGUF({ read: async (o, n) => { const b = Buffer.alloc(n); readSync(fd, b, 0, n, o); return new Uint8Array(b.buffer, b.byteOffset, n); } });
  const tok = Tokenizer.fromGGUF(g.kv);
  for (const { text, ids } of cases) {
    assert.deepEqual(tok.encode(text), ids, JSON.stringify(text));
    assert.equal(tok.decode(ids), text, 'decode ' + JSON.stringify(text));
  }
});
