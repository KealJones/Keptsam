import test from 'node:test';
import assert from 'node:assert/strict';

// A fake Cache API that, like the real one, ignores #fragments when matching requests.
const store = new Map();
const keyOf = (r) => r.url.split('#')[0];
globalThis.caches = {
  open: async () => ({
    match: async (r) => (store.has(keyOf(r)) ? new Response(store.get(keyOf(r))) : undefined),
    put: async (r, res) => { store.set(keyOf(r), new Uint8Array(await res.arrayBuffer())); },
  }),
};
const { cachedSource } = await import('../src/cache.js');

test('cached reads return the right bytes across 16 MB pieces, from network and from cache', async () => {
  const file = new Uint8Array(40 << 20);
  for (let i = 0; i < file.length; i += 4096) file[i] = (i >> 12) & 255;
  file.forEach((_, i) => { if (i % 4096) file[i] = i & 255; });
  let fetched = 0;
  const src = { name: 'f', read: async (o, n) => { const b = file.slice(o, Math.min(o + n, file.length)); fetched += b.length; return b; } };
  const ranges = [[0, 100], [16777200, 100], [5, 16 << 20], [30e6, 10e6], [33554431, 2], [(40 << 20) - 10, 10]];
  for (const pass of ['network', 'cache']) {
    const c = await cachedSource(src, { key: 'https://example.com/model.gguf' });
    for (const [o, n] of ranges) assert.deepEqual(await c.read(o, n), file.subarray(o, o + n), `${pass} ${o}+${n}`);
  }
  assert.ok(fetched <= 40 << 20, `downloaded ${fetched} bytes for a 40 MB file`);
});
