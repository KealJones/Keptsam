// Caches a byte source in fixed 16 MB pieces with the Cache API, so a model downloads once.
// Only one or two pieces are in memory at a time.

const PIECE = 16 << 20;

/**
 * @param {import('./gguf.js').ByteSource} src
 * @param {{ key: string, cacheName?: string, size?: number }} opts  `key` identifies the file (usually its URL).
 */
export async function cachedSource(src, { key, cacheName = 'keptsam-v1', size } = {}) {
  if (typeof caches === 'undefined') return src;
  let cache;
  try { cache = await caches.open(cacheName); } catch { return src; }
  // The Cache API ignores #fragments when matching, so each piece gets its own path.
  const reqFor = (i) => new Request(`https://keptsam.cache/${encodeURIComponent(key)}/${PIECE}/${i}`);
  const piece = async (i) => {
    const hit = await cache.match(reqFor(i));
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    const len = size ? Math.min(PIECE, size - i * PIECE) : PIECE;
    const bytes = await readUpTo(src, i * PIECE, len);
    try { await cache.put(reqFor(i), new Response(bytes)); } catch { /* storage full: keep going uncached */ }
    return bytes;
  };
  return {
    name: src.name,
    async read(offset, length) {
      const first = Math.floor(offset / PIECE), last = Math.floor((offset + length - 1) / PIECE);
      if (first === last) {
        const p = await piece(first);
        return p.subarray(offset - first * PIECE, offset - first * PIECE + length);
      }
      const out = new Uint8Array(length);
      for (let i = first; i <= last; i++) {
        const p = await piece(i);
        const from = Math.max(offset, i * PIECE), to = Math.min(offset + length, i * PIECE + p.length);
        out.set(p.subarray(from - i * PIECE, to - i * PIECE), from - offset);
      }
      return out;
    },
  };
}

// The last piece of a file is short; a source may refuse to read past the end, so ask for less on failure.
async function readUpTo(src, offset, length) {
  try { return await src.read(offset, length); } catch (e) {
    if (!src.size) throw e;
    return src.read(offset, Math.min(length, src.size - offset));
  }
}

/** Deletes every cached model piece. */
export async function clearCache(cacheName = 'keptsam-v1') {
  if (typeof caches !== 'undefined') await caches.delete(cacheName);
}
