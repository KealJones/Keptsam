// Caches a byte source in fixed 16 MB pieces with the Cache API, so a model downloads once.
// Only one or two pieces are in memory at a time.
// The cache name keptsam-v1 and the keptsam.cache keys keep the old name on purpose: renaming them would orphan
// every model already downloaded, and Squiggle's service worker keeps keptsam-v1 and would delete a new name.

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

const sizeReq = (key) => new Request(`https://keptsam.cache/${encodeURIComponent(key)}/size`);
const pieceReq = (key, i) => new Request(`https://keptsam.cache/${encodeURIComponent(key)}/${PIECE}/${i}`);

/**
 * Downloads a whole file into the cache, piece by piece (resumable: pieces already there are skipped), so a later
 * load reads it from disk. `source` is a ByteSource for `url` (default: Range requests).
 * @param {string} url
 * `size`, when the caller knows it, saves asking the server (pieces a load already cached then need no network at all).
 * @param {{ key?: string, cacheName?: string, source?: import('./gguf.js').ByteSource, size?: number,
 *   onProgress?: (done: number, total: number) => void, signal?: AbortSignal }} [opts]
 */
export async function cacheFile(url, { key = url, cacheName = 'keptsam-v1', source, size: known = 0, onProgress, signal } = {}) {
  const cache = await caches.open(cacheName);
  const src = source || (await import('./gguf.js')).httpSource(url);
  let size = Number(await (await cache.match(sizeReq(key)))?.text()) || known || 0;
  if (!size) { await src.read(0, 1); size = src.size; }
  if (!size) throw new Error(`${url}: the server didn't say how big the file is`);
  const n = Math.ceil(size / PIECE);
  let done = 0;
  for (let i = 0; i < n; i++) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    const len = Math.min(PIECE, size - i * PIECE);
    if (!(await cache.match(pieceReq(key, i)))) await cache.put(pieceReq(key, i), new Response(await readUpTo(src, i * PIECE, len)));
    done += len;
    onProgress?.(done, size);
  }
  await cache.put(sizeReq(key), new Response(String(size)));
}

/**
 * How much of a file is cached: { size, stored, complete } (size 0 if it was never fully downloaded and no `size` is
 * given). Pieces cached by a load (cachedSource) carry no size record, so they count only when the caller says the size.
 */
export async function cachedFileInfo(key, cacheName = 'keptsam-v1', { size: known = 0 } = {}) {
  if (typeof caches === 'undefined') return { size: 0, stored: 0, complete: false };
  const cache = await caches.open(cacheName);
  const size = Number(await (await cache.match(sizeReq(key)))?.text()) || known || 0;
  if (!size) return { size: 0, stored: 0, complete: false };
  let stored = 0;
  for (let i = 0; i < Math.ceil(size / PIECE); i++) if (await cache.match(pieceReq(key, i))) stored += Math.min(PIECE, size - i * PIECE);
  return { size, stored, complete: stored === size };
}

/** Deletes one file's pieces from the cache. */
export async function removeCachedFile(key, cacheName = 'keptsam-v1') {
  if (typeof caches === 'undefined') return;
  const cache = await caches.open(cacheName);
  const prefix = `https://keptsam.cache/${encodeURIComponent(key)}/`;
  for (const req of await cache.keys()) if (req.url.startsWith(prefix)) await cache.delete(req);
}
