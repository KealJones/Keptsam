// GGUF reader that never holds the whole file: the header is read in growing chunks,
// tensor data is read later, one tensor (or part of one) at a time.

/** ggml tensor types we know the size of: [values per block, bytes per block]. */
export const GGML_TYPES = {
  0: { name: 'F32', block: 1, bytes: 4 },
  1: { name: 'F16', block: 1, bytes: 2 },
  2: { name: 'Q4_0', block: 32, bytes: 18 },
  3: { name: 'Q4_1', block: 32, bytes: 20 },
  8: { name: 'Q8_0', block: 32, bytes: 34 },
  12: { name: 'Q4_K', block: 256, bytes: 144 },
  13: { name: 'Q5_K', block: 256, bytes: 176 },
  14: { name: 'Q6_K', block: 256, bytes: 210 },
  30: { name: 'BF16', block: 1, bytes: 2 },
};

/**
 * A byte source: `read(offset, length)` resolves to exactly those bytes.
 * @typedef {{ read(offset: number, length: number): Promise<Uint8Array>, name?: string }} ByteSource
 */

/** Byte source over HTTP range requests. */
export function httpSource(url, { fetchImpl = fetch } = {}) {
  return {
    name: url,
    async read(offset, length) {
      const res = await fetchImpl(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
      if (res.status !== 206 && res.status !== 200) throw new Error(`GGUF read ${res.status} for ${url}`);
      const buf = new Uint8Array(await res.arrayBuffer());
      // A server that ignores Range sends the whole file; slice it rather than fail.
      if (res.status === 200 && buf.length > length) return buf.subarray(offset, offset + length);
      if (buf.length !== length) throw new Error(`GGUF short read: wanted ${length} bytes at ${offset}, got ${buf.length}`);
      return buf;
    },
  };
}

/** Byte source over an in-memory buffer (tests). */
export function bufferSource(bytes) {
  return { name: 'buffer', read: async (o, n) => bytes.subarray(o, o + n) };
}

const DEC = new TextDecoder();

/**
 * Parses a GGUF header.
 * @param {ByteSource} src
 * @returns {Promise<{ version: number, kv: Record<string, any>, tensors: Tensor[], dataOffset: number, alignment: number }>}
 * @typedef {{ name: string, dims: number[], type: number, typeName: string, count: number, offset: number, nbytes: number }} Tensor
 */
export async function readGGUF(src, { chunk = 4 << 20 } = {}) {
  let buf = await src.read(0, chunk);
  let end = buf.length;
  let p = 0;

  // Reads past the buffer fetch the next chunk and retry.
  const need = async (n) => {
    while (p + n > end) {
      const more = await src.read(end, Math.max(chunk, n));
      const grown = new Uint8Array(end + more.length);
      grown.set(buf.subarray(0, end));
      grown.set(more, end);
      buf = grown;
      end = grown.length;
    }
  };
  const view = () => new DataView(buf.buffer, buf.byteOffset, end);
  const u32 = async () => { await need(4); const v = view().getUint32(p, true); p += 4; return v; };
  const u64 = async () => {
    await need(8);
    const v = view().getBigUint64(p, true); p += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('GGUF value too large');
    return Number(v);
  };
  const str = async () => { const n = await u64(); await need(n); const s = DEC.decode(buf.subarray(p, p + n)); p += n; return s; };

  const scalar = {
    0: [1, (d, o) => d.getUint8(o)], 1: [1, (d, o) => d.getInt8(o)],
    2: [2, (d, o) => d.getUint16(o, true)], 3: [2, (d, o) => d.getInt16(o, true)],
    4: [4, (d, o) => d.getUint32(o, true)], 5: [4, (d, o) => d.getInt32(o, true)],
    6: [4, (d, o) => d.getFloat32(o, true)], 7: [1, (d, o) => d.getUint8(o) !== 0],
    10: [8, (d, o) => Number(d.getBigUint64(o, true))], 11: [8, (d, o) => Number(d.getBigInt64(o, true))],
    12: [8, (d, o) => d.getFloat64(o, true)],
  };
  const value = async (t) => {
    if (t === 8) return str();
    if (t === 9) {
      const et = await u32();
      const n = await u64();
      if (et === 8) { const out = new Array(n); for (let i = 0; i < n; i++) out[i] = await str(); return out; }
      const [size, get] = scalar[et] || [];
      if (!size) throw new Error(`GGUF array of unknown type ${et}`);
      await need(size * n);
      const d = view(); const out = new Array(n);
      for (let i = 0; i < n; i++) out[i] = get(d, p + i * size);
      p += size * n;
      return out;
    }
    const [size, get] = scalar[t] || [];
    if (!size) throw new Error(`GGUF value of unknown type ${t}`);
    await need(size); const v = get(view(), p); p += size; return v;
  };

  await need(4);
  if (DEC.decode(buf.subarray(0, 4)) !== 'GGUF') throw new Error(`${src.name || 'file'} is not a GGUF file`);
  p = 4;
  const version = await u32();
  if (version < 2) throw new Error(`GGUF version ${version} is not supported`);
  const nTensors = await u64();
  const nKV = await u64();

  const kv = {};
  for (let i = 0; i < nKV; i++) { const k = await str(); kv[k] = await value(await u32()); }

  const tensors = [];
  for (let i = 0; i < nTensors; i++) {
    const name = await str();
    const nd = await u32();
    const dims = [];
    for (let j = 0; j < nd; j++) dims.push(await u64());
    const type = await u32();
    const rel = await u64();
    const info = GGML_TYPES[type];
    const count = dims.reduce((a, b) => a * b, 1);
    tensors.push({
      name, dims, type, typeName: info ? info.name : `type${type}`, count, offset: rel,
      nbytes: info ? (count / info.block) * info.bytes : NaN,
    });
  }

  const alignment = kv['general.alignment'] || 32;
  const dataOffset = Math.ceil(p / alignment) * alignment;
  for (const t of tensors) t.offset += dataOffset;
  return { version, kv, tensors, dataOffset, alignment };
}

/** Architecture-prefixed key, e.g. arch('lfm3', 'block_count'). */
export const archKey = (gguf, key) => gguf.kv[`${gguf.kv['general.architecture']}.${key}`];
