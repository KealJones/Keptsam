// LFM3 (Liquid AI): 24 layers of short conv / gated GQA attention (some sliding window), sandwich norms,
// SwiGLU FFN, tied Q6_K embedding/head. See docs/design.md for the math.

import { readGGUF, archKey } from '../gguf.js';
import { repackQ4_0, repackQ6_K } from '../quant.js';
import * as K from '../kernels.js';

const WINDOW_READ = 16 << 20; // bytes per HTTP/file read
const PIECE = 8 << 20; // max bytes repacked at once
const MM = {}; // matmul source by tokens-per-pass

/** Reads a source through one moving window so neighbouring small tensors share a request. */
function windowed(src, fileEnd) {
  let start = 0, buf = new Uint8Array(0);
  return async (offset, length) => {
    if (offset >= start && offset + length <= start + buf.length) return buf.subarray(offset - start, offset - start + length);
    buf = null; // drop the old window before reading the next one
    const n = Math.min(Math.max(length, WINDOW_READ), fileEnd - offset);
    buf = await src.read(offset, n);
    start = offset;
    return buf.subarray(0, length);
  };
}

export class LFM3 {
  /**
   * Streams an LFM3 GGUF into GPU memory.
   * @param {import('../gpu.js').Gpu} gpu
   * @param {import('../gguf.js').ByteSource} src
   * @param {{ ctx?: number, chunk?: number, onProgress?: (p: {loaded: number, total: number, tensor: string}) => void }} opts
   */
  static async load(gpu, src, { ctx = 4096, chunk = 64, onProgress } = {}) {
    const g = await readGGUF(src);
    if (g.kv['general.architecture'] !== 'lfm3') throw new Error(`not an LFM3 model: ${g.kv['general.architecture']}`);
    const m = new LFM3(gpu, g, { ctx, chunk });
    await m.#stream(src, onProgress);
    m.#allocState();
    return m;
  }

  constructor(gpu, g, { ctx, chunk }) {
    this.gpu = gpu;
    this.gguf = g;
    this.ctx = ctx;
    this.chunk = chunk;
    this.nLayer = archKey(g, 'block_count');
    this.nEmbd = archKey(g, 'embedding_length');
    this.nFF = archKey(g, 'feed_forward_length');
    this.eps = archKey(g, 'attention.layer_norm_rms_epsilon');
    this.ropeBase = archKey(g, 'rope.freq_base');
    this.ropeBaseSWA = archKey(g, 'rope.freq_base_swa');
    this.window = archKey(g, 'attention.sliding_window');
    const kvHeads = archKey(g, 'attention.head_count_kv');
    const swa = archKey(g, 'attention.sliding_window_pattern');
    this.layers = Array.from({ length: this.nLayer }, (_, i) => ({ attn: kvHeads[i] > 0, swa: !!swa[i] }));
    this.tokenizer = {
      tokens: g.kv['tokenizer.ggml.tokens'], merges: g.kv['tokenizer.ggml.merges'],
      bos: g.kv['tokenizer.ggml.bos_token_id'], eos: g.kv['tokenizer.ggml.eos_token_id'],
      template: g.kv['tokenizer.chat_template'],
    };
    this.vocab = this.tokenizer.tokens.length;
    this.w = {};
    this.pos = 0;
  }

  async #stream(src, onProgress) {
    const gpu = this.gpu, q = gpu.device.queue;
    const tensors = [...this.gguf.tensors].sort((a, b) => a.offset - b.offset);
    const end = tensors.at(-1).offset + tensors.at(-1).nbytes;
    const total = tensors.reduce((s, t) => s + t.nbytes, 0);
    const read = windowed(src, end);
    let loaded = 0;
    for (const t of tensors) {
      let buf, rec;
      if (t.typeName === 'Q4_0') {
        const blocks = t.count / 32;
        buf = gpu.buffer(blocks * 16 + Math.ceil(blocks / 2) * 4, 'weights');
        rec = { buf, type: 'Q4_0', dOff: blocks * 4, rows: t.dims[1], K: t.dims[0] };
        const step = Math.floor(PIECE / 36) * 2; // even block count per piece
        for (let b0 = 0; b0 < blocks; b0 += step) {
          const n = Math.min(step, blocks - b0);
          const { d, qs } = repackQ4_0(await read(t.offset + b0 * 18, n * 18));
          q.writeBuffer(buf, b0 * 16, qs);
          q.writeBuffer(buf, blocks * 16 + b0 * 2, d);
        }
      } else if (t.typeName === 'Q6_K') {
        const blocks = t.count / 256;
        buf = gpu.buffer(blocks * 212, 'weights');
        rec = { buf, type: 'Q6_K', rows: t.dims[1], K: t.dims[0] };
        const step = Math.floor(PIECE / 210);
        for (let b0 = 0; b0 < blocks; b0 += step) {
          const n = Math.min(step, blocks - b0);
          q.writeBuffer(buf, b0 * 212, repackQ6_K(await read(t.offset + b0 * 210, n * 210)).words);
        }
      } else if (t.typeName === 'F32') {
        buf = gpu.buffer(t.nbytes, 'weights');
        rec = { buf, type: 'F32' };
        const bytes = await read(t.offset, t.nbytes);
        q.writeBuffer(buf, 0, bytes.slice());
      } else {
        throw new Error(`${t.name}: ${t.typeName} weights are not supported yet`);
      }
      this.w[t.name] = rec;
      loaded += t.nbytes;
      // Let the GPU take the writes before reading more, so staging copies don't pile up.
      if (loaded % (64 << 20) < t.nbytes) await q.onSubmittedWorkDone();
      onProgress?.({ loaded, total, tensor: t.name });
    }
    await q.onSubmittedWorkDone();
  }

  #allocState() {
    const gpu = this.gpu, T = this.chunk, E = this.nEmbd, F = this.nFF;
    const f = (n, tag = 'activations') => gpu.buffer(n * 4, tag);
    this.swaSlots = this.window + T;
    this.a = {
      x: f(T * E), h: f(T * E), y: f(T * E), att: f(T * E), bcx: f(T * 3 * E),
      q: f(T * E), qn: f(T * E), k: f(T * 512), kn: f(T * 512), v: f(T * 512), gate: f(T * 16),
      g: f(T * F), u: f(T * F), S: f(T * 16 * this.ctx), headIn: f(E), logits: f(this.vocab),
      tok: gpu.buffer(T * 4, 'activations'), ones: f(64),
    };
    gpu.device.queue.writeBuffer(this.a.ones, 0, new Float32Array(64).fill(1));
    this.cache = this.layers.map((L) => {
      if (L.attn) {
        const slots = L.swa ? this.swaSlots : this.ctx;
        return { slots, k: f(slots * 512, 'kv'), v: f(slots * 512, 'kv') };
      }
      return { state: f(6 * E, 'conv-state') };
    });
  }

  /** Forgets the conversation (positions, KV, conv state). */
  reset() {
    this.pos = 0;
    const q = this.gpu.device.queue;
    for (const c of this.cache) if (c.state) q.writeBuffer(c.state, 0, new Float32Array(6 * this.nEmbd));
  }

  #u(...vals) {
    const a = new Uint32Array(Math.ceil(vals.length / 4) * 4);
    const fv = new Float32Array(a.buffer);
    vals.forEach((v, i) => { if (Number.isInteger(v)) a[i] = v; else fv[i] = v; });
    return this.#keep(this.gpu.uniform(a));
  }
  #keep(b) { (this._uniforms ||= []).push(b); return b; }
  #uf(ints, floats) { // explicit float slots
    const a = new Uint32Array(Math.ceil(ints.length / 4) * 4);
    const fv = new Float32Array(a.buffer);
    ints.forEach((v, i) => { if (floats.includes(i)) fv[i] = v; else a[i] = v; });
    return this.#keep(this.gpu.uniform(a));
  }

  #matmul(enc, name, X, Y, T, { yStride, yOff = 0, xStride, accumulate = 0 } = {}) {
    const w = this.w[name];
    const rows = w.rows;
    const mt = T === 1 ? 1 : K.MAXT;
    this.gpu.dispatch(enc, (MM[mt] ||= K.matmulQ4_0(mt)), [this.#u(rows, w.K, T, w.dOff, yStride ?? rows, yOff, xStride ?? w.K, accumulate), w.buf, X, Y],
      [Math.min(rows, 32768), Math.ceil(rows / 32768), Math.ceil(T / mt)]);
  }
  #norm(enc, X, w, Y, rows, N, { mode = 0, xOff = 0, yOff = 0 } = {}) {
    this.gpu.dispatch(enc, K.rmsnorm, [this.#uf([N, mode, this.eps, N, xOff, yOff, 0, 0], [2]), X, w, Y], rows);
  }

  /** Runs tokens (≤ chunk) through the model at the current position. */
  async forward(tokens) {
    const T = tokens.length, E = this.nEmbd, F = this.nFF, a = this.a, gpu = this.gpu, pos0 = this.pos;
    if (T > this.chunk) throw new Error(`forward takes at most ${this.chunk} tokens`);
    if (pos0 + T > this.ctx) throw new Error('context is full');
    gpu.device.queue.writeBuffer(a.tok, 0, new Uint32Array(tokens));
    const pre = (i) => `blk.${i}.`;
    const enc = gpu.device.createCommandEncoder();
    gpu.dispatch(enc, K.embedQ6_K, [this.#u(E, T, 0, 0), this.w['token_embd.weight'].buf, a.tok, a.x], T);
    for (let i = 0; i < this.nLayer; i++) {
      const L = this.layers[i], p = pre(i), c = this.cache[i];
      this.#norm(enc, a.x, this.w[p + 'attn_norm.weight'].buf, a.h, T, E);
      if (!L.attn) {
        this.#matmul(enc, p + 'shortconv.in_proj.weight', a.h, a.bcx, T);
        gpu.dispatch(enc, K.shortconv, [this.#u(T, E, 0, 0), a.bcx, this.w[p + 'shortconv.conv.weight'].buf, c.state, a.att], Math.ceil(E / 64));
        this.#matmul(enc, p + 'shortconv.out_proj.weight', a.att, a.y, T);
      } else {
        this.#matmul(enc, p + 'attn_q.weight', a.h, a.q, T);
        this.#matmul(enc, p + 'attn_k.weight', a.h, a.k, T);
        this.#matmul(enc, p + 'attn_v.weight', a.h, a.v, T);
        this.#matmul(enc, p + 'attn_gate.weight', a.h, a.gate, T);
        this.#norm(enc, a.q, this.w[p + 'attn_q_norm.weight'].buf, a.qn, T * 16, 128);
        this.#norm(enc, a.k, this.w[p + 'attn_k_norm.weight'].buf, a.kn, T * 4, 128);
        const base = L.swa ? this.ropeBaseSWA : this.ropeBase;
        const factors = L.swa ? a.ones : this.w['rope_freqs.weight'].buf;
        gpu.dispatch(enc, K.rope, [this.#uf([16, pos0, base, 0], [2]), a.qn, factors], T * 16);
        gpu.dispatch(enc, K.rope, [this.#uf([4, pos0, base, 0], [2]), a.kn, factors], T * 4);
        gpu.dispatch(enc, K.kvWrite, [this.#u(pos0, c.slots, 0, 0), a.kn, a.v, c.k, c.v], T);
        const Lk = Math.min(pos0 + T, c.slots);
        const win = L.swa ? this.window : 0xffffffff;
        gpu.dispatch(enc, K.attnScores, [this.#u(pos0, T, c.slots, Lk, win, this.ctx, 0, 0), a.qn, c.k, a.S], [Math.ceil(Lk / 64), T * 16]);
        gpu.dispatch(enc, K.softmax, [this.#u(Lk, this.ctx, 0, 0), a.S], T * 16);
        gpu.dispatch(enc, K.attnOut, [this.#u(Lk, this.ctx, 0, 0), a.S, c.v, a.gate, a.att], T * 16);
        this.#matmul(enc, p + 'attn_output.weight', a.att, a.y, T);
      }
      this.#norm(enc, a.y, this.w[p + 'post_attention_norm.weight'].buf, a.x, T, E, { mode: 1 });
      this.#norm(enc, a.x, this.w[p + 'ffn_norm.weight'].buf, a.h, T, E);
      this.#matmul(enc, p + 'ffn_gate.weight', a.h, a.g, T);
      this.#matmul(enc, p + 'ffn_up.weight', a.h, a.u, T);
      const n = T * F;
      gpu.dispatch(enc, K.swiglu, [this.#u(n, 0, 0, 0), a.g, a.u], [Math.min(Math.ceil(n / 256), 65535), Math.ceil(n / 256 / 65535)]);
      this.#matmul(enc, p + 'ffn_down.weight', a.g, a.y, T);
      this.#norm(enc, a.y, this.w[p + 'post_ffw_norm.weight'].buf, a.x, T, E, { mode: 1 });
    }
    this.#norm(enc, a.x, this.w['output_norm.weight'].buf, a.headIn, 1, E, { xOff: (T - 1) * E });
    const head = this.w['token_embd.weight'];
    gpu.dispatch(enc, K.matmulQ6_K, [this.#u(head.rows, E, 0, 0), head.buf, a.headIn, a.logits], [Math.min(head.rows, 32768), Math.ceil(head.rows / 32768)]);
    const t0 = performance.now();
    gpu.device.queue.submit([enc.finish()]);
    await gpu.device.queue.onSubmittedWorkDone();
    gpu.longestSubmit = Math.max(gpu.longestSubmit, performance.now() - t0);
    for (const b of this._uniforms.splice(0)) b.destroy();
    this.pos += T;
  }

  /** Logits of the last token from the latest forward(). */
  async logits() {
    return new Float32Array(await this.gpu.read(this.a.logits, this.vocab * 4));
  }
}
