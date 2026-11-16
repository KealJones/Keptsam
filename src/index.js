// Keptsam public API.
//
//   const llm = await Keptsam.load('https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-QAD-Q4_0.gguf');
//   const text = await llm.chat([{ role: 'user', content: 'hi' }], { onToken: (t) => print(t) });

import { openGpu } from './gpu.js';
import { httpSource } from './gguf.js';
import { Tokenizer } from './tokenizer.js';
import { cachedSource, clearCache } from './cache.js';
import { readGGUF } from './gguf.js';
import { LFM3 } from './models/lfm3.js';
import { LFM3Vision } from './models/lfm3vl.js';
import { patchify } from './image.js';

export { openGpu, httpSource, Tokenizer, LFM3, LFM3Vision, cachedSource, clearCache, patchify };

export class Keptsam {
  /**
   * Loads a GGUF model onto the GPU, streaming it in.
   * @param {string | import('./gguf.js').ByteSource} source URL or byte source
   * @param {{ ctx?: number, chunk?: number, cache?: boolean, gpu?: import('./gpu.js').Gpu, onProgress?: Function,
   *   mmproj?: string, imageMaxPixels?: number, keepEncoders?: boolean }} [opts]
   *   `cache` (default true for URLs) keeps files in the browser's Cache Storage so they download once.
   *   `mmproj` (Omni/VL models) is the encoder file; encoders load only when an image arrives and are freed after
   *   unless `keepEncoders`. `imageMaxPixels` caps image size (default 512·512, at most 256 image tokens).
   */
  static async load(source, { ctx = 4096, chunk = 64, cache = true, gpu, onProgress, mmproj, imageMaxPixels = 512 * 512, keepEncoders = false } = {}) {
    gpu ||= await openGpu();
    const open = async (s) => (typeof s === 'string' && cache ? cachedSource(httpSource(s), { key: s }) : typeof s === 'string' ? httpSource(s) : s);
    const model = await LFM3.load(gpu, await open(source), { ctx, chunk, onProgress });
    const k = new Keptsam(gpu, model);
    if (mmproj) {
      const src = await open(mmproj);
      k.encoders = { src, gguf: await readGGUF(src) };
    }
    k.imageMaxPixels = imageMaxPixels;
    k.keepEncoders = keepEncoders;
    return k;
  }

  constructor(gpu, model) {
    this.gpu = gpu;
    this.model = model;
    this.tokenizer = Tokenizer.fromGGUF(model.gguf.kv);
    const s = model.gguf.kv;
    this.defaults = {
      temperature: s['general.sampling.temp'] ?? 0.1,
      topK: s['general.sampling.top_k'] ?? 50,
      repeatPenalty: s['general.sampling.penalty_repeat'] ?? 1.1,
    };
    /** Tokens the model has already read, in order. */
    this.history = [];
  }

  /**
   * Formats messages with the model's chat format (system/user/assistant; tools are left to the caller).
   * Content is a string, or an array of parts: strings, { type: 'text', text }, { type: 'image', image }.
   * Returns a string, or (when there are images) an array of strings and { image } parts.
   * @param {{ role: string, content: string | any[] }[]} messages
   */
  formatChat(messages, { thinking = true } = {}) {
    const parts = [this.tokenizer.tokens[this.tokenizer.bos]];
    const push = (x) => { if (typeof x === 'string' && typeof parts.at(-1) === 'string') parts[parts.length - 1] += x; else parts.push(x); };
    const content = (c) => {
      if (!Array.isArray(c)) return push(String(c ?? ''));
      for (const p of c) {
        if (typeof p === 'string') push(p);
        else if (p.type === 'image') { push('<|vision_start|>'); push({ image: p.image }); push('<|vision_end|>'); }
        else push(String(p.text ?? ''));
      }
    };
    let lastUser = -1;
    messages.forEach((m, i) => { if (m.role === 'user') lastUser = i; });
    messages.forEach((m, i) => {
      push(`<|im_start|>${m.role}\n`);
      if (m.role === 'assistant' && i < lastUser) push('<think></think>' + String(m.content).split('</think>').pop().trim());
      else content(m.content);
      push('<|im_end|>\n');
    });
    push('<|im_start|>assistant\n' + (thinking ? '<think>' : '<think></think>'));
    return parts.length === 1 ? parts[0] : parts;
  }

  /** Chat completion: formats, generates, returns the reply text. */
  chat(messages, opts = {}) {
    return this.generate(this.formatChat(messages, opts), opts);
  }

  /**
   * Generates text after `prompt`: a string, token ids, or an array of strings and { image } parts.
   * If the prompt starts with everything the model has already read (same tokens, same image objects),
   * only the new part is read; otherwise the model starts over.
   * @param {string | number[] | (string | { image: any })[]} prompt
   * @param {{ maxTokens?: number, temperature?: number, topK?: number, topP?: number, repeatPenalty?: number,
   *   stop?: string[], onToken?: (text: string, id: number) => void, onStatus?: (s: string) => void, signal?: AbortSignal }} [opts]
   */
  async generate(prompt, opts = {}) {
    const o = { maxTokens: 512, topP: 1, stop: [], ...this.defaults, ...opts };
    // Items: token ids, or image parts (one item each, compared by identity).
    const items = typeof prompt === 'string' ? this.tokenizer.encode(prompt)
      : prompt.flatMap((p) => (typeof p === 'number' ? [p] : typeof p === 'string' ? this.tokenizer.encode(p) : [p]));
    const m = this.model;
    const h = this.history;
    const reuse = h.length < items.length && h.every((t, i) => t === items[i]);
    if (!reuse) { m.reset(); this.history = []; }
    const fresh = items.slice(this.history.length);
    for (let i = 0; i < fresh.length;) {
      if (o.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      if (typeof fresh[i] !== 'number') { await this.#readImage(fresh[i].image, o); i++; continue; }
      let j = i;
      while (j < fresh.length && j - i < m.chunk && typeof fresh[j] === 'number') j++;
      await m.forward(fresh.slice(i, j));
      i = j;
    }
    this.history = [...items];

    const out = [];
    let text = '';
    for (let n = 0; n < o.maxTokens; n++) {
      if (o.signal?.aborted) break;
      const id = sampleCandidates(await m.candidates(), this.history.slice(-64).filter((t) => typeof t === 'number'), o);
      if (id === this.tokenizer.eos || this.tokenizer.tokens[id] === '<|im_end|>') break;
      out.push(id);
      const piece = this.tokenizer.decode(out).slice(text.length);
      text += piece;
      o.onToken?.(piece, id);
      if (o.stop.some((s) => text.endsWith(s))) break;
      if (m.pos >= m.ctx) break;
      await m.forward([id]);
      this.history.push(id);
    }
    return text;
  }

  /** Encodes an image (loading the vision encoder if needed) and feeds its embeddings to the model. */
  async #readImage(image, o) {
    if (!this.encoders) throw new Error('This model was loaded without an mmproj, so it cannot read images.');
    if (!this.vision) {
      o.onStatus?.('loading vision encoder');
      this.vision = await LFM3Vision.load(this.gpu, this.encoders.src, this.encoders.gguf);
    }
    o.onStatus?.('reading image');
    const pt = patchify(image, { ...this.vision.cfg, maxPixels: Math.min(this.vision.cfg.maxPixels, this.imageMaxPixels) });
    const { buf, tokens } = await this.vision.encode(pt);
    if (!this.keepEncoders) { this.vision.free(); this.vision = null; }
    try {
      for (let i = 0; i < tokens; i += this.model.chunk) await this.model.forwardEmbeds(buf, i, Math.min(this.model.chunk, tokens - i));
    } finally { this.gpu.free(buf); }
    o.onStatus?.(`read image (${tokens} tokens)`);
  }

  /** Memory and timing report. */
  stats() { return this.gpu.report(); }
}

/**
 * Samples from GPU top candidates ({ values, ids }): repeat penalty, temperature, top-k, top-p; temperature 0 = greedy.
 * The penalty only lowers scores, so the penalized top-k is still among the candidates.
 */
export function sampleCandidates({ values, ids }, recent, o) {
  const rec = new Set(recent);
  const pen = o.repeatPenalty && o.repeatPenalty !== 1 ? o.repeatPenalty : 0;
  const order = [...values.keys()].filter((i) => values[i] > -1e38);
  const score = (i) => (pen && rec.has(ids[i]) ? (values[i] > 0 ? values[i] / pen : values[i] * pen) : values[i]);
  const sc = new Float32Array(values.length);
  for (const i of order) sc[i] = score(i);
  order.sort((a, b) => sc[b] - sc[a]);
  if (!o.temperature) return ids[order[0]];
  const k = Math.min(o.topK || order.length, order.length);
  const top = order.slice(0, k);
  const mx = sc[top[0]];
  let p = top.map((i) => Math.exp((sc[i] - mx) / o.temperature));
  const sum = p.reduce((a, b) => a + b, 0);
  p = p.map((x) => x / sum);
  let cut = top.length, acc = 0;
  if (o.topP < 1) for (let i = 0; i < p.length; i++) { acc += p[i]; if (acc >= o.topP) { cut = i + 1; break; } }
  let r = Math.random() * p.slice(0, cut).reduce((a, b) => a + b, 0);
  for (let i = 0; i < cut; i++) { r -= p[i]; if (r <= 0) return ids[top[i]]; }
  return ids[top[cut - 1]];
}

/** Repeat penalty, temperature, top-k, top-p over full logits. temperature 0 = greedy. */
export function sample(logits, recent, { temperature, topK, topP, repeatPenalty }) {
  const l = logits;
  if (repeatPenalty && repeatPenalty !== 1) {
    for (const t of new Set(recent)) l[t] = l[t] > 0 ? l[t] / repeatPenalty : l[t] * repeatPenalty;
  }
  if (!temperature) { let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i; return b; }
  const k = Math.min(topK || l.length, l.length);
  // Partial selection of the top k.
  const idx = [];
  for (let i = 0; i < l.length; i++) {
    if (idx.length < k) { idx.push(i); if (idx.length === k) idx.sort((a, b) => l[b] - l[a]); continue; }
    if (l[i] > l[idx[k - 1]]) { let j = k - 1; while (j > 0 && l[idx[j - 1]] < l[i]) { idx[j] = idx[j - 1]; j--; } idx[j] = i; }
  }
  if (idx.length < k) idx.sort((a, b) => l[b] - l[a]);
  const mx = l[idx[0]];
  let p = idx.map((i) => Math.exp((l[i] - mx) / temperature));
  const sum = p.reduce((a, b) => a + b, 0);
  p = p.map((x) => x / sum);
  let cut = idx.length, acc = 0;
  if (topP < 1) for (let i = 0; i < p.length; i++) { acc += p[i]; if (acc >= topP) { cut = i + 1; break; } }
  let r = Math.random() * p.slice(0, cut).reduce((a, b) => a + b, 0);
  for (let i = 0; i < cut; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
  return idx[cut - 1];
}
