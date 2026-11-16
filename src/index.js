// Keptsam public API.
//
//   const llm = await Keptsam.load('https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-QAD-Q4_0.gguf');
//   const text = await llm.chat([{ role: 'user', content: 'hi' }], { onToken: (t) => print(t) });

import { openGpu } from './gpu.js';
import { httpSource } from './gguf.js';
import { Tokenizer } from './tokenizer.js';
import { cachedSource, clearCache } from './cache.js';
import { LFM3 } from './models/lfm3.js';

export { openGpu, httpSource, Tokenizer, LFM3, cachedSource, clearCache };

const MODELS = { lfm3: LFM3 };

export class Keptsam {
  /**
   * Loads a GGUF model onto the GPU, streaming it in.
   * @param {string | import('./gguf.js').ByteSource} source URL or byte source
   * @param {{ ctx?: number, chunk?: number, cache?: boolean, gpu?: import('./gpu.js').Gpu, onProgress?: Function }} [opts]
   *   `cache` (default true for URLs) keeps the file in the browser's Cache Storage so it downloads once.
   */
  static async load(source, { ctx = 4096, chunk = 64, cache = true, gpu, onProgress } = {}) {
    gpu ||= await openGpu();
    let src = typeof source === 'string' ? httpSource(source) : source;
    if (typeof source === 'string' && cache) src = await cachedSource(src, { key: source });
    const model = await LFM3.load(gpu, src, { ctx, chunk, onProgress });
    return new Keptsam(gpu, model);
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
   * @param {{ role: string, content: string }[]} messages
   */
  formatChat(messages, { thinking = true } = {}) {
    const t = this.tokenizer.tokens;
    let s = t[this.tokenizer.bos];
    let lastUser = -1;
    messages.forEach((m, i) => { if (m.role === 'user') lastUser = i; });
    messages.forEach((m, i) => {
      if (m.role === 'assistant') {
        const body = i > lastUser ? m.content : '<think></think>' + String(m.content).split('</think>').pop().trim();
        s += `<|im_start|>assistant\n${body}<|im_end|>\n`;
      } else {
        s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
      }
    });
    return s + '<|im_start|>assistant\n' + (thinking ? '<think>' : '<think></think>');
  }

  /** Chat completion: formats, generates, returns the reply text. */
  chat(messages, opts = {}) {
    return this.generate(this.formatChat(messages, opts), opts);
  }

  /**
   * Generates text after `prompt` (a string, or token ids). If the prompt starts with everything the model has
   * already read, only the new part is read; otherwise the model starts over.
   * @param {string | number[]} prompt
   * @param {{ maxTokens?: number, temperature?: number, topK?: number, topP?: number, repeatPenalty?: number,
   *   stop?: string[], onToken?: (text: string, id: number) => void, signal?: AbortSignal }} [opts]
   */
  async generate(prompt, opts = {}) {
    const o = { maxTokens: 512, topP: 1, stop: [], ...this.defaults, ...opts };
    const ids = typeof prompt === 'string' ? this.tokenizer.encode(prompt) : prompt;
    const m = this.model;
    const reuse = this.history.length <= ids.length && this.history.every((t, i) => t === ids[i]) && this.history.length < ids.length;
    if (!reuse) { m.reset(); this.history = []; }
    const fresh = ids.slice(this.history.length);
    for (let i = 0; i < fresh.length; i += m.chunk) {
      if (o.signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      await m.forward(fresh.slice(i, i + m.chunk));
    }
    this.history = [...ids];

    const out = [];
    let text = '';
    for (let n = 0; n < o.maxTokens; n++) {
      if (o.signal?.aborted) break;
      const logits = await m.logits();
      const id = sample(logits, [...this.history.slice(-64)], o);
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

  /** Memory and timing report. */
  stats() { return this.gpu.report(); }
}

/** Repeat penalty, temperature, top-k, top-p. temperature 0 = greedy. */
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
