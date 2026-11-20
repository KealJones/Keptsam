// Crash black box: learn from deaths instead of probing for them.
//
// iOS Safari gives no warning before it kills a tab for memory: committed GPU memory past the tab's ~1.5–2 GB kills
// it in well under a second, with no device.lost and no error event, and probing for the limit either shows a
// fake number (allocations are lazy until written) or kills the tab. The limit also moves with what else is running.
// So Tilde writes a tiny record of what it is doing to localStorage as it goes (a synchronous write survives an iOS
// out-of-memory kill), and on the next load reads how the last session ended and backs off if it died there.
// Findings this relies on: nearform/crashbox docs/research 01–03 and 06 (iPhone 15 Pro, iOS 18.7).
//
//   - a clean exit is only `pagehide` with persisted false (app switches and BFCache parks are not exits)
//   - `document.wasDiscarded` is set by an iOS discard, never by a crash
//   - a crash fires nothing; Safari reloads within a second, so heartbeat age tells nothing either
//
// How a session ended decides the next one's settings:
//   killed   (no clean exit, not discarded): memory. Each level halves maxChunk (to 16 at most twice), then ctx.
//   gpu-lost (device lost during a GPU pass): the GPU watchdog. Each level takes 30% off the submit budget.
// Three good sessions in a row give one level back.

const PREFIX = 'tilde:blackbox:';
const MAX_LEVEL = 3;
const CLEAN_TO_RELAX = 3;

/**
 * How the previous session ended.
 * @param {object | undefined} prev the previous session record
 * @param {boolean} wasDiscarded document.wasDiscarded on this load
 * @returns {'none' | 'clean' | 'discarded' | 'gpu-lost' | 'killed'}
 */
export function classify(prev, wasDiscarded = false) {
  if (!prev) return 'none';
  if (prev.lost) return 'gpu-lost';
  if (prev.closed) return 'clean';
  if (wasDiscarded) return 'discarded';
  return 'killed';
}

/**
 * Next levels from the stored ones and how the last session ended.
 * @param {{ mem: number, gpu: number, good: number }} levels
 * @param {ReturnType<typeof classify>} verdict
 * @param {object | undefined} prev
 */
export function nextLevels({ mem = 0, gpu = 0, good = 0 } = {}, verdict, prev) {
  if (verdict === 'killed') return { mem: Math.min(MAX_LEVEL, mem + 1), gpu, good: 0 };
  if (verdict === 'gpu-lost') return { mem, gpu: Math.min(MAX_LEVEL, gpu + 1), good: 0 };
  // A session counts as good only if the model got as far as running.
  if ((verdict === 'clean' || verdict === 'discarded') && prev?.ran) {
    good++;
    if (good >= CLEAN_TO_RELAX && (mem || gpu)) return gpu ? { mem, gpu: gpu - 1, good: 0 } : { mem: mem - 1, gpu, good: 0 };
  }
  return { mem, gpu, good };
}

/**
 * Settings after backing off by the given levels.
 * @param {{ ctx: number, chunk: number, maxChunk: number, submitBudgetMs: number }} want
 * @param {{ mem: number, gpu: number }} levels
 */
export function backOff(want, { mem = 0, gpu = 0 }) {
  const maxChunk = Math.max(Math.min(16, want.maxChunk), want.maxChunk >> Math.min(mem, 2));
  const ctx = Math.max(Math.min(1024, want.ctx), want.ctx >> Math.max(0, mem - 1));
  const submitBudgetMs = Math.max(Math.min(300, want.submitBudgetMs), Math.round(want.submitBudgetMs * 0.7 ** gpu));
  return { ctx, chunk: Math.min(want.chunk, maxChunk), maxChunk, submitBudgetMs };
}

export class BlackBox {
  /**
   * Reads how the last session for `name` ended, works out the settings to use, and starts a new record.
   * @param {string} name model identity (one record per model)
   * @param {{ ctx: number, chunk: number, maxChunk: number, submitBudgetMs: number }} want requested settings
   * @param {{ storage?: Storage, apply?: boolean, wasDiscarded?: boolean, target?: EventTarget }} [o]
   *   `apply: false` records and reports but keeps the requested settings (benchmarks).
   */
  constructor(name, want, { storage = globalThis.localStorage, apply = true, wasDiscarded = globalThis.document?.wasDiscarded ?? false, target = globalThis } = {}) {
    this.storage = storage;
    this.key = PREFIX + name;
    let stored = {};
    try { stored = JSON.parse(storage?.getItem(this.key) || '{}') || {}; } catch { stored = {}; }
    const prev = stored.session;
    /** How the previous session ended. */
    this.verdict = classify(prev, wasDiscarded);
    /** The previous session's last record (stage, MB, settings), if any. */
    this.previous = prev;
    this.levels = nextLevels(stored.levels, this.verdict, prev);
    /** Settings to use. */
    this.settings = apply ? backOff(want, this.levels) : { ...want };
    this.session = { t: Date.now(), stage: 'loading', ...this.settings };
    this.#write();
    this.last = 0;
    if (target?.addEventListener) {
      // Only a non-persisted pagehide is an exit; a BFCache park comes back with pageshow and runs on.
      target.addEventListener('pagehide', (e) => { if (!e.persisted) this.close(); });
      target.addEventListener('pageshow', (e) => { if (e.persisted && this.session.closed) { delete this.session.closed; this.#write(); } });
    }
  }

  /** Merges fields into the record. Writes at once on a stage change or `force`, else at most every 250 ms. */
  note(fields, force = false) {
    const changed = fields.stage !== undefined && fields.stage !== this.session.stage;
    Object.assign(this.session, fields);
    const now = Date.now();
    if (changed || force || now - this.last >= 250) { this.last = now; this.#write(); }
  }

  /** Marks a clean end (pagehide, or the app is done with the model). */
  close() { this.session.closed = true; this.#write(); }

  /** A plain summary for logs and benches. */
  report() {
    return { verdict: this.verdict, previous: this.previous, levels: { mem: this.levels.mem, gpu: this.levels.gpu }, settings: this.settings };
  }

  #write() {
    try { this.storage?.setItem(this.key, JSON.stringify({ levels: this.levels, session: this.session })); } catch { /* private mode, quota: memory is best-effort */ }
  }
}
