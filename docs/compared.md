# Tilde next to the other browser runtimes

Every runtime below ran the same model, Liquid AI's **LFM3 1.7B** (4-bit), in Safari on one iPhone 16, Nov 2026.
The rows link to where each result is recorded. Results that were seen but never written down are left out on
purpose; [Next runs](#next-runs) lists what would fill the gaps.

## LFM3 1.7B, by runtime

| Runtime | Result | Speed | Memory | How it ended |
|---|---|---|---|---|
| **Tilde** (Nov 16) | ✅ runs | prompt **51 tok/s**, writing **12.6 tok/s**, reload from cache ~5 s | **1,123 MB** counted at a 4K context | Top 5 tokens match llama.cpp and 12/12 greedy tokens are identical. ([design.md](design.md#measured-on-an-iphone-16-safari-27-nov-16-2026)) |
| **WebLLM**, our own MLC 4-bit build (Nov 10) | ✅ with a patch | writing **13.7 tok/s**, first word in 15 s | – | A 3K-token prompt killed the tab while answering. ([Squiggle CLAUDE.md](https://github.com/KealJones/squiggle/blob/main/CLAUDE.md), [b9b9f87](https://github.com/KealJones/squiggle/commit/b9b9f87)) |
| **WebLLM**, QAD weights, one-sequence library (Nov 15) | ✅ up to 6K, with a patch | prompt ~100 tok/s unpaced (256-token pieces, ~2.5 s each) | 4K: 1,021 MB GPU, 1,112 peak. 6K: 1,065 MB, 1,200 peak | **8K killed the tab while loading.** ([a44fe37](https://github.com/KealJones/squiggle/commit/a44fe37)) |
| **wllama 3.13** (llama.cpp in WebAssembly) | ❌ no tokens | – | – | On its own it picks a build iOS 27.2 can't compile ("Memory64 is not enabled"). With its Safari build forced, the run got the tab killed, and Safari's auto-reload turned that into a crash loop. ([8115f27](https://github.com/KealJones/tilde/commit/8115f27), [b64a483](https://github.com/KealJones/tilde/commit/b64a483), [vs-wllama.html](../bench/vs-wllama.html)) |
| **wllama 3.8.1** | ❌ | – | – | Its llama.cpp predates LFM3. ([a694a1f](https://github.com/KealJones/tilde/commit/a694a1f)) |
| **wllama on the CPU** (memtest) | ⚠ short prompts only | 1.7 tok/s | 736 MB left on a short prompt | A full 3K-token prompt killed the tab while loading. ([LFM3 demo](https://github.com/KealJones/lm-experiments/blob/main/site/demos/lfm3-webllm/demo.js)) |
| **transformers.js**, ONNX on WebGPU (1.3 GB export) | ❌ never finished downloading | – | – | Three of four tries killed the tab. It holds a download in memory about twice over while loading it. ([LFM3 demo](https://github.com/KealJones/lm-experiments/blob/main/site/demos/lfm3-webllm/demo.js)) |

## What WebLLM needed to survive iOS 27

WebLLM is fast, but on iOS 27 it only stays up with patches reaching into its internals. Tilde avoids each of these
by design.

- **GPU watchdog.** From iOS 27.2.1 on, WebLLM lost the GPU on a plain "Hi" (a ~1,000-token prompt). It queues its
  256-token prefill pieces without waiting, ~2.5 s each, and WebKit kills the GPU process when a command buffer
  waits more than [3 s](https://github.com/WebKit/WebKit/blob/main/Source/WebGPU/WebGPU/Queue.mm#L492-L494).
  Squiggle cuts the pieces to 128 tokens and syncs between them by reaching into WebLLM's pipeline
  ([61ad3b4](https://github.com/KealJones/squiggle/commit/61ad3b4)). Tilde sizes every submit ahead of time from
  per-batch-size measurements instead.
- **"Object has already been disposed."** After a GPU loss, WebLLM's TVM runtime disposes itself and every later
  call fails until the model is loaded again ([688aaae](https://github.com/KealJones/squiggle/commit/688aaae)).
- **Prompt reuse.** A same-buffer `copyBufferToBuffer` lost the GPU device. With a workaround in place, reuse kept 61%
  of prompt tokens (3,392 of 5,536) from being read again, and the first word came at 7.2 s
  ([lab notebook](https://github.com/KealJones/squiggle/blob/main/docs/lab.md)). Tilde snapshots state at a safe cut
  and reads only the rest.
- **Compile jobs.** WebLLM had no LFM3 build at all. Making one took eight Hugging Face jobs and TVM built from source
  at an exact commit ([lfm3-mlc](https://github.com/KealJones/lm-experiments/blob/main/experiments/lfm3-mlc/README.md)).
  Tilde reads Liquid's official GGUF directly.

## Other phone-killers

- **Liquid's own LFM3 Omni 2B WebGPU demo** crashed the iPhone tab at ~80% loaded, because it loads every sense at
  once (~1.6 GB of weights). Tilde's Omni peaks at **1,280 MB** with vision swapped in and settles at **1,067 MB**
  ([4097517](https://github.com/KealJones/tilde/commit/4097517)).
- **Before LFM3:** Gemma 4 E2B and E4B on LiteRT got the tab killed, and Qwen3.5 2B on WebLLM loaded but died on its
  first reply ([lab notebook](https://github.com/KealJones/squiggle/blob/main/docs/lab.md)).
- **The 1.9 GB budget** comes from memcap: five rounds on the iPhone landed within about 0.1 GB of each other, and
  JavaScript, WebAssembly and the GPU all draw from the same budget
  ([memcap](https://github.com/KealJones/lm-experiments/blob/main/experiments/memcap/README.md)).

## Read these numbers fairly

- **Tilde doesn't use less memory than WebLLM.** At 4K it's 1,123 MB against WebLLM's 1,021 MB, and the two are
  counted differently. Tilde's case is that it survives iOS 27 without patches, needs no compile jobs, runs official
  GGUFs directly and swaps senses in and out.
- **WebLLM reads prompts faster** (~100 against 51 tok/s), but that ~100 is its unpaced speed, which iOS 27.2.1 now
  kills. Its paced speed hasn't been measured.
- **WebLLM's 13.7 tok/s was measured on Nov 10, before iOS 27.2.1,** and Tilde's 12.6 on Nov 16. A fair comparison
  needs both on the same day and the same iOS.
- **The wllama run doesn't record whether it used the GPU.** `result.webgpu` is only saved when a run finishes, so
  it was lost with the tab.

## Next runs

- An `engine=webllm` option in [`bench/vs-wllama.html`](../bench/vs-wllama.html), so all three runtimes run on the
  same day, phone and iOS.
- Save `result.webgpu` at the `loaded` stage, so a killed wllama run still says where it ran.
- Tilde with a 1,024-token prompt, and Tilde at 6K and 8K. If Tilde loads at 8K, that's a direct win over WebLLM,
  which died loading there.
