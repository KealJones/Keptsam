# Tilde: notes for agents

WebGPU engine for small language models that must survive iPhone Safari (~1.9 GB per tab shared by JS, Wasm and
GPU; ~3 s per GPU command buffer). Read docs/design.md first; it is the contract.

## Rules

- Plain ES modules, no build step, JSDoc for types. Must run as-is from GitHub Pages.
- Never hold a whole model file in memory. Weights stream tensor by tensor through a small staging buffer.
- Every GPU buffer goes through the engine's allocator so memory is counted.
- GPU only. No CPU fallback path.
- Kernels must not need `shader-f16`: store halves in u32 and use `unpack2x16float`.
- Model-specific code lives in one file per architecture (src/models/<arch>.js).
- Every kernel change is checked against the CPU reference (src/quant.js, src/ref/) and, for whole-model changes,
  against llama.cpp logits for the same tokens.

## Commands

- `npm test`: CPU tests (GGUF parsing, repacking, reference math). The header test uses a local GGUF if
  `TILDE_GGUF` (or /home/claude/models/LFM3-1.7B-QAD-Q4_0.gguf) exists, otherwise it is skipped.
- `npm run gpu-test`: kernels in headless Chrome on SwiftShader WebGPU (needs Chrome; set CHROME to its path).

## Git

Push straight to main. Commits are authored as Keal Jones with Claude as co-author (Co-Authored-By trailer).

## Checking against llama.cpp

`bench/lfm3-check.html` runs a prompt through Tilde and compares the last-position logits and a greedy
continuation with a dump from llama.cpp (`[n][tokens][vocab][logits n×vocab][n_gen][gen]`, all little-endian
int32/f32). Mount the model and the dump into the bench server:

    TILDE_MOUNTS="/models=/path/to/ggufs /ref=/path/to/dumps" npm run gpu-test -- "lfm3-check.html?gen=4"

SwiftShader is slow (~80 s per token for LFM3 1.7B on 2 cores), so keep `gen` small. Baseline (Nov 16, 2026):
"The capital of France is" → top 5 identical, max |logit diff| 0.50, 12/12 greedy tokens identical.
