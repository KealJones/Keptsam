# Keptsam: notes for agents

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
  `KEPTSAM_GGUF` (or /home/claude/models/LFM3-1.7B-QAD-Q4_0.gguf) exists, otherwise it is skipped.
- `npm run gpu-test`: kernels in headless Chrome on SwiftShader WebGPU (needs Chrome; set CHROME to its path).

## Git

Push straight to main. Commits are authored as Keal Jones with Claude as co-author (Co-Authored-By trailer).
