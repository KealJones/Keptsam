# compare.html export: Qwen3 0.6B second pass, iPhone 16, Safari 27.2, Nov 19, 2026

The runtimes the first pass ([2026-11-19-iphone16-qwen3.md](2026-11-19-iphone16-qwen3.md)) couldn't test fairly,
plus Tilde again with its f16 KV cache (commit 1d8ed9f; the first pass had f32). Copied as exported by
[bench/compare.html](../../bench/compare.html). LiteRT-LM's prompt goes through Qwen3's chat template, so it reads a
few more tokens than the others, and its speeds are LiteRT-LM's own benchmark counts.

### Qwen3 0.6B, iPhone Safari 27.2, 2026-11-19

GPU apple (shader-f16); context 4096; 64 tokens written, greedy; one warm-up request each. Tilde: this page's copy.

| Runtime | Prompt | Outcome | Load s | Read tok/s | Write tok/s | First token s | GPU MB | Worst frame ms | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Tilde | 256 | ran | 2.97 | 134.4 | 26.9 | 1.92 | 873 | 25 | Tilde counted 873 MB |
| wllama 3.13.0 WebGPU | 256 | error | – | – | – | – | – | – | reading + writing: (ABORT) ; Safari compat build |
| transformers.js 4.4.0-next.0 | 256 | skipped | – | – | – | – | – | – | couldn't get the model ready (died) |
| LiteRT-LM 0.19.0 | 256 | ran | 12.32 | 609.7 | 37.8 | 0.46 | 1,492 | 45 | Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm |
| Tilde | 1024 | ran | 3 | 116.4 | 21.7 | 8.81 | 873 | 26 | Tilde counted 873 MB |
| wllama 3.13.0 WebGPU | 1024 | skipped | – | – | – | – | – | – | error at 256 tokens |
| transformers.js 4.4.0-next.0 | 1024 | skipped | – | – | – | – | – | – | couldn't get the model ready (died) |
| LiteRT-LM 0.19.0 | 1024 | ran | 13.07 | 802.5 | 33.4 | 1.32 | 1,496 | 34 | Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm |
| Tilde | 3072 | ran | 2.59 | 82.7 | 14.6 | 37.15 | 873 | 30 | Tilde counted 873 MB |
| wllama 3.13.0 WebGPU | 3072 | skipped | – | – | – | – | – | – | error at 256 tokens |
| transformers.js 4.4.0-next.0 | 3072 | skipped | – | – | – | – | – | – | couldn't get the model ready (died) |
| LiteRT-LM 0.19.0 | 3072 | lost the GPU | – | – | – | – | 1,496 | – | reading: ABORTED: The timeout was reached while reading back data.; GPU lost (destroyed) during reading; Qwen3-0.6B_dynamic_wi4b32_afp32.litertlm |
