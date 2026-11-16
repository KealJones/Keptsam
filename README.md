# Keptsam

**The model iOS didn't throw overboard.**

Keptsam is a WebGPU engine for running small language models in a browser tab, built to survive iPhone Safari.
On iOS, a tab that uses too much memory gets killed by Jetsam; Keptsam is what it keeps.

It streams weights straight into GPU memory instead of holding the whole file, counts every byte it allocates,
splits GPU work into short submits so WebKit's watchdog never fires, and (for multimodal models) only keeps
loaded the senses it is using right now.

- First model: [LFM3 1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) from Liquid AI, read straight from the
  official GGUF.
- Next: [LFM3 Omni 2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF): text, images and voice in one model,
  with each sense swapped in when needed.
- GPU only (WebGPU). No build step: plain ES modules.

Status: early, under construction. See [docs/design.md](docs/design.md) for how it works and why.

## License

MIT for the code. Model weights have their own licenses (LFM3: LFM Open License v1.0).
