# Kimi K3 (AirLLM)

Run Moonshot’s **Kimi K3** (2.8T parameters, ~104B active) on very low VRAM using AirLLM expert streaming.

## Special Requirements

Kimi K3 needs extra packages and specific versions:

```bash
pip install airllm compressed-tensors flash-attn
# Important: use CUDA 12 build of torch + transformers==4.56.x
```

- `flash-attn` requires **CUDA 12** (no CUDA 13 wheels yet at time of writing)
- `transformers==4.56.x` is required (remote code does not load cleanly on 5.x)

## Quick Start

```bash
pip install -r requirements.txt
python inference.py
```

## Optimized Disk Caching

Same optimizations as Inkling-Small:

- Custom `shards_path` on fast NVMe
- `delete_original=True`
- Prefetching enabled

Because Kimi K3 is extremely large, the first-time layer/expert splitting will take a long time and require a lot of disk space.

## Notes

- AirLLM streams only the active experts per token.
- Measured VRAM can be as low as ~3.7–4 GB on some setups.
- This is still primarily text-oriented through AirLLM.
