# AirLLM Runners

Low-VRAM inference for very large models using [AirLLM](https://github.com/lyogavin/airllm).

## Supported Models (Sub-repos)

| Model | Path | Parameters | Notes |
|-------|------|------------|-------|
| **Inkling-Small** | [`inkling-small/`](./inkling-small) | 276B total / 12B active | Text-only via AirLLM |
| **Kimi K3** | [`kimi-k3/`](./kimi-k3) | 2.8T total / ~104B active | Special deps required |

## Shared

- [`shared/airllm_loader.py`](./shared/airllm_loader.py) — Optimized disk caching loader used by both models

## Disk Caching Optimizations (applied to both)

- Custom `layer_shards_saving_path` → put on fast NVMe
- `delete_original=True` → saves ~50% disk after first split
- Prefetching enabled
- 4-bit compression where supported

## Quick Start

```bash
git clone https://github.com/anorakadrian/airllm-runners.git
cd airllm-runners

# For Inkling-Small
cd inkling-small && pip install -r requirements.txt
python inference.py

# For Kimi K3 (see kimi-k3/README.md for special deps)
cd ../kimi-k3 && pip install -r requirements.txt
python inference.py
```

## License

Apache-2.0
