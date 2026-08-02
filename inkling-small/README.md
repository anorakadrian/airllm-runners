# Inkling-Small (AirLLM)

Text-only inference of `thinkingmachines/Inkling-Small` using AirLLM with optimized disk caching.

> Full multimodal (image + audio) is **not** supported through AirLLM.

## Quick Start

```bash
pip install -r requirements.txt
python inference.py
```

## Optimized Disk Caching

The loader uses:

- Custom `shards_path` (set to a fast NVMe)
- `delete_original=True`
- `compression="4bit"`
- Prefetching

Edit `SHARDS_PATH` in `inference.py` to point to your fastest drive.
