"""
Shared optimized AirLLM loader.

Used by both Inkling-Small and Kimi K3 runners.
Focuses on better disk caching behavior.
"""

from pathlib import Path
from airllm import AutoModel

def load_model(
    model_id: str,
    compression: str | None = "4bit",
    shards_path: str | None = None,
    delete_original: bool = True,
    prefetching: bool = True,
    hf_token: str | None = None,
    **extra_kwargs
):
    """
    Load any supported model with optimized disk caching.

    Args:
        model_id: Hugging Face model ID
        compression: "4bit", "8bit", or None
        shards_path: Directory for layer shards (put on fast NVMe)
        delete_original: Delete original model after splitting (~50% disk save)
        prefetching: Overlap loading + compute when supported
        hf_token: Optional Hugging Face token
        **extra_kwargs: Passed through to AutoModel.from_pretrained
    """

    kwargs = {
        "delete_original": delete_original,
        "prefetching": prefetching,
        **extra_kwargs,
    }

    if compression is not None:
        kwargs["compression"] = compression

    if shards_path is not None:
        Path(shards_path).mkdir(parents=True, exist_ok=True)
        kwargs["layer_shards_saving_path"] = shards_path

    if hf_token is not None:
        kwargs["hf_token"] = hf_token

    print(f"Loading {model_id}")
    print(f"  compression     : {compression}")
    print(f"  shards_path     : {shards_path or 'default HF cache'}")
    print(f"  delete_original : {delete_original}")
    print(f"  prefetching     : {prefetching}")
    print("-" * 55)

    model = AutoModel.from_pretrained(model_id, **kwargs)
    return model
