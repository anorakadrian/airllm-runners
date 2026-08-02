"""
Inkling-Small text-only inference with optimized AirLLM disk caching.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from shared.airllm_loader import load_model

MODEL_ID = "thinkingmachines/Inkling-Small"
COMPRESSION = "4bit"
SHARDS_PATH = "./airllm_shards"          # ← change to fast NVMe path
DELETE_ORIGINAL = True
MAX_NEW_TOKENS = 256

def main():
    model = load_model(
        model_id=MODEL_ID,
        compression=COMPRESSION,
        shards_path=SHARDS_PATH,
        delete_original=DELETE_ORIGINAL,
        prefetching=True,
    )

    prompt = "Explain what a Mixture-of-Experts model is in simple terms."
    print("\nPrompt:", prompt)
    print("\nGenerating...\n")

    input_tokens = model.tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=False
    )

    with torch.inference_mode():
        generation_output = model.generate(
            input_tokens["input_ids"].cuda(),
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True
        )

    output = model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    print(output)

if __name__ == "__main__":
    main()
