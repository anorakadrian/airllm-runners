"""
Kimi K3 inference with optimized AirLLM disk caching + expert streaming.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from shared.airllm_loader import load_model

MODEL_ID = "moonshotai/Kimi-K3"          # official ID used by AirLLM
COMPRESSION = None                       # Kimi K3 often works best without extra compression flag
SHARDS_PATH = "./airllm_shards_kimi"     # ← put on fast NVMe
DELETE_ORIGINAL = True
MAX_NEW_TOKENS = 256

def main():
    print("Note: First run will take a long time (model is ~2.8T and needs splitting).")
    print("Make sure you have enough disk space.\n")

    model = load_model(
        model_id=MODEL_ID,
        compression=COMPRESSION,
        shards_path=SHARDS_PATH,
        delete_original=DELETE_ORIGINAL,
        prefetching=True,
    )

    prompt = "What is the capital of France? Answer concisely."
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
