# src/inspect_model.py

import torch
from transformers import AutoModelForCausalLM

# Use the exact model name from your config.py
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

print(f"--- Inspecting full layer names for: {MODEL_NAME} ---")

try:
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("\n--- Found Targetable Linear Layers ---")
    # We are interested in the layers within the transformer blocks
    for name, module in model.named_modules():
        if "self_attn" in name or "mlp" in name:
            if isinstance(module, torch.nn.Linear):
                print(name)

except Exception as e:
    print(f"\nAn error occurred: {e}")