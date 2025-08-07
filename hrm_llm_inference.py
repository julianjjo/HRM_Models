"""
HRM-Text1 Inference Script

This script allows running inference with a trained HRM-Text1 model. It takes a model weights file as input, prompts for text, and generates a short sequence of tokens.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer

from hrm_text1_modeling import HRMText1

# Model Config (Make sure these are the same as the training if you tweaked those params)
T5_TOKENIZER_REPO = "t5-small"
MODEL_CONFIG = {"d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1}
BLOCK_SIZE = 512
MAX_HALT_STEPS = 8

def generate_text(model, tokenizer, prompt_text, max_new_tokens=15, temperature=0.7, top_k=50):
    """Generates text using the HRMText1 model."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    generated_ids = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # The model expects the full sequence for causal attention
            current_input_ids = input_ids
            current_attention_mask = attention_mask

            outputs = model(current_input_ids, attention_mask=current_attention_mask)
            next_token_logits = outputs["logits"][:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / max(temperature, 1e-6)

            # Top-K filtering
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                mask = torch.full_like(next_token_logits, float("-inf"))
                mask.scatter_(1, topk_idx, topk_vals)
                next_token_logits = mask

            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Stop if EOS token is generated
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append the new token for the next generation step
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
            generated_ids.append(next_token_id.item())

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(description="Run inference with a trained HRM-Text1 model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights file (e.g., pytorch_model.bin)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=15,
        help="Maximum number of new tokens to generate (default: 15)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k value for filtering (default: 50)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer '{T5_TOKENIZER_REPO}'...")
    tokenizer = T5Tokenizer.from_pretrained(T5_TOKENIZER_REPO, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"

    print("Initializing model...")
    config = {
        "vocab_size": len(tokenizer),
        "block_size": BLOCK_SIZE,
        "d_model": MODEL_CONFIG["d_model"],
        "n_heads": MODEL_CONFIG["n_heads"],
        "d_ff": MODEL_CONFIG["d_ff"],
        "dropout": MODEL_CONFIG["dropout"],
        "halt_max_steps": MAX_HALT_STEPS,
        "ponder_loss_weight": 0.0,
        "halt_bias_init": 0.0
    }
    model = HRMText1(config).to(device)

    print(f"Loading model weights from '{args.model_path}'...")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params:,}")

    try:
        prompt = input("Enter your prompt: ")
        print("\nGenerating...")

        generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

        print("\n--- Generated Text ---")
        print(generated_text)
        print("----------------------\n")

    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"\nAn error occurred during generation: {e}")

if __name__ == "__main__":
    main()
