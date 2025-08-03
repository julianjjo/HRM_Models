import torch
import argparse
from transformers import AutoTokenizer
from HRM import HRM

def load_model(checkpoint_path, device):
    """Load the HRM model from checkpoint with the same configuration as training."""
    
    # Model configuration (must match training script)
    TOKENIZER_NAME = "Qwen/Qwen3-32B"
    HRM_DIM = 1024
    HRM_DEPTH_L = 6
    HRM_DEPTH_H = 8
    ATTENTION_DIM_HEAD_L = 64
    HEADS_L = 8
    ATTENTION_DIM_HEAD_H = 128
    HEADS_H = 16
    REASONING_STEPS = 10
    USE_RMSNORM_L = True
    USE_RMSNORM_H = True
    ROTARY_POS_EMB_L = True
    ROTARY_POS_EMB_H = True
    PRE_NORM_L = False
    PRE_NORM_H = False
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    VOCAB_SIZE = tokenizer.vocab_size
    
    # Initialize model with same architecture
    model = HRM(
        networks=[
            dict(
                dim=HRM_DIM,
                depth=HRM_DEPTH_L,
                attn_dim_head=ATTENTION_DIM_HEAD_L,
                heads=HEADS_L,
                use_rmsnorm=USE_RMSNORM_L,
                rotary_pos_emb=ROTARY_POS_EMB_L,
                pre_norm=PRE_NORM_L
            ),
            dict(
                dim=HRM_DIM,
                depth=HRM_DEPTH_H,
                attn_dim_head=ATTENTION_DIM_HEAD_H,
                heads=HEADS_H,
                use_rmsnorm=USE_RMSNORM_H,
                rotary_pos_emb=ROTARY_POS_EMB_H,
                pre_norm=PRE_NORM_H
            )
        ],
        num_tokens=VOCAB_SIZE,
        dim=HRM_DIM,
        reasoning_steps=REASONING_STEPS
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, num_tokens, reasoning_steps, device):
    """Generate text using the loaded HRM model."""
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Initialize generation
    generated_ids = input_ids.clone()
    
    print(f"Generating {num_tokens} tokens with {reasoning_steps} reasoning steps per token...")
    print(f"Prompt: {prompt}")
    print("Generated text:")
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(num_tokens):
            # Get model predictions with specified reasoning steps
            output = model(generated_ids, reasoning_steps=reasoning_steps)
            
            # Unpack output - assuming it returns (logits, hiddens)
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output
            
            # Get next token logits (last position)
            next_token_logits = logits[0, -1, :]
            
            # Simple greedy sampling (argmax)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # confidence = next_token_logits[next_token]
            # print(confidence)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # Decode and print the new token
            new_text = tokenizer.decode(next_token.item())
            print(new_text, end="", flush=True)
            
            # Optional: break on end-of-sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print("\n")  # New line after generation
    
    # Return full generated text
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained HRM model')
    parser.add_argument('-p', '--prompt', type=str, required=True, 
                        help='Text prompt to start generation')
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file')
    parser.add_argument('-n', '--num_tokens', type=int, default=50,
                        help='Number of tokens to generate (default: 50)')
    parser.add_argument('--reasoning_steps', type=int, default=10,
                        help='Number of reasoning steps for each generation step (default: 10)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        print(f"Loading model from {args.checkpoint}...")
        model, tokenizer = load_model(args.checkpoint, device)
        print("Model loaded successfully!")
        
        # Generate text
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            num_tokens=args.num_tokens,
            reasoning_steps=args.reasoning_steps,
            device=device
        )
        
        print("\n" + "="*50)
        print("Full generated text:")
        print(generated_text)
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()