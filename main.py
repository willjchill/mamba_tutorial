import torch
from transformers import AutoTokenizer
import argparse
import os
from mamba_10m_pretrain import MambaModel, Config10M

def generate(model, tokenizer, prompt, max_new_tokens=50, device="cpu"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Evaluate current context
            logits = model(input_ids)
            
            # Predict the next token (greedy decoding)
            # Only care about the last logit in the sequence
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            
            # Append token and continue
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
    generated_text = tokenizer.decode(input_ids[0].cpu().tolist())
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Mamba Inference Script")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt to generate text")
    parser.add_argument("--weights", type=str, default="mamba_10m.pt", help="Path to model weights")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = Config10M()
    config.B = 1 # Update batch size for inference
    model = MambaModel(config).to(device)
    
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}...")
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    else:
        print(f"Warning: Model weights {args.weights} not found. Running inference with untrained weights.")
        
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    output = generate(model, tokenizer, args.prompt, args.max_tokens, device)
    
    print("Output:")
    print(output)
    print("-" * 50)

if __name__ == "__main__":
    main()
