import torch
import torch.nn as nn
from transformers import AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
import argparse
from mamba_10m_pretrain import MambaModel, Config10M

class SimpleMambaWrapper(LM):
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.config = Config10M()
        self.config.B = 1 # batch size for evaluation passes
        self.model = MambaModel(self.config)
        self.device = device
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"Loaded {model_path} successfully.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            print("Running in untrained/eval mode.")
            
        self.model = self.model.to(self.device).eval()
        
    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string)
        
    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)
        
    def loglikelihood(self, requests):
        res = []
        for context, continuation in requests:
            ctx_enc = self.tok_encode(context)
            cont_enc = self.tok_encode(continuation)
            
            inp = torch.tensor([ctx_enc + cont_enc[:-1]], dtype=torch.long).to(self.device)
            target = torch.tensor([ctx_enc[1:] + cont_enc], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                logits = self.model(inp)
                
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            # Flatten to shape (seq_len, vocab) and (seq_len)
            loss = loss_fn(logits[0], target[0]) 
            
            # Sum the loss only over the continuation tokens
            cont_loss = loss[-len(cont_enc):].sum().item()
            is_greedy = True # Simplified
            res.append((-cont_loss, is_greedy))
            
        return res
        
    def loglikelihood_rolling(self, requests):
        res = []
        for string in requests:
            enc = self.tok_encode(string)
            inp = torch.tensor([enc[:-1]], dtype=torch.long).to(self.device)
            target = torch.tensor([enc[1:]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                logits = self.model(inp)
                
            loss_fn = nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fn(logits[0], target[0]).item()
            res.append(-loss)
            
        return res
        
    def generate_until(self, requests):
        res = []
        for context, gen_kwargs in requests:
            max_new = gen_kwargs.get("max_gen_toks", 50)
            
            inp = torch.tensor([self.tok_encode(context)], dtype=torch.long).to(self.device)
            with torch.no_grad():
                for _ in range(max_new):
                    logits = self.model(inp)
                    next_token = torch.argmax(logits[0, -1]).unsqueeze(0).unsqueeze(0)
                    inp = torch.cat([inp, next_token], dim=1)
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            generated = inp[0, len(self.tok_encode(context)):]
            res.append(self.tok_decode(generated.cpu().tolist()))
            
        return res

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mamba")
    parser.add_argument("--weights", type=str, default="mamba_10m.pt", help="Path to weights")
    parser.add_argument("--tasks", type=str, default="hellaswag", help="Tasks to run (comma separated)")
    parser.add_argument("--limit", type=int, default=10, help="Max number of samples to evaluate")
    args = parser.parse_args()
    
    wrapper = SimpleMambaWrapper(args.weights)
    
    results = simple_evaluate(
        model=wrapper,
        tasks=args.tasks.split(","),
        limit=args.limit
    )
    
    print("\n" + "="*50)
    print(f"Results for {args.tasks} ({args.limit} samples):")
    print("="*50)
    
    # Simple print out
    for task_name, task_res in results['results'].items():
        print(f"\nTask: {task_name}")
        for metric, val in task_res.items():
            if isinstance(val, (int, float)):
                if "stderr" not in metric:
                    print(f"  {metric}: {val:.4f}")

if __name__ == "__main__":
    main()
