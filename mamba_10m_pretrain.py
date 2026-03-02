import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import os

# Configuration
class Config10M:
    B = 8          # Batch size
    L = 128        # Sequence length
    D_outer = 256  # Model dimension
    D = 512        # Expanded dimension in MixerBlock
    N = 16         # SSM state dimension
    kernel_size = 4
    n_layers = 12   # Number of blocks
    vocab_size = 50254 # EleutherAI/gpt-neox-20b vocab size

class SSMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        A_init = torch.arange(1, self.config.N + 1, dtype=torch.float32).repeat(self.config.D, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
    
    def forward(self, X, B, C, delta, H_prev=None):
        A = -torch.exp(self.A_log)
        delta_A = torch.einsum('BLD,DN->BLDN', delta, A)
        A_bar = torch.exp(delta_A)
        
        delta_B = torch.einsum('BLD,BLN->BLDN', delta, B)
        zoh_factor = (A_bar - 1.0) / delta_A 
        B_bar = zoh_factor * delta_B
        
        Y = []
        if H_prev is not None:
            H_current = H_prev
        else:
            H_current = torch.zeros((X.size(0), self.config.D, self.config.N), device=X.device)
            
        for t in range(X.size(1)):
            X_t = X[:, t, :]
            C_t = C[:, t, :]
            A_bar_t = A_bar[:, t, :, :]
            B_bar_t = B_bar[:, t, :, :]
            
            U_decay = torch.einsum('BD,BDN->BDN', X_t, B_bar_t)
            H_decay = torch.einsum('BDN,BDN->BDN', A_bar_t, H_current)
            H_current = U_decay + H_decay
            
            Y_next = torch.einsum('BN,BDN->BD', C_t, H_current)
            Y.append(Y_next)
            
        Y_stacked = torch.stack(Y, dim=1)
        return H_current, Y_stacked

class MixerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layernorm = nn.LayerNorm(self.config.D_outer)
        self.up_projection_1 = nn.Linear(self.config.D_outer, self.config.D, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.config.D,
            out_channels=self.config.D,
            kernel_size=self.config.kernel_size,
            groups=self.config.D,
            padding=self.config.kernel_size - 1
        )
        self.silu_1 = nn.SiLU()
        self.ll = nn.Linear(self.config.D, self.config.N * 2 + self.config.D)
        self.sp = nn.Softplus()
        self.ssm = SSMBlock(config=self.config)
        self.up_projection_2 = nn.Linear(self.config.D_outer, self.config.D, bias=False)
        self.silu_2 = nn.SiLU()
        self.down_projection = nn.Linear(self.config.D, self.config.D_outer)
    
    def forward(self, X):
        X_res = X
        X = self.layernorm(X)
        X_main = self.up_projection_1(X)
        X_main = X_main.transpose(1, 2)
        X_main = self.conv1d(X_main)
        X_main = X_main[..., :X.size(1)]
        X_main = X_main.transpose(1, 2)
        X_main = self.silu_1(X_main)
        
        param_proj = self.ll(X_main)
        delta_i = param_proj[..., :self.config.D]
        B = param_proj[..., self.config.D : self.config.D + self.config.N]
        C = param_proj[..., self.config.D + self.config.N :]
        delta = self.sp(delta_i)
        
        _, Y_ssm = self.ssm(X_main, B, C, delta)
        
        X_gate = self.up_projection_2(X)
        X_gate = self.silu_2(X_gate)
        Y = Y_ssm * X_gate
        Y_proj = self.down_projection(Y)
        return X_res + Y_proj

class MambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.D_outer)
        self.layers = nn.ModuleList([MixerBlock(config=self.config) for _ in range(self.config.n_layers)])
        self.layernorm = nn.LayerNorm(self.config.D_outer)
        self.lm_head = nn.Linear(self.config.D_outer, self.config.vocab_size, bias=False)
    
    def forward(self, input_ids):
        X = self.embedding(input_ids)
        for layer in self.layers:
            X = layer(X)
        X = self.layernorm(X)
        logits = self.lm_head(X)
        return logits

def main():
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.eos_token
    
    raw_data_stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", streaming=True)
    shuffled_stream = raw_data_stream.shuffle(buffer_size=10000)
    raw_data = list(shuffled_stream['train'].take(1000))
    hf_dataset = Dataset.from_list(raw_data)
    hf_split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    
    def tokenize(dataset):
        encode = lambda ds: tokenizer(ds['text'], truncation=True, max_length=128)
        return dataset.map(encode, batched=True, remove_columns=dataset.column_names)
    
    res = tokenize(hf_split['train'])
    hf_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    loader = DataLoader(res, collate_fn=hf_collator, batch_size=Config10M.B, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config10M()
    model = MambaModel(config).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 1
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "mamba_10m.pt")

if __name__ == "__main__":
    main()
