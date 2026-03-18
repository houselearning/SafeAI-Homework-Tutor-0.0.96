from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from .config import GPTConfig, TrainConfig
from .model import GPT

class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len=512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + 1 + self.seq_len]
        return x.clone().detach(), y.clone().detach()

def train():
    cfg = GPTConfig()
    tcfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = torch.load("data/tokens.pt")
    ds = TokenDataset(tokens, seq_len=cfg.max_seq_len)
    dl = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=2)

    model = GPT(cfg).to(device)
    opt = AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    Path(tcfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(tcfg.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}")
        opt.zero_grad()
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / tcfg.grad_accum_steps).backward()
            if (i + 1) % tcfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                pbar.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), f"{tcfg.ckpt_dir}/gpt-epoch{epoch+1}.pt")

if __name__ == "__main__":
    train()
