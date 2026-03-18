from pathlib import Path
import torch
from .tokenizer import train_tokenizer, encode

RAW_DIR = Path("data/raw")
TOKENS_PATH = Path("data/tokens.pt")

def build_tokens(max_tokens: int = 500_000):
    texts = []
    for path in RAW_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        texts.append(text)
    all_ids: list[int] = []
    for t in texts:
        all_ids.extend(encode(t))
    tokens = torch.tensor(all_ids[:max_tokens], dtype=torch.long)
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, TOKENS_PATH)

if __name__ == "__main__":
    # 1) Put your 2 books + 1 textbook as .txt into data/raw/
    # 2) Train tokenizer
    train_tokenizer(str(RAW_DIR))
    # 3) Build token dataset
    build_tokens()
    print("Tokenizer and tokens.pt created.")
