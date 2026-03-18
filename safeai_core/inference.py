from pathlib import Path
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from .config import GPTConfig
from .model import GPT
from .safety import SYSTEM_PREFIX, check_request
from .tokenizer import encode, decode

app = FastAPI()

cfg = GPTConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = Path("models/gpt-epoch3.pt")
if not CKPT_PATH.exists():
    raise FileNotFoundError("Model checkpoint not found. Train the model first.")

model = GPT(cfg).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/generate")
def generate(q: Query):
    violation = check_request(q.prompt)
    if violation:
        return {"completion": violation}

    prompt = SYSTEM_PREFIX + "\n\nStudent: " + q.prompt + "\nTutor:"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(q.max_new_tokens):
            logits, _ = model(idx)
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

    text = decode(idx[0].tolist())
    completion = text.split("Tutor:", 1)[-1].strip()
    return {"completion": completion}
