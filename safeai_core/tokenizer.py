from pathlib import Path
import sentencepiece as spm

SENTENCEPIECE_MODEL = Path("data/spm.model")

def train_tokenizer(raw_dir: str, vocab_size: int = 32000):
    raw_dir = Path(raw_dir)
    input_files = ",".join(str(p) for p in raw_dir.glob("*.txt"))
    SENTENCEPIECE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        f"--input={input_files} "
        f"--model_prefix={SENTENCEPIECE_MODEL.with_suffix('')} "
        f"--vocab_size={vocab_size} "
        "--model_type=bpe --character_coverage=0.9995"
    )

def get_sp():
    if not SENTENCEPIECE_MODEL.exists():
        raise FileNotFoundError("SentencePiece model not found. Run data pipeline first.")
    return spm.SentencePieceProcessor(model_file=str(SENTENCEPIECE_MODEL))

def encode(text: str) -> list[int]:
    sp = get_sp()
    return sp.encode(text, out_type=int)

def decode(ids: list[int]) -> str:
    sp = get_sp()
    return sp.decode(ids)
