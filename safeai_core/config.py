from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 256
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 1
    grad_accum_steps: int = 4
    lr: float = 3e-4
    epochs: int = 1
    weight_decay: float = 0.01
    ckpt_dir: str = "models"
