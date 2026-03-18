"""
SafeAI Homework Tutor 0.0.96
Core package initialization.
"""

from .config import GPTConfig, TrainConfig
from .model import GPT
from .tokenizer import encode, decode, train_tokenizer
from .safety import check_request, SYSTEM_PREFIX
