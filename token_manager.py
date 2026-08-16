"""
Token rotation manager for HuggingFace Inference API keys.

Reads tokens from config.py (which loads from .env).
The HF_TOKENS list supports multiple comma-separated tokens for rotation.
"""

from config import HF_TOKENS


class TokenManager:
    """Manages HuggingFace API token rotation across multiple tokens."""

    def __init__(self):
        self.tokens = HF_TOKENS
        self.current = 0

    def get_token(self):
        if not self.tokens:
            raise ValueError(
                "No HF_TOKENS configured. Set HF_TOKEN or HF_TOKENS in .env."
            )
        return self.tokens[self.current]

    def rotate(self):
        if not self.tokens:
            raise ValueError(
                "No HF_TOKENS configured. Set HF_TOKEN or HF_TOKENS in .env."
            )
        self.current = (self.current + 1) % len(self.tokens)
        return self.tokens[self.current]