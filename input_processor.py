"""Stage 1: normalize and validate incoming user input.

Tokenization is intentionally delegated to the configured Ollama model. This
module does not pretend to reproduce the model tokenizer.
"""
from datetime import datetime


def process_input(raw_text: str) -> dict:
    if not isinstance(raw_text, str):
        raise TypeError("사용자 입력은 문자열이어야 합니다.")
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("사용자 입력이 비어 있습니다.")
    return {
        "raw_text": raw_text,
        "text": normalized,
        "received_at": datetime.now().astimezone().isoformat(),
        "input_type": "text",
        "tokenization": "delegated_to_ollama",
    }
