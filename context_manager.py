"""Stage 2: bounded, in-memory conversation context.

Context is session-local and is not persisted to disk. Only completed turns
are included in the next request.
"""
from collections import deque


class ContextManager:
    def __init__(self, max_turns: int = 4, max_chars: int = 4000):
        self.turns = deque(maxlen=max(1, int(max_turns)))
        self.max_chars = max(200, int(max_chars))

    def build_context(self) -> str:
        if not self.turns:
            return ""
        blocks = []
        for turn in self.turns:
            blocks.append(
                f"사용자: {turn['user']}\n에이전트: {turn['assistant']}"
            )
        context = "\n\n".join(blocks)
        if len(context) <= self.max_chars:
            return context
        return context[-self.max_chars:]

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text or not assistant_text:
            return
        self.turns.append({"user": user_text, "assistant": assistant_text})

    def clear(self) -> None:
        self.turns.clear()
