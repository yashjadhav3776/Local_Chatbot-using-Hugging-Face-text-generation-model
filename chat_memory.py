from collections import deque
from typing import Deque, Tuple, List

class ChatMemory:
    def __init__(self, max_messages: int = 8):
        self.max_messages = max_messages
        self.buffer: Deque[Tuple[str, str]] = deque(maxlen=max_messages)

    def add_user(self, text: str):
        self.buffer.append(("User", text.strip()))

    def add_bot(self, text: str):
        self.buffer.append(("Bot", text.strip()))

    def get_messages(self) -> List[Tuple[str, str]]:
        return list(self.buffer)

    def format_for_model(self) -> str:
        conversation = ""
        for role, text in self.buffer:
            conversation += f"{role}: {text}\n"
        conversation += "Bot:"
        return conversation

    def clear(self):
        self.buffer.clear()
