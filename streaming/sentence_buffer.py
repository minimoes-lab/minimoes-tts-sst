import re
from typing import List, Optional


class SentenceBuffer:
    """Accumulates streaming LLM tokens and yields complete sentences."""

    SPLIT_PATTERN = re.compile(r'(?<=[.!?;])\s+|(?<=\n)')

    def __init__(self, min_chars: int = 20, max_chars: int = 200):
        self._buffer = ""
        self._min_chars = min_chars
        self._max_chars = max_chars

    def add_token(self, token: str) -> List[str]:
        """Add a token to the buffer. Returns list of complete sentences (0 or more)."""
        self._buffer += token
        sentences: List[str] = []

        while True:
            match = self.SPLIT_PATTERN.search(self._buffer)
            if match and match.start() >= self._min_chars:
                sentence = self._buffer[:match.end()].strip()
                self._buffer = self._buffer[match.end():]
                if sentence:
                    sentences.append(sentence)
            elif len(self._buffer) > self._max_chars:
                last_space = self._buffer.rfind(' ', 0, self._max_chars)
                if last_space > self._min_chars:
                    sentence = self._buffer[:last_space].strip()
                    self._buffer = self._buffer[last_space:]
                    if sentence:
                        sentences.append(sentence)
                else:
                    break
            else:
                break

        return sentences

    def flush(self) -> Optional[str]:
        """Flush remaining buffer content as a final sentence."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None
