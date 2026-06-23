import re
from typing import List, Optional

_ABBREV_RE = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|est|vol|art|fig|ex|no|st|rd|nd|th)\.$',
    re.IGNORECASE,
)
_INITIAL_RE = re.compile(r'\b[A-Z]\.$')
_DECIMAL_RE = re.compile(r'\d\.$')


class SentenceBuffer:
    """Accumulates streaming LLM tokens and yields complete sentences."""

    SPLIT_PATTERN = re.compile(r'(?<=[.!?;:])\s+|(?<=—)\s*|(?<=\n)')

    def __init__(self, min_chars: int = 40, max_chars: int = 160):
        self._buffer = ""
        self._min_chars = min_chars
        self._max_chars = max_chars

    def _is_abbreviation_boundary(self, text: str) -> bool:
        """Return True if text ends with an abbreviation (not a real sentence end)."""
        stripped = text.rstrip()
        return bool(
            _ABBREV_RE.search(stripped) or
            _INITIAL_RE.search(stripped) or
            _DECIMAL_RE.search(stripped)
        )

    def add_token(self, token: str) -> List[str]:
        """Add a token to the buffer. Returns list of complete sentences (0 or more)."""
        self._buffer += token
        sentences: List[str] = []

        while True:
            match = self.SPLIT_PATTERN.search(self._buffer)
            if match and match.start() >= self._min_chars:
                candidate = self._buffer[:match.end()].strip()
                # Don't split on abbreviation boundaries
                if candidate and self._is_abbreviation_boundary(candidate):
                    break
                self._buffer = self._buffer[match.end():]
                if candidate:
                    sentences.append(candidate)
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
