from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
FILE_PATH_RE = re.compile(r"(^|\s)([\w.-]+/)+[\w.-]+\.[a-zA-Z0-9]+")
STACK_TRACE_RE = re.compile(r"(traceback|exception|error:|^\s*at\s+\S+\()", re.I | re.M)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


@dataclass(frozen=True, slots=True)
class HeuristicResult:
    score_adjustments: dict[str, float]
    signals: dict[str, Any]
    reasons: tuple[str, ...]


class StructuralHeuristics:
    def evaluate(self, text: str) -> HeuristicResult:
        normalized = text.strip().lower()
        words = normalized.split()
        adjustments: dict[str, float] = {}
        signals: dict[str, Any] = {"word_count": len(words)}
        reasons: list[str] = []

        if self._looks_like_greeting(normalized, len(words)):
            self._boost(adjustments, "trivial.respond", 0.55)
            signals["looks_like_greeting"] = True
            reasons.append("looks_like_greeting:+trivial.respond")

        if len(words) <= 4 and not URL_RE.search(text) and not CODE_BLOCK_RE.search(text):
            self._boost(adjustments, "trivial.respond", 0.2)
            signals["very_short_input"] = True
            reasons.append("very_short_input:+trivial.respond")

        if CODE_BLOCK_RE.search(text):
            self._boost(adjustments, "code.implement", 0.45)
            signals["contains_code_block"] = True
            reasons.append("contains_code_block:+code.implement")

        if STACK_TRACE_RE.search(text):
            self._boost(adjustments, "code.verify", 0.55)
            signals["contains_stack_trace"] = True
            reasons.append("contains_stack_trace:+code.verify")

        if FILE_PATH_RE.search(text):
            self._boost(adjustments, "code.implement", 0.25)
            self._boost(adjustments, "code.verify", 0.2)
            signals["contains_file_path"] = True
            reasons.append("contains_file_path:+code")

        if URL_RE.search(text):
            self._boost(adjustments, "web.research", 0.45)
            signals["contains_url"] = True
            reasons.append("contains_url:+web.research")

        if self._contains_any(
            normalized,
            ("build", "create", "implement", "refactor", "write"),
        ):
            self._boost(adjustments, "code.implement", 0.3)
            signals["implementation_style"] = True
            reasons.append("implementation_style:+code.implement")

        if self._contains_any(
            normalized,
            ("architecture", "design", "scalable", "tradeoff", "trade-off"),
        ):
            self._boost(adjustments, "architecture.design", 0.35)
            signals["architecture_language"] = True
            reasons.append("architecture_language:+architecture.design")

        if self._contains_any(normalized, ("latest", "current", "recent", "today")):
            self._boost(adjustments, "web.research", 0.3)
            signals["time_sensitive_language"] = True
            reasons.append("time_sensitive_language:+web.research")

        return HeuristicResult(
            score_adjustments=adjustments,
            signals=signals,
            reasons=tuple(reasons),
        )

    def _boost(
        self,
        adjustments: dict[str, float],
        capability: str,
        amount: float,
    ) -> None:
        adjustments[capability] = adjustments.get(capability, 0.0) + amount

    def _contains_any(self, text: str, phrases: tuple[str, ...]) -> bool:
        return any(phrase in text for phrase in phrases)

    def _looks_like_greeting(self, text: str, word_count: int) -> bool:
        if word_count > 6:
            return False
        return text in {"hello", "hi", "hey", "yo"} or text.startswith(
            ("hello ", "hi ", "hey ")
        )
