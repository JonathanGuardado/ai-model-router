from __future__ import annotations

import re
from typing import Protocol

from .models import CapabilityCandidate, CapabilityDefinition

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#.-]*")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "is",
    "it",
    "of",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


class SemanticMatcher(Protocol):
    def match(
        self,
        text: str,
        capabilities: tuple[CapabilityDefinition, ...],
    ) -> tuple[CapabilityCandidate, ...]:
        ...


class LocalSemanticMatcher:
    """Small deterministic lexical matcher behind a semantic-style interface."""

    def match(
        self,
        text: str,
        capabilities: tuple[CapabilityDefinition, ...],
    ) -> tuple[CapabilityCandidate, ...]:
        text_tokens = self._tokens(text)
        candidates = [
            self._score_capability(text_tokens, capability)
            for capability in capabilities
        ]
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.capability)))

    def _score_capability(
        self,
        text_tokens: set[str],
        capability: CapabilityDefinition,
    ) -> CapabilityCandidate:
        positive_text = " ".join(
            (capability.name, capability.description, *capability.examples)
        )
        negative_text = " ".join(capability.anti_examples)
        positive_tokens = self._tokens(positive_text)
        negative_tokens = self._tokens(negative_text)

        overlap = text_tokens.intersection(positive_tokens)
        anti_overlap = text_tokens.intersection(negative_tokens)
        union_size = len(text_tokens.union(positive_tokens)) or 1
        base_score = len(overlap) / union_size
        example_bonus = self._example_bonus(text_tokens, capability.examples)
        anti_penalty = min(0.3, len(anti_overlap) * 0.04)
        score = max(0.0, base_score + example_bonus - anti_penalty)

        reasons = [
            f"semantic_overlap:{','.join(sorted(overlap)) or 'none'}",
            f"semantic_base:{base_score:.3f}",
        ]
        if example_bonus:
            reasons.append(f"example_bonus:+{example_bonus:.3f}")
        if anti_penalty:
            reasons.append(f"anti_example_penalty:-{anti_penalty:.3f}")

        return CapabilityCandidate(
            capability=capability.name,
            score=round(score, 6),
            reasons=tuple(reasons),
        )

    def _example_bonus(self, text_tokens: set[str], examples: tuple[str, ...]) -> float:
        if not text_tokens:
            return 0.0

        best = 0.0
        for example in examples:
            example_tokens = self._tokens(example)
            if not example_tokens:
                continue
            overlap = len(text_tokens.intersection(example_tokens))
            best = max(best, overlap / len(text_tokens.union(example_tokens)))

        return min(0.25, best * 0.35)

    def _tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in TOKEN_RE.findall(text)
            if token.lower() not in STOP_WORDS
        }
