from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ResolutionSource = Literal["semantic", "ambiguous", "default"]


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    name: str
    description: str
    examples: tuple[str, ...]
    anti_examples: tuple[str, ...]
    default_signals: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CapabilityCandidate:
    capability: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityResolution:
    capability: str
    confidence: float
    source: ResolutionSource
    ambiguous: bool
    signals: dict[str, Any]
    candidates: tuple[CapabilityCandidate, ...]
    debug: tuple[str, ...]
