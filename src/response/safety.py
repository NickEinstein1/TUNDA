"""Safety checks and crisis routing for responses."""

from dataclasses import dataclass
from typing import Optional
import re

from ..utils.config import config


@dataclass
class SafetyResult:
    is_crisis: bool
    confidence: float
    reason: Optional[str] = None
    response: Optional[str] = None


class SafetyGuard:
    def __init__(self):
        self.enabled = config.response_generation.safety_enabled
        self.threshold = config.response_generation.safety_confidence_threshold
        self.crisis_message = config.response_generation.crisis_message
        self._patterns = self._compile_patterns()

    def _compile_patterns(self):
        phrases = [
            r"\bi want to die\b",
            r"\bi don't want to live\b",
            r"\bkill myself\b",
            r"\bend my life\b",
            r"\bsuicidal\b",
            r"\bself harm\b",
            r"\bhurt myself\b",
            r"\bno reason to live\b",
            r"\bcan't go on\b",
            r"\bworthless\b.*\blife\b",
        ]
        return [re.compile(p, re.IGNORECASE) for p in phrases]

    def assess(self, text: str) -> SafetyResult:
        if not self.enabled or not text:
            return SafetyResult(is_crisis=False, confidence=0.0)

        matches = sum(1 for p in self._patterns if p.search(text))
        confidence = min(1.0, matches * 0.4)
        is_crisis = confidence >= self.threshold
        response = self.crisis_message if is_crisis else None
        reason = "self_harm_detected" if is_crisis else None
        return SafetyResult(is_crisis=is_crisis, confidence=confidence, reason=reason, response=response)
