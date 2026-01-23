from dataclasses import dataclass
from typing import Dict

@dataclass
class EmotionPrediction:
    """Result of emotion prediction."""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    features_used: int
