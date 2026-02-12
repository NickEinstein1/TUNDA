import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.response.safety import SafetyGuard


def test_self_harm_detection():
    guard = SafetyGuard()
    result = guard.assess("I want to kill myself")
    assert result.is_crisis is True
    assert result.response is not None
