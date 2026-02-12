import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.performance import LatencyBudgetManager


def test_latency_budget_adjusts():
    manager = LatencyBudgetManager()
    base = manager.get_budget_ms()
    for _ in range(5):
        manager.observe("stt", base * 1.5)
    increased = manager.get_budget_ms()
    assert increased >= base

    for _ in range(5):
        manager.observe("stt", base * 0.3)
    decreased = manager.get_budget_ms()
    assert decreased <= increased
