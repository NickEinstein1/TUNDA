"""Performance utilities for adaptive latency budgeting."""

import threading
import time
from typing import Dict

from .config import config


class LatencyBudgetManager:
    """Tracks inference time and adapts latency budget."""

    def __init__(self):
        self._lock = threading.Lock()
        self._budget_ms = config.performance.latency_budget_ms
        self._min_ms = config.performance.latency_budget_min_ms
        self._max_ms = config.performance.latency_budget_max_ms
        self._alpha = config.performance.latency_ewma_alpha
        self._enabled = config.performance.adaptive_latency_enabled
        self._ewma_ms = None
        self._stage_ewma: Dict[str, float] = {}

    def get_budget_ms(self) -> int:
        with self._lock:
            return int(self._budget_ms)

    def observe(self, stage: str, duration_ms: float):
        if not self._enabled:
            return
        with self._lock:
            if self._ewma_ms is None:
                self._ewma_ms = duration_ms
            else:
                self._ewma_ms = (self._alpha * duration_ms) + ((1 - self._alpha) * self._ewma_ms)

            stage_ewma = self._stage_ewma.get(stage)
            if stage_ewma is None:
                self._stage_ewma[stage] = duration_ms
            else:
                self._stage_ewma[stage] = (self._alpha * duration_ms) + ((1 - self._alpha) * stage_ewma)

            if self._ewma_ms > self._budget_ms * 1.1:
                self._budget_ms = min(self._max_ms, self._budget_ms * 1.1)
            elif self._ewma_ms < self._budget_ms * 0.7:
                self._budget_ms = max(self._min_ms, self._budget_ms * 0.9)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "budget_ms": float(self._budget_ms),
                "ewma_ms": float(self._ewma_ms or 0.0),
                "stage_ewma": dict(self._stage_ewma),
            }


latency_manager = LatencyBudgetManager()
