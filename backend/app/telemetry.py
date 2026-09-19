"""Per-stage timing and counters (contract §2 GET /metrics).

Everything here is measured. Nothing is defaulted to a flattering value and nothing is
carried over between requests (directive 10).
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List


class StageTimer:
    def __init__(self) -> None:
        self.stages: Dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000, 3)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 3)


class Metrics:
    """Process-wide counters. Percentiles are computed from raw samples, never estimated."""

    def __init__(self) -> None:
        self.latencies: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.cost_usd_total: float = 0.0

    def record(self, path: str, total_ms: float, stages: Dict[str, float],
               cache_hit: bool, cost_usd: float = 0.0) -> None:
        self.latencies[path].append(total_ms)
        for name, ms in stages.items():
            self.latencies[f"stage:{name}"].append(ms)
        self.counters["requests"] += 1
        self.counters["cache_hits" if cache_hit else "cache_misses"] += 1
        self.cost_usd_total += cost_usd

    @staticmethod
    def _pct(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
        return round(ordered[idx], 3)

    def snapshot(self) -> Dict[str, object]:
        return {
            "counters": dict(self.counters),
            "cost_usd_total": round(self.cost_usd_total, 6),
            "latency_ms": {
                key: {"n": len(v), "p50": self._pct(v, 0.50), "p95": self._pct(v, 0.95)}
                for key, v in self.latencies.items()
            },
        }


METRICS = Metrics()
