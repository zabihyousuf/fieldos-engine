"""Trace collection and sampling."""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core.models import TraceMode
from .engine import SimulationTrace


class TraceSampler:
    """Manages trace sampling according to policy."""

    def __init__(
        self,
        mode: TraceMode,
        top_n: Optional[int] = None,
        sample_rate: Optional[float] = None,
        seed: Optional[int] = None
    ):
        self.mode = mode
        self.top_n = top_n
        self.sample_rate = sample_rate
        self.rng = np.random.Generator(np.random.PCG64(seed))

        self.traces: List[SimulationTrace] = []
        self.all_traces: List[SimulationTrace] = []

    def should_record(self, episode_num: int) -> bool:
        """Determine if this episode should be traced."""
        if self.mode == TraceMode.NONE:
            return False
        elif self.mode == TraceMode.TOP_N:
            # Always record for TOP_N, we'll filter later
            return True
        elif self.mode == TraceMode.SAMPLE_RATE:
            if self.sample_rate is None:
                return False
            return self.rng.random() < self.sample_rate
        return False

    def add_trace(self, trace: SimulationTrace):
        """Add a trace."""
        self.all_traces.append(trace)

    def finalize(self) -> List[SimulationTrace]:
        """Finalize and return sampled traces."""
        if self.mode == TraceMode.NONE:
            return []
        elif self.mode == TraceMode.TOP_N:
            # Sort by yards gained and return top N
            sorted_traces = sorted(
                self.all_traces,
                key=lambda t: t.outcome.yards_gained,
                reverse=True
            )
            n = self.top_n or 10
            return sorted_traces[:n]
        elif self.mode == TraceMode.SAMPLE_RATE:
            # Already sampled during recording
            return self.all_traces
        return []

    def clear(self):
        """Clear all traces."""
        self.traces.clear()
        self.all_traces.clear()


def serialize_trace(trace: SimulationTrace) -> dict:
    """Serialize trace to JSON-friendly dict."""
    return {
        "play_id": trace.play_id,
        "scenario_id": trace.scenario_id,
        "outcome": {
            "outcome": trace.outcome.outcome.value,
            "yards_gained": trace.outcome.yards_gained,
            "time_to_throw_ms": trace.outcome.time_to_throw_ms,
            "target_role": trace.outcome.target_role.value if trace.outcome.target_role else None,
            "completion_probability": trace.outcome.completion_probability,
            "separation_at_throw": trace.outcome.separation_at_throw,
            "separation_at_catch": trace.outcome.separation_at_catch,
            "failure_modes": [fm.value for fm in trace.outcome.failure_modes]
        },
        "num_states": len(trace.states) if trace.states else 0,
        "num_events": len(trace.events) if trace.events else 0,
        # Note: Full states not serialized for size - would include in detailed export
    }
