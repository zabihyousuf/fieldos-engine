"""Metrics aggregation and analysis."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.models import (
    PlayOutcome, OutcomeType, FailureMode,
    DownDistanceBucket, GameSituation, Role
)


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple play simulations."""

    num_plays: int = 0

    # Outcome counts
    completions: int = 0
    incompletions: int = 0
    sacks: int = 0
    interceptions: int = 0

    # Yards
    yards_total: float = 0.0
    yards_list: List[float] = field(default_factory=list)

    # Time to throw
    time_to_throw_list: List[float] = field(default_factory=list)

    # Target distribution
    target_counts: Dict[Role, int] = field(default_factory=lambda: defaultdict(int))

    # Failure modes
    failure_mode_counts: Dict[FailureMode, int] = field(default_factory=lambda: defaultdict(int))

    def add_outcome(self, outcome: PlayOutcome):
        """Add a play outcome to metrics."""
        self.num_plays += 1

        if outcome.outcome == OutcomeType.COMPLETE:
            self.completions += 1
        elif outcome.outcome == OutcomeType.INCOMPLETE:
            self.incompletions += 1
        elif outcome.outcome == OutcomeType.SACK:
            self.sacks += 1
        elif outcome.outcome == OutcomeType.INTERCEPT:
            self.interceptions += 1

        self.yards_total += outcome.yards_gained
        self.yards_list.append(outcome.yards_gained)

        if outcome.time_to_throw_ms is not None:
            self.time_to_throw_list.append(outcome.time_to_throw_ms)

        if outcome.target_role:
            self.target_counts[outcome.target_role] += 1

        for fm in outcome.failure_modes:
            self.failure_mode_counts[fm] += 1

    @property
    def completion_rate(self) -> float:
        """Completion percentage."""
        if self.num_plays == 0:
            return 0.0
        return self.completions / self.num_plays

    @property
    def sack_rate(self) -> float:
        """Sack percentage."""
        if self.num_plays == 0:
            return 0.0
        return self.sacks / self.num_plays

    @property
    def intercept_rate(self) -> float:
        """Interception percentage."""
        if self.num_plays == 0:
            return 0.0
        return self.interceptions / self.num_plays

    @property
    def yards_mean(self) -> float:
        """Mean yards per play."""
        if not self.yards_list:
            return 0.0
        return float(np.mean(self.yards_list))

    @property
    def yards_p50(self) -> float:
        """Median yards per play."""
        if not self.yards_list:
            return 0.0
        return float(np.median(self.yards_list))

    @property
    def yards_p90(self) -> float:
        """90th percentile yards."""
        if not self.yards_list:
            return 0.0
        return float(np.percentile(self.yards_list, 90))

    @property
    def time_to_throw_mean(self) -> float:
        """Mean time to throw in ms."""
        if not self.time_to_throw_list:
            return 0.0
        return float(np.mean(self.time_to_throw_list))

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_plays": self.num_plays,
            "completion_rate": self.completion_rate,
            "sack_rate": self.sack_rate,
            "intercept_rate": self.intercept_rate,
            "yards_mean": self.yards_mean,
            "yards_p50": self.yards_p50,
            "yards_p90": self.yards_p90,
            "time_to_throw_mean": self.time_to_throw_mean,
            "target_distribution": {
                role.value: count for role, count in self.target_counts.items()
            },
            "failure_modes": {
                mode.value: count for mode, count in self.failure_mode_counts.items()
            }
        }


@dataclass
class SlicedMetrics:
    """Metrics sliced by scenario buckets."""

    overall: AggregateMetrics = field(default_factory=AggregateMetrics)
    by_bucket: Dict[DownDistanceBucket, AggregateMetrics] = field(
        default_factory=lambda: defaultdict(AggregateMetrics)
    )

    def add_outcome(
        self,
        outcome: PlayOutcome,
        situation: Optional[GameSituation] = None
    ):
        """Add outcome to overall and bucket-specific metrics."""
        self.overall.add_outcome(outcome)

        if situation:
            bucket = situation.bucket
            self.by_bucket[bucket].add_outcome(outcome)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "overall": self.overall.to_dict(),
            "by_bucket": {
                bucket.value: metrics.to_dict()
                for bucket, metrics in self.by_bucket.items()
            }
        }


class MetricsCollector:
    """Collects and aggregates metrics across simulation runs."""

    def __init__(self):
        self.metrics = SlicedMetrics()
        # Event: (play_id, outcome, situation)
        self.events: List[Tuple[str, PlayOutcome, Optional[GameSituation]]] = []

    def record(self, play_id: str, outcome: PlayOutcome, situation: Optional[GameSituation] = None):
        """Record a play outcome."""
        self.events.append((play_id, outcome, situation))
        self.metrics.add_outcome(outcome, situation)

    def get_metrics(self) -> SlicedMetrics:
        """Get aggregated metrics."""
        return self.metrics

    def get_best_plays_by_bucket(self) -> Dict[str, str]:
        """
        Determine best play for each situation bucket.
        Returns map of bucket_name -> play_id.
        Measure: Mean yards gained.
        """
        # Bucket -> PlayId -> List[outcomes]
        bucket_play_outcomes: Dict[DownDistanceBucket, Dict[str, List[PlayOutcome]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for play_id, outcome, situation in self.events:
            if not situation:
                continue
            bucket_play_outcomes[situation.bucket][play_id].append(outcome)

        best_plays = {}

        for bucket, plays in bucket_play_outcomes.items():
            best_play_id = None
            best_score = -float('inf')

            for play_id, outcomes in plays.items():
                if not outcomes:
                    continue
                # Score = mean yards
                mean_yards = np.mean([o.yards_gained for o in outcomes])
                
                # Tiny penalty for low sample size? No, keep simple.
                if mean_yards > best_score:
                    best_score = mean_yards
                    best_play_id = play_id
            
            if best_play_id:
                best_plays[bucket.value] = best_play_id

        return best_plays

    def reset(self):
        """Reset all metrics."""
        self.metrics = SlicedMetrics()
        self.events.clear()
