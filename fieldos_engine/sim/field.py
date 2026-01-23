"""Field coordinate system and utilities."""

import numpy as np
from typing import Tuple
from ..core.models import FieldConfig, Point2D


class FieldCoordinates:
    """
    Coordinate system utilities.
    Origin at LOS center: (x=0, y=0)
    x+ toward offense endzone (downfield)
    y+ toward offense right
    """

    def __init__(self, config: FieldConfig):
        self.config = config
        self.half_width = config.width_yards / 2.0
        self.half_length = config.total_length_yards / 2.0

    def in_bounds(self, x: float, y: float) -> bool:
        """Check if position is in bounds."""
        return (abs(y) <= self.half_width and
                -self.half_length <= x <= self.half_length)

    def clip_to_bounds(self, x: float, y: float) -> Tuple[float, float]:
        """Clip position to field bounds."""
        x = np.clip(x, -self.half_length, self.half_length)
        y = np.clip(y, -self.half_width, self.half_width)
        return x, y

    def distance(self, p1: Point2D, p2: Point2D) -> float:
        """Euclidean distance between two points."""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return np.sqrt(dx*dx + dy*dy)

    def distance_xy(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance between two (x,y) coordinates."""
        dx = x2 - x1
        dy = y2 - y1
        return np.sqrt(dx*dx + dy*dy)

    def to_endzone_distance(self, x: float) -> float:
        """Distance from x position to offense endzone."""
        endzone_start = self.half_length - self.config.endzone_depth_yards
        return max(0.0, endzone_start - x)

    def is_in_endzone(self, x: float) -> bool:
        """Check if x position is in offense endzone."""
        endzone_start = self.half_length - self.config.endzone_depth_yards
        return x >= endzone_start

    def is_in_no_run_zone(self, x: float) -> bool:
        """Check if x position is in no-run zone."""
        if not self.config.no_run_zone_depth_yards:
            return False
        endzone_start = self.half_length - self.config.endzone_depth_yards
        no_run_start = endzone_start - self.config.no_run_zone_depth_yards
        return x >= no_run_start
