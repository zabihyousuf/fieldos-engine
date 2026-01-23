"""Route execution and player motion."""

import numpy as np
from typing import List, Tuple, Optional
from ..core.models import Route, RouteBreakpoint, PlayerAttributes, Point2D


def interpolate_route(
    route: Route,
    start_position: Point2D,
    time_ms: float,
    player_speed: float,
    timestep_ms: float = 50.0
) -> Tuple[float, float]:
    """
    Interpolate player position along route at given time.

    Args:
        route: Route definition
        start_position: Starting position at snap
        time_ms: Current time since snap
        player_speed: Player speed attribute (0-100)
        timestep_ms: Simulation timestep

    Returns:
        (x, y) position at time_ms

    Speed constraint: player can only move at max speed determined by attributes.
    If route requires faster movement, player lags behind.
    """
    if time_ms <= 0:
        return start_position.x, start_position.y

    # Convert route breakpoints to absolute positions (relative to start)
    abs_breakpoints = [
        RouteBreakpoint(
            x_yards=start_position.x + bp.x_yards,
            y_yards=start_position.y + bp.y_yards,
            time_ms=bp.time_ms
        )
        for bp in route.breakpoints
    ]

    # Add start position as first breakpoint
    abs_breakpoints.insert(
        0,
        RouteBreakpoint(x_yards=start_position.x, y_yards=start_position.y, time_ms=0.0)
    )

    # Find which segment we're in
    for i in range(len(abs_breakpoints) - 1):
        bp_start = abs_breakpoints[i]
        bp_end = abs_breakpoints[i + 1]

        if bp_start.time_ms <= time_ms <= bp_end.time_ms:
            # Interpolate within this segment
            t_ratio = (time_ms - bp_start.time_ms) / max(1.0, bp_end.time_ms - bp_start.time_ms)

            # Target position (ideal)
            target_x = bp_start.x_yards + t_ratio * (bp_end.x_yards - bp_start.x_yards)
            target_y = bp_start.y_yards + t_ratio * (bp_end.y_yards - bp_start.y_yards)

            # Apply speed constraint
            # Speed: ~5-10 yards/sec for speed 50-100
            max_speed_yps = 5.0 + (player_speed / 100.0) * 5.0
            elapsed_s = time_ms / 1000.0
            max_distance = max_speed_yps * elapsed_s

            # Distance from start to target
            actual_dist = np.sqrt(
                (target_x - start_position.x)**2 +
                (target_y - start_position.y)**2
            )

            if actual_dist > max_distance:
                # Player can't reach target yet, scale back
                scale = max_distance / actual_dist
                x = start_position.x + scale * (target_x - start_position.x)
                y = start_position.y + scale * (target_y - start_position.y)
                return x, y

            return target_x, target_y

    # Past last breakpoint - hold position
    last_bp = abs_breakpoints[-1]
    return last_bp.x_yards, last_bp.y_yards


def compute_velocity(
    prev_pos: Tuple[float, float],
    curr_pos: Tuple[float, float],
    timestep_ms: float
) -> Tuple[float, float]:
    """Compute velocity vector from position change."""
    dt_s = timestep_ms / 1000.0
    if dt_s <= 0:
        return 0.0, 0.0

    vx = (curr_pos[0] - prev_pos[0]) / dt_s
    vy = (curr_pos[1] - prev_pos[1]) / dt_s
    return vx, vy


def compute_separation(
    receiver_pos: Tuple[float, float],
    defender_positions: List[Tuple[float, float]]
) -> float:
    """
    Compute minimum separation between receiver and all defenders.

    Returns:
        Minimum distance to nearest defender in yards
    """
    if not defender_positions:
        return 999.0  # No defenders

    min_dist = float('inf')
    rx, ry = receiver_pos

    for dx, dy in defender_positions:
        dist = np.sqrt((rx - dx)**2 + (ry - dy)**2)
        min_dist = min(min_dist, dist)

    return min_dist


def project_trajectory(
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    time_ahead_ms: float
) -> Tuple[float, float]:
    """Project position forward based on current velocity."""
    dt_s = time_ahead_ms / 1000.0
    x = start_pos[0] + velocity[0] * dt_s
    y = start_pos[1] + velocity[1] * dt_s
    return x, y


def compute_throw_distance(qb_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> float:
    """Compute throw distance in yards."""
    dx = target_pos[0] - qb_pos[0]
    dy = target_pos[1] - qb_pos[1]
    return np.sqrt(dx*dx + dy*dy)


def get_throw_category(distance: float) -> str:
    """Categorize throw as short/mid/deep."""
    if distance < 10.0:
        return "short"
    elif distance < 20.0:
        return "mid"
    else:
        return "deep"
