"""Defensive coverage logic."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..core.models import (
    CoverageType, CoverageShell, Role, Point2D,
    PlayerAttributes, DefenseCall
)


class CoverageAssignment:
    """Tracks defensive assignments for a play."""

    def __init__(
        self,
        defense_call: DefenseCall,
        defender_positions: Dict[Role, Point2D],
        offensive_receivers: List[Role]
    ):
        self.defense_call = defense_call
        self.initial_positions = defender_positions
        self.offensive_receivers = offensive_receivers

        # Determine assignments based on coverage type
        if defense_call.type == CoverageType.MAN:
            self.assignments = self._assign_man_coverage()
        else:
            self.assignments = self._assign_zone_coverage()

    def _assign_man_coverage(self) -> Dict[Role, Optional[Role]]:
        """
        Assign defenders to receivers in man coverage.

        Returns:
            Map of defender_role -> receiver_role (or None for rusher)
        """
        assignments = {}

        # Rushers have no assignment
        if Role.RUSHER in self.initial_positions:
            assignments[Role.RUSHER] = None

        # Get non-rusher defenders
        coverage_defenders = [
            role for role in self.initial_positions.keys()
            if role != Role.RUSHER
        ]

        # Simple assignment: match defenders to receivers in order
        # In real implementation, would consider alignment, formations, etc.
        receivers = self.offensive_receivers.copy()

        for i, defender in enumerate(coverage_defenders):
            if i < len(receivers):
                assignments[defender] = receivers[i]
            else:
                # Extra defender (shouldn't happen in 5v5 man)
                assignments[defender] = receivers[0] if receivers else None

        return assignments

    def _assign_zone_coverage(self) -> Dict[Role, Optional[str]]:
        """
        Assign defenders to zones.

        Returns:
            Map of defender_role -> zone_name
        """
        assignments = {}
        shell = self.defense_call.shell

        # Rushers have no assignment
        if Role.RUSHER in self.initial_positions:
            assignments[Role.RUSHER] = None

        # Get non-rusher defenders
        coverage_defenders = [
            role for role in self.initial_positions.keys()
            if role != Role.RUSHER
        ]

        if shell == CoverageShell.COVER2:
            # 2 deep halves, 3 underneath
            # Simple: safeties/CBs deep, others underneath
            deep_roles = {Role.SAFETY, Role.CB1, Role.CB2}
            deep_assigned = 0
            for defender in coverage_defenders:
                if defender in deep_roles and deep_assigned < 2:
                    assignments[defender] = f"deep_{deep_assigned}"
                    deep_assigned += 1
                else:
                    assignments[defender] = "underneath"

        elif shell == CoverageShell.COVER3:
            # 3 deep thirds, 4 underneath
            deep_roles = {Role.SAFETY, Role.CB1, Role.CB2}
            deep_assigned = 0
            for defender in coverage_defenders:
                if defender in deep_roles and deep_assigned < 3:
                    assignments[defender] = f"deep_third_{deep_assigned}"
                    deep_assigned += 1
                else:
                    assignments[defender] = "underneath"

        elif shell == CoverageShell.COVER1:
            # 1 deep safety, others man
            if Role.SAFETY in coverage_defenders:
                assignments[Role.SAFETY] = "deep_center"
                # Others play man
                receivers = self.offensive_receivers.copy()
                other_defenders = [d for d in coverage_defenders if d != Role.SAFETY]
                for i, defender in enumerate(other_defenders):
                    if i < len(receivers):
                        assignments[defender] = receivers[i]
                    else:
                        assignments[defender] = None
            else:
                # Fallback to man
                return self._assign_man_coverage()

        else:  # COVER0
            # Pure man, no deep help
            return self._assign_man_coverage()

        return assignments


def update_defender_position(
    defender_role: Role,
    current_pos: Tuple[float, float],
    assignment: Optional[any],
    coverage_type: CoverageType,
    receiver_positions: Dict[Role, Tuple[float, float]],
    defender_attrs: PlayerAttributes,
    timestep_ms: float,
    rng: np.random.Generator
) -> Tuple[float, float]:
    """
    Update defender position based on coverage assignment.

    Args:
        defender_role: Defender role
        current_pos: Current (x, y) position
        assignment: Man assignment (Role) or zone (str) or None (rusher)
        coverage_type: MAN or ZONE
        receiver_positions: Current receiver positions
        defender_attrs: Defender attributes
        timestep_ms: Simulation timestep
        rng: Random number generator

    Returns:
        New (x, y) position
    """
    if assignment is None:
        # Rusher - handled separately
        return current_pos

    # Man coverage
    if coverage_type == CoverageType.MAN and isinstance(assignment, Role):
        target_role = assignment
        if target_role not in receiver_positions:
            return current_pos

        target_pos = receiver_positions[target_role]

        # Defender chases receiver with reaction lag
        reaction_factor = defender_attrs.man_coverage / 100.0
        speed_factor = defender_attrs.closing_speed / 100.0

        # Max speed: 5-10 yds/s
        max_speed = 5.0 + speed_factor * 5.0
        dt_s = timestep_ms / 1000.0

        # Direction to target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)

        if dist < 0.1:
            return current_pos

        # Move toward target with speed and reaction constraints
        move_distance = max_speed * dt_s * reaction_factor

        if move_distance >= dist:
            # Can reach target
            new_x, new_y = target_pos
        else:
            # Partial move
            ratio = move_distance / dist
            new_x = current_pos[0] + ratio * dx
            new_y = current_pos[1] + ratio * dy

        # Add small jitter
        jitter = rng.normal(0, 0.1, 2)
        new_x += jitter[0]
        new_y += jitter[1]

        return new_x, new_y

    # Zone coverage
    elif coverage_type == CoverageType.ZONE and isinstance(assignment, str):
        zone = assignment

        # Determine zone landmark based on assignment
        zone_center = _get_zone_center(zone, current_pos)

        # Find nearest threat in zone
        nearest_threat = _find_nearest_threat_in_zone(
            zone, zone_center, receiver_positions
        )

        if nearest_threat:
            # Drift toward threat
            target_pos = receiver_positions[nearest_threat]
        else:
            # No threat, hold zone center
            target_pos = zone_center

        # Move toward target
        speed_factor = defender_attrs.zone_coverage / 100.0
        max_speed = 5.0 + speed_factor * 4.0  # Zone defenders slightly slower
        dt_s = timestep_ms / 1000.0

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)

        if dist < 0.1:
            return current_pos

        move_distance = max_speed * dt_s * speed_factor

        if move_distance >= dist:
            new_x, new_y = target_pos
        else:
            ratio = move_distance / dist
            new_x = current_pos[0] + ratio * dx
            new_y = current_pos[1] + ratio * dy

        # Add jitter
        jitter = rng.normal(0, 0.15, 2)
        new_x += jitter[0]
        new_y += jitter[1]

        return new_x, new_y

    return current_pos


def _get_zone_center(zone: str, current_pos: Tuple[float, float]) -> Tuple[float, float]:
    """Get zone landmark position."""
    # Simplified zone centers
    if "deep" in zone:
        depth = 20.0
        if "0" in zone or "left" in zone:
            return (depth, -10.0)
        elif "1" in zone or "right" in zone:
            return (depth, 10.0)
        else:  # center
            return (depth, 0.0)
    else:  # underneath
        return (5.0, 0.0)


def _find_nearest_threat_in_zone(
    zone: str,
    zone_center: Tuple[float, float],
    receiver_positions: Dict[Role, Tuple[float, float]]
) -> Optional[Role]:
    """Find nearest receiver threatening the zone."""
    min_dist = float('inf')
    nearest = None

    for role, pos in receiver_positions.items():
        # Simple: any receiver near zone center
        dx = pos[0] - zone_center[0]
        dy = pos[1] - zone_center[1]
        dist = np.sqrt(dx*dx + dy*dy)

        # Zone radius ~15 yards
        if dist < 15.0 and dist < min_dist:
            min_dist = dist
            nearest = role

    return nearest
