"""Defensive coverage logic for 5v5 flag football."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ..core.models import (
    CoverageType, CoverageShell, Role, Point2D,
    PlayerAttributes, DefenseCall, RusherPosition
)


# Zone definitions with (x_center, y_center, radius) - using sim coords (x=downfield, y=lateral)
ZONE_DEFINITIONS = {
    # Cover 2 zones
    "deep_left_half": (20.0, -8.0, 12.0),
    "deep_right_half": (20.0, 8.0, 12.0),
    "underneath_left": (5.0, -8.0, 8.0),
    "underneath_middle": (5.0, 0.0, 8.0),
    "underneath_right": (5.0, 8.0, 8.0),

    # Cover 3 zones
    "deep_left_third": (18.0, -10.0, 10.0),
    "deep_middle_third": (20.0, 0.0, 10.0),
    "deep_right_third": (18.0, 10.0, 10.0),

    # Cover 1 zones
    "deep_center": (18.0, 0.0, 15.0),

    # Generic zones
    "flat_left": (3.0, -10.0, 8.0),
    "flat_right": (3.0, 10.0, 8.0),
    "hook_left": (8.0, -5.0, 8.0),
    "hook_right": (8.0, 5.0, 8.0),
    "middle_hook": (7.0, 0.0, 8.0),
}


class CoverageAssignment:
    """
    Tracks defensive assignments for a play.

    In 5v5 flag football with D1-D5 defenders:
    - D1: Outside left (typically corner/flat)
    - D2: Inside left (typically LB/hook)
    - D3: Middle (typically MLB/safety/rusher)
    - D4: Inside right (typically LB/hook) or deep safety
    - D5: Outside right (typically corner/flat)
    """

    def __init__(
        self,
        defense_call: DefenseCall,
        defender_positions: Dict[Role, Point2D],
        offensive_receivers: List[Role],
        rng: Optional[np.random.Generator] = None
    ):
        self.defense_call = defense_call
        self.initial_positions = defender_positions
        self.offensive_receivers = offensive_receivers
        self.rng = rng or np.random.default_rng()

        # Determine which defender is the rusher (if any)
        self.rusher_role = self._determine_rusher()

        # Determine assignments based on coverage scheme
        self.assignments = self._assign_coverage()
        self.zone_assignments = {}  # For visualization

    def _determine_rusher(self) -> Optional[Role]:
        """
        Determine which defender is the rusher based on defense call.
        Randomizes position (L/C/R) if not specified.
        """
        if self.defense_call.rushers_count == 0:
            return None

        # Get rusher position
        rusher_pos = self.defense_call.rusher_position
        if rusher_pos is None:
            # Randomize
            rusher_pos = self.rng.choice([RusherPosition.LEFT, RusherPosition.CENTER, RusherPosition.RIGHT])

        # Map position to defender role
        if rusher_pos == RusherPosition.LEFT:
            # D1 or D2 rushes from left
            return Role.D1 if Role.D1 in self.initial_positions else None
        elif rusher_pos == RusherPosition.RIGHT:
            # D5 or D4 rushes from right
            return Role.D5 if Role.D5 in self.initial_positions else None
        else:  # CENTER
            # D3 rushes from center
            return Role.D3 if Role.D3 in self.initial_positions else None

    def _assign_coverage(self) -> Dict[Role, Optional[Union[Role, str]]]:
        """
        Assign coverage based on defense call.

        Returns:
            Map of defender_role -> assignment
            - None: rusher
            - Role: man coverage on that receiver
            - str: zone name
        """
        shell = self.defense_call.shell
        coverage_type = self.defense_call.type

        if coverage_type == CoverageType.MAN:
            if shell == CoverageShell.COVER0:
                return self._assign_man_cover0()
            elif shell == CoverageShell.COVER1:
                return self._assign_man_cover1()
            else:
                # Default man assignment
                return self._assign_man_cover1()
        else:  # ZONE
            if shell == CoverageShell.COVER2:
                return self._assign_zone_cover2()
            elif shell == CoverageShell.COVER3:
                return self._assign_zone_cover3()
            elif shell == CoverageShell.COVER1:
                return self._assign_zone_cover1()
            else:
                return self._assign_zone_cover2()

    def _assign_man_cover0(self) -> Dict[Role, Optional[Union[Role, str]]]:
        """
        Man Cover 0: All 5 defenders in man coverage, no deep help.
        Used when no rush.
        """
        assignments = {}

        # All defenders get man assignments (no deep safety)
        coverage_defenders = list(self.initial_positions.keys())
        if self.rusher_role:
            assignments[self.rusher_role] = None
            coverage_defenders = [d for d in coverage_defenders if d != self.rusher_role]

        # Match defenders to receivers based on position alignment
        receivers = self.offensive_receivers.copy()

        # Sort defenders by lateral position (y) to match with receivers
        def_positions = [(d, self.initial_positions[d].y) for d in coverage_defenders]
        def_positions.sort(key=lambda x: x[1])

        # Pair with receivers (sorted by their expected position)
        for i, (defender, _) in enumerate(def_positions):
            if i < len(receivers):
                assignments[defender] = receivers[i]
            else:
                # Extra defender covers CENTER or last receiver
                if Role.CENTER in receivers:
                    assignments[defender] = Role.CENTER
                elif receivers:
                    assignments[defender] = receivers[-1]
                else:
                    assignments[defender] = "middle_hook"

        return assignments

    def _assign_man_cover1(self) -> Dict[Role, Optional[Union[Role, str]]]:
        """
        Man Cover 1: 3 man coverage, 1 deep safety, optionally 1 rusher.
        """
        assignments = {}

        # Rusher (if any)
        if self.rusher_role:
            assignments[self.rusher_role] = None

        coverage_defenders = [d for d in self.initial_positions.keys() if d != self.rusher_role]

        # Find the deepest defender to be safety
        deepest = max(coverage_defenders, key=lambda d: self.initial_positions[d].x)
        assignments[deepest] = "deep_center"
        coverage_defenders.remove(deepest)

        # Remaining defenders get man assignments
        receivers = [r for r in self.offensive_receivers if r != Role.CENTER]

        for i, defender in enumerate(sorted(coverage_defenders, key=lambda d: self.initial_positions[d].y)):
            if i < len(receivers):
                assignments[defender] = receivers[i]
            else:
                # Extra defender plays zone
                if Role.CENTER in self.offensive_receivers:
                    assignments[defender] = Role.CENTER
                else:
                    assignments[defender] = "middle_hook"

        return assignments

    def _assign_zone_cover2(self) -> Dict[Role, Optional[Union[Role, str]]]:
        """
        Cover 2: Two deep halves (D2, D4), three underneath zones (D1, D3, D5).
        No rush in true Cover 2 (3-2).
        """
        assignments = {}

        if self.rusher_role:
            assignments[self.rusher_role] = None

        coverage_defenders = [d for d in self.initial_positions.keys() if d != self.rusher_role]

        # Identify defenders by lateral position
        sorted_by_lateral = sorted(coverage_defenders, key=lambda d: self.initial_positions[d].y)

        # Deep safeties (typically the two near-side players who are deep)
        # In 3-2, D2 and D4 are the deep halves
        deep_defenders = []
        underneath_defenders = []

        for d in coverage_defenders:
            pos = self.initial_positions[d]
            if pos.x >= 12.0:  # Deep player
                deep_defenders.append(d)
            else:
                underneath_defenders.append(d)

        # Assign deep halves
        deep_defenders.sort(key=lambda d: self.initial_positions[d].y)
        for i, d in enumerate(deep_defenders[:2]):
            if i == 0:
                assignments[d] = "deep_left_half"
            else:
                assignments[d] = "deep_right_half"

        # Assign underneath zones
        underneath_defenders.sort(key=lambda d: self.initial_positions[d].y)
        zones = ["flat_left", "middle_hook", "flat_right"]
        for i, d in enumerate(underneath_defenders[:3]):
            assignments[d] = zones[i] if i < len(zones) else "middle_hook"

        # Any remaining defenders
        for d in coverage_defenders:
            if d not in assignments:
                assignments[d] = "middle_hook"

        return assignments

    def _assign_zone_cover3(self) -> Dict[Role, Optional[Union[Role, str]]]:
        """
        Cover 3: One deep safety (middle third), four underneath.
        Corners drop back to help in 4-1 look.
        """
        assignments = {}

        if self.rusher_role:
            assignments[self.rusher_role] = None

        coverage_defenders = [d for d in self.initial_positions.keys() if d != self.rusher_role]

        # Find the deepest central defender for deep middle
        deepest = max(coverage_defenders, key=lambda d: self.initial_positions[d].x)
        assignments[deepest] = "deep_middle_third"
        coverage_defenders.remove(deepest)

        # Remaining 4 play underneath zones
        coverage_defenders.sort(key=lambda d: self.initial_positions[d].y)
        zones = ["flat_left", "hook_left", "hook_right", "flat_right"]
        for i, d in enumerate(coverage_defenders):
            assignments[d] = zones[i] if i < len(zones) else "middle_hook"

        return assignments

    def _assign_zone_cover1(self) -> Dict[Role, Optional[Union[Role, str]]]:
        """
        Cover 1 Zone: 1 deep safety, 3 underneath, 1 rusher.
        Or 3-1 with rush.
        """
        assignments = {}

        if self.rusher_role:
            assignments[self.rusher_role] = None

        coverage_defenders = [d for d in self.initial_positions.keys() if d != self.rusher_role]

        # Find the deepest defender for deep center
        deepest = max(coverage_defenders, key=lambda d: self.initial_positions[d].x)
        assignments[deepest] = "deep_center"
        coverage_defenders.remove(deepest)

        # Remaining play underneath zones
        coverage_defenders.sort(key=lambda d: self.initial_positions[d].y)
        zones = ["flat_left", "middle_hook", "flat_right"]
        for i, d in enumerate(coverage_defenders):
            assignments[d] = zones[i] if i < len(zones) else "middle_hook"

        return assignments


def get_rusher_start_position(
    defense_call: DefenseCall,
    scenario_positions: Dict[Role, Point2D],
    rusher_role: Role,
    rng: np.random.Generator
) -> Tuple[float, float]:
    """
    Get the starting position for the rusher, randomizing L/C/R if needed.
    """
    rush_distance = 7.0  # Standard rush distance

    rusher_pos = defense_call.rusher_position
    if rusher_pos is None:
        rusher_pos = rng.choice([RusherPosition.LEFT, RusherPosition.CENTER, RusherPosition.RIGHT])

    if rusher_pos == RusherPosition.LEFT:
        return (rush_distance, -5.0)
    elif rusher_pos == RusherPosition.RIGHT:
        return (rush_distance, 5.0)
    else:  # CENTER
        return (rush_distance, 0.0)


def update_defender_position(
    defender_role: Role,
    current_pos: Tuple[float, float],
    assignment: Optional[Union[Role, str]],
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
        # Rusher - handled separately in engine
        return current_pos

    # Man coverage on a receiver
    if isinstance(assignment, Role):
        return _update_man_coverage_position(
            current_pos, assignment, receiver_positions,
            defender_attrs, timestep_ms, rng
        )

    # Zone coverage
    elif isinstance(assignment, str):
        return _update_zone_coverage_position(
            current_pos, assignment, receiver_positions,
            defender_attrs, timestep_ms, rng
        )

    return current_pos


def _update_man_coverage_position(
    current_pos: Tuple[float, float],
    target_role: Role,
    receiver_positions: Dict[Role, Tuple[float, float]],
    defender_attrs: PlayerAttributes,
    timestep_ms: float,
    rng: np.random.Generator
) -> Tuple[float, float]:
    """Update position for man coverage."""
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


def _update_zone_coverage_position(
    current_pos: Tuple[float, float],
    zone_name: str,
    receiver_positions: Dict[Role, Tuple[float, float]],
    defender_attrs: PlayerAttributes,
    timestep_ms: float,
    rng: np.random.Generator
) -> Tuple[float, float]:
    """Update position for zone coverage."""
    # Get zone center and radius
    zone_info = ZONE_DEFINITIONS.get(zone_name)
    if zone_info is None:
        # Fallback to generic zone
        zone_center = (5.0, 0.0)
        zone_radius = 8.0
    else:
        zone_center = (zone_info[0], zone_info[1])
        zone_radius = zone_info[2]

    # Find nearest threat in zone
    nearest_threat = _find_nearest_threat_in_zone(zone_center, zone_radius, receiver_positions)

    if nearest_threat:
        # Drift toward threat but stay in zone
        threat_pos = receiver_positions[nearest_threat]
        # Move to a point between zone center and threat
        target_x = zone_center[0] + 0.6 * (threat_pos[0] - zone_center[0])
        target_y = zone_center[1] + 0.6 * (threat_pos[1] - zone_center[1])
        target_pos = (target_x, target_y)
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


def _find_nearest_threat_in_zone(
    zone_center: Tuple[float, float],
    zone_radius: float,
    receiver_positions: Dict[Role, Tuple[float, float]]
) -> Optional[Role]:
    """Find nearest receiver threatening the zone."""
    min_dist = float('inf')
    nearest = None

    for role, pos in receiver_positions.items():
        dx = pos[0] - zone_center[0]
        dy = pos[1] - zone_center[1]
        dist = np.sqrt(dx*dx + dy*dy)

        # Is receiver in or near the zone?
        if dist < zone_radius * 1.5 and dist < min_dist:
            min_dist = dist
            nearest = role

    return nearest
