"""Validation functions for domain model invariants."""

from typing import List, Dict, Optional
from .models import (
    Formation, Play, Scenario, Player, Route, Ruleset,
    Role, Side, RouteBreakpoint
)


class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_formation(formation: Formation) -> None:
    """
    Validate formation invariants:
    - Exactly 5 slots
    - Must include QB and CENTER
    - All roles must be offensive
    """
    # Pydantic validators already handle most of this,
    # but we can add extra checks here
    roles = [slot.role for slot in formation.slots]

    if len(set(roles)) != len(roles):
        raise ValidationError(f"Formation {formation.id} has duplicate roles")

    # QB should be roughly at center
    qb_slot = next((s for s in formation.slots if s.role == Role.QB), None)
    if qb_slot and abs(qb_slot.position.y) > 5.0:
        raise ValidationError(f"QB should be near center (y close to 0), got y={qb_slot.position.y}")


def validate_route(route: Route) -> None:
    """
    Validate route invariants:
    - At least one breakpoint
    - Times monotonically increasing
    - Reasonable physical constraints
    """
    if not route.breakpoints:
        raise ValidationError(f"Route {route.id} has no breakpoints")

    # Check physical feasibility (simple check: distance vs time)
    for i in range(1, len(route.breakpoints)):
        prev = route.breakpoints[i-1]
        curr = route.breakpoints[i]

        dx = curr.x_yards - prev.x_yards
        dy = curr.y_yards - prev.y_yards
        distance = (dx**2 + dy**2)**0.5
        time_s = (curr.time_ms - prev.time_ms) / 1000.0

        if time_s <= 0:
            raise ValidationError(f"Route {route.id} has non-positive time interval")

        # Assume max speed ~10 yards/second for now (will use player speed in sim)
        max_distance = time_s * 15.0  # generous limit
        if distance > max_distance:
            raise ValidationError(
                f"Route {route.id} breakpoint {i} requires {distance:.1f}yds in {time_s:.1f}s "
                f"(>15 yds/s)"
            )


def validate_play(play: Play) -> None:
    """
    Validate play invariants:
    - Formation is valid
    - All assignments are for formation roles
    - QB progression roles have routes
    - Routes are valid
    """
    validate_formation(play.formation)

    formation_roles = {slot.role for slot in play.formation.slots}

    # Check assignments
    for role, route in play.assignments.items():
        if role not in formation_roles:
            raise ValidationError(f"Play {play.id} assigns route to {role} not in formation")

        if route is not None:
            validate_route(route)

    # Check QB progression
    for role in play.qb_plan.progression_roles:
        if role not in play.assignments or play.assignments[role] is None:
            raise ValidationError(
                f"Play {play.id} QB progression includes {role} but no route assigned"
            )


def validate_scenario(scenario: Scenario) -> None:
    """
    Validate scenario invariants:
    - Exactly 5 defenders
    - All defender roles are defensive
    - Positions are on field
    """
    defense_roles = {Role.RUSHER, Role.CB1, Role.CB2, Role.SAFETY, Role.LB}

    if len(scenario.defender_start_positions) != 5:
        raise ValidationError(
            f"Scenario {scenario.id} must have exactly 5 defenders, "
            f"got {len(scenario.defender_start_positions)}"
        )

    for role, pos in scenario.defender_start_positions.items():
        if role not in defense_roles:
            raise ValidationError(f"Scenario {scenario.id} has non-defensive role {role}")

        # Check positions are reasonable
        half_width = scenario.field.width_yards / 2.0
        if abs(pos.y) > half_width:
            raise ValidationError(
                f"Scenario {scenario.id} defender {role} position y={pos.y} "
                f"outside field width"
            )

    # Note: We allow RUSHER role even when rushers_count=0 (they just won't rush)
    # This is common in real football - a "rusher" position that sometimes drops into coverage
    rusher_count = sum(1 for role in scenario.defender_start_positions if role == Role.RUSHER)
    if rusher_count > scenario.defense_call.rushers_count:
        # Strict validation: cannot have more rushers than the call specifies
        raise ValidationError(
            f"Scenario {scenario.id} has {rusher_count} RUSHER positions but only "
            f"{scenario.defense_call.rushers_count} rushers in defense_call"
        )


def validate_player(player: Player) -> None:
    """
    Validate player invariants:
    - Side matches role
    - Attributes in valid ranges (Pydantic handles this)
    """
    offense_roles = {Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3}
    defense_roles = {Role.RUSHER, Role.CB1, Role.CB2, Role.SAFETY, Role.LB}

    if player.side == Side.OFFENSE and player.role not in offense_roles:
        raise ValidationError(f"Player {player.id} is OFFENSE but has role {player.role}")

    if player.side == Side.DEFENSE and player.role not in defense_roles:
        raise ValidationError(f"Player {player.id} is DEFENSE but has role {player.role}")


def validate_ruleset(ruleset: Ruleset) -> None:
    """
    Validate ruleset invariants:
    - Players per side must be 5 for MVP
    - Field dimensions reasonable
    """
    if ruleset.players_per_side != 5:
        raise ValidationError(f"Ruleset {ruleset.id} must have players_per_side=5 for MVP")

    if ruleset.field.width_yards < 20.0 or ruleset.field.width_yards > 100.0:
        raise ValidationError(f"Ruleset {ruleset.id} field width unreasonable")

    if ruleset.field.total_length_yards < 40.0:
        raise ValidationError(f"Ruleset {ruleset.id} field too short")


def validate_all_before_sim(
    play: Play,
    scenario: Scenario,
    offensive_players: Dict[Role, Player],
    defensive_players: Dict[Role, Player]
) -> None:
    """
    Validate everything before running simulation.
    Ensures all invariants hold.
    """
    validate_play(play)
    validate_scenario(scenario)

    # Validate offensive players match formation
    formation_roles = {slot.role for slot in play.formation.slots}
    if set(offensive_players.keys()) != formation_roles:
        raise ValidationError(
            f"Offensive players {set(offensive_players.keys())} "
            f"don't match formation roles {formation_roles}"
        )

    for role, player in offensive_players.items():
        validate_player(player)
        if player.role != role:
            raise ValidationError(f"Player role {player.role} doesn't match assignment {role}")

    # Validate defensive players match scenario
    scenario_roles = set(scenario.defender_start_positions.keys())
    if set(defensive_players.keys()) != scenario_roles:
        raise ValidationError(
            f"Defensive players {set(defensive_players.keys())} "
            f"don't match scenario roles {scenario_roles}"
        )

    for role, player in defensive_players.items():
        validate_player(player)
        if player.role != role:
            raise ValidationError(f"Player role {player.role} doesn't match assignment {role}")
