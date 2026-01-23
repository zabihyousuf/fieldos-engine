"""Main simulation engine for play execution."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from ..core.models import (
    Play, Player, Scenario, Role, OutcomeType, PlayOutcome,
    FailureMode, Point2D, SimMode
)
from ..core.validation import validate_all_before_sim
from .field import FieldCoordinates
from .motion import (
    interpolate_route, compute_velocity, compute_separation,
    compute_throw_distance, get_throw_category
)
from .coverage import CoverageAssignment, update_defender_position
from .outcome import (
    compute_completion_probability, determine_outcome,
    compute_yards_after_catch, check_sack
)


@dataclass
class SimulationState:
    """State at a single timestep."""
    time_ms: float
    offensive_positions: Dict[Role, Tuple[float, float]]
    defensive_positions: Dict[Role, Tuple[float, float]]
    qb_position: Tuple[float, float]
    rusher_position: Optional[Tuple[float, float]] = None


@dataclass
class SimulationTrace:
    """Complete trace of a simulation run."""
    play_id: str
    scenario_id: str
    outcome: PlayOutcome
    states: List[SimulationState] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)


class SimulationEngine:
    """Main simulation engine."""

    def __init__(
        self,
        timestep_ms: float = 50.0,
        seed: Optional[int] = None
    ):
        self.timestep_ms = timestep_ms
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def simulate_play(
        self,
        play: Play,
        scenario: Scenario,
        offensive_players: Dict[Role, Player],
        defensive_players: Dict[Role, Player],
        mode: SimMode = SimMode.EVAL,
        target_override: Optional[Role] = None,
        record_trace: bool = False
    ) -> Tuple[PlayOutcome, Optional[SimulationTrace]]:
        """
        Simulate a single play.

        Args:
            play: Play definition
            scenario: Scenario (field, defense, etc.)
            offensive_players: Map of offensive players
            defensive_players: Map of defensive players
            mode: EVAL (auto-select target) or POLICY (use target_override)
            target_override: Forced target role (for RL training)
            record_trace: Whether to record full trace

        Returns:
            (PlayOutcome, Optional[SimulationTrace])
        """
        # Validate inputs
        validate_all_before_sim(play, scenario, offensive_players, defensive_players)

        # Initialize
        field = FieldCoordinates(scenario.field)
        coverage = CoverageAssignment(
            scenario.defense_call,
            scenario.defender_start_positions,
            list(play.qb_plan.progression_roles)
        )

        # Get players
        qb_player = offensive_players[Role.QB]
        qb_slot = next(s for s in play.formation.slots if s.role == Role.QB)

        # Initial positions
        off_positions = {
            slot.role: (
                slot.position.x + self.rng.normal(0, scenario.randomness.position_jitter_yards),
                slot.position.y + self.rng.normal(0, scenario.randomness.position_jitter_yards)
            )
            for slot in play.formation.slots
        }

        def_positions = {
            role: (
                pos.x + self.rng.normal(0, scenario.randomness.position_jitter_yards),
                pos.y + self.rng.normal(0, scenario.randomness.position_jitter_yards)
            )
            for role, pos in scenario.defender_start_positions.items()
        }

        # Simulation loop
        max_time = play.qb_plan.max_time_to_throw_ms + 500.0  # buffer
        current_time = 0.0
        states = [] if record_trace else None
        events = [] if record_trace else None

        # Rush timing
        rush_start_time = scenario.rules.rush.rush_delay_seconds * 1000.0
        rusher_started = False

        throw_time = None
        throw_target = None
        throw_decision_made = False

        # Simulation loop
        while current_time <= max_time:
            # Update offensive positions (run routes)
            for role, route in play.assignments.items():
                if route is not None and role in off_positions:
                    player = offensive_players[role]
                    start_pos = next(s.position for s in play.formation.slots if s.role == role)
                    new_pos = interpolate_route(
                        route,
                        start_pos,
                        current_time,
                        player.attributes.speed,
                        self.timestep_ms
                    )
                    off_positions[role] = new_pos

            # Update defensive positions
            receiver_positions = {
                role: pos for role, pos in off_positions.items()
                if role in play.qb_plan.progression_roles
            }

            for def_role, def_pos in def_positions.items():
                if def_role == Role.RUSHER:
                    # Handle rusher separately
                    # Only rush if defense call mandates it
                    if current_time >= rush_start_time and scenario.defense_call.rushers_count > 0:
                        rusher_started = True
                        # Move toward QB
                        qb_pos = off_positions[Role.QB]
                        rusher_player = defensive_players[Role.RUSHER]

                        dx = qb_pos[0] - def_pos[0]
                        dy = qb_pos[1] - def_pos[1]
                        dist = np.sqrt(dx*dx + dy*dy)

                        if dist > 0.1:
                            speed_factor = rusher_player.attributes.pass_rush / 100.0
                            max_speed = 6.0 + speed_factor * 4.0
                            dt_s = self.timestep_ms / 1000.0
                            move_dist = max_speed * dt_s

                            ratio = min(1.0, move_dist / dist)
                            new_x = def_pos[0] + ratio * dx
                            new_y = def_pos[1] + ratio * dy
                            def_positions[def_role] = (new_x, new_y)
                else:
                    # Coverage defender
                    assignment = coverage.assignments.get(def_role)
                    defender_player = defensive_players[def_role]
                    new_pos = update_defender_position(
                        def_role,
                        def_pos,
                        assignment,
                        scenario.defense_call.type,
                        receiver_positions,
                        defender_player.attributes,
                        self.timestep_ms,
                        self.rng
                    )
                    def_positions[def_role] = new_pos

            # Check for sack
            if rusher_started and not throw_decision_made:
                qb_pos = off_positions[Role.QB]
                rusher_pos = def_positions.get(Role.RUSHER)
                if rusher_pos:
                    dx = qb_pos[0] - rusher_pos[0]
                    dy = qb_pos[1] - rusher_pos[1]
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < 2.0:
                        # Sack!
                        outcome = PlayOutcome(
                            outcome=OutcomeType.SACK,
                            yards_gained=-3.0,
                            time_to_throw_ms=current_time,
                            failure_modes=[FailureMode.SACK_BEFORE_THROW]
                        )

                        trace = None
                        if record_trace:
                            trace = SimulationTrace(
                                play_id=play.id,
                                scenario_id=scenario.id,
                                outcome=outcome,
                                states=states,
                                events=events
                            )

                        return outcome, trace

            # QB decision logic
            if not throw_decision_made:
                # Check if time to throw
                decision_time = qb_player.attributes.decision_latency_ms

                if current_time >= decision_time:
                    # Evaluate targets
                    if mode == SimMode.POLICY and target_override:
                        # Use override from RL policy
                        throw_target = target_override
                        throw_time = current_time
                        throw_decision_made = True
                    else:
                        # EVAL mode: evaluate progression
                        best_target, best_score = self._evaluate_progression(
                            play.qb_plan.progression_roles,
                            off_positions,
                            def_positions,
                            qb_player,
                            offensive_players,
                            defensive_players,
                            scenario,
                            current_time
                        )

                        # Threshold to throw
                        if best_score > 0.3 or current_time >= play.qb_plan.max_time_to_throw_ms * 0.9:
                            throw_target = best_target
                            throw_time = current_time
                            throw_decision_made = True

            # If throw made, simulate outcome
            if throw_decision_made and throw_time is not None:
                outcome = self._simulate_throw(
                    throw_target,
                    throw_time,
                    off_positions,
                    def_positions,
                    qb_player,
                    offensive_players,
                    defensive_players,
                    scenario,
                    play,
                    field
                )

                trace = None
                if record_trace:
                    trace = SimulationTrace(
                        play_id=play.id,
                        scenario_id=scenario.id,
                        outcome=outcome,
                        states=states,
                        events=events
                    )

                return outcome, trace

            # Record state
            if record_trace:
                states.append(SimulationState(
                    time_ms=current_time,
                    offensive_positions=off_positions.copy(),
                    defensive_positions=def_positions.copy(),
                    qb_position=off_positions[Role.QB],
                    rusher_position=def_positions.get(Role.RUSHER)
                ))

            current_time += self.timestep_ms

        # Timeout - no throw made
        outcome = PlayOutcome(
            outcome=OutcomeType.SACK,
            yards_gained=-2.0,
            time_to_throw_ms=current_time,
            failure_modes=[FailureMode.SACK_BEFORE_THROW, FailureMode.LATE_THROW]
        )

        trace = None
        if record_trace:
            trace = SimulationTrace(
                play_id=play.id,
                scenario_id=scenario.id,
                outcome=outcome,
                states=states,
                events=events
            )

        return outcome, trace

    def _evaluate_progression(
        self,
        progression: List[Role],
        off_positions: Dict[Role, Tuple[float, float]],
        def_positions: Dict[Role, Tuple[float, float]],
        qb: Player,
        offensive_players: Dict[Role, Player],
        defensive_players: Dict[Role, Player],
        scenario: Scenario,
        current_time: float
    ) -> Tuple[Optional[Role], float]:
        """Evaluate targets in progression and return best."""
        best_target = None
        best_score = -1.0

        qb_pos = off_positions[Role.QB]

        for target_role in progression:
            if target_role not in off_positions:
                continue

            target_pos = off_positions[target_role]
            all_def_positions = list(def_positions.values())
            separation = compute_separation(target_pos, all_def_positions)

            # Simple score: separation weighted by distance
            distance = compute_throw_distance(qb_pos, target_pos)

            # Prefer closer targets with good separation
            score = separation - (distance * 0.1)

            if score > best_score:
                best_score = score
                best_target = target_role

        return best_target, best_score

    def _simulate_throw(
        self,
        target_role: Role,
        throw_time: float,
        off_positions: Dict[Role, Tuple[float, float]],
        def_positions: Dict[Role, Tuple[float, float]],
        qb: Player,
        offensive_players: Dict[Role, Player],
        defensive_players: Dict[Role, Player],
        scenario: Scenario,
        play: Play,
        field: FieldCoordinates
    ) -> PlayOutcome:
        """Simulate throw outcome."""
        qb_pos = off_positions[Role.QB]
        target_pos = off_positions[target_role]

        # Throw parameters
        distance = compute_throw_distance(qb_pos, target_pos)

        # Flight time (simplified: ~1 yard per 50ms)
        flight_time_ms = distance * 50.0
        catch_time = throw_time + qb.attributes.release_time_ms + flight_time_ms

        # Separation at throw and catch
        all_def_pos = list(def_positions.values())
        sep_throw = compute_separation(target_pos, all_def_pos)

        # Project catch position (assume receiver keeps moving on route)
        # Simplified: use current position
        catch_pos = target_pos
        sep_catch = compute_separation(catch_pos, all_def_pos)

        # Completion probability
        qb_moving = False  # Simplified for MVP
        comp_prob, failure_modes = compute_completion_probability(
            qb.attributes,
            distance,
            sep_throw,
            sep_catch,
            scenario.defense_call.shell,
            qb_moving,
            throw_time,
            play.qb_plan.max_time_to_throw_ms
        )

        # Determine outcome
        receiver = offensive_players[target_role]

        # Find nearest defender
        nearest_def_dist = sep_catch
        nearest_defender = None
        for def_role, def_pos in def_positions.items():
            dist = np.sqrt((catch_pos[0] - def_pos[0])**2 + (catch_pos[1] - def_pos[1])**2)
            if dist < nearest_def_dist:
                nearest_def_dist = dist
                nearest_defender = def_role

        defender_attrs = defensive_players[nearest_defender].attributes if nearest_defender else defensive_players[list(defensive_players.keys())[0]].attributes

        outcome_type = determine_outcome(
            comp_prob,
            receiver.attributes,
            defender_attrs,
            sep_catch,
            self.rng
        )

        # Compute yards
        if outcome_type == OutcomeType.COMPLETE:
            base_yards = catch_pos[0]  # x coordinate from LOS
            yac = compute_yards_after_catch(
                catch_pos,
                receiver.attributes,
                nearest_def_dist,
                self.rng
            )
            total_yards = base_yards + yac
        elif outcome_type == OutcomeType.INTERCEPT:
            total_yards = -5.0  # Penalty
        else:  # INCOMPLETE
            total_yards = 0.0

        return PlayOutcome(
            outcome=outcome_type,
            yards_gained=total_yards,
            time_to_throw_ms=throw_time,
            target_role=target_role,
            completion_probability=comp_prob,
            separation_at_throw=sep_throw,
            separation_at_catch=sep_catch,
            failure_modes=failure_modes
        )
