"""Play outcome determination logic."""

import numpy as np
from typing import Tuple, List, Optional
from ..core.models import (
    OutcomeType, FailureMode, PlayerAttributes, CoverageShell, Role
)
from .motion import get_throw_category


def compute_completion_probability(
    qb_attrs: PlayerAttributes,
    throw_distance: float,
    separation_at_throw: float,
    separation_at_catch: float,
    coverage_shell: CoverageShell,
    qb_is_moving: bool,
    time_to_throw_ms: float,
    max_time_to_throw_ms: float
) -> Tuple[float, List[FailureMode]]:
    """
    Compute probability of completion and identify failure modes.

    Args:
        qb_attrs: QB attributes
        throw_distance: Distance of throw in yards
        separation_at_throw: Receiver separation when ball released
        separation_at_catch: Receiver separation at catch point
        coverage_shell: Defensive coverage shell
        qb_is_moving: Whether QB is scrambling/moving
        time_to_throw_ms: Actual time to throw
        max_time_to_throw_ms: Max allowed time

    Returns:
        (probability, failure_modes)
    """
    failure_modes = []

    # Base accuracy from QB skill
    throw_category = get_throw_category(throw_distance)
    if throw_category == "short":
        base_acc = qb_attrs.short_acc / 100.0
    elif throw_category == "mid":
        base_acc = qb_attrs.mid_acc / 100.0
    else:
        base_acc = qb_attrs.deep_acc / 100.0

    prob = base_acc

    # Penalty for throwing on run
    if qb_is_moving:
        prob *= 0.85
        if prob < 0.5:
            failure_modes.append(FailureMode.LOW_QB_ACCURACY)

    # Separation penalty
    if separation_at_throw < 2.0:
        failure_modes.append(FailureMode.TIGHT_WINDOW)
        sep_factor = max(0.3, separation_at_throw / 5.0)
        prob *= sep_factor

    if separation_at_catch < 1.5:
        failure_modes.append(FailureMode.TIGHT_WINDOW)
        prob *= 0.7

    # Coverage shell adjustment
    if coverage_shell in {CoverageShell.COVER2, CoverageShell.COVER3}:
        # Zone coverage provides help over the top
        if throw_category == "deep":
            prob *= 0.85

    # Late throw penalty
    time_pressure = time_to_throw_ms / max_time_to_throw_ms
    if time_pressure > 0.9:
        failure_modes.append(FailureMode.LATE_THROW)
        prob *= 0.9

    # Route timing (simplified: assume ideal timing for MVP)
    # Could add more sophisticated timing checks here

    # Clamp probability
    prob = np.clip(prob, 0.05, 0.95)

    return prob, failure_modes


def determine_outcome(
    completion_prob: float,
    receiver_attrs: PlayerAttributes,
    defender_attrs: PlayerAttributes,
    separation_at_catch: float,
    rng: np.random.Generator
) -> OutcomeType:
    """
    Determine actual outcome based on probabilities.

    Args:
        completion_prob: Probability of completion
        receiver_attrs: Receiver attributes
        defender_attrs: Defender attributes
        separation_at_catch: Separation at catch point
        rng: Random number generator

    Returns:
        OutcomeType
    """
    # Roll for completion
    roll = rng.random()

    if roll < completion_prob:
        # Success factors
        receiver_hands = receiver_attrs.hands / 100.0
        catch_roll = rng.random()

        if catch_roll < receiver_hands * 0.95 + 0.05:
            return OutcomeType.COMPLETE
        else:
            return OutcomeType.INCOMPLETE
    else:
        # Failed completion - check for INT
        if separation_at_catch < 1.0:
            defender_ball_skills = defender_attrs.ball_skills / 100.0
            int_prob = (1.0 - separation_at_catch) * defender_ball_skills * 0.15

            int_roll = rng.random()
            if int_roll < int_prob:
                return OutcomeType.INTERCEPT

        return OutcomeType.INCOMPLETE


def compute_yards_after_catch(
    catch_position: Tuple[float, float],
    receiver_attrs: PlayerAttributes,
    defender_distance: float,
    rng: np.random.Generator
) -> float:
    """
    Compute YAC (simplified).

    Args:
        catch_position: (x, y) of catch
        receiver_attrs: Receiver attributes
        defender_distance: Distance to nearest defender
        rng: Random generator

    Returns:
        Additional yards gained after catch
    """
    # Simple YAC model
    if defender_distance < 1.0:
        # Immediate tackle
        return 0.0

    # Base YAC on receiver speed and separation
    speed_factor = receiver_attrs.speed / 100.0
    agility_factor = receiver_attrs.agility / 100.0

    base_yac = speed_factor * defender_distance * 0.5
    agility_bonus = agility_factor * rng.random() * 2.0

    yac = base_yac + agility_bonus

    # Cap YAC
    return min(yac, 15.0)


def check_sack(
    rusher_distance_to_qb: float,
    rusher_attrs: PlayerAttributes,
    qb_attrs: PlayerAttributes,
    time_since_rush_start_ms: float
) -> bool:
    """
    Check if rusher gets sack.

    Args:
        rusher_distance_to_qb: Current distance to QB
        rusher_attrs: Rusher attributes
        qb_attrs: QB attributes
        time_since_rush_start_ms: Time since rush started

    Returns:
        True if sack occurs
    """
    # Sack if rusher within sack radius
    sack_radius = 2.0  # yards

    if rusher_distance_to_qb <= sack_radius:
        # Probability based on rush skill vs QB mobility
        rush_skill = rusher_attrs.pass_rush / 100.0
        qb_mobility = (qb_attrs.speed + qb_attrs.agility) / 200.0

        # Simple: higher rush skill = higher sack chance
        sack_prob = rush_skill * 0.8 + 0.1 - qb_mobility * 0.3

        # For MVP, just use distance threshold
        return True

    return False
