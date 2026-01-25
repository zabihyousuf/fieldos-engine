"""Visualization utilities for FieldOS Engine."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..rl.evaluate import EvaluationReport
from ..sim.engine import SimulationTrace, SimulationState
from ..core.models import Role, Play, Scenario, CoverageType
import matplotlib.animation as animation
import matplotlib.patches as patches


logger = logging.getLogger("fieldos_engine.viz")

# Color scheme
OFFENSE_COLOR = '#2563eb'  # Blue
DEFENSE_COLOR = '#dc2626'  # Red
FIELD_COLOR = '#22c55e'    # Green
LINE_COLOR = '#ffffff'     # White
ROUTE_COLOR = '#60a5fa'    # Light blue
COVERAGE_COLOR = '#f87171' # Light red
LOS_COLOR = '#fbbf24'      # Yellow/gold


def plot_policy_performance(report: EvaluationReport, output_path: str) -> str:
    """
    Generate a bar chart of play performance per bucket.

    Args:
        report: Evaluation report from evaluate_policy_detailed
        output_path: Path to save the PNG file

    Returns:
        Absolute path to the saved file
    """

    # Extract data
    buckets = [bp.bucket for bp in report.bucket_performance]

    # Collect all plays seen across all buckets
    all_plays = set()
    for bp in report.bucket_performance:
        all_plays.update(bp.play_performance.keys())

    sorted_plays = sorted(list(all_plays))

    # Prepare data for grouped bar chart
    x = np.arange(len(buckets))
    width = 0.8 / len(sorted_plays)  # Distribute width among plays

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, play_id in enumerate(sorted_plays):
        # Get mean reward for this play in each bucket
        scores = []
        for bp in report.bucket_performance:
            scores.append(bp.play_performance.get(play_id, 0.0))

        offset = width * i
        rects = ax.bar(x + offset, scores, width, label=play_id)

    # Add labels
    ax.set_ylabel('Mean Reward (Yards + Bonuses)')
    ax.set_title(f'Play Performance by Situation - {report.policy_name}')
    ax.set_xticks(x + width * (len(sorted_plays) - 1) / 2)
    ax.set_xticklabels(buckets, rotation=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    # Save
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p))
    plt.close()

    logger.info(f"Saved performance plot to {p}")
    return str(p)


def animate_play(
    trace: SimulationTrace,
    output_path: str,
    play: Optional[Play] = None,
    scenario: Optional[Scenario] = None,
    coverage_assignments: Optional[Dict[Role, Any]] = None
) -> str:
    """
    Generate an animated GIF of a play simulation.

    Args:
        trace: SimulationTrace object
        output_path: Path to save the GIF file
        play: Optional Play object for route visualization
        scenario: Optional Scenario object for defense info
        coverage_assignments: Optional coverage assignments dict

    Returns:
        Absolute path to the saved file
    """
    if not trace.states:
        logger.warning("No states in trace, cannot animate.")
        return ""

    frames = len(trace.states)

    # Field dimensions based on typical flag football
    # Coordinate system: X = downfield (positive toward endzone), Y = lateral (positive = right)
    field_half_width = 20.0  # -20 to +20 yards lateral
    field_length_behind_los = 10.0  # 10 yards behind LOS for QB
    field_length_ahead_los = 40.0   # 40 yards downfield to show routes

    # Figure setup - landscape orientation showing field from above
    fig, ax = plt.subplots(figsize=(14, 10))

    # Set axis limits: X = lateral (-20 to +20), Y = downfield (-10 to +40)
    ax.set_xlim(-field_half_width, field_half_width)
    ax.set_ylim(-field_length_behind_los, field_length_ahead_los)

    # Draw field
    _draw_field(ax, field_half_width, field_length_behind_los, field_length_ahead_los)

    # Get initial positions for route drawing
    initial_state = trace.states[0]

    # Draw routes if play provided
    route_lines = {}
    if play:
        route_lines = _draw_routes(ax, play, initial_state)

    # Draw coverage assignments if provided
    coverage_lines = []
    if coverage_assignments and scenario:
        coverage_lines = _draw_coverage_assignments(
            ax, coverage_assignments, initial_state, scenario
        )

    # Initialize player markers and labels
    off_markers = {}
    def_markers = {}
    off_labels = {}
    def_labels = {}

    # Create markers for each player
    for role in initial_state.offensive_positions.keys():
        marker, = ax.plot([], [], 'o', color=OFFENSE_COLOR, markersize=15,
                          markeredgecolor='white', markeredgewidth=2, zorder=10)
        label = ax.text(0, 0, _get_role_abbrev(role), fontsize=8, fontweight='bold',
                       ha='center', va='center', color='white', zorder=11)
        off_markers[role] = marker
        off_labels[role] = label

    for role in initial_state.defensive_positions.keys():
        marker, = ax.plot([], [], 'o', color=DEFENSE_COLOR, markersize=15,
                          markeredgecolor='white', markeredgewidth=2, zorder=10)
        label = ax.text(0, 0, _get_role_abbrev(role), fontsize=8, fontweight='bold',
                       ha='center', va='center', color='white', zorder=11)
        def_markers[role] = marker
        def_labels[role] = label

    # Ball marker
    ball_marker, = ax.plot([], [], 'o', color='#8B4513', markersize=10,
                           markeredgecolor='white', markeredgewidth=1.5, zorder=15)

    # Title and info text
    title_text = ax.set_title('', fontsize=14, fontweight='bold', pad=10)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Info text for outcome
    outcome_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, fontsize=11,
                           verticalalignment='top', horizontalalignment='right',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Build title
    play_name = play.name if play else trace.play_id
    scenario_name = scenario.name if scenario else trace.scenario_id
    coverage_info = ""
    if scenario:
        coverage_info = f" ({scenario.defense_call.type.value} {scenario.defense_call.shell.value})"

    def init():
        for marker in off_markers.values():
            marker.set_data([], [])
        for marker in def_markers.values():
            marker.set_data([], [])
        for label in off_labels.values():
            label.set_position((0, 0))
            label.set_visible(False)
        for label in def_labels.values():
            label.set_position((0, 0))
            label.set_visible(False)
        ball_marker.set_data([], [])
        time_text.set_text('')
        outcome_text.set_text('')
        title_text.set_text(f'{play_name} vs {scenario_name}{coverage_info}')
        return list(off_markers.values()) + list(def_markers.values()) + [ball_marker, time_text, outcome_text, title_text]

    def update(frame):
        state: SimulationState = trace.states[frame]

        # Update offensive positions
        # Coordinates: sim (x=downfield, y=lateral) -> plot (x=lateral, y=downfield)
        for role, pos in state.offensive_positions.items():
            if role in off_markers:
                plot_x, plot_y = pos[1], pos[0]  # Swap: lateral->X, downfield->Y
                off_markers[role].set_data([plot_x], [plot_y])
                off_labels[role].set_position((plot_x, plot_y))
                off_labels[role].set_visible(True)

        # Update defensive positions
        for role, pos in state.defensive_positions.items():
            if role in def_markers:
                plot_x, plot_y = pos[1], pos[0]  # Swap: lateral->X, downfield->Y
                def_markers[role].set_data([plot_x], [plot_y])
                def_labels[role].set_position((plot_x, plot_y))
                def_labels[role].set_visible(True)

        # Update ball position if in flight
        if hasattr(state, 'ball_position') and state.ball_position is not None and state.ball_in_flight:
            ball_x, ball_y = state.ball_position[1], state.ball_position[0]  # Swap coords
            ball_marker.set_data([ball_x], [ball_y])
        else:
            ball_marker.set_data([], [])

        # Update time text
        time_text.set_text(f'Time: {state.time_ms/1000:.2f}s')

        # Show outcome on last frame
        if frame == frames - 1:
            outcome = trace.outcome
            outcome_text.set_text(
                f'Result: {outcome.outcome.value}\n'
                f'Yards: {outcome.yards_gained:.1f}\n'
                f'Target: {outcome.target_role.value if outcome.target_role else "N/A"}'
            )

        return list(off_markers.values()) + list(def_markers.values()) + [ball_marker, time_text, outcome_text, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=100
    )

    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Save as GIF
    try:
        ani.save(str(p), writer='pillow', fps=10)
        logger.info(f"Saved animation to {p}")
    except Exception as e:
        logger.error(f"Failed to save animation: {e}")
        plt.close()
        return ""

    plt.close()
    return str(p)


def _draw_field(ax, half_width: float, behind_los: float, ahead_los: float):
    """Draw the football field."""
    # Field background
    field_rect = patches.Rectangle(
        (-half_width, -behind_los),
        half_width * 2,
        behind_los + ahead_los,
        facecolor=FIELD_COLOR,
        alpha=0.3,
        zorder=0
    )
    ax.add_patch(field_rect)

    # Line of scrimmage at y=0
    ax.axhline(y=0, color=LOS_COLOR, linewidth=3, linestyle='-', zorder=2, label='LOS')
    ax.text(-half_width + 0.5, 0.5, 'LOS', fontsize=10, color=LOS_COLOR, fontweight='bold')

    # Yard lines every 5 yards
    for y in range(-5, int(ahead_los) + 5, 5):
        if y == 0:
            continue  # Skip LOS
        alpha = 0.8 if y % 10 == 0 else 0.4
        ax.axhline(y=y, color=LINE_COLOR, linewidth=1 if y % 10 == 0 else 0.5,
                   linestyle='-', alpha=alpha, zorder=1)
        if y > 0 and y % 10 == 0:
            ax.text(-half_width + 0.5, y + 0.3, f'{y}', fontsize=8, color='gray')

    # Hash marks (simplified)
    for y in range(0, int(ahead_los) + 5, 5):
        ax.plot([-3, -2.5], [y, y], 'w-', linewidth=1, alpha=0.5, zorder=1)
        ax.plot([2.5, 3], [y, y], 'w-', linewidth=1, alpha=0.5, zorder=1)

    # Sidelines
    ax.axvline(x=-half_width, color=LINE_COLOR, linewidth=2, zorder=1)
    ax.axvline(x=half_width, color=LINE_COLOR, linewidth=2, zorder=1)

    # Labels
    ax.set_xlabel('Lateral Position (yards)', fontsize=11)
    ax.set_ylabel('Downfield (yards from LOS)', fontsize=11)

    # Add legend for teams
    off_patch = patches.Patch(color=OFFENSE_COLOR, label='Offense')
    def_patch = patches.Patch(color=DEFENSE_COLOR, label='Defense')
    ax.legend(handles=[off_patch, def_patch], loc='lower right', fontsize=10)

    ax.set_aspect('equal')
    ax.grid(False)


def _draw_routes(ax, play: Play, initial_state: SimulationState) -> Dict[Role, Any]:
    """Draw receiver routes on the field."""
    route_lines = {}

    for role, route in play.assignments.items():
        if route is None:
            continue
        if role not in initial_state.offensive_positions:
            continue

        start_pos = initial_state.offensive_positions[role]
        start_x, start_y = start_pos[1], start_pos[0]  # Convert to plot coords

        # Build route points
        route_xs = [start_x]
        route_ys = [start_y]

        for bp in route.breakpoints:
            # Breakpoints are relative to start position in sim coords
            abs_x = start_pos[0] + bp.x_yards  # Sim X = downfield
            abs_y = start_pos[1] + bp.y_yards  # Sim Y = lateral
            route_xs.append(abs_y)  # Plot X = lateral
            route_ys.append(abs_x)  # Plot Y = downfield

        # Draw route line
        line, = ax.plot(route_xs, route_ys, color=ROUTE_COLOR, linewidth=2,
                        linestyle='--', alpha=0.7, zorder=3)

        # Add arrow at end
        if len(route_xs) >= 2:
            ax.annotate('', xy=(route_xs[-1], route_ys[-1]),
                       xytext=(route_xs[-2], route_ys[-2]),
                       arrowprops=dict(arrowstyle='->', color=ROUTE_COLOR, lw=2),
                       zorder=3)

        route_lines[role] = line

    return route_lines


def _draw_coverage_assignments(
    ax,
    coverage_assignments: Dict[Role, Any],
    initial_state: SimulationState,
    scenario: Scenario
) -> List:
    """Draw coverage assignment lines."""
    lines = []

    for def_role, assignment in coverage_assignments.items():
        if def_role not in initial_state.defensive_positions:
            continue

        def_pos = initial_state.defensive_positions[def_role]
        def_x, def_y = def_pos[1], def_pos[0]  # Convert to plot coords

        if assignment is None:
            # Rusher - draw arrow toward QB
            if Role.QB in initial_state.offensive_positions:
                qb_pos = initial_state.offensive_positions[Role.QB]
                qb_x, qb_y = qb_pos[1], qb_pos[0]
                line, = ax.plot([def_x, qb_x], [def_y, qb_y],
                               color='orange', linewidth=1.5, linestyle=':',
                               alpha=0.5, zorder=2)
                lines.append(line)

        elif isinstance(assignment, Role):
            # Man coverage - draw line to receiver
            if assignment in initial_state.offensive_positions:
                rec_pos = initial_state.offensive_positions[assignment]
                rec_x, rec_y = rec_pos[1], rec_pos[0]
                line, = ax.plot([def_x, rec_x], [def_y, rec_y],
                               color=COVERAGE_COLOR, linewidth=1.5, linestyle=':',
                               alpha=0.5, zorder=2)
                lines.append(line)

        elif isinstance(assignment, str):
            # Zone coverage - draw to zone center
            if "deep" in assignment:
                zone_y = 20.0  # Deep zone depth
                if "0" in assignment or "left" in assignment:
                    zone_x = -10.0
                elif "1" in assignment or "right" in assignment:
                    zone_x = 10.0
                else:
                    zone_x = 0.0
            else:  # underneath
                zone_y = 5.0
                zone_x = 0.0

            # Draw zone indicator
            circle = patches.Circle((zone_x, zone_y), 3, fill=False,
                                    edgecolor=COVERAGE_COLOR, linewidth=1.5,
                                    linestyle='--', alpha=0.4, zorder=2)
            ax.add_patch(circle)
            line, = ax.plot([def_x, zone_x], [def_y, zone_y],
                           color=COVERAGE_COLOR, linewidth=1, linestyle=':',
                           alpha=0.3, zorder=2)
            lines.append(line)

    return lines


def _get_role_abbrev(role: Role) -> str:
    """Get short abbreviation for role."""
    abbrevs = {
        # Offense
        Role.QB: 'QB',
        Role.CENTER: 'C',
        Role.WR1: 'W1',
        Role.WR2: 'W2',
        Role.WR3: 'W3',
        # New defense (D1-D5)
        Role.D1: 'D1',
        Role.D2: 'D2',
        Role.D3: 'D3',
        Role.D4: 'D4',
        Role.D5: 'D5',
        # Legacy defense
        Role.RUSHER: 'R',
        Role.CB1: 'C1',
        Role.CB2: 'C2',
        Role.SAFETY: 'S',
        Role.LB: 'LB'
    }
    return abbrevs.get(role, str(role.value)[:2])


def visualize_play_static(
    play: Play,
    scenario: Scenario,
    output_path: str,
    coverage_assignments: Optional[Dict[Role, Any]] = None
) -> str:
    """
    Generate a static image of a play diagram showing:
    - Offensive formation and routes
    - Defensive alignment and coverage assignments

    Args:
        play: Play definition
        scenario: Scenario with defensive setup
        output_path: Path to save PNG
        coverage_assignments: Optional coverage assignments

    Returns:
        Absolute path to saved file
    """
    # Field dimensions
    field_half_width = 20.0
    field_length_behind_los = 10.0
    field_length_ahead_los = 30.0

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-field_half_width, field_half_width)
    ax.set_ylim(-field_length_behind_los, field_length_ahead_los)

    # Draw field
    _draw_field(ax, field_half_width, field_length_behind_los, field_length_ahead_los)

    # Plot offensive formation
    for slot in play.formation.slots:
        pos = slot.position
        plot_x, plot_y = pos.y, pos.x  # Convert: lateral->X, downfield->Y

        ax.plot(plot_x, plot_y, 'o', color=OFFENSE_COLOR, markersize=20,
                markeredgecolor='white', markeredgewidth=2, zorder=10)
        ax.text(plot_x, plot_y, _get_role_abbrev(slot.role), fontsize=9,
                fontweight='bold', ha='center', va='center', color='white', zorder=11)

        # Draw route if exists
        route = play.assignments.get(slot.role)
        if route:
            route_xs = [plot_x]
            route_ys = [plot_y]
            for bp in route.breakpoints:
                route_xs.append(pos.y + bp.y_yards)
                route_ys.append(pos.x + bp.x_yards)

            ax.plot(route_xs, route_ys, color=ROUTE_COLOR, linewidth=2.5,
                    linestyle='--', alpha=0.8, zorder=5)
            if len(route_xs) >= 2:
                ax.annotate('', xy=(route_xs[-1], route_ys[-1]),
                           xytext=(route_xs[-2], route_ys[-2]),
                           arrowprops=dict(arrowstyle='->', color=ROUTE_COLOR, lw=2.5),
                           zorder=5)

    # Plot defensive positions
    for def_role, pos in scenario.defender_start_positions.items():
        plot_x, plot_y = pos.y, pos.x  # Convert: lateral->X, downfield->Y

        ax.plot(plot_x, plot_y, 'o', color=DEFENSE_COLOR, markersize=20,
                markeredgecolor='white', markeredgewidth=2, zorder=10)
        ax.text(plot_x, plot_y, _get_role_abbrev(def_role), fontsize=9,
                fontweight='bold', ha='center', va='center', color='white', zorder=11)

        # Draw coverage assignment if provided
        if coverage_assignments and def_role in coverage_assignments:
            assignment = coverage_assignments[def_role]
            if isinstance(assignment, Role):
                # Man coverage - find receiver position
                for slot in play.formation.slots:
                    if slot.role == assignment:
                        rec_x, rec_y = slot.position.y, slot.position.x
                        ax.plot([plot_x, rec_x], [plot_y, rec_y],
                               color=COVERAGE_COLOR, linewidth=1.5, linestyle=':',
                               alpha=0.6, zorder=4)
                        break

    # Title
    coverage_info = f"{scenario.defense_call.type.value} {scenario.defense_call.shell.value}"
    ax.set_title(f'{play.name} vs {scenario.name}\n({coverage_info})',
                 fontsize=14, fontweight='bold', pad=10)

    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved static play diagram to {p}")
    return str(p)
