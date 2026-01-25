"""
Play Generation System for RL Discovery.

This module generates new play combinations by:
1. Creating random routes from a library of route types
2. Combining routes into formations
3. Generating QB progressions
4. Creating new formations (spread, bunch, trips, etc.)

The RL system can then test these generated plays to discover
which combinations work best against different defenses.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

from ..core.models import (
    Play, Formation, FormationSlot, Route, RouteBreakpoint,
    QBPlan, Role, Point2D
)


# =============================================================================
# Route Library - Base routes that can be combined
# =============================================================================

ROUTE_LIBRARY = {
    # Quick routes (0-5 yards)
    "quick_out": [
        RouteBreakpoint(x_yards=3.0, y_yards=0.0, time_ms=500),
        RouteBreakpoint(x_yards=3.0, y_yards=5.0, time_ms=900)
    ],
    "quick_in": [
        RouteBreakpoint(x_yards=3.0, y_yards=0.0, time_ms=500),
        RouteBreakpoint(x_yards=3.0, y_yards=-5.0, time_ms=900)
    ],
    "hitch": [
        RouteBreakpoint(x_yards=5.0, y_yards=0.0, time_ms=800),
        RouteBreakpoint(x_yards=4.0, y_yards=0.0, time_ms=1000)
    ],
    "slant": [
        RouteBreakpoint(x_yards=2.0, y_yards=0.0, time_ms=350),
        RouteBreakpoint(x_yards=8.0, y_yards=-6.0, time_ms=1200)
    ],
    "flat": [
        RouteBreakpoint(x_yards=1.0, y_yards=6.0, time_ms=600)
    ],
    "arrow": [
        RouteBreakpoint(x_yards=3.0, y_yards=-5.0, time_ms=700)
    ],

    # Medium routes (5-12 yards)
    "out": [
        RouteBreakpoint(x_yards=8.0, y_yards=0.0, time_ms=1300),
        RouteBreakpoint(x_yards=8.0, y_yards=6.0, time_ms=1700)
    ],
    "in": [
        RouteBreakpoint(x_yards=8.0, y_yards=0.0, time_ms=1300),
        RouteBreakpoint(x_yards=8.0, y_yards=-6.0, time_ms=1700)
    ],
    "curl": [
        RouteBreakpoint(x_yards=10.0, y_yards=0.0, time_ms=1600),
        RouteBreakpoint(x_yards=8.0, y_yards=0.0, time_ms=1900)
    ],
    "dig": [
        RouteBreakpoint(x_yards=10.0, y_yards=0.0, time_ms=1600),
        RouteBreakpoint(x_yards=10.0, y_yards=-8.0, time_ms=2200)
    ],
    "comeback": [
        RouteBreakpoint(x_yards=12.0, y_yards=0.0, time_ms=1900),
        RouteBreakpoint(x_yards=10.0, y_yards=3.0, time_ms=2400)
    ],
    "stick": [
        RouteBreakpoint(x_yards=5.0, y_yards=0.0, time_ms=800),
        RouteBreakpoint(x_yards=6.0, y_yards=2.0, time_ms=1100)
    ],

    # Deep routes (12+ yards)
    "go": [
        RouteBreakpoint(x_yards=10.0, y_yards=0.0, time_ms=1600),
        RouteBreakpoint(x_yards=25.0, y_yards=0.0, time_ms=4000)
    ],
    "post": [
        RouteBreakpoint(x_yards=10.0, y_yards=0.0, time_ms=1600),
        RouteBreakpoint(x_yards=18.0, y_yards=-6.0, time_ms=2800)
    ],
    "corner": [
        RouteBreakpoint(x_yards=10.0, y_yards=0.0, time_ms=1600),
        RouteBreakpoint(x_yards=16.0, y_yards=8.0, time_ms=2600)
    ],
    "wheel": [
        RouteBreakpoint(x_yards=0.0, y_yards=-3.0, time_ms=400),
        RouteBreakpoint(x_yards=5.0, y_yards=-5.0, time_ms=1100),
        RouteBreakpoint(x_yards=18.0, y_yards=-5.0, time_ms=3000)
    ],
    "seam": [
        RouteBreakpoint(x_yards=8.0, y_yards=0.0, time_ms=1300),
        RouteBreakpoint(x_yards=20.0, y_yards=0.0, time_ms=3200)
    ],

    # Center-specific routes (shorter, quicker)
    "center_drag": [
        RouteBreakpoint(x_yards=3.0, y_yards=0.0, time_ms=500),
        RouteBreakpoint(x_yards=5.0, y_yards=-6.0, time_ms=1100)
    ],
    "center_arrow": [
        RouteBreakpoint(x_yards=4.0, y_yards=-4.0, time_ms=800)
    ],
    "center_seam": [
        RouteBreakpoint(x_yards=6.0, y_yards=0.0, time_ms=1000),
        RouteBreakpoint(x_yards=12.0, y_yards=0.0, time_ms=2000)
    ],
    "center_swing": [
        RouteBreakpoint(x_yards=-1.0, y_yards=-4.0, time_ms=500),
        RouteBreakpoint(x_yards=4.0, y_yards=-8.0, time_ms=1200)
    ],
}

# Route categories for smart generation
QUICK_ROUTES = ["quick_out", "quick_in", "hitch", "slant", "flat", "arrow", "stick"]
MEDIUM_ROUTES = ["out", "in", "curl", "dig", "comeback"]
DEEP_ROUTES = ["go", "post", "corner", "wheel", "seam"]
CENTER_ROUTES = ["center_drag", "center_arrow", "center_seam", "center_swing"]


# =============================================================================
# Formation Templates
# =============================================================================

FORMATION_TEMPLATES = {
    "trips_right": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=10.0),
        Role.WR2: Point2D(x=0.0, y=7.0),
        Role.WR3: Point2D(x=0.0, y=4.0),
    },
    "trips_left": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=-10.0),
        Role.WR2: Point2D(x=0.0, y=-7.0),
        Role.WR3: Point2D(x=0.0, y=-4.0),
    },
    "bunch_tight": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=2.0),
        Role.WR2: Point2D(x=0.0, y=1.0),
        Role.WR3: Point2D(x=0.5, y=1.5),
    },
    "bunch_left": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=-2.0),
        Role.WR2: Point2D(x=0.0, y=-1.0),
        Role.WR3: Point2D(x=0.5, y=-1.5),
    },
    "stack_right": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=8.0),
        Role.WR2: Point2D(x=1.0, y=8.0),
        Role.WR3: Point2D(x=0.0, y=4.0),
    },
    "twins_right": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=10.0),
        Role.WR2: Point2D(x=0.0, y=7.0),
        Role.WR3: Point2D(x=0.0, y=-10.0),
    },
    "spread": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=12.0),
        Role.WR2: Point2D(x=0.0, y=6.0),
        Role.WR3: Point2D(x=0.0, y=-12.0),
    },
    "empty_spread": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=14.0),
        Role.WR2: Point2D(x=0.0, y=7.0),
        Role.WR3: Point2D(x=0.0, y=-7.0),
    },
    # TIGHT BUNCH - all receivers touching shoulders next to center
    "tight_bunch_right": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=1.5),
        Role.WR2: Point2D(x=0.0, y=0.75),
        Role.WR3: Point2D(x=0.0, y=0.25),
    },
    "tight_bunch_left": {
        Role.QB: Point2D(x=-5.0, y=0.0),
        Role.CENTER: Point2D(x=0.0, y=0.0),
        Role.WR1: Point2D(x=0.0, y=-1.5),
        Role.WR2: Point2D(x=0.0, y=-0.75),
        Role.WR3: Point2D(x=0.0, y=-0.25),
    },
}


@dataclass
class GeneratedPlay:
    """A generated play with metadata."""
    play: Play
    formation_type: str
    route_types: Dict[Role, str]
    generation_id: str


class PlayGenerator:
    """
    Generates new play combinations for RL exploration.

    The generator creates plays by:
    1. Selecting a formation template
    2. Assigning routes to each receiver
    3. Creating a QB progression based on route timing
    4. Optionally mirroring plays to create left/right variants
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.generation_counter = 0

    def generate_random_play(
        self,
        formation_type: Optional[str] = None,
        route_depth: Optional[str] = None,  # "quick", "medium", "deep", "mixed"
        include_center_route: bool = True
    ) -> GeneratedPlay:
        """
        Generate a random play.

        Args:
            formation_type: Specific formation or None for random
            route_depth: Preferred route depth category
            include_center_route: Whether center runs a route

        Returns:
            GeneratedPlay with the generated play and metadata
        """
        # Select formation
        if formation_type is None:
            formation_type = self.rng.choice(list(FORMATION_TEMPLATES.keys()))

        formation_template = FORMATION_TEMPLATES[formation_type]

        # Create formation
        slots = [
            FormationSlot(role=role, position=pos)
            for role, pos in formation_template.items()
        ]
        formation = Formation(
            id=f"form_gen_{self.generation_counter}",
            name=f"Generated {formation_type.replace('_', ' ').title()}",
            slots=slots
        )

        # Select routes for each receiver
        route_types = {}
        assignments = {Role.QB: None}

        # Determine route depth preferences
        if route_depth == "quick":
            wr_route_pool = QUICK_ROUTES
        elif route_depth == "medium":
            wr_route_pool = MEDIUM_ROUTES
        elif route_depth == "deep":
            wr_route_pool = DEEP_ROUTES
        else:
            # Mixed - combine all
            wr_route_pool = QUICK_ROUTES + MEDIUM_ROUTES + DEEP_ROUTES

        # Assign routes to WRs
        for role in [Role.WR1, Role.WR2, Role.WR3]:
            route_name = self.rng.choice(wr_route_pool)
            route_types[role] = route_name

            # Create route with mirroring based on receiver position
            receiver_pos = formation_template[role]
            route = self._create_route(route_name, role, receiver_pos.y < 0)
            assignments[role] = route

        # Assign center route
        if include_center_route:
            center_route_name = self.rng.choice(CENTER_ROUTES)
            route_types[Role.CENTER] = center_route_name
            assignments[Role.CENTER] = self._create_route(center_route_name, Role.CENTER, False)
        else:
            assignments[Role.CENTER] = None

        # Create QB plan with progression based on route timing
        progression = self._create_progression(assignments)

        qb_plan = QBPlan(
            progression_roles=progression,
            max_time_to_throw_ms=3000.0,
            scramble_allowed=False
        )

        # Create play
        play = Play(
            id=f"play_gen_{self.generation_counter}",
            name=f"Gen Play {self.generation_counter}",
            formation=formation,
            assignments=assignments,
            qb_plan=qb_plan
        )

        self.generation_counter += 1

        return GeneratedPlay(
            play=play,
            formation_type=formation_type,
            route_types=route_types,
            generation_id=f"gen_{self.generation_counter - 1}"
        )

    def generate_play_batch(
        self,
        count: int,
        formation_types: Optional[List[str]] = None,
        route_depths: Optional[List[str]] = None
    ) -> List[GeneratedPlay]:
        """Generate a batch of random plays."""
        plays = []

        for _ in range(count):
            formation = None
            if formation_types:
                formation = self.rng.choice(formation_types)

            depth = None
            if route_depths:
                depth = self.rng.choice(route_depths)

            plays.append(self.generate_random_play(
                formation_type=formation,
                route_depth=depth
            ))

        return plays

    def generate_concept_play(
        self,
        concept: str,
        formation_type: str = "trips_right"
    ) -> GeneratedPlay:
        """
        Generate a play based on a known concept.

        Concepts:
        - "flood": Three receivers to same side at different depths
        - "mesh": Crossing routes
        - "smash": Corner + hitch
        - "stick": Flat + stick + go
        - "verticals": All go routes
        - "slants": All slant routes
        """
        formation_template = FORMATION_TEMPLATES.get(formation_type, FORMATION_TEMPLATES["trips_right"])

        slots = [
            FormationSlot(role=role, position=pos)
            for role, pos in formation_template.items()
        ]
        formation = Formation(
            id=f"form_concept_{concept}_{self.generation_counter}",
            name=f"{concept.title()} {formation_type.replace('_', ' ').title()}",
            slots=slots
        )

        assignments = {Role.QB: None}
        route_types = {}

        if concept == "flood":
            route_types = {Role.WR1: "corner", Role.WR2: "out", Role.WR3: "flat"}
        elif concept == "mesh":
            route_types = {Role.WR1: "dig", Role.WR2: "in", Role.WR3: "flat"}
        elif concept == "smash":
            route_types = {Role.WR1: "corner", Role.WR2: "hitch", Role.WR3: "go"}
        elif concept == "stick":
            route_types = {Role.WR1: "go", Role.WR2: "stick", Role.WR3: "flat"}
        elif concept == "verticals":
            route_types = {Role.WR1: "go", Role.WR2: "seam", Role.WR3: "go"}
        elif concept == "slants":
            route_types = {Role.WR1: "slant", Role.WR2: "slant", Role.WR3: "slant"}
        else:
            # Default to flood
            route_types = {Role.WR1: "corner", Role.WR2: "out", Role.WR3: "flat"}

        for role, route_name in route_types.items():
            receiver_pos = formation_template[role]
            route = self._create_route(route_name, role, receiver_pos.y < 0)
            assignments[role] = route

        # Add center route
        route_types[Role.CENTER] = "center_drag"
        assignments[Role.CENTER] = self._create_route("center_drag", Role.CENTER, False)

        progression = self._create_progression(assignments)

        qb_plan = QBPlan(
            progression_roles=progression,
            max_time_to_throw_ms=3000.0,
            scramble_allowed=False
        )

        play = Play(
            id=f"play_concept_{concept}_{self.generation_counter}",
            name=f"{concept.title()} Concept",
            formation=formation,
            assignments=assignments,
            qb_plan=qb_plan
        )

        self.generation_counter += 1

        return GeneratedPlay(
            play=play,
            formation_type=formation_type,
            route_types=route_types,
            generation_id=f"concept_{concept}_{self.generation_counter - 1}"
        )

    def _create_route(
        self,
        route_name: str,
        role: Role,
        mirror: bool = False
    ) -> Route:
        """Create a route from the library, optionally mirroring."""
        breakpoints = ROUTE_LIBRARY.get(route_name, ROUTE_LIBRARY["hitch"])

        if mirror:
            # Mirror Y coordinates for left-side receivers
            breakpoints = [
                RouteBreakpoint(
                    x_yards=bp.x_yards,
                    y_yards=-bp.y_yards,
                    time_ms=bp.time_ms
                )
                for bp in breakpoints
            ]
        else:
            # Deep copy to avoid modifying library
            breakpoints = [
                RouteBreakpoint(
                    x_yards=bp.x_yards,
                    y_yards=bp.y_yards,
                    time_ms=bp.time_ms
                )
                for bp in breakpoints
            ]

        return Route(
            id=f"route_{route_name}_{role.value}_{self.generation_counter}",
            name=f"{route_name.replace('_', ' ').title()}",
            breakpoints=breakpoints
        )

    def _create_progression(
        self,
        assignments: Dict[Role, Optional[Route]]
    ) -> List[Role]:
        """
        Create QB progression based on route timing.
        Quick routes first, deep routes last.
        """
        routes_with_timing = []

        for role, route in assignments.items():
            if route is not None and role != Role.QB:
                # Use first breakpoint time as "open" time
                open_time = route.breakpoints[0].time_ms if route.breakpoints else 1000
                routes_with_timing.append((role, open_time))

        # Sort by timing (quick to slow)
        routes_with_timing.sort(key=lambda x: x[1])

        return [role for role, _ in routes_with_timing]


def generate_expanded_playbook(
    base_plays: List[Play],
    num_generated: int = 20,
    seed: Optional[int] = None
) -> List[Play]:
    """
    Expand a playbook with generated plays.

    Args:
        base_plays: Existing plays to include
        num_generated: Number of new plays to generate
        seed: Random seed

    Returns:
        Combined list of base and generated plays
    """
    generator = PlayGenerator(seed=seed)

    # Start with base plays
    all_plays = list(base_plays)

    # Generate concept plays (known good combinations)
    concepts = ["flood", "mesh", "smash", "stick", "verticals", "slants"]
    formations = ["trips_right", "bunch_tight", "tight_bunch_right", "spread"]

    for concept in concepts:
        for formation in formations[:2]:  # Limit combinations
            gen = generator.generate_concept_play(concept, formation)
            all_plays.append(gen.play)

    # Generate random plays
    remaining = max(0, num_generated - len(concepts) * 2)
    random_plays = generator.generate_play_batch(
        remaining,
        formation_types=list(FORMATION_TEMPLATES.keys()),
        route_depths=["quick", "medium", "deep", "mixed"]
    )

    for gen in random_plays:
        all_plays.append(gen.play)

    return all_plays
