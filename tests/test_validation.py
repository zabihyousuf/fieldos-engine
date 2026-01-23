"""Test validation logic."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.core.models import (
    Formation, FormationSlot, Role, Point2D, Route, RouteBreakpoint
)
from fieldos_engine.core.validation import (
    validate_formation, validate_route, ValidationError
)


def test_valid_formation():
    """Test that valid formation passes."""
    formation = Formation(
        id="test",
        name="Test",
        slots=[
            FormationSlot(role=Role.QB, position=Point2D(x=-5.0, y=0.0)),
            FormationSlot(role=Role.CENTER, position=Point2D(x=0.0, y=0.0)),
            FormationSlot(role=Role.WR1, position=Point2D(x=0.0, y=10.0)),
            FormationSlot(role=Role.WR2, position=Point2D(x=0.0, y=-10.0)),
            FormationSlot(role=Role.WR3, position=Point2D(x=0.0, y=5.0))
        ]
    )
    # Should not raise
    validate_formation(formation)


def test_formation_missing_qb():
    """Test that formation without QB fails."""
    with pytest.raises(ValueError):
        Formation(
            id="test",
            name="Test",
            slots=[
                FormationSlot(role=Role.CENTER, position=Point2D(x=0.0, y=0.0)),
                FormationSlot(role=Role.WR1, position=Point2D(x=0.0, y=10.0)),
                FormationSlot(role=Role.WR2, position=Point2D(x=0.0, y=-10.0)),
                FormationSlot(role=Role.WR3, position=Point2D(x=0.0, y=5.0)),
                FormationSlot(role=Role.WR1, position=Point2D(x=0.0, y=7.0))
            ]
        )


def test_formation_missing_center():
    """Test that formation without CENTER fails."""
    with pytest.raises(ValueError):
        Formation(
            id="test",
            name="Test",
            slots=[
                FormationSlot(role=Role.QB, position=Point2D(x=-5.0, y=0.0)),
                FormationSlot(role=Role.WR1, position=Point2D(x=0.0, y=10.0)),
                FormationSlot(role=Role.WR2, position=Point2D(x=0.0, y=-10.0)),
                FormationSlot(role=Role.WR3, position=Point2D(x=0.0, y=5.0)),
                FormationSlot(role=Role.WR1, position=Point2D(x=0.0, y=7.0))
            ]
        )


def test_formation_wrong_count():
    """Test that formation with wrong number of slots fails."""
    with pytest.raises(ValueError):
        Formation(
            id="test",
            name="Test",
            slots=[
                FormationSlot(role=Role.QB, position=Point2D(x=-5.0, y=0.0)),
                FormationSlot(role=Role.CENTER, position=Point2D(x=0.0, y=0.0)),
                FormationSlot(role=Role.WR1, position=Point2D(x=0.0, y=10.0))
            ]
        )


def test_valid_route():
    """Test that valid route passes."""
    route = Route(
        id="test",
        name="Test",
        breakpoints=[
            RouteBreakpoint(x_yards=0.0, y_yards=0.0, time_ms=0.0),
            RouteBreakpoint(x_yards=5.0, y_yards=0.0, time_ms=1000.0),
            RouteBreakpoint(x_yards=10.0, y_yards=3.0, time_ms=2000.0)
        ]
    )
    # Should not raise
    validate_route(route)


def test_route_non_monotonic_time():
    """Test that route with non-monotonic times fails."""
    with pytest.raises(ValueError):
        Route(
            id="test",
            name="Test",
            breakpoints=[
                RouteBreakpoint(x_yards=0.0, y_yards=0.0, time_ms=0.0),
                RouteBreakpoint(x_yards=5.0, y_yards=0.0, time_ms=2000.0),
                RouteBreakpoint(x_yards=10.0, y_yards=3.0, time_ms=1000.0)
            ]
        )


def test_route_properties():
    """Test route derived properties."""
    route = Route(
        id="test",
        name="Test",
        breakpoints=[
            RouteBreakpoint(x_yards=0.0, y_yards=0.0, time_ms=0.0),
            RouteBreakpoint(x_yards=5.0, y_yards=0.0, time_ms=1000.0),
            RouteBreakpoint(x_yards=10.0, y_yards=3.0, time_ms=2000.0)
        ]
    )

    assert route.target_depth_yards == 10.0
    assert route.duration_ms == 2000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
