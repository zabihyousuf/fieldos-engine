"""ID generation and validation utilities."""

import uuid
from typing import NewType

# Type aliases for different ID types
PlayId = NewType("PlayId", str)
PlayerId = NewType("PlayerId", str)
FormationId = NewType("FormationId", str)
RouteId = NewType("RouteId", str)
ScenarioId = NewType("ScenarioId", str)
RulesetId = NewType("RulesetId", str)
TrainingId = NewType("TrainingId", str)
RunId = NewType("RunId", str)


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique = str(uuid.uuid4())[:8]
    return f"{prefix}_{unique}" if prefix else unique


def validate_id(id_value: str) -> bool:
    """Validate that an ID is non-empty and reasonable length."""
    return bool(id_value) and len(id_value) > 0 and len(id_value) < 128
