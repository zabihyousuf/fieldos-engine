"""In-memory registry for storing domain entities (MVP)."""

from typing import Dict, List, Optional, TypeVar, Generic
from threading import Lock
from .models import Play, Player, Route, Formation, Scenario, Ruleset
from .validation import (
    validate_play, validate_player, validate_route,
    validate_formation, validate_scenario, validate_ruleset
)

T = TypeVar('T')


class Registry(Generic[T]):
    """Thread-safe in-memory registry for entities."""

    def __init__(self, name: str, validator=None):
        self.name = name
        self.validator = validator
        self._data: Dict[str, T] = {}
        self._lock = Lock()

    def create(self, id: str, entity: T) -> T:
        """Create a new entity."""
        if self.validator:
            self.validator(entity)

        with self._lock:
            if id in self._data:
                raise ValueError(f"{self.name} with id {id} already exists")
            self._data[id] = entity
        return entity

    def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        with self._lock:
            return self._data.get(id)

    def update(self, id: str, entity: T) -> T:
        """Update existing entity."""
        if self.validator:
            self.validator(entity)

        with self._lock:
            if id not in self._data:
                raise ValueError(f"{self.name} with id {id} not found")
            self._data[id] = entity
        return entity

    def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        with self._lock:
            if id in self._data:
                del self._data[id]
                return True
            return False

    def list(self) -> List[T]:
        """List all entities."""
        with self._lock:
            return list(self._data.values())

    def clear(self) -> None:
        """Clear all entities (for testing)."""
        with self._lock:
            self._data.clear()

    def count(self) -> int:
        """Count entities."""
        with self._lock:
            return len(self._data)


class GlobalRegistry:
    """Global singleton registry for all entity types."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize all registries."""
        self.plays = Registry[Play]("Play", validate_play)
        self.players = Registry[Player]("Player", validate_player)
        self.routes = Registry[Route]("Route", validate_route)
        self.formations = Registry[Formation]("Formation", validate_formation)
        self.scenarios = Registry[Scenario]("Scenario", validate_scenario)
        self.rulesets = Registry[Ruleset]("Ruleset", validate_ruleset)

    def clear_all(self):
        """Clear all registries (for testing)."""
        self.plays.clear()
        self.players.clear()
        self.routes.clear()
        self.formations.clear()
        self.scenarios.clear()
        self.rulesets.clear()


# Global instance
registry = GlobalRegistry()
