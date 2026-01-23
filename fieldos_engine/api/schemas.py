"""API request/response schemas."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from ..core.models import (
    Play, Player, Scenario, Ruleset, Formation, Route,
    GameSituation, TraceMode, SimMode, Role
)


# ============================================================================
# Health
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    ok: bool = True
    version: str = "0.1.0"


# ============================================================================
# Simulation
# ============================================================================

class TracePolicyRequest(BaseModel):
    """Trace sampling policy."""
    mode: TraceMode = TraceMode.NONE
    top_n: Optional[int] = None
    sample_rate: Optional[float] = None


class SimulateRequest(BaseModel):
    """Simulate play request."""
    play_id: str
    scenario_ids: List[str] = Field(min_length=1)
    num_episodes: int = Field(ge=1, le=10000, default=10)
    seed: Optional[int] = None
    mode: SimMode = SimMode.EVAL
    trace_policy: TracePolicyRequest = Field(default_factory=lambda: TracePolicyRequest(mode=TraceMode.NONE))
    situation_distribution: Optional[List[GameSituation]] = None
    # Player mappings
    offensive_players: Optional[Dict[Role, str]] = None  # role -> player_id
    defensive_players: Optional[Dict[Role, str]] = None


class SimulateResponse(BaseModel):
    """Simulate play response."""
    run_id: str
    play_id: str
    num_episodes: int
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# RL Training
# ============================================================================

class TrainRequest(BaseModel):
    """Train RL policy request."""
    play_ids: List[str] = Field(min_length=1)
    scenario_ids: List[str] = Field(min_length=1)
    offensive_players: Dict[Role, str]  # role -> player_id
    defensive_players: Dict[Role, str]
    seed: Optional[int] = None
    steps: int = Field(ge=100, le=100000, default=1000)
    algo: str = Field(default="BANDIT")  # BANDIT, UCB, PPO
    # Algo-specific params
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    ucb_c: float = Field(default=2.0, ge=0.0)


class TrainResponse(BaseModel):
    """Train RL policy response."""
    training_id: str
    summary: Dict[str, Any]
    policy_artifact_id: Optional[str] = None


# ============================================================================
# RL Evaluation
# ============================================================================

class EvaluateRequest(BaseModel):
    """Evaluate policy request."""
    policy_id: str  # or "baseline"
    play_ids: List[str]
    scenario_ids: List[str]
    offensive_players: Dict[Role, str]
    defensive_players: Dict[Role, str]
    num_episodes: int = Field(ge=10, le=1000, default=100)
    seed: Optional[int] = None


class EvaluateResponse(BaseModel):
    """Evaluate policy response."""
    policy_id: str
    report: Dict[str, Any]
    plot_url: Optional[str] = None


# ============================================================================
# CRUD
# ============================================================================

class CreatePlayRequest(BaseModel):
    """Create play request."""
    play: Play


class UpdatePlayRequest(BaseModel):
    """Update play request."""
    play: Play


class CreatePlayerRequest(BaseModel):
    """Create player request."""
    player: Player


class CreateScenarioRequest(BaseModel):
    """Create scenario request."""
    scenario: Scenario


class CreateRulesetRequest(BaseModel):
    """Create ruleset request."""
    ruleset: Ruleset


class CreateFormationRequest(BaseModel):
    """Create formation request."""
    formation: Formation


class CreateRouteRequest(BaseModel):
    """Create route request."""
    route: Route


# ============================================================================
# Seed Data
# ============================================================================

class SeedDataResponse(BaseModel):
    """Seed demo data response."""
    plays_loaded: int
    players_loaded: int
    routes_loaded: int
    formations_loaded: int
    scenarios_loaded: int
    rulesets_loaded: int
    message: str
