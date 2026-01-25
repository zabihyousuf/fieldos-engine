"""Canonical data models for FieldOS Engine."""

from enum import Enum
from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================

class Side(str, Enum):
    OFFENSE = "OFFENSE"
    DEFENSE = "DEFENSE"


class Role(str, Enum):
    # Offense
    QB = "QB"
    CENTER = "CENTER"
    WR1 = "WR1"
    WR2 = "WR2"
    WR3 = "WR3"

    # Defense - 5 defenders for 5v5 flag football
    # D1-D5 are generic defender slots that can be assigned different roles
    D1 = "D1"  # Typically outside left (corner/flat)
    D2 = "D2"  # Typically inside left (LB/hook)
    D3 = "D3"  # Typically middle (MLB/rusher/safety)
    D4 = "D4"  # Typically inside right (LB/hook)
    D5 = "D5"  # Typically outside right (corner/flat)

    # Legacy roles for backward compatibility
    RUSHER = "RUSHER"
    CB1 = "CB1"
    CB2 = "CB2"
    SAFETY = "SAFETY"
    LB = "LB"


class RusherPosition(str, Enum):
    """Position where the rusher lines up."""
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class CoverageShell(str, Enum):
    COVER0 = "COVER0"
    COVER1 = "COVER1"
    COVER2 = "COVER2"
    COVER3 = "COVER3"


class CoverageType(str, Enum):
    MAN = "MAN"
    ZONE = "ZONE"


class DownDistanceBucket(str, Enum):
    FIRST_ANY = "1ST_ANY"
    SECOND_SHORT = "2ND_SHORT"
    SECOND_LONG = "2ND_LONG"
    THIRD_SHORT = "3RD_SHORT"
    THIRD_LONG = "3RD_LONG"
    REDZONE = "REDZONE"
    GOALLINE = "GOALLINE"


class OutcomeType(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    INTERCEPT = "INTERCEPT"
    SACK = "SACK"


class FailureMode(str, Enum):
    SACK_BEFORE_THROW = "SACK_BEFORE_THROW"
    TIGHT_WINDOW = "TIGHT_WINDOW"
    LATE_THROW = "LATE_THROW"
    LOW_QB_ACCURACY = "LOW_QB_ACCURACY"
    ROUTE_TIMING_MISMATCH = "ROUTE_TIMING_MISMATCH"


class TraceMode(str, Enum):
    NONE = "NONE"
    TOP_N = "TOP_N"
    SAMPLE_RATE = "SAMPLE_RATE"


class SimMode(str, Enum):
    EVAL = "EVAL"
    POLICY = "POLICY"


# ============================================================================
# Field and Rules
# ============================================================================

class FieldConfig(BaseModel):
    """Field dimensions and boundaries."""
    width_yards: float = Field(default=40.0, ge=20.0, le=100.0)
    total_length_yards: float = Field(default=80.0, ge=40.0, le=120.0)
    endzone_depth_yards: float = Field(default=10.0, ge=5.0, le=20.0)
    no_run_zone_depth_yards: float = Field(default=5.0, ge=0.0, le=10.0)

    @property
    def los_to_endzone_yards(self) -> float:
        """Distance from LOS to endzone (used for play length)."""
        return self.total_length_yards / 2.0 - self.endzone_depth_yards


class DownsConfig(BaseModel):
    """Downs and line-to-gain configuration."""
    use_midfield_line_to_gain: bool = True
    line_to_gain_yards: float = 25.0
    reset_at_midfield: bool = True
    downs_to_convert: int = 3


class NoRunZoneConfig(BaseModel):
    """No-run zone rules."""
    enabled: bool = True
    depth_from_goal_yards: float = 5.0


class RushConfig(BaseModel):
    """Pass rush rules."""
    rusher_distance_yards: float = 7.0
    rush_delay_seconds: float = 0.0
    max_rushers: int = 1
    qb_can_run: bool = False


class MotionConfig(BaseModel):
    """Pre-snap motion rules."""
    allow_motion: bool = True
    lateral_only: bool = True


class ScoringConfig(BaseModel):
    """Point values."""
    td_points: int = 6
    xp1_points: int = 1
    xp2_points: int = 2
    safety_points: int = 2


class Ruleset(BaseModel):
    """Complete ruleset configuration."""
    id: str
    name: str
    players_per_side: int = Field(default=5, ge=5, le=5)
    field: FieldConfig = Field(default_factory=FieldConfig)
    downs: DownsConfig = Field(default_factory=DownsConfig)
    no_run_zone: NoRunZoneConfig = Field(default_factory=NoRunZoneConfig)
    rush: RushConfig = Field(default_factory=RushConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


# ============================================================================
# Player Attributes
# ============================================================================

class PlayerAttributes(BaseModel):
    """Player physical and skill attributes (0-100 scale, ms for timing)."""

    # Physical
    speed: float = Field(default=70.0, ge=0.0, le=100.0)
    acceleration: float = Field(default=70.0, ge=0.0, le=100.0)
    agility: float = Field(default=70.0, ge=0.0, le=100.0)
    change_of_direction: float = Field(default=70.0, ge=0.0, le=100.0)
    stamina: float = Field(default=70.0, ge=0.0, le=100.0)

    # Receiving
    hands: float = Field(default=70.0, ge=0.0, le=100.0)
    route_running: float = Field(default=70.0, ge=0.0, le=100.0)
    separation: float = Field(default=70.0, ge=0.0, le=100.0)

    # Passing
    throw_power: float = Field(default=70.0, ge=0.0, le=100.0)
    short_acc: float = Field(default=70.0, ge=0.0, le=100.0)
    mid_acc: float = Field(default=70.0, ge=0.0, le=100.0)
    deep_acc: float = Field(default=70.0, ge=0.0, le=100.0)
    release_time_ms: float = Field(default=400.0, ge=200.0, le=800.0)
    decision_latency_ms: float = Field(default=300.0, ge=100.0, le=1000.0)

    # Coverage
    man_coverage: float = Field(default=70.0, ge=0.0, le=100.0)
    zone_coverage: float = Field(default=70.0, ge=0.0, le=100.0)
    reaction_time_ms: float = Field(default=300.0, ge=100.0, le=800.0)
    ball_skills: float = Field(default=70.0, ge=0.0, le=100.0)
    closing_speed: float = Field(default=70.0, ge=0.0, le=100.0)

    # Pass rush
    pass_rush: float = Field(default=70.0, ge=0.0, le=100.0)
    get_off: float = Field(default=70.0, ge=0.0, le=100.0)
    contain: float = Field(default=70.0, ge=0.0, le=100.0)


class Player(BaseModel):
    """Player definition."""
    id: str
    name: str
    side: Side
    role: Role
    attributes: PlayerAttributes = Field(default_factory=PlayerAttributes)

    @model_validator(mode='after')
    def validate_side_role_match(self):
        """Ensure role matches side."""
        offense_roles = {Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3}
        # Support both new D1-D5 roles and legacy roles
        defense_roles = {
            Role.D1, Role.D2, Role.D3, Role.D4, Role.D5,
            Role.RUSHER, Role.CB1, Role.CB2, Role.SAFETY, Role.LB
        }

        if self.side == Side.OFFENSE and self.role not in offense_roles:
            raise ValueError(f"Offense player cannot have role {self.role}")
        if self.side == Side.DEFENSE and self.role not in defense_roles:
            raise ValueError(f"Defense player cannot have role {self.role}")

        return self


# ============================================================================
# Routes and Formation
# ============================================================================

class Point2D(BaseModel):
    """2D coordinate (yards)."""
    x: float
    y: float


class RouteBreakpoint(BaseModel):
    """Route waypoint with timing."""
    x_yards: float
    y_yards: float
    time_ms: float = Field(ge=0.0)


class Route(BaseModel):
    """Receiver route definition."""
    id: str
    name: str
    breakpoints: List[RouteBreakpoint] = Field(min_length=1)

    @field_validator('breakpoints')
    @classmethod
    def validate_monotonic_time(cls, v: List[RouteBreakpoint]) -> List[RouteBreakpoint]:
        """Ensure breakpoint times are monotonically increasing."""
        for i in range(1, len(v)):
            if v[i].time_ms <= v[i-1].time_ms:
                raise ValueError(f"Breakpoint times must be monotonically increasing")
        return v

    @property
    def target_depth_yards(self) -> float:
        """Maximum x-coordinate (depth downfield)."""
        return max(bp.x_yards for bp in self.breakpoints)

    @property
    def duration_ms(self) -> float:
        """Total route duration."""
        return self.breakpoints[-1].time_ms if self.breakpoints else 0.0


class FormationSlot(BaseModel):
    """Player position at snap."""
    role: Role
    position: Point2D


class Formation(BaseModel):
    """Offensive formation (5 players)."""
    id: str
    name: str
    slots: List[FormationSlot] = Field(min_length=5, max_length=5)

    @model_validator(mode='after')
    def validate_formation(self):
        """Ensure exactly 5 offensive slots with QB and CENTER."""
        roles = [slot.role for slot in self.slots]

        if len(roles) != 5:
            raise ValueError("Formation must have exactly 5 slots")

        if Role.QB not in roles:
            raise ValueError("Formation must include QB")

        if Role.CENTER not in roles:
            raise ValueError("Formation must include CENTER")

        # All roles must be offensive
        offense_roles = {Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3}
        for role in roles:
            if role not in offense_roles:
                raise ValueError(f"Formation cannot include defensive role {role}")

        return self


# ============================================================================
# Play Definition
# ============================================================================

class QBPlan(BaseModel):
    """QB read progression and timing."""
    progression_roles: List[Role] = Field(min_length=1)
    max_time_to_throw_ms: float = Field(default=3000.0, ge=500.0, le=10000.0)
    scramble_allowed: bool = False

    @field_validator('progression_roles')
    @classmethod
    def validate_receiver_roles(cls, v: List[Role]) -> List[Role]:
        """Ensure progression only includes eligible receiver roles."""
        # CENTER can run routes after snapping the ball (common in flag football)
        receiver_roles = {Role.WR1, Role.WR2, Role.WR3, Role.CENTER}
        for role in v:
            if role not in receiver_roles:
                raise ValueError(f"QB progression can only include receiver roles (WR1, WR2, WR3, CENTER), got {role}")
        return v


class Play(BaseModel):
    """Complete play definition."""
    id: str
    name: str
    formation: Formation
    assignments: Dict[Role, Optional[Route]]
    qb_plan: QBPlan

    @model_validator(mode='after')
    def validate_play(self):
        """Ensure assignments match formation and progression is valid."""
        formation_roles = {slot.role for slot in self.formation.slots}

        # All assignments must be for roles in formation
        for role in self.assignments:
            if role not in formation_roles:
                raise ValueError(f"Assignment for {role} but not in formation")

        # All progression roles must have routes
        for role in self.qb_plan.progression_roles:
            if role not in self.assignments or self.assignments[role] is None:
                raise ValueError(f"QB progression includes {role} but no route assigned")

        return self


# ============================================================================
# Defense
# ============================================================================

class DefenseCall(BaseModel):
    """Defensive coverage and rush configuration."""
    type: CoverageType
    shell: CoverageShell
    rushers_count: int = Field(ge=0, le=1)  # 5v5 typically has 0 or 1 rusher
    rusher_position: Optional[RusherPosition] = None  # L/C/R - randomized if None
    notes: Optional[str] = None


class RandomnessConfig(BaseModel):
    """Randomness parameters for simulation."""
    position_jitter_yards: float = Field(default=0.5, ge=0.0, le=3.0)
    reaction_jitter_ms: float = Field(default=50.0, ge=0.0, le=500.0)


class Scenario(BaseModel):
    """Complete scenario definition (field, rules, defense setup)."""
    id: str
    name: str
    field: FieldConfig
    rules: Ruleset
    defense_call: DefenseCall
    defender_start_positions: Dict[Role, Point2D]
    randomness: RandomnessConfig = Field(default_factory=RandomnessConfig)

    @model_validator(mode='after')
    def validate_defender_positions(self):
        """Ensure defender positions are for defensive roles only."""
        # New D1-D5 roles plus legacy roles for backward compatibility
        defense_roles = {
            Role.D1, Role.D2, Role.D3, Role.D4, Role.D5,
            Role.RUSHER, Role.CB1, Role.CB2, Role.SAFETY, Role.LB
        }

        for role in self.defender_start_positions:
            if role not in defense_roles:
                raise ValueError(f"Defender position for non-defensive role {role}")

        # Should have 5 defenders
        if len(self.defender_start_positions) != 5:
            raise ValueError(f"Must have exactly 5 defender positions, got {len(self.defender_start_positions)}")

        return self


# ============================================================================
# Game Situation
# ============================================================================

class GameSituation(BaseModel):
    """Game situation context (down/distance)."""
    down: int = Field(ge=1, le=4)
    yards_to_gain: float = Field(ge=0.0)
    yardline_to_goal: float = Field(ge=0.0)
    time_remaining_seconds: Optional[float] = Field(default=None, ge=0.0)
    score_diff: Optional[int] = None

    @property
    def bucket(self) -> DownDistanceBucket:
        """Compute situation bucket."""
        # Goalline
        if self.yardline_to_goal <= 5.0:
            return DownDistanceBucket.GOALLINE

        # Redzone
        if self.yardline_to_goal <= 20.0:
            return DownDistanceBucket.REDZONE

        # Down/distance buckets
        if self.down == 1:
            return DownDistanceBucket.FIRST_ANY
        elif self.down == 2:
            return DownDistanceBucket.SECOND_SHORT if self.yards_to_gain <= 5.0 else DownDistanceBucket.SECOND_LONG
        else:  # down >= 3
            return DownDistanceBucket.THIRD_SHORT if self.yards_to_gain <= 5.0 else DownDistanceBucket.THIRD_LONG


# ============================================================================
# Simulation Results
# ============================================================================

class PlayOutcome(BaseModel):
    """Result of a single play simulation."""
    outcome: OutcomeType
    yards_gained: float
    time_to_throw_ms: Optional[float] = None
    target_role: Optional[Role] = None
    completion_probability: Optional[float] = None
    separation_at_throw: Optional[float] = None
    separation_at_catch: Optional[float] = None
    failure_modes: List[FailureMode] = Field(default_factory=list)
