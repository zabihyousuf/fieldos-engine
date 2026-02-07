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


# ============================================================================
# Game Simulation Models
# ============================================================================

class PlayerSpecialty(str, Enum):
    """Player specialty - what they're best at."""
    OFFENSE_ONLY = "OFFENSE_ONLY"
    DEFENSE_ONLY = "DEFENSE_ONLY"
    TWO_WAY = "TWO_WAY"  # Plays both sides


class DualRolePlayerAttributes(BaseModel):
    """Player attributes for both offense and defense (for game simulation).

    Physical attributes (speed, acceleration, agility) are unified - a player
    has ONE speed rating that applies to both offense and defense.
    """

    # Physical attributes (UNIFIED - same for offense and defense)
    speed: float = Field(default=70.0, ge=0.0, le=100.0, description="Raw speed")
    acceleration: float = Field(default=70.0, ge=0.0, le=100.0, description="Burst/acceleration")
    agility: float = Field(default=70.0, ge=0.0, le=100.0, description="Change of direction")

    # Physical measurements
    height_inches: float = Field(default=72.0, ge=60.0, le=84.0, description="Height in inches (5'0\" to 7'0\")")
    weight_lbs: float = Field(default=180.0, ge=120.0, le=300.0, description="Weight in pounds")

    # Offensive skill attributes
    hands: float = Field(default=70.0, ge=0.0, le=100.0, description="Catching ability")
    route_running: float = Field(default=70.0, ge=0.0, le=100.0, description="Route precision and timing")

    # Passing (all players can throw for trick plays)
    throw_power: float = Field(default=60.0, ge=0.0, le=100.0, description="Arm strength")
    short_acc: float = Field(default=60.0, ge=0.0, le=100.0, description="Short pass accuracy (0-10 yds)")
    mid_acc: float = Field(default=55.0, ge=0.0, le=100.0, description="Medium pass accuracy (10-20 yds)")
    deep_acc: float = Field(default=50.0, ge=0.0, le=100.0, description="Deep pass accuracy (20+ yds)")
    release_time_ms: float = Field(default=500.0, ge=200.0, le=1000.0, description="Time to release ball")
    decision_latency_ms: float = Field(default=400.0, ge=100.0, le=1000.0, description="Read progression speed")

    # Defensive skill attributes
    man_coverage: float = Field(default=65.0, ge=0.0, le=100.0, description="Man-to-man coverage ability")
    zone_coverage: float = Field(default=65.0, ge=0.0, le=100.0, description="Zone coverage awareness")
    ball_skills: float = Field(default=65.0, ge=0.0, le=100.0, description="Ability to make plays on the ball")
    closing_speed: float = Field(default=70.0, ge=0.0, le=100.0, description="Speed when breaking on the ball")
    pass_rush: float = Field(default=60.0, ge=0.0, le=100.0, description="Pass rushing ability")
    reaction_time_ms: float = Field(default=350.0, ge=100.0, le=800.0, description="Reaction to ball in air")

    # Player specialty
    specialty: PlayerSpecialty = Field(default=PlayerSpecialty.TWO_WAY, description="Offense only, defense only, or two-way player")

    def to_offensive_attributes(self) -> PlayerAttributes:
        """Convert to offensive PlayerAttributes."""
        return PlayerAttributes(
            speed=self.speed,
            acceleration=self.acceleration,
            agility=self.agility,
            hands=self.hands,
            route_running=self.route_running,
            throw_power=self.throw_power,
            short_acc=self.short_acc,
            mid_acc=self.mid_acc,
            deep_acc=self.deep_acc,
            release_time_ms=self.release_time_ms,
            decision_latency_ms=self.decision_latency_ms,
        )

    def to_defensive_attributes(self) -> PlayerAttributes:
        """Convert to defensive PlayerAttributes."""
        return PlayerAttributes(
            speed=self.speed,
            acceleration=self.acceleration,
            agility=self.agility,
            man_coverage=self.man_coverage,
            zone_coverage=self.zone_coverage,
            ball_skills=self.ball_skills,
            closing_speed=self.closing_speed,
            pass_rush=self.pass_rush,
            reaction_time_ms=self.reaction_time_ms,
        )

    def can_play_offense(self) -> bool:
        """Check if player can play offense."""
        return self.specialty in (PlayerSpecialty.OFFENSE_ONLY, PlayerSpecialty.TWO_WAY)

    def can_play_defense(self) -> bool:
        """Check if player can play defense."""
        return self.specialty in (PlayerSpecialty.DEFENSE_ONLY, PlayerSpecialty.TWO_WAY)

    def qb_score(self) -> float:
        """Calculate QB suitability score (0-100). Higher is better."""
        if not self.can_play_offense():
            return 0.0

        # QB needs: throwing ability, quick decisions, some mobility
        accuracy_avg = (self.short_acc + self.mid_acc + self.deep_acc) / 3
        # Lower release/decision time is better - normalize to 0-100 where lower ms = higher score
        release_score = max(0, 100 - (self.release_time_ms - 200) / 6)  # 200ms = 100, 800ms = 0
        decision_score = max(0, 100 - (self.decision_latency_ms - 100) / 9)  # 100ms = 100, 1000ms = 0

        return (
            self.throw_power * 0.25 +
            accuracy_avg * 0.35 +
            release_score * 0.15 +
            decision_score * 0.15 +
            self.speed * 0.05 +
            self.agility * 0.05
        )

    def wr_score(self, slot: str = "outside") -> float:
        """Calculate WR suitability score (0-100). Higher is better.

        Args:
            slot: "outside" for X/Z receivers, "slot" for slot receiver
        """
        if not self.can_play_offense():
            return 0.0

        base_score = (
            self.speed * 0.25 +
            self.hands * 0.30 +
            self.route_running * 0.30 +
            self.agility * 0.10 +
            self.acceleration * 0.05
        )

        # Height modifier: taller better for outside (contested catches), shorter better for slot (agility)
        if slot == "outside":
            # Favor height 72"+ for outside, max bonus at 78"
            height_bonus = min(10, max(0, (self.height_inches - 70) * 1.5))
        else:
            # Favor height under 72" for slot, quicker in traffic
            height_bonus = min(10, max(0, (74 - self.height_inches) * 1.5))

        return min(100, base_score + height_bonus)

    def center_score(self) -> float:
        """Calculate Center suitability score (0-100). Higher is better."""
        if not self.can_play_offense():
            return 0.0

        # Center needs: reliable hands, route running for outlet passes, some size
        return (
            self.hands * 0.35 +
            self.route_running * 0.25 +
            self.speed * 0.15 +
            self.agility * 0.15 +
            min(100, self.weight_lbs / 2.5) * 0.10  # Slight favor for size
        )

    def defender_score(self, coverage_type: str = "man") -> float:
        """Calculate defender suitability score (0-100). Higher is better.

        Args:
            coverage_type: "man" or "zone"
        """
        if not self.can_play_defense():
            return 0.0

        if coverage_type == "man":
            return (
                self.man_coverage * 0.35 +
                self.speed * 0.25 +
                self.ball_skills * 0.20 +
                self.acceleration * 0.10 +
                self.closing_speed * 0.10
            )
        else:  # zone
            return (
                self.zone_coverage * 0.35 +
                self.ball_skills * 0.25 +
                self.speed * 0.15 +
                self.closing_speed * 0.15 +
                self.acceleration * 0.10
            )

    def rusher_score(self) -> float:
        """Calculate pass rusher suitability score (0-100). Higher is better."""
        if not self.can_play_defense():
            return 0.0

        return (
            self.pass_rush * 0.40 +
            self.speed * 0.25 +
            self.acceleration * 0.20 +
            self.closing_speed * 0.15
        )

    def overall_offense_score(self) -> float:
        """Calculate overall offensive ability score."""
        if not self.can_play_offense():
            return 0.0

        # Weight QB, WR, and Center scores
        return max(
            self.qb_score(),
            self.wr_score("outside"),
            self.wr_score("slot"),
            self.center_score()
        )

    def overall_defense_score(self) -> float:
        """Calculate overall defensive ability score."""
        if not self.can_play_defense():
            return 0.0

        return max(
            self.defender_score("man"),
            self.defender_score("zone"),
            self.rusher_score()
        )

    def height_formatted(self) -> str:
        """Return height as feet'inches\" format."""
        feet = int(self.height_inches // 12)
        inches = int(self.height_inches % 12)
        return f"{feet}'{inches}\""


class GamePlayer(BaseModel):
    """Player with dual role capabilities for game simulation."""
    id: str
    name: str
    number: int = Field(ge=0, le=99)
    attributes: DualRolePlayerAttributes = Field(default_factory=DualRolePlayerAttributes)

    def as_offense_player(self, role: Role) -> Player:
        """Return a Player configured for offense."""
        return Player(
            id=f"{self.id}_off",
            name=self.name,
            side=Side.OFFENSE,
            role=role,
            attributes=self.attributes.to_offensive_attributes()
        )

    def as_defense_player(self, role: Role) -> Player:
        """Return a Player configured for defense."""
        return Player(
            id=f"{self.id}_def",
            name=self.name,
            side=Side.DEFENSE,
            role=role,
            attributes=self.attributes.to_defensive_attributes()
        )


class Team(BaseModel):
    """Team with 5-15 players and a playbook."""
    id: str
    name: str
    players: List[GamePlayer] = Field(min_length=5, max_length=15)
    playbook: List[str] = Field(default_factory=list)  # Play IDs

    def get_player_by_number(self, number: int) -> Optional[GamePlayer]:
        """Get player by jersey number."""
        return next((p for p in self.players if p.number == number), None)


class DriveResult(str, Enum):
    """Possible drive outcomes."""
    TOUCHDOWN = "TOUCHDOWN"
    TURNOVER_ON_DOWNS = "TURNOVER_ON_DOWNS"
    INTERCEPTION = "INTERCEPTION"
    IN_PROGRESS = "IN_PROGRESS"


class FieldZone(str, Enum):
    """Field position zones."""
    OWN_TERRITORY = "OWN_TERRITORY"
    MIDFIELD = "MIDFIELD"
    OPPONENT_TERRITORY = "OPPONENT_TERRITORY"
    REDZONE = "REDZONE"
    GOALLINE = "GOALLINE"


class ExtraPointChoice(str, Enum):
    """Choice for extra point attempt after touchdown."""
    ONE_POINT = "ONE_POINT"   # From 5 yard line
    TWO_POINT = "TWO_POINT"   # From 12 yard line


class GameConfig(BaseModel):
    """Configuration for a game simulation."""
    field_length: float = Field(default=60.0, description="Total field length in yards")
    endzone_depth: float = Field(default=7.0, description="Endzone depth in yards")
    field_width: float = Field(default=25.0, description="Field width in yards")
    playing_field: float = Field(default=46.0, description="Playing field length (excludes endzones)")

    downs_to_first_down: int = Field(default=3, description="Downs to get first down")
    downs_to_score: int = Field(default=3, description="Downs to score after first down")
    first_down_yards: float = Field(default=20.0, description="Yards needed for first down")

    drives_per_team: int = Field(default=10, description="Number of drives per team")
    td_points: int = Field(default=6, description="Points for touchdown")

    # Extra point configuration
    xp1_distance: float = Field(default=5.0, description="Distance for 1-point conversion")
    xp2_distance: float = Field(default=12.0, description="Distance for 2-point conversion")
    xp1_points: int = Field(default=1, description="Points for 1-point conversion")
    xp2_points: int = Field(default=2, description="Points for 2-point conversion")

    start_position: float = Field(default=20.0, description="Starting field position from own goal")

    @property
    def yards_to_midfield(self) -> float:
        """Yards to midfield from starting position."""
        return (self.playing_field / 2) - self.start_position


class GameState(BaseModel):
    """Current state of a game."""
    home_team_id: str
    away_team_id: str
    home_score: int = 0
    away_score: int = 0

    current_drive: int = 1
    possession: Literal["home", "away"] = "home"

    # Field position (yards from own goal, 0 = own goal, playing_field = opponent goal)
    field_position: float = 20.0

    down: int = 1
    yards_to_first: float = 20.0
    first_down_achieved: bool = False

    # Game progress
    total_drives: int = 20  # 10 per team
    game_over: bool = False

    @property
    def yards_to_goal(self) -> float:
        """Yards to opponent's goal line."""
        return 46.0 - self.field_position  # Assuming 46 yard playing field

    @property
    def field_zone(self) -> FieldZone:
        """Current field zone."""
        ytg = self.yards_to_goal
        if ytg <= 5:
            return FieldZone.GOALLINE
        elif ytg <= 20:
            return FieldZone.REDZONE
        elif self.field_position >= 23:
            return FieldZone.OPPONENT_TERRITORY
        elif self.field_position >= 18:
            return FieldZone.MIDFIELD
        else:
            return FieldZone.OWN_TERRITORY

    def to_game_situation(self) -> GameSituation:
        """Convert to GameSituation for play simulation."""
        return GameSituation(
            down=self.down,
            yards_to_gain=self.yards_to_first,
            yardline_to_goal=self.yards_to_goal,
        )


class PlayResult(BaseModel):
    """Result of a single play in a game context."""
    play_id: str
    play_name: str
    outcome: OutcomeType
    yards_gained: float

    down: int
    yards_to_first: float
    field_position_before: float
    field_position_after: float

    target_role: Optional[Role] = None
    passer_id: Optional[str] = None
    passer_name: Optional[str] = None
    receiver_id: Optional[str] = None
    receiver_name: Optional[str] = None

    # Full lineup for this play (role -> player name)
    offensive_lineup: Dict[str, str] = Field(default_factory=dict)
    defensive_lineup: Dict[str, str] = Field(default_factory=dict)

    resulted_in_first_down: bool = False
    resulted_in_touchdown: bool = False
    resulted_in_turnover: bool = False

    time_to_throw_ms: Optional[float] = None
    completion_probability: Optional[float] = None

    def get_player_at_position(self, role: str) -> Optional[str]:
        """Get the player name at a given offensive position."""
        return self.offensive_lineup.get(role)


class DriveRecord(BaseModel):
    """Record of a single drive."""
    drive_number: int
    team_id: str
    starting_field_position: float
    ending_field_position: float
    plays: List[PlayResult] = Field(default_factory=list)
    result: DriveResult = DriveResult.IN_PROGRESS
    points_scored: int = 0
    total_yards: float = 0.0

    # Extra point tracking
    extra_point_choice: Optional[str] = None  # "ONE_POINT" or "TWO_POINT"
    extra_point_success: Optional[bool] = None
    extra_point_play: Optional[PlayResult] = None

    @property
    def num_plays(self) -> int:
        return len(self.plays)

    @property
    def scoring_play(self) -> Optional[PlayResult]:
        """Get the play that scored the touchdown."""
        for play in self.plays:
            if play.resulted_in_touchdown:
                return play
        return None


class PlayerGameStats(BaseModel):
    """Individual player stats for a game."""
    player_id: str
    player_name: str
    player_number: int

    # Passing
    pass_attempts: int = 0
    completions: int = 0
    passing_yards: float = 0.0
    touchdowns_thrown: int = 0
    interceptions_thrown: int = 0

    # Receiving
    targets: int = 0
    receptions: int = 0
    receiving_yards: float = 0.0
    touchdowns_receiving: int = 0

    # Defense
    coverage_snaps: int = 0
    passes_defended: int = 0
    interceptions_caught: int = 0

    @property
    def completion_pct(self) -> float:
        if self.pass_attempts == 0:
            return 0.0
        return (self.completions / self.pass_attempts) * 100

    @property
    def catch_pct(self) -> float:
        if self.targets == 0:
            return 0.0
        return (self.receptions / self.targets) * 100


class PlayGameStats(BaseModel):
    """Stats for a specific play across a game."""
    play_id: str
    play_name: str

    times_called: int = 0
    completions: int = 0
    attempts: int = 0
    total_yards: float = 0.0
    touchdowns: int = 0
    turnovers: int = 0

    # Situational stats
    times_called_by_down: Dict[int, int] = Field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    success_by_down: Dict[int, int] = Field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    times_called_by_zone: Dict[str, int] = Field(default_factory=dict)

    first_down_conversions: int = 0
    third_down_attempts: int = 0
    third_down_conversions: int = 0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return (self.completions / self.attempts) * 100

    @property
    def avg_yards(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_yards / self.attempts

    @property
    def third_down_conversion_rate(self) -> float:
        if self.third_down_attempts == 0:
            return 0.0
        return (self.third_down_conversions / self.third_down_attempts) * 100


class TeamGameStats(BaseModel):
    """Aggregate team stats for a game."""
    team_id: str
    team_name: str

    # Scoring
    total_points: int = 0
    touchdowns: int = 0

    # Offense
    total_plays: int = 0
    total_yards: float = 0.0
    completions: int = 0
    attempts: int = 0
    interceptions_thrown: int = 0

    # Drives
    drives: int = 0
    scoring_drives: int = 0
    turnovers: int = 0

    # Situational
    first_downs: int = 0
    third_down_attempts: int = 0
    third_down_conversions: int = 0

    # By player
    player_stats: Dict[str, PlayerGameStats] = Field(default_factory=dict)

    # By play
    play_stats: Dict[str, PlayGameStats] = Field(default_factory=dict)

    @property
    def completion_pct(self) -> float:
        if self.attempts == 0:
            return 0.0
        return (self.completions / self.attempts) * 100

    @property
    def yards_per_play(self) -> float:
        if self.total_plays == 0:
            return 0.0
        return self.total_yards / self.total_plays

    @property
    def third_down_pct(self) -> float:
        if self.third_down_attempts == 0:
            return 0.0
        return (self.third_down_conversions / self.third_down_attempts) * 100


class GameResult(BaseModel):
    """Complete result of a game simulation."""
    game_id: str
    home_team_id: str
    away_team_id: str

    home_score: int
    away_score: int
    winner: Optional[str] = None  # team_id or None for tie

    total_plays: int = 0
    total_drives: int = 0

    home_stats: TeamGameStats
    away_stats: TeamGameStats

    drive_records: List[DriveRecord] = Field(default_factory=list)

    @property
    def is_tie(self) -> bool:
        return self.home_score == self.away_score
