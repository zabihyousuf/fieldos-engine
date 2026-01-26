# Game Simulation Feature - Implementation Plan

## Overview

This document outlines the implementation plan for a comprehensive game simulation feature that enables RL-trained team vs team matches in 5v5 flag football. The system will simulate complete games with drives, field position, downs, and detailed statistics to determine optimal play-calling strategies.

## Goals

1. **Train RL algorithm** to simulate matches between Team 1 and Team 2
2. **Create new plays** with motion and trick elements (pitch back, double QB)
3. **Simulate complete games** with realistic field mechanics
4. **Generate detailed reports** on what worked, when, and why
5. **Optimize for winning** - one team's goal is to win the game

---

## Field Specifications

| Dimension | Value |
|-----------|-------|
| Field Length | 60 yards |
| End Zones | 7 yards each |
| Field Width | 25 yards |
| Playing Field | 46 yards (60 - 7 - 7) |

---

## Game Rules

| Rule | Value |
|------|-------|
| Downs to First Down | 3 |
| Downs to Score (after 1st down) | 3 |
| Drives per Team | 10 (replaces 12-minute halves) |
| First Down Line | 20 yards from start |
| Play Ends | On catch, incompletion, or sack |
| Teams | 5v5 |

---

## New Play Types Required

### 1. Motion Plays (Pre-Snap Motion)
Players in motion before the snap to create confusion and mismatches.

| Play Name | Motion | Formation | Concept |
|-----------|--------|-----------|---------|
| Jet Motion Left | WR3 motions left across formation | Spread | Speed option, quick pass |
| Orbit Motion | WR2 comes behind QB | Trips | Orbit screen, play action |
| Bunch Motion | WR1 motions into bunch | Twins | Stack release, mesh |
| Center Motion | CENTER motions out wide | Spread | Misdirection, center screen |
| Trade Motion | WR1/WR2 swap positions | Twins | Crossing routes |

### 2. Trick Plays (Pitch Back / Double QB)
Plays where the ball is pitched to create additional throwing options.

| Play Name | Mechanic | Purpose |
|-----------|----------|---------|
| Shovel Option | QB pitches to WR1 who throws | Extra throwing lane |
| Reverse Pass | WR3 motions, gets pitch, throws back | Escape rush, deep shot |
| Double Move Pitch | QB fakes throw, pitches to WR2 | Buy time, new angle |
| Center Throwback | CENTER gets pitch, throws to QB | Ultimate misdirection |
| Halfback Pass | WR1 (in backfield) gets pitch, throws | Play action bomb |

### 3. Receiver Adaptation
When pitch occurs:
- Receivers may need to block briefly
- Routes adjust timing based on pitch delay
- Hot routes activate if pitch is threatened

---

## Implementation Phases

### Phase 1: Data Models (New/Updated)

#### 1.1 Team Model
```python
class Team(BaseModel):
    id: str
    name: str
    players: List[Player]  # 5 players with offense + defense attributes
    playbook: List[str]    # Play IDs (5-7 plays)

class PlayerWithDualRole(BaseModel):
    """Player with both offensive and defensive attributes"""
    id: str
    name: str
    # Offensive attributes
    off_speed: int
    off_hands: int
    off_route_running: int
    throw_power: int       # All players can throw for trick plays
    short_acc: int
    mid_acc: int
    deep_acc: int
    # Defensive attributes
    def_speed: int
    man_coverage: int
    zone_coverage: int
    ball_skills: int
    pass_rush: int
```

#### 1.2 Game State Model
```python
class GameState(BaseModel):
    home_team: Team
    away_team: Team
    home_score: int = 0
    away_score: int = 0
    current_drive: int = 1
    total_drives: int = 10
    possession: str  # "home" or "away"
    field_position: float  # Yards to goal (0 = in end zone)
    down: int = 1
    yards_to_first: float = 20.0
    yards_to_goal: float
    first_down_achieved: bool = False
    drive_result: Optional[str]  # "TD", "TURNOVER", "PUNT"
```

#### 1.3 Motion/Pitch Play Models
```python
class MotionEvent(BaseModel):
    """Pre-snap motion definition"""
    mover: Role
    start_position: Point2D
    end_position: Point2D
    duration_ms: float

class PitchEvent(BaseModel):
    """Ball pitch/handoff definition"""
    passer: Role           # Who pitches
    receiver: Role         # Who catches pitch
    trigger_time_ms: float # When pitch happens
    new_qb_role: Role      # Who becomes the thrower

class TrickPlay(Play):
    """Extended play with motion and pitch support"""
    pre_snap_motion: Optional[List[MotionEvent]]
    pitch_events: Optional[List[PitchEvent]]
    route_adjustments: Dict[Role, Route]  # Post-pitch routes
```

### Phase 2: Motion & Trick Plays Implementation

#### 2.1 Create Motion Plays
Location: `fieldos_engine/data/demo/motion_plays.json`

5 motion plays:
1. **Jet_Motion_Sweep** - WR3 motions across, quick pass or pitch
2. **Orbit_Screen** - WR2 orbits behind QB for screen
3. **Bunch_Stack** - WR1 motions into bunch, stack release
4. **Center_Swap** - CENTER motions wide, another player takes center
5. **Trade_Cross** - WR1/WR2 trade positions pre-snap

#### 2.2 Create Trick/Pitch Plays
Location: `fieldos_engine/data/demo/trick_plays.json`

5 trick plays:
1. **Shovel_Option** - QB pitches to WR1 who throws
2. **Reverse_Pass** - WR3 gets pitch, throws deep
3. **Double_Move_Pitch** - QB fakes, pitches to WR2
4. **Center_Throwback** - CENTER gets pitch, throws to QB running route
5. **Halfback_Pass** - WR1 in backfield gets pitch, throws deep

#### 2.3 Update Engine for Pitch Mechanics
- Handle pitch timing
- Switch active passer mid-play
- Adjust route timings based on pitch delay
- Track pitch success/failure

### Phase 3: Game Simulation Engine

#### 3.1 GameSimulator Class
Location: `fieldos_engine/sim/game_simulator.py`

```python
class GameSimulator:
    def __init__(self, home_team: Team, away_team: Team, field_config: FieldConfig):
        self.home = home_team
        self.away = away_team
        self.field = field_config
        self.state = GameState(...)
        self.play_engine = SimulationEngine()
        self.stats = GameStats()

    def simulate_game(self) -> GameResult:
        """Simulate complete game with 10 drives per team"""
        for drive_num in range(self.state.total_drives * 2):
            self.simulate_drive()
            self.switch_possession()
        return self.generate_result()

    def simulate_drive(self) -> DriveResult:
        """Simulate a single drive (3 downs to first, 3 to score)"""
        while not self.drive_over():
            play = self.select_play()
            result = self.simulate_play(play)
            self.update_state(result)
            self.record_stats(result)
        return self.drive_result()

    def select_play(self) -> Play:
        """Select play based on situation (or RL policy)"""
        pass

    def update_field_position(self, yards_gained: float):
        """Update field position, check for first down, TD"""
        self.state.field_position -= yards_gained
        if self.state.field_position <= 0:
            self.score_touchdown()
        elif yards_gained >= self.state.yards_to_first:
            self.award_first_down()
```

#### 3.2 Field Position Mechanics
```python
class FieldPositionTracker:
    """Track field position as it compresses near goal"""

    def __init__(self, field_length: float = 60.0, endzone: float = 7.0):
        self.field_length = field_length
        self.endzone = endzone
        self.playing_field = field_length - (2 * endzone)  # 46 yards

    def yards_to_goal(self, position: float) -> float:
        """Current yards to goal line"""
        return max(0, position)

    def yards_to_first(self, current_pos: float, first_down_marker: float) -> float:
        """Yards needed for first down"""
        return max(0, current_pos - first_down_marker)

    def is_redzone(self, position: float) -> bool:
        """Within 20 yards of goal"""
        return position <= 20

    def is_goalline(self, position: float) -> bool:
        """Within 5 yards of goal"""
        return position <= 5
```

### Phase 4: RL Game Environment

#### 4.1 GameRL Environment
Location: `fieldos_engine/rl/game_env.py`

```python
class GameRLEnv(gymnasium.Env):
    """Full game simulation environment for RL training"""

    def __init__(self, home_team: Team, away_team: Team,
                 optimize_for: str = "home"):
        self.home = home_team
        self.away = away_team
        self.optimize_for = optimize_for
        self.game_sim = GameSimulator(home_team, away_team)

        # State space: game situation + field position + score
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        # Action space: play selection
        self.action_space = spaces.Discrete(len(home_team.playbook))

    def _get_obs(self) -> np.ndarray:
        """Return observation vector"""
        return np.array([
            self.state.down / 3,
            self.state.yards_to_first / 20,
            self.state.yards_to_goal / 46,
            self.state.home_score / 42,  # Max ~6 TDs
            self.state.away_score / 42,
            self.state.current_drive / 20,
            1.0 if self.state.possession == self.optimize_for else 0.0,
            # ... defense shell one-hot
            # ... field zone one-hot (own territory, midfield, redzone, goalline)
        ])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute play and return reward"""
        play = self.current_team_playbook[action]
        result = self.game_sim.simulate_play(play)

        # Reward shaping
        reward = self._calculate_reward(result)

        done = self.game_sim.game_over()
        return self._get_obs(), reward, done, False, {"result": result}

    def _calculate_reward(self, result: PlayOutcome) -> float:
        """Calculate reward based on play outcome"""
        reward = 0.0

        # Yards gained
        reward += result.yards_gained * 0.1

        # First down
        if result.achieved_first_down:
            reward += 5.0

        # Touchdown
        if result.touchdown:
            reward += 20.0

        # Turnover
        if result.turnover:
            reward -= 15.0

        # Win bonus at game end
        if self.game_sim.game_over():
            if self.we_won():
                reward += 100.0
            else:
                reward -= 50.0

        return reward
```

#### 4.2 Self-Play Training
```python
class SelfPlayTrainer:
    """Train two policies against each other"""

    def __init__(self, team1: Team, team2: Team):
        self.team1 = team1
        self.team2 = team2
        self.policy1 = None
        self.policy2 = None

    def train(self, num_games: int = 1000):
        """Train both policies through self-play"""
        for game_num in range(num_games):
            # Play game
            result = self.play_game()

            # Update policies based on outcome
            self.update_policy(self.policy1, result, is_team1=True)
            self.update_policy(self.policy2, result, is_team1=False)

            # Record stats
            self.record_game(result)
```

### Phase 5: Statistics & Reporting

#### 5.1 Player Statistics Model
```python
class PlayerGameStats(BaseModel):
    player_id: str
    player_name: str

    # Passing stats
    pass_attempts: int = 0
    completions: int = 0
    passing_yards: float = 0.0
    touchdowns_thrown: int = 0
    interceptions: int = 0
    completion_pct: float = 0.0

    # Receiving stats
    targets: int = 0
    receptions: int = 0
    receiving_yards: float = 0.0
    touchdowns_receiving: int = 0
    avg_separation: float = 0.0

    # Defensive stats
    coverage_snaps: int = 0
    passes_defended: int = 0
    interceptions_def: int = 0
    sacks: int = 0

    # Advanced stats
    expected_yards: float = 0.0
    yards_over_expected: float = 0.0
```

#### 5.2 Play Statistics Model
```python
class PlayGameStats(BaseModel):
    play_id: str
    play_name: str

    # Usage stats
    times_called: int = 0
    times_called_by_down: Dict[int, int]  # {1: 5, 2: 3, 3: 2}
    times_called_by_field_zone: Dict[str, int]  # {"redzone": 3, "midfield": 5}

    # Success stats
    completions: int = 0
    success_rate: float = 0.0
    avg_yards: float = 0.0
    touchdowns: int = 0
    turnovers: int = 0

    # Situational effectiveness
    third_down_conversions: int = 0
    third_down_attempts: int = 0
    redzone_tds: int = 0
    redzone_attempts: int = 0

    # Best/worst situations
    best_down_distance: Optional[str]  # "3rd & short"
    worst_down_distance: Optional[str]
    best_field_zone: Optional[str]
    worst_field_zone: Optional[str]
```

#### 5.3 Game Report Generator
```python
class GameReportGenerator:
    """Generate detailed post-game analysis"""

    def generate_report(self, game_result: GameResult) -> GameReport:
        return GameReport(
            summary=self.generate_summary(game_result),
            player_stats=self.compile_player_stats(game_result),
            play_effectiveness=self.analyze_play_effectiveness(game_result),
            situational_analysis=self.situational_breakdown(game_result),
            key_moments=self.identify_key_moments(game_result),
            recommendations=self.generate_recommendations(game_result)
        )

    def analyze_play_effectiveness(self, game: GameResult) -> PlayEffectivenessReport:
        """Analyze when each play was most/least effective"""
        effectiveness = {}

        for play_stat in game.play_stats:
            effectiveness[play_stat.play_name] = {
                "overall_success_rate": play_stat.success_rate,
                "best_situation": {
                    "down_distance": play_stat.best_down_distance,
                    "field_zone": play_stat.best_field_zone,
                    "success_rate_in_best": self.calc_best_rate(play_stat)
                },
                "worst_situation": {
                    "down_distance": play_stat.worst_down_distance,
                    "field_zone": play_stat.worst_field_zone,
                    "success_rate_in_worst": self.calc_worst_rate(play_stat)
                },
                "recommendation": self.generate_play_recommendation(play_stat)
            }

        return effectiveness
```

---

## File Structure

```
fieldos_engine/
├── core/
│   └── models.py              # Add Team, GameState, MotionEvent, PitchEvent
├── sim/
│   ├── engine.py              # Update for pitch mechanics
│   ├── game_simulator.py      # NEW: Full game simulation
│   └── field_position.py      # NEW: Field position tracking
├── rl/
│   ├── game_env.py            # NEW: Game-level RL environment
│   ├── self_play.py           # NEW: Self-play training
│   └── game_trainer.py        # NEW: Game training orchestration
├── stats/
│   ├── __init__.py            # NEW
│   ├── player_stats.py        # NEW: Player statistics tracking
│   ├── play_stats.py          # NEW: Play statistics tracking
│   └── report_generator.py    # NEW: Report generation
├── data/
│   └── demo/
│       ├── motion_plays.json  # NEW: Motion plays
│       ├── trick_plays.json   # NEW: Trick plays
│       ├── team1_playbook.json # NEW: Team 1 plays (5-7)
│       └── team2_playbook.json # NEW: Team 2 plays (5-7)
└── visualizations/
    └── motion/                # NEW: Motion play visualizations
```

---

## Implementation Order

### Week 1: Core Models & Plays
1. ✅ Create this implementation plan
2. Add Team, GameState, MotionEvent, PitchEvent models to `models.py`
3. Create 5 motion plays with visualizations
4. Create 5 trick/pitch plays with visualizations
5. Update engine.py to handle motion and pitch

### Week 2: Game Simulation
6. Implement FieldPositionTracker
7. Implement GameSimulator
8. Implement drive logic (3 downs to first, 3 to score)
9. Implement field compression mechanics
10. Create team playbooks (5-7 plays each)

### Week 3: RL & Statistics
11. Implement GameRLEnv
12. Implement PlayerGameStats and PlayGameStats
13. Implement GameReportGenerator
14. Implement self-play training loop
15. Run test game (10 drives each team)

### Week 4: Optimization & Reporting
16. Train RL to optimize for Team 1 winning
17. Simulate batch of games
18. Generate comprehensive reports
19. Identify optimal play-calling strategies
20. Document findings

---

## Testing Plan

### Test 1: Single Game Simulation
```python
# Simulate one complete game
game = GameSimulator(team1, team2, field_config)
result = game.simulate_game()

# Expected output:
# - Final score
# - All player stats
# - All play stats
# - Drive-by-drive breakdown
```

### Test 2: RL Training
```python
# Train for 100 games
trainer = SelfPlayTrainer(team1, team2)
trainer.train(num_games=100)

# Expected output:
# - Trained policy
# - Win rate improvement curve
# - Play selection evolution
```

### Test 3: Batch Analysis
```python
# Simulate 1000 games for statistical significance
results = simulate_batch(team1, team2, num_games=1000)

# Expected output:
# - Aggregate statistics
# - Play effectiveness breakdown
# - Situational recommendations
```

---

## Success Criteria

1. ✅ Motion plays execute correctly with pre-snap movement
2. ✅ Trick plays handle pitch/handoff mechanics
3. ✅ Games complete with proper drive/down logic
4. ✅ Field position compresses correctly near goal
5. ✅ Player stats track all relevant metrics
6. ✅ Play stats identify situational effectiveness
7. ✅ RL policy learns to improve win rate
8. ✅ Reports provide actionable insights

---

## Notes

- All players have QB attributes for trick plays
- Motion timing affects route timing
- Pitch adds delay but creates new throwing lanes
- RL reward emphasizes winning over yards
- Statistical significance requires many games

---

## Next Steps

After implementation:
1. Expand playbooks based on effectiveness data
2. Add more complex trick plays
3. Implement defensive play-calling RL
4. Add real player data integration
5. Create interactive game viewer
