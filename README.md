# FieldOS Engine

5v5 flag football simulation and reinforcement learning optimization engine.

## Overview

FieldOS Engine is a Python backend that provides:

- **Canonical data model** for 5v5 flag football (plays, formations, routes, players, scenarios)
- **Physics-based simulation** of play execution with realistic coverage, route running, and outcomes
- **RL optimization** to learn optimal play-calling strategies using contextual bandits and PPO
- **HTTP API** for integration with frontends and external tools

Built with proper ML engineering principles, the system is deterministic, reproducible, and architected for growth.

## Features

### Data Model
- Plays with formations, route assignments, and QB progression
- Player attributes (speed, agility, coverage skills, etc.)
- Defensive coverages (Man, Zone, Cover 0/1/2/3)
- Game situations (down/distance buckets: 1st down, 3rd & short, redzone, etc.)
- Field configurations and rulesets
- **CENTER can run routes** after snapping the ball (common in flag football)

### Simulation
- Discrete-time simulation (50ms timesteps)
- Route interpolation with speed constraints
- Man and zone coverage logic
- Pass rush and sack mechanics
- Completion probability based on separation, QB accuracy, coverage help
- Yards after catch (YAC)
- Comprehensive metrics: completion rate, yards (mean/p50/p90), sack rate, INT rate
- Failure mode detection (sack, tight window, late throw, etc.)
- Optional trace sampling (top-N, sample rate)

### Reinforcement Learning
- **Play-calling as contextual bandit**: Learn which plays work best in different situations
- Gymnasium-compatible environment
- Baseline policies: Random, Epsilon-Greedy, UCB
- Optional PPO support via stable-baselines3
- Evaluation reports with per-bucket performance

### API
- FastAPI with automatic OpenAPI docs
- CRUD for all entities (plays, players, scenarios, etc.)
- `/simulate` - Run batch simulations
- `/train` - Train RL policies
- `/evaluate` - Evaluate policy performance
- Full error handling and validation

---

## UI Integration Guide

This section documents how to build a frontend UI that allows users to create and manage plays, routes, formations, players, and scenarios.

### Architecture for UI Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Future)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │ Play     │  │ Route    │  │Formation │  │  Simulation      ││
│  │ Designer │  │ Editor   │  │ Builder  │  │  Visualizer      ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘│
│       │             │             │                  │          │
│       └─────────────┴─────────────┴──────────────────┘          │
│                              │                                   │
│                         HTTP/JSON                                │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────┐
│                     FIELDOS ENGINE API                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     FastAPI Server                          ││
│  │  /plays  /routes  /formations  /players  /scenarios         ││
│  │  /simulate  /train  /evaluate  /seed-demo-data              ││
│  └──────────────────────────┬──────────────────────────────────┘│
│                              │                                   │
│  ┌───────────────┬───────────┴───────────┬────────────────────┐ │
│  │ Core Models   │    Simulation         │    RL              │ │
│  │ & Validation  │    Engine             │    Training        │ │
│  └───────────────┴───────────────────────┴────────────────────┘ │
│                              │                                   │
│  ┌───────────────────────────┴──────────────────────────────────┐│
│  │                     In-Memory Registry                       ││
│  │     (Thread-safe storage for all entities)                   ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
                               │
                    (Future: Database Layer)
```

### Data Model Reference

All data structures use Pydantic v2 models with full validation. Here's a reference for building UI forms:

#### 1. Route (receiver path on the field)

```json
{
  "id": "route_slant_8",
  "name": "8-Yard Slant",
  "breakpoints": [
    {"x_yards": 0.0, "y_yards": 0.0, "time_ms": 0.0},
    {"x_yards": 3.0, "y_yards": 0.0, "time_ms": 300.0},
    {"x_yards": 8.0, "y_yards": 5.0, "time_ms": 800.0}
  ]
}
```

**Field Reference:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `route_slant_8`) |
| `name` | string | Human-readable name |
| `breakpoints` | array | Ordered list of waypoints |
| `breakpoints[].x_yards` | float | Distance downfield from LOS (positive = forward) |
| `breakpoints[].y_yards` | float | Lateral position (0 = center, positive = right) |
| `breakpoints[].time_ms` | float | Milliseconds from snap to reach this point |

**Validation Rules:**
- At least 1 breakpoint required
- Times must be monotonically increasing
- Physical constraint: Cannot travel >15 yards/second between breakpoints

**UI Suggestions:**
- Visual route editor with draggable waypoints on a field diagram
- Time slider to animate route progression
- Speed calculator showing yards/second between points

#### 2. Formation (player starting positions)

```json
{
  "id": "form_trips_right",
  "name": "Trips Right",
  "slots": [
    {"role": "QB", "position": {"x": -5.0, "y": 0.0}},
    {"role": "CENTER", "position": {"x": 0.0, "y": 0.0}},
    {"role": "WR1", "position": {"x": 0.0, "y": 12.0}},
    {"role": "WR2", "position": {"x": 0.0, "y": 8.0}},
    {"role": "WR3", "position": {"x": 0.0, "y": 4.0}}
  ]
}
```

**Field Reference:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `name` | string | Human-readable name |
| `slots` | array | Exactly 5 player positions |
| `slots[].role` | enum | One of: `QB`, `CENTER`, `WR1`, `WR2`, `WR3` |
| `slots[].position.x` | float | X position (negative = behind LOS) |
| `slots[].position.y` | float | Y position (0 = center) |

**Validation Rules:**
- Exactly 5 slots required
- Must include QB and CENTER
- All roles must be offensive
- QB should be near center (|y| <= 5.0)

**UI Suggestions:**
- Drag-and-drop formation builder
- Field grid with snap-to-grid
- Preset templates (Trips, Twins, Bunch, Spread, Empty)

#### 3. Player (individual with attributes)

```json
{
  "id": "player_wr1_1",
  "name": "Jaylen Davis",
  "side": "OFFENSE",
  "role": "WR1",
  "attributes": {
    "speed": 88.0,
    "acceleration": 85.0,
    "agility": 82.0,
    "change_of_direction": 80.0,
    "stamina": 78.0,
    "hands": 85.0,
    "route_running": 88.0,
    "separation": 86.0,
    "throw_power": 70.0,
    "short_acc": 70.0,
    "mid_acc": 70.0,
    "deep_acc": 70.0,
    "release_time_ms": 400.0,
    "decision_latency_ms": 300.0,
    "man_coverage": 70.0,
    "zone_coverage": 70.0,
    "reaction_time_ms": 300.0,
    "ball_skills": 70.0,
    "closing_speed": 70.0,
    "pass_rush": 70.0,
    "get_off": 70.0,
    "contain": 70.0
  }
}
```

**Attribute Ranges (all 0.0 - 100.0 scale except timing in ms):**

| Category | Attributes | Relevant For |
|----------|------------|--------------|
| Physical | `speed`, `acceleration`, `agility`, `change_of_direction`, `stamina` | All players |
| Receiving | `hands`, `route_running`, `separation` | WRs, CENTER |
| Passing | `throw_power`, `short_acc`, `mid_acc`, `deep_acc`, `release_time_ms` (200-800), `decision_latency_ms` (100-1000) | QB |
| Coverage | `man_coverage`, `zone_coverage`, `reaction_time_ms` (100-800), `ball_skills`, `closing_speed` | Defenders |
| Pass Rush | `pass_rush`, `get_off`, `contain` | RUSHER |

**Roles:**
- Offense: `QB`, `CENTER`, `WR1`, `WR2`, `WR3`
- Defense: `RUSHER`, `CB1`, `CB2`, `SAFETY`, `LB`

**UI Suggestions:**
- Slider-based attribute editor with radar charts
- Position-specific attribute templates
- Import/export player profiles

#### 4. Play (complete offensive play design)

```json
{
  "id": "play_trips_flood",
  "name": "Trips Flood",
  "formation": { /* Formation object */ },
  "assignments": {
    "QB": null,
    "CENTER": null,
    "WR1": { /* Route object for WR1 */ },
    "WR2": { /* Route object for WR2 */ },
    "WR3": { /* Route object for WR3 */ }
  },
  "qb_plan": {
    "progression_roles": ["WR2", "WR3", "WR1"],
    "max_time_to_throw_ms": 3000.0,
    "scramble_allowed": false
  }
}
```

**Key Points:**
- `assignments` maps each role to either a Route or `null`
- QB and CENTER can have `null` (don't run routes) OR a Route (CENTER can run routes after snap)
- `progression_roles` defines QB read order - can include `CENTER`!
- All progression roles must have routes assigned

**UI Suggestions:**
- Play designer combining formation + route assignment
- Visual QB progression order editor
- Drag routes from a route library onto player positions

#### 5. Scenario (defensive setup and game context)

```json
{
  "id": "scenario_man_cover1_1rush",
  "name": "Man Cover 1 (1 Rush)",
  "field": {
    "width_yards": 40.0,
    "total_length_yards": 80.0,
    "endzone_depth_yards": 10.0,
    "no_run_zone_depth_yards": 5.0
  },
  "rules": { /* Ruleset object */ },
  "defense_call": {
    "type": "MAN",
    "shell": "COVER1",
    "rushers_count": 1,
    "notes": "Man coverage with 1 deep safety, 1 rusher"
  },
  "defender_start_positions": {
    "CB1": {"x": 0.0, "y": 10.0},
    "CB2": {"x": 0.0, "y": -10.0},
    "SAFETY": {"x": 15.0, "y": 0.0},
    "LB": {"x": -1.0, "y": 5.0},
    "RUSHER": {"x": -7.0, "y": 0.0}
  },
  "randomness": {
    "position_jitter_yards": 0.5,
    "reaction_jitter_ms": 50.0
  }
}
```

**Coverage Types:**
- `MAN` - Defenders assigned to specific receivers
- `ZONE` - Defenders assigned to field zones

**Coverage Shells:**
- `COVER0` - No deep safety, all man
- `COVER1` - 1 deep safety
- `COVER2` - 2 deep safeties
- `COVER3` - 3 deep zones

**UI Suggestions:**
- Defensive formation builder with coverage type selector
- Visual coverage shell diagrams
- Rush count slider (0-4)

### API Endpoints Reference

All endpoints accept/return JSON. Base URL: `http://localhost:8000`

#### Entity CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/plays` | List all plays |
| POST | `/plays` | Create a play |
| GET | `/plays/{id}` | Get a specific play |
| PUT | `/plays/{id}` | Update a play |
| GET | `/routes` | List all routes |
| POST | `/routes` | Create a route |
| GET | `/formations` | List all formations |
| POST | `/formations` | Create a formation |
| GET | `/players` | List all players |
| POST | `/players` | Create a player |
| GET | `/scenarios` | List all scenarios |
| POST | `/scenarios` | Create a scenario |
| GET | `/rules` | List all rulesets |
| POST | `/rules` | Create a ruleset |

#### Simulation & Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/simulate` | Run play simulation |
| POST | `/train` | Train RL policy |
| POST | `/evaluate` | Evaluate policy performance |
| POST | `/seed-demo-data` | Load demo data |
| GET | `/health` | Health check |

### Example: Creating a Custom Play via API

```bash
# 1. Create a custom route
curl -X POST http://localhost:8000/routes \
  -H "Content-Type: application/json" \
  -d '{
    "route": {
      "id": "route_custom_wheel",
      "name": "Wheel Route",
      "breakpoints": [
        {"x_yards": 0.0, "y_yards": 0.0, "time_ms": 0.0},
        {"x_yards": 2.0, "y_yards": -3.0, "time_ms": 300.0},
        {"x_yards": 10.0, "y_yards": -5.0, "time_ms": 900.0},
        {"x_yards": 20.0, "y_yards": -2.0, "time_ms": 1800.0}
      ]
    }
  }'

# 2. Create a play using that route
curl -X POST http://localhost:8000/plays \
  -H "Content-Type: application/json" \
  -d '{
    "play": {
      "id": "play_my_custom",
      "name": "My Custom Play",
      "formation": {
        "id": "form_spread_custom",
        "name": "Custom Spread",
        "slots": [
          {"role": "QB", "position": {"x": -5.0, "y": 0.0}},
          {"role": "CENTER", "position": {"x": 0.0, "y": 0.0}},
          {"role": "WR1", "position": {"x": 0.0, "y": 15.0}},
          {"role": "WR2", "position": {"x": 0.0, "y": -15.0}},
          {"role": "WR3", "position": {"x": 1.0, "y": 5.0}}
        ]
      },
      "assignments": {
        "QB": null,
        "CENTER": null,
        "WR1": {"id": "route_go", "name": "Go", "breakpoints": [
          {"x_yards": 0.0, "y_yards": 0.0, "time_ms": 0.0},
          {"x_yards": 25.0, "y_yards": 0.0, "time_ms": 2500.0}
        ]},
        "WR2": {"id": "route_custom_wheel", "name": "Wheel Route", "breakpoints": [
          {"x_yards": 0.0, "y_yards": 0.0, "time_ms": 0.0},
          {"x_yards": 2.0, "y_yards": -3.0, "time_ms": 300.0},
          {"x_yards": 10.0, "y_yards": -5.0, "time_ms": 900.0},
          {"x_yards": 20.0, "y_yards": -2.0, "time_ms": 1800.0}
        ]},
        "WR3": {"id": "route_slant_8", "name": "Slant", "breakpoints": [
          {"x_yards": 0.0, "y_yards": 0.0, "time_ms": 0.0},
          {"x_yards": 3.0, "y_yards": 0.0, "time_ms": 300.0},
          {"x_yards": 8.0, "y_yards": 5.0, "time_ms": 800.0}
        ]}
      },
      "qb_plan": {
        "progression_roles": ["WR3", "WR2", "WR1"],
        "max_time_to_throw_ms": 3500.0,
        "scramble_allowed": false
      }
    }
  }'

# 3. Simulate the play
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "play_id": "play_my_custom",
    "scenario_ids": ["scenario_man_cover1_1rush", "scenario_zone_cover2"],
    "num_episodes": 100,
    "seed": 42,
    "mode": "EVAL"
  }'
```

### Example: CENTER Running a Route

The CENTER position can now run routes after snapping the ball, which is common in flag football where the center is an eligible receiver.

```json
{
  "id": "play_center_release",
  "name": "Center Release",
  "formation": {
    "id": "form_spread",
    "name": "Spread",
    "slots": [
      {"role": "QB", "position": {"x": -5.0, "y": 0.0}},
      {"role": "CENTER", "position": {"x": 0.0, "y": 0.0}},
      {"role": "WR1", "position": {"x": 0.0, "y": 15.0}},
      {"role": "WR2", "position": {"x": 0.0, "y": -15.0}},
      {"role": "WR3", "position": {"x": 1.0, "y": 8.0}}
    ]
  },
  "assignments": {
    "QB": null,
    "CENTER": {
      "id": "route_center_seam",
      "name": "Center Seam",
      "breakpoints": [
        {"x_yards": 0.0, "y_yards": 0.0, "time_ms": 0.0},
        {"x_yards": 5.0, "y_yards": 0.0, "time_ms": 600.0},
        {"x_yards": 12.0, "y_yards": 0.0, "time_ms": 1200.0}
      ]
    },
    "WR1": { /* go route */ },
    "WR2": { /* out route */ },
    "WR3": { /* drag route */ }
  },
  "qb_plan": {
    "progression_roles": ["CENTER", "WR3", "WR2", "WR1"],
    "max_time_to_throw_ms": 3000.0,
    "scramble_allowed": false
  }
}
```

### Recommended UI Features

#### Phase 1: Core Editors
1. **Route Editor** - Visual waypoint editor on field grid
2. **Formation Builder** - Drag-and-drop player positioning
3. **Play Designer** - Combine formation + route assignments
4. **Player Manager** - CRUD with attribute sliders

#### Phase 2: Simulation & Analysis
5. **Scenario Builder** - Defense setup and coverage selection
6. **Simulation Runner** - Run plays and view results
7. **Play Visualizer** - Animate play execution with traces

#### Phase 3: Advanced Features
8. **Playbook Manager** - Organize plays into categories
9. **RL Training Dashboard** - Train and compare policies
10. **Performance Analytics** - Charts and comparisons

### Frontend Technology Recommendations

**State Management:**
- React Query or SWR for API data fetching
- Zustand or Redux for complex local state
- Optimistic updates for smooth UX

**Visualization:**
- Canvas/SVG for field and route rendering
- D3.js or Recharts for performance charts
- Framer Motion for animations

**Form Handling:**
- React Hook Form for complex forms
- Zod for client-side validation (matches Pydantic)

### Data Persistence Roadmap

Currently, the engine uses an in-memory registry. For production:

1. **Add database layer** (PostgreSQL recommended)
2. **User authentication** for personal playbooks
3. **Team/organization support** for shared plays
4. **Version history** for play iterations

---

## Installation

### Requirements
- Python 3.11+
- pip

### Setup

```bash
# Clone repository
git clone <repo-url>
cd fieldos-engine

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Load Demo Data

```bash
python scripts/seed_demo_data.py
```

This loads:
- 6 plays (Trips Flood, Bunch Quick Slants, Twins Smash, etc.)
- 11 routes (slant, hitch, corner, post, go, block, etc.)
- 8 formations (Trips, Twins, Bunch, Spread, Empty, etc.)
- 10 players (QB, CENTER, WRs, CBs, Safety, LB, Rusher)
- 8 scenarios (Man Cover 0/1, Zone Cover 2/3, 3rd & short/long, redzone, goalline)
- 2 rulesets (standard, compressed field)

### 2. Start API Server

```bash
uvicorn fieldos_engine.api.main:app --reload --port 8000
```

Or use the convenience script:

```bash
./scripts/run_dev.sh
```

Server starts at `http://localhost:8000`

API docs at `http://localhost:8000/docs`

### 3. Run Tests

```bash
pytest tests/ -v
```

## Interpreting Metrics

### Overall Metrics
- **completion_rate**: Percentage of completed passes (0.0 - 1.0)
- **sack_rate**: Percentage of plays ending in sack
- **intercept_rate**: Percentage of interceptions
- **yards_mean**: Average yards gained per play
- **yards_p50**: Median yards (50th percentile)
- **yards_p90**: 90th percentile yards (big play potential)
- **time_to_throw_mean**: Average QB decision time (ms)

### Target Distribution
Shows which receivers were targeted:
```json
"target_distribution": {
  "WR1": 20,
  "WR2": 25,
  "WR3": 5,
  "CENTER": 10
}
```

### Failure Modes
- **SACK_BEFORE_THROW**: Rusher reached QB
- **TIGHT_WINDOW**: Low separation at throw/catch
- **LATE_THROW**: Threw near max time limit
- **LOW_QB_ACCURACY**: QB accuracy too low for distance
- **ROUTE_TIMING_MISMATCH**: Route timing not optimal

### By-Bucket Metrics
Same metrics sliced by game situation:
- **1ST_ANY**: First down, any distance
- **2ND_SHORT**: 2nd down, <=5 yards
- **2ND_LONG**: 2nd down, >5 yards
- **3RD_SHORT**: 3rd down, <=5 yards
- **3RD_LONG**: 3rd down, >5 yards
- **REDZONE**: Inside 20 yards
- **GOALLINE**: Inside 5 yards

## Architecture

```
fieldos-engine/
├── fieldos_engine/
│   ├── core/          # Data models, validation, registry
│   │   ├── models.py  # Pydantic models for all entities
│   │   ├── validation.py # Cross-entity validation
│   │   ├── registry.py # Thread-safe in-memory storage
│   │   └── ids.py     # ID generation
│   ├── sim/           # Simulation engine
│   │   ├── engine.py  # Main simulation loop
│   │   ├── motion.py  # Route interpolation
│   │   ├── coverage.py # Man/zone coverage logic
│   │   ├── outcome.py # Completion probability
│   │   └── trace.py   # Trace sampling
│   ├── rl/            # RL environment, policies, training
│   │   ├── env.py     # Gymnasium environment
│   │   ├── policy.py  # Bandit policies
│   │   └── train.py   # Training loops
│   ├── api/           # FastAPI endpoints
│   │   ├── main.py    # App setup
│   │   ├── routes.py  # All endpoints
│   │   └── schemas.py # Request/response DTOs
│   └── data/demo/     # Demo JSON data files
├── tests/             # Comprehensive tests
├── scripts/           # Utility scripts
└── README.md
```

### Key Design Principles

1. **Determinism**: All simulation and training accept a `seed` parameter for reproducibility
2. **Validation**: All entities validated before creation and before simulation
3. **Separation of concerns**: Clear boundaries between data model, simulation, RL, and API
4. **Extensibility**: Easy to add new plays, formations, coverage schemes
5. **Type safety**: Full Pydantic models with validation
6. **UI-Ready**: RESTful CRUD API designed for frontend integration

## Development

### Run Linter

```bash
black fieldos_engine/ tests/
ruff check fieldos_engine/ tests/
```

### Run Type Checker

```bash
mypy fieldos_engine/
```

### Clear Registry (for testing)

```python
from fieldos_engine.core.registry import registry
registry.clear_all()
```

## Roadmap / Future Enhancements

- [ ] Database persistence layer (PostgreSQL/SQLite)
- [ ] User authentication and authorization
- [ ] Frontend UI (React recommended)
- [ ] Replay visualization with animated traces
- [ ] Pre-snap motion support
- [ ] Running plays (currently pass-only)
- [ ] Multi-down drive simulation
- [ ] Advanced coverage schemes
- [ ] Player fatigue model
- [ ] Team/organization playbook sharing

## License

MIT

## Contact

For questions or feedback, open an issue on GitHub.

---

Built with passion for football and machine learning.
