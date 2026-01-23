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
- 10 routes (slant, hitch, corner, post, go, etc.)
- 8 formations (Trips, Twins, Bunch, Spread, Empty, etc.)
- 10 players (QB, WRs, CBs, Safety, LB, Rusher)
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

Tests cover:
- Determinism (same seed = identical results)
- Validation (formation/route invariants)
- API smoke tests
- RL training smoke tests

## Usage Examples

### Example 1: List Available Plays

```bash
curl http://localhost:8000/plays
```

Returns:
```json
[
  {
    "id": "play_trips_flood",
    "name": "Trips Flood",
    "formation": {...},
    "assignments": {...},
    "qb_plan": {...}
  },
  ...
]
```

### Example 2: Simulate a Play

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "play_id": "play_trips_flood",
    "scenario_ids": ["scenario_man_cover0", "scenario_zone_cover2"],
    "num_episodes": 50,
    "seed": 42,
    "mode": "EVAL",
    "trace_policy": {"mode": "NONE"}
  }'
```

Returns:
```json
{
  "run_id": "run_abc123",
  "play_id": "play_trips_flood",
  "num_episodes": 50,
  "metrics": {
    "overall": {
      "num_plays": 50,
      "completion_rate": 0.68,
      "sack_rate": 0.08,
      "intercept_rate": 0.02,
      "yards_mean": 8.4,
      "yards_p50": 7.2,
      "yards_p90": 14.1,
      "time_to_throw_mean": 1850.5,
      "target_distribution": {"WR1": 20, "WR2": 25, "WR3": 5},
      "failure_modes": {"TIGHT_WINDOW": 12, "SACK_BEFORE_THROW": 4}
    },
    "by_bucket": {...}
  },
  "artifacts": {"traces": []}
}
```

### Example 3: Simulate with Trace Sampling

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "play_id": "play_spread_vertical",
    "scenario_ids": ["scenario_zone_cover3"],
    "num_episodes": 100,
    "seed": 123,
    "mode": "EVAL",
    "trace_policy": {
      "mode": "TOP_N",
      "top_n": 5
    }
  }'
```

Returns top 5 best plays with full position traces.

### Example 4: Train RL Policy

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "play_ids": [
      "play_trips_flood",
      "play_bunch_quick_slants",
      "play_twins_smash",
      "play_tight_levels"
    ],
    "scenario_ids": [
      "scenario_3rd_short_man",
      "scenario_3rd_long_zone"
    ],
    "offensive_players": {
      "QB": "player_qb1",
      "CENTER": "player_center1",
      "WR1": "player_wr1_1",
      "WR2": "player_wr2_1",
      "WR3": "player_wr3_1"
    },
    "defensive_players": {
      "RUSHER": "player_rusher1",
      "CB1": "player_cb1_1",
      "CB2": "player_cb2_1",
      "SAFETY": "player_safety1",
      "LB": "player_lb1"
    },
    "seed": 42,
    "steps": 1000,
    "algo": "BANDIT",
    "epsilon": 0.1,
    "learning_rate": 0.1
  }'
```

Returns:
```json
{
  "training_id": "train_xyz789",
  "summary": {
    "algorithm": "BANDIT",
    "total_steps": 1000,
    "final_reward_mean": 7.2,
    "final_reward_std": 4.1,
    "reward_history_length": 1000,
    "eval_history": [...],
    "best_actions_per_bucket": {
      "0": 1,
      "3": 2,
      "4": 0
    }
  },
  "policy_artifact_id": "train_xyz789"
}
```

### Example 5: Evaluate Policy

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": "baseline",
    "play_ids": [
      "play_trips_flood",
      "play_spread_vertical"
    ],
    "scenario_ids": ["scenario_redzone_man", "scenario_goalline_cover0"],
    "offensive_players": {...},
    "defensive_players": {...},
    "num_episodes": 100,
    "seed": 42
  }'
```

Returns per-bucket performance analysis.

### Example 6: Create Custom Play

```bash
curl -X POST http://localhost:8000/plays \
  -H "Content-Type: application/json" \
  -d '{
    "play": {
      "id": "play_custom_1",
      "name": "My Custom Play",
      "formation": {
        "id": "form_trips_right",
        "name": "Trips Right",
        "slots": [...]
      },
      "assignments": {
        "QB": null,
        "CENTER": null,
        "WR1": {"id": "route_go", "name": "Go", "breakpoints": [...]},
        "WR2": {"id": "route_post_15", "name": "Post", "breakpoints": [...]},
        "WR3": {"id": "route_drag_10", "name": "Drag", "breakpoints": [...]}
      },
      "qb_plan": {
        "progression_roles": ["WR1", "WR2", "WR3"],
        "max_time_to_throw_ms": 3000.0,
        "scramble_allowed": false
      }
    }
  }'
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
  "WR1": 20,  // WR1 targeted 20 times
  "WR2": 25,  // WR2 targeted 25 times
  "WR3": 5    // WR3 targeted 5 times
}
```

### Failure Modes
Counts of specific failure causes:
- **SACK_BEFORE_THROW**: Rusher reached QB
- **TIGHT_WINDOW**: Low separation at throw/catch
- **LATE_THROW**: Threw near max time limit
- **LOW_QB_ACCURACY**: QB accuracy too low for distance
- **ROUTE_TIMING_MISMATCH**: Route timing not optimal

### By-Bucket Metrics
Same metrics sliced by game situation:
- **1ST_ANY**: First down, any distance
- **2ND_SHORT**: 2nd down, ≤5 yards
- **2ND_LONG**: 2nd down, >5 yards
- **3RD_SHORT**: 3rd down, ≤5 yards
- **3RD_LONG**: 3rd down, >5 yards
- **REDZONE**: Inside 20 yards
- **GOALLINE**: Inside 5 yards

Use this to identify which plays work in which situations!

## Architecture

```
fieldos-engine/
├── fieldos_engine/
│   ├── core/          # Data models, validation, registry
│   ├── sim/           # Simulation engine
│   ├── rl/            # RL environment, policies, training
│   ├── api/           # FastAPI endpoints
│   └── data/demo/     # Demo data files
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

- [ ] Add QB decision-making RL (when to throw)
- [ ] Implement actual YAC pursuit physics
- [ ] Add pre-snap motion
- [ ] Support for running plays (currently pass-only)
- [ ] Multi-down drive simulation
- [ ] Persistence layer (PostgreSQL/SQLite)
- [ ] Frontend integration
- [ ] Replay visualization
- [ ] Advanced coverage schemes
- [ ] Player fatigue model

## License

MIT

## Contact

For questions or feedback, open an issue on GitHub.

---

Built with passion for football and machine learning.
