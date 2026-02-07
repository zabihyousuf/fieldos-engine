# FieldOS Engine - Codebase Report

## Overview

5v5 flag football simulation engine with RL-based play optimization, game simulation, scouting reports, and an AI play-calling advisor that learns opponent tendencies.

**Total Files**: ~50 Python files, ~25k+ lines of code
**Stack**: Python 3.9+, FastAPI, Pydantic, NumPy, Gymnasium, Stable-Baselines3, PyTorch, LangChain

---

## Feature Status

| Feature | Status | Quality | Notes |
|---------|--------|---------|-------|
| Core data models | COMPLETE | GOOD | Pydantic v2, comprehensive types |
| Play simulation engine | COMPLETE | GOOD | Physics-based, 50ms timesteps |
| Defensive coverage | COMPLETE | GOOD | Man/Zone schemes, situation adjustments |
| Game simulation (drives/downs) | COMPLETE | GOOD | Full game flow, extra points, scoring |
| Dynamic lineup selection | COMPLETE | GOOD | Position scoring, specialty filtering |
| Player names in reports | COMPLETE | GOOD | Names instead of WR1/WR2 |
| 10 players per team | COMPLETE | GOOD | Offense-only, defense-only, two-way |
| RL training (bandit/PPO) | COMPLETE | GOOD | Gymnasium env, multiple policies |
| Play generation system | COMPLETE | GOOD | Generates plays for RL training |
| Visualization | COMPLETE | GOOD | Static + animated play diagrams |
| REST API (CRUD) | COMPLETE | GOOD | Full CRUD for plays, players, etc. |
| Scouting reports | COMPLETE | GOOD | Player analysis, opponent awareness |
| Opponent model training | COMPLETE | GOOD | Learns per-situation play effectiveness |
| AI play advisor | COMPLETE | GOOD | Situation-specific recommendations |
| LangChain chat endpoint | COMPLETE | GOOD | Natural language with Claude fallback |
| API endpoints for advisor | COMPLETE | GOOD | /recommend, /chat, /train, /model |
| Test suite | COMPLETE | GOOD | 4 test files, ~21k lines |

---

## Directory Breakdown

### `fieldos_engine/core/` - Data Models & Registry

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `models.py` | 1022 | All Pydantic models (players, plays, game state) | None |
| `registry.py` | 109 | Thread-safe in-memory entity storage | None |
| `validation.py` | 201 | Input validation for all domain objects | None |
| `ids.py` | 26 | UUID-based ID generation | None |

**Assessment**: Solid foundation. Models are well-typed with Pydantic v2. Supports both legacy roles (CB1, CB2, SAFETY) and new generic D1-D5 roles. `DualRolePlayerAttributes` has unified speed/accel/agility with `PlayerSpecialty` enum.

### `fieldos_engine/sim/` - Simulation Engine

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `engine.py` | ~500 | Main simulation loop (discrete-time physics) | None |
| `coverage.py` | 646 | Man/Zone coverage schemes (Cover 0/1/2/3) | None |
| `motion.py` | 154 | Route interpolation, velocity, separation | None |
| `outcome.py` | 201 | Completion probability, YAC, sacks | None |
| `field.py` | 61 | Field coordinates, bounds checking | None |
| `game_simulator.py` | ~800 | Full game sim (drives, downs, scoring) | None |
| `metrics.py` | 224 | Stats aggregation and bucketed analysis | Minor: missing `Tuple` import |
| `trace.py` | 90 | Simulation trace sampling | None |

**Assessment**: The strongest part of the codebase. Physics-based simulation with 50ms timesteps, realistic coverage schemes, probabilistic outcomes. Game simulator handles drives, field position, downs, extra points (1-pt from 5yds, 2-pt from 12yds). Dynamic lineup selection picks the best 5 from 10 players per play.

### `fieldos_engine/rl/` - Reinforcement Learning

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `env.py` | 402 | Gymnasium environment for play calling | None |
| `game_env.py` | 487 | Game-level RL environment | None |
| `policy.py` | 263 | Random, EpsilonGreedy, UCB policies | None |
| `train.py` | 246 | Training loops (bandit + PPO) | None |
| `evaluate.py` | 176 | Policy evaluation and reporting | None |
| `play_generator.py` | 530 | Generates new plays for RL training | None |

**Assessment**: Fully functional RL pipeline. Gymnasium-compatible environments, multiple policy types, PPO via Stable-Baselines3. Play generator creates novel plays. All well-integrated with the simulation engine.

### `fieldos_engine/api/` - REST API

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `main.py` | 132 | FastAPI app, startup, CORS | None |
| `routes.py` | 639 | CRUD endpoints (plays, players, simulate, train) | None |
| `advisor_routes.py` | 344 | AI advisor endpoints (/recommend, /chat, /train) | None |
| `schemas.py` | 155 | Request/response Pydantic schemas | None |

**Assessment**: Clean REST API. CRUD for all entities, simulation endpoint, RL training endpoint. The advisor routes add `/api/advisor/recommend`, `/api/advisor/chat`, `/api/advisor/train`, `/api/advisor/model/{id}`, `/api/advisor/opponents`. Chat uses `ANTHROPIC_API_KEY` env var for Claude-powered responses, falls back to rule-based.

### `fieldos_engine/ai/` - AI Play Advisor (NEW, untracked)

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `opponent_model.py` | 562 | Per-opponent learning through simulation | None |
| `play_advisor.py` | 414 | Situation-specific play recommendations | None |
| `chat.py` | 287 | LangChain + Claude chatbot interface | None |

**Assessment**: Complete opponent-specific learning pipeline. Trains by simulating N games, records play effectiveness by situation key (down + distance bucket + field zone + score bucket). Models persist to JSON. PlayAdvisor generates recommendations with reasoning. Chat module injects trained model statistics into Claude's context for conversational play calling.

### `fieldos_engine/stats/` - Reports & Scouting

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `report_generator.py` | ~600 | Game reports (text + JSON) | None |
| `scouting.py` | 779 | Team scouting, play recommendations | None (untracked) |

**Assessment**: Comprehensive reporting. Game reports show drive-by-drive details with player names and lineups. Scouting analyzes player strengths/weaknesses and provides opponent-aware play recommendations.

### `fieldos_engine/utils/` - Utilities

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `viz.py` | 656 | Play visualization (static + animated) | None |
| `__init__.py` | MISSING | Package marker | **MISSING FILE** |

**Assessment**: Good visualization. Missing `__init__.py` could cause import issues in some contexts.

### `fieldos_engine/teams.py` - Team Definitions (NEW)

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `teams.py` | 303 | Shared team rosters (Butterfingers, Godbods) | None |

**Assessment**: Extracts team definitions from `run_game_simulation.py` so both the CLI and API can use them. Clean separation.

### `scripts/` - CLI Tools

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `run_game_simulation.py` | ~1210 | Game sim, scouting, advisor CLI | None |
| `train_policy.py` | 1020 | RL policy training scripts | None |
| `visualize_play.py` | 181 | Play visualization | None |
| `visualize_new_plays.py` | 133 | New play visualization | None |
| `seed_demo_data.py` | 88 | Demo data loader | None |
| `demo.sh` | - | Shell script runner | None |

### `tests/` - Test Suite

| File | Lines | Purpose |
|------|-------|---------|
| `test_validation.py` | 4110 | Model validation tests |
| `test_sim_determinism.py` | 6949 | Simulation determinism tests |
| `test_api_smoke.py` | 5535 | API endpoint smoke tests |
| `test_rl_smoke.py` | 4618 | RL pipeline smoke tests |

### `fieldos_engine/data/demo/` - Demo Data

8 JSON files (formations, plays, routes, players, scenarios, rules, motion/trick plays). All valid. Covers multiple formations (trips, bunch, twins, spread) and defensive scenarios.

---

## Bugs & Issues to Fix

### Must Fix

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `fieldos_engine/utils/__init__.py` | Missing file - could break imports | Create empty `__init__.py` |
| 2 | `fieldos_engine/sim/metrics.py` | Missing `Tuple` in typing import | Add `Tuple` to import line |

### Should Do

| # | Item | Why |
|---|------|-----|
| 3 | Git add `fieldos_engine/ai/` module | Currently untracked, will be lost |
| 4 | Git add `fieldos_engine/stats/scouting.py` | Currently untracked |
| 5 | Git add `fieldos_engine/teams.py` | Currently untracked |
| 6 | Git add `fieldos_engine/api/advisor_routes.py` | Currently untracked |

---

## Feature vs Goal Assessment

| Original Goal | Met? | Evidence |
|---------------|------|----------|
| Dynamic position assignment based on stats | YES | `game_simulator.py` scores players for QB/WR/DEF and picks best 5 |
| Height and weight on players | YES | `height_inches`, `weight_lbs` on `DualRolePlayerAttributes` |
| Player names in reports (not WR1/WR3) | YES | Reports show actual names via `offensive_lineup`/`defensive_lineup` |
| 10 players per team | YES | Both teams have 10 players with specialties |
| Offense-only / defense-only players | YES | `PlayerSpecialty` enum with `can_play_offense()`/`can_play_defense()` |
| Scouting report on a team | YES | `scouting.py` with full player analysis |
| "3rd and 5, what play should I run?" | YES | Both CLI (`--ai`, `--advisor`) and API (`/api/advisor/recommend`, `/api/advisor/chat`) |
| Your team never changes (Zabih's team) | YES | Butterfingers hardcoded, opponents are separate |
| Model per opponent that learns | YES | `OpponentModel` trains through N simulated games, saves to JSON |
| LLM-powered chat interface | YES | `chat.py` with LangChain + Claude, falls back to rule-based |
| Learn over many games/iterations | YES | `OpponentModelTrainer.train()` runs 100+ games, tracks by situation |

---

## Architecture Diagram

```
scripts/
  run_game_simulation.py  ─── CLI entry point
                                │
fieldos_engine/                 │
  teams.py ─────────────────────┤── Team definitions (shared)
  core/                         │
    models.py ──────────────────┤── All data types
    registry.py                 │── In-memory storage
    validation.py               │── Input validation
  sim/                          │
    engine.py ──────────────────┤── Physics simulation
    coverage.py                 │── Defensive AI
    game_simulator.py ──────────┤── Full game loop
  rl/                           │
    env.py ─────────────────────┤── RL environments
    train.py                    │── Policy training
  stats/                        │
    report_generator.py ────────┤── Game reports
    scouting.py                 │── Team scouting
  ai/                           │
    opponent_model.py ──────────┤── Per-opponent learning
    play_advisor.py             │── Recommendations
    chat.py ────────────────────┤── LangChain chatbot
  api/                          │
    main.py ────────────────────┤── FastAPI app
    routes.py                   │── CRUD endpoints
    advisor_routes.py ──────────┘── /recommend, /chat, /train
```

---

## API Endpoints Quick Reference

### Existing CRUD
- `GET /health`
- `GET|POST /plays`, `GET|PUT /plays/{id}`
- `GET|POST /players`, `GET|POST /scenarios`, `GET|POST /rules`, `GET|POST /routes`, `GET|POST /formations`
- `POST /simulate`, `POST /train`, `POST /evaluate`

### AI Advisor (NEW)
- `GET  /api/advisor/opponents` - List opponents and trained models
- `POST /api/advisor/train` - Train model (body: `{opponent_id, num_games}`)
- `GET  /api/advisor/model/{opponent_id}` - Model stats
- `POST /api/advisor/recommend` - Play recommendation (body: `{down, yards_to_go, field_position, score_diff, opponent_id}`)
- `POST /api/advisor/chat` - Chat (body: `{message, opponent_id}`) - set `ANTHROPIC_API_KEY` env var for Claude
- `POST /api/advisor/chat/clear` - Clear conversation

---

## What's Well Built

1. **Simulation engine** - Physics-based with proper timesteps, coverage schemes, probabilistic outcomes
2. **Data models** - Pydantic v2 with full validation, clean type system
3. **Test suite** - 21k+ lines of tests covering validation, determinism, API, RL
4. **Game simulation** - Drives, downs, field position, scoring, extra points all work
5. **Opponent learning** - Trains through game sim, tracks by situation, persists to disk
6. **API design** - Clean REST endpoints, proper schemas, CORS configured

## What Could Be Improved

1. **Opponent model tracks score_diff=0 for all plays in a drive** - `record_game()` sets `running_score_diff = 0` and never updates it during the drive (`opponent_model.py:240`). Should track actual score differential per play.
2. **Teams are hardcoded** - No API to create/manage teams dynamically. Adding a new opponent requires code changes in `teams.py`.
3. **Single-threaded training** - `POST /api/advisor/train` blocks the server during training. Should be a background task.
4. **No database** - Everything is in-memory (registry) or JSON files. Supabase is in dependencies but not wired up.
5. **Chat has no session management** - Single global `_chat_instance` means all API callers share conversation history.
6. **Duplicate team definitions** - `run_game_simulation.py` still has its own copy of team definitions alongside `teams.py`.
