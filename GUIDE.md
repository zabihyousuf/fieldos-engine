# FieldOS Engine - Complete Guide

This guide explains how the simulation and reinforcement learning system works, how to run plays, and how to train the system.

## Table of Contents

1. [Quick Start: Running the Demo](#quick-start-running-the-demo)
2. [Running a Single Play](#running-a-single-play)
3. [Running All Plays and Getting a Report](#running-all-plays-and-getting-a-report)
4. [How the Simulation Works](#how-the-simulation-works)
5. [Understanding the RL System](#understanding-the-rl-system)
6. [Training Your System](#training-your-system)
7. [Getting Play Recommendations](#getting-play-recommendations)
8. [API Reference](#api-reference)

---

## Quick Start: Running the Demo

```bash
# Run the full demo
./scripts/demo.sh
```

This will:
1. Set up the virtual environment
2. Install dependencies
3. Run tests
4. Start the API server (demo data loads automatically)
5. Run example API calls showing simulation and training

---

## Running a Single Play

### Option 1: Using curl (API)

Start the server first:
```bash
source venv/bin/activate
uvicorn fieldos_engine.api.main:app --port 8000
```

Then run a single play simulation:
```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "play_id": "play_trips_flood",
    "scenario_ids": ["scenario_man_cover1_1rush"],
    "num_episodes": 50,
    "seed": 42,
    "mode": "EVAL"
  }'
```

### Option 2: Using Python directly

```python
from fieldos_engine.core.registry import registry
from fieldos_engine.core.models import GameSituation
from fieldos_engine.sim.engine import SimulationEngine
from fieldos_engine.sim.metrics import MetricsCollector

# Load demo data (if not already loaded)
# The API server does this automatically on startup

# Get play and scenario
play = registry.plays.get("play_trips_flood")
scenario = registry.scenarios.get("scenario_man_cover1_1rush")

# Get players
off_players = {slot.role: registry.players.list()[i]
               for i, slot in enumerate(play.formation.slots)}
def_players = {role: p for p in registry.players.list()
               if p.side.value == "DEFENSE" for role in [p.role]}

# Run simulation
engine = SimulationEngine(seed=42)
outcome, trace = engine.simulate_play(play, scenario, off_players, def_players)

print(f"Outcome: {outcome.outcome}")
print(f"Yards: {outcome.yards_gained}")
print(f"Completion Probability: {outcome.completion_probability}")
```

### Understanding the Response

```json
{
  "run_id": "run_abc123",
  "play_id": "play_trips_flood",
  "num_episodes": 50,
  "metrics": {
    "overall": {
      "num_plays": 50,
      "completion_rate": 0.68,      // 68% of passes completed
      "sack_rate": 0.08,            // 8% ended in sacks
      "intercept_rate": 0.02,       // 2% were intercepted
      "yards_mean": 8.4,            // Average yards per play
      "yards_p50": 7.2,             // Median yards (50th percentile)
      "yards_p90": 14.1,            // Big play potential (90th percentile)
      "time_to_throw_mean": 1850.5, // Average QB decision time in ms
      "target_distribution": {       // Who got targeted
        "WR1": 20,
        "WR2": 25,
        "WR3": 5
      },
      "failure_modes": {             // Why plays failed
        "TIGHT_WINDOW": 12,          // Defender too close
        "SACK_BEFORE_THROW": 4       // QB got sacked
      }
    }
  }
}
```

---

## Running All Plays and Getting a Report

### Run All Plays Against All Scenarios

```bash
# Start the server
uvicorn fieldos_engine.api.main:app --port 8000 &

# Run each play against each scenario
for PLAY in play_trips_flood play_bunch_quick_slants play_twins_smash play_tight_levels play_spread_vertical play_empty_all_go; do
  for SCENARIO in scenario_man_cover0 scenario_man_cover1_1rush scenario_zone_cover2 scenario_zone_cover3; do
    echo "=== $PLAY vs $SCENARIO ==="
    curl -s -X POST "http://localhost:8000/simulate" \
      -H "Content-Type: application/json" \
      -d "{
        \"play_id\": \"$PLAY\",
        \"scenario_ids\": [\"$SCENARIO\"],
        \"num_episodes\": 100,
        \"seed\": 42,
        \"mode\": \"EVAL\"
      }" | python3 -c "
import sys, json
d = json.load(sys.stdin)
m = d['metrics']['overall']
print(f\"  Completion: {m['completion_rate']:.0%}  Yards: {m['yards_mean']:.1f}  P90: {m['yards_p90']:.1f}\")
"
  done
done
```

### Get a Full Report with Python Script

Create `scripts/full_report.py`:
```python
#!/usr/bin/env python3
"""Generate a full report of all plays vs all scenarios."""

import sys
sys.path.insert(0, '.')

from fieldos_engine.core.registry import registry
from fieldos_engine.sim.engine import SimulationEngine
from fieldos_engine.sim.metrics import MetricsCollector
from fieldos_engine.core.models import GameSituation, Role

# Load demo data
from fieldos_engine.api.main import load_demo_data
load_demo_data()

# Get all plays and scenarios
plays = registry.plays.list()
scenarios = registry.scenarios.list()

# Get default players
off_roles = [Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3]
def_roles = [Role.RUSHER, Role.CB1, Role.CB2, Role.SAFETY, Role.LB]
all_players = registry.players.list()

off_players = {role: next(p for p in all_players if p.role == role) for role in off_roles}
def_players = {role: next(p for p in all_players if p.role == role) for role in def_roles}

# Run simulations
print("=" * 80)
print("FIELDOS ENGINE - FULL PLAY ANALYSIS REPORT")
print("=" * 80)
print()

results = []
engine = SimulationEngine(seed=42)

for play in plays:
    print(f"\n{'='*60}")
    print(f"PLAY: {play.name} ({play.id})")
    print(f"{'='*60}")

    for scenario in scenarios:
        collector = MetricsCollector()
        situations = [
            GameSituation(down=1, yards_to_gain=25.0, yardline_to_goal=40.0),
            GameSituation(down=3, yards_to_gain=5.0, yardline_to_goal=30.0),
        ]

        for episode in range(50):
            situation = situations[episode % len(situations)]
            outcome, _ = engine.simulate_play(play, scenario, off_players, def_players)
            collector.record(play.id, outcome, situation)

        metrics = collector.get_metrics()
        overall = metrics.overall

        results.append({
            'play': play.name,
            'scenario': scenario.name,
            'completion_rate': overall.completion_rate,
            'yards_mean': overall.yards_mean,
            'yards_p90': overall.yards_p90,
            'sack_rate': overall.sack_rate
        })

        print(f"  vs {scenario.name:30} | Comp: {overall.completion_rate:5.0%} | "
              f"Yards: {overall.yards_mean:5.1f} | P90: {overall.yards_p90:5.1f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: BEST PLAYS BY SCENARIO")
print("=" * 80)

for scenario in scenarios:
    scenario_results = [r for r in results if r['scenario'] == scenario.name]
    best = max(scenario_results, key=lambda x: x['yards_mean'])
    print(f"\n{scenario.name}:")
    print(f"  Best Play: {best['play']}")
    print(f"  Expected Yards: {best['yards_mean']:.1f}")
    print(f"  Completion Rate: {best['completion_rate']:.0%}")
```

Run it:
```bash
python scripts/full_report.py
```

---

## How the Simulation Works

### The Simulation Loop (50ms timesteps)

```
SNAP (time=0)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  For each timestep (every 50ms until throw or sack):   │
│                                                         │
│  1. UPDATE RECEIVER POSITIONS                           │
│     - Each WR/CENTER follows their assigned route       │
│     - Position interpolated based on route breakpoints  │
│     - Speed limited by player attributes                │
│                                                         │
│  2. UPDATE DEFENDER POSITIONS                           │
│     - Man coverage: Chase assigned receiver             │
│     - Zone coverage: Move toward zone, react to threats │
│     - Rusher: Move toward QB (if rushers_count > 0)     │
│                                                         │
│  3. CHECK FOR SACK                                      │
│     - If rusher within 2 yards of QB → SACK             │
│                                                         │
│  4. QB DECISION                                         │
│     - After decision_latency_ms, evaluate targets       │
│     - Check separation for each receiver in progression │
│     - If good window (>0.3 score) or near max_time → THROW │
│                                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
THROW DECISION MADE
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  COMPUTE OUTCOME:                                       │
│                                                         │
│  1. Calculate throw distance                            │
│  2. Calculate separation at throw and at catch          │
│  3. Compute completion probability based on:            │
│     - QB accuracy (short/mid/deep)                      │
│     - Separation from defenders                         │
│     - Coverage shell/type                               │
│     - Time pressure                                     │
│                                                         │
│  4. Roll for outcome:                                   │
│     - COMPLETE → Receiver catches, compute YAC          │
│     - INCOMPLETE → Ball falls incomplete                │
│     - INTERCEPT → Defender catches ball                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
RETURN PlayOutcome
```

### Key Factors That Affect Outcomes

| Factor | Impact |
|--------|--------|
| **Route Design** | Breakpoints determine where receiver will be at each moment |
| **Player Speed** | Faster receivers create more separation |
| **QB Accuracy** | Higher accuracy = better completion % at all distances |
| **QB Decision Time** | Slower decision = more time for rush/coverage |
| **Coverage Type** | Man is tighter, Zone has holes |
| **Rush Count** | More rushers = less time to throw |
| **Separation** | Most important factor for completion probability |

---

## Understanding the RL System

### What is the RL System?

The RL (Reinforcement Learning) system learns **which plays work best in which situations**. It's a "contextual bandit" - meaning it learns a mapping from game situations to optimal play choices.

### The Problem It Solves

```
SITUATION                          →  BEST PLAY
─────────────────────────────────────────────────
3rd & Short, Man Coverage          →  ?
3rd & Long, Zone Coverage          →  ?
Redzone, Cover 0                   →  ?
1st & 10, Cover 2                  →  ?
```

The RL system figures out these mappings through trial and error.

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  For each training step:                                │
│                                                         │
│  1. OBSERVE SITUATION                                   │
│     - What's the down/distance?                         │
│     - What coverage is the defense playing?             │
│     - Where on the field are we?                        │
│                                                         │
│  2. SELECT PLAY (exploration vs exploitation)           │
│     - With probability ε: pick random play (explore)    │
│     - Otherwise: pick best known play (exploit)         │
│                                                         │
│  3. SIMULATE THE PLAY                                   │
│     - Run the full physics simulation                   │
│     - Get outcome (yards, completion, sack, etc.)       │
│                                                         │
│  4. COMPUTE REWARD                                      │
│     - Base: yards gained (clipped -5 to +25)            │
│     - Bonus: +10 for converting on 3rd/4th down         │
│     - Penalty: -5 for sack, -10 for interception        │
│                                                         │
│  5. UPDATE Q-VALUES                                     │
│     - Q(situation, play) += α × (reward - Q(situation, play)) │
│     - This is how the system "learns"                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### The Q-Table

After training, you get a Q-table that looks like this:

```
                    Play 0    Play 1    Play 2    Play 3
                    (Trips)   (Bunch)   (Twins)   (Spread)
─────────────────────────────────────────────────────────
1st & Any           7.2       6.8       8.1*      5.4
2nd & Short         6.5       7.9*      6.2       5.8
2nd & Long          8.5*      7.2       7.8       6.1
3rd & Short         5.2       9.1*      6.8       4.9
3rd & Long          9.8*      8.4       8.9       7.2
Redzone             6.1       5.8       7.2*      5.5
Goalline            4.8       6.2*      5.1       4.2

* = best play for that situation
```

### Available Algorithms

1. **Epsilon-Greedy Bandit** (default)
   - Simple and effective
   - Explores with probability ε, exploits otherwise
   - Good for: Quick learning, simple situations

2. **UCB (Upper Confidence Bound)**
   - Smarter exploration
   - Tries plays that haven't been tested much
   - Good for: Thorough exploration, finding hidden gems

3. **PPO (Proximal Policy Optimization)**
   - Deep learning approach
   - Handles complex state representations
   - Good for: Large playbooks, nuanced situations

---

## Training Your System

### Basic Training (via API)

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "play_ids": [
      "play_trips_flood",
      "play_bunch_quick_slants",
      "play_twins_smash",
      "play_tight_levels"
    ],
    "scenario_ids": [
      "scenario_man_cover0",
      "scenario_man_cover1_1rush",
      "scenario_zone_cover2",
      "scenario_zone_cover3"
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
    "steps": 5000,
    "algo": "BANDIT",
    "epsilon": 0.1,
    "learning_rate": 0.1
  }'
```

### Training Response Explained

```json
{
  "training_id": "train_xyz789",
  "summary": {
    "algorithm": "BANDIT",
    "total_steps": 5000,
    "final_reward_mean": 8.4,      // Average reward after training
    "final_reward_std": 3.2,       // Consistency of results
    "best_actions_per_bucket": {
      "0": 2,  // 1ST_ANY → Play #2 (Twins Smash)
      "1": 1,  // 2ND_SHORT → Play #1 (Bunch Quick Slants)
      "3": 0,  // 3RD_SHORT → Play #0 (Trips Flood)
      "4": 2,  // 3RD_LONG → Play #2 (Twins Smash)
      "5": 1,  // REDZONE → Play #1 (Bunch Quick Slants)
      "6": 3   // GOALLINE → Play #3 (Tight Levels)
    }
  },
  "policy_artifact_id": "train_xyz789.pkl"
}
```

### Training with Python

```python
from fieldos_engine.rl.env import PlayCallingEnv
from fieldos_engine.rl.train import train_bandit, TrainingConfig
from fieldos_engine.core.models import Role

# Define playbook
playbook = [
    "play_trips_flood",
    "play_bunch_quick_slants",
    "play_twins_smash",
    "play_tight_levels"
]

# Define scenarios to train against
scenarios = [
    "scenario_man_cover1_1rush",
    "scenario_zone_cover2"
]

# Create environment
env = PlayCallingEnv(
    playbook=playbook,
    scenarios=scenarios,
    offensive_players={
        Role.QB: "player_qb1",
        Role.CENTER: "player_center1",
        Role.WR1: "player_wr1_1",
        Role.WR2: "player_wr2_1",
        Role.WR3: "player_wr3_1"
    },
    defensive_players={
        Role.RUSHER: "player_rusher1",
        Role.CB1: "player_cb1_1",
        Role.CB2: "player_cb2_1",
        Role.SAFETY: "player_safety1",
        Role.LB: "player_lb1"
    },
    seed=42
)

# Train
config = TrainingConfig(
    total_steps=5000,
    algorithm="BANDIT",
    epsilon=0.1,
    learning_rate=0.1,
    seed=42
)

result, policy = train_bandit(env, config)

# See results
print("Best plays per situation:")
for bucket_id, play_idx in result.best_actions_per_bucket.items():
    bucket_names = ['1ST_ANY', '2ND_SHORT', '2ND_LONG', '3RD_SHORT', '3RD_LONG', 'REDZONE', 'GOALLINE']
    print(f"  {bucket_names[bucket_id]}: {playbook[play_idx]}")

# Save the trained policy
policy.save("my_trained_policy.pkl")
```

---

## Getting Play Recommendations

Once trained, you can use the policy to get recommendations:

```python
from fieldos_engine.rl.policy import EpsilonGreedyBandit
import numpy as np

# Load trained policy
policy = EpsilonGreedyBandit(num_actions=4)
policy.load("my_trained_policy.pkl")

# Create observation for current situation
# Format: [bucket_onehot(7), yards_to_gain(norm), yardline(norm), shell_onehot(4), type_onehot(2)]

def get_play_recommendation(situation_bucket, defense_shell, defense_type, playbook):
    """Get recommended play for a situation."""

    # Build observation vector
    obs = np.zeros(15, dtype=np.float32)

    # Bucket one-hot (positions 0-6)
    bucket_map = {
        '1ST_ANY': 0, '2ND_SHORT': 1, '2ND_LONG': 2,
        '3RD_SHORT': 3, '3RD_LONG': 4, 'REDZONE': 5, 'GOALLINE': 6
    }
    obs[bucket_map[situation_bucket]] = 1.0

    # Normalized yards/yardline (positions 7-8) - use defaults
    obs[7] = 0.4  # yards_to_gain normalized
    obs[8] = 0.6  # yardline normalized

    # Shell one-hot (positions 9-12)
    shell_map = {'COVER0': 9, 'COVER1': 10, 'COVER2': 11, 'COVER3': 12}
    obs[shell_map[defense_shell]] = 1.0

    # Type one-hot (positions 13-14)
    type_map = {'MAN': 13, 'ZONE': 14}
    obs[type_map[defense_type]] = 1.0

    # Get recommendation
    action = policy.predict(obs, deterministic=True)
    return playbook[action]

# Example usage
playbook = ["play_trips_flood", "play_bunch_quick_slants", "play_twins_smash", "play_tight_levels"]

print("Recommendations:")
print(f"  3rd & Short vs Man: {get_play_recommendation('3RD_SHORT', 'COVER1', 'MAN', playbook)}")
print(f"  3rd & Long vs Zone: {get_play_recommendation('3RD_LONG', 'COVER3', 'ZONE', playbook)}")
print(f"  Redzone vs Cover 0: {get_play_recommendation('REDZONE', 'COVER0', 'MAN', playbook)}")
```

---

## API Reference

### Simulation

**POST /simulate**
```json
{
  "play_id": "string",           // Required: Play to simulate
  "scenario_ids": ["string"],    // Required: Defensive scenarios
  "num_episodes": 50,            // Number of simulations
  "seed": 42,                    // For reproducibility
  "mode": "EVAL",                // EVAL or POLICY
  "trace_policy": {
    "mode": "NONE"               // NONE, TOP_N, or SAMPLE_RATE
  }
}
```

### Training

**POST /train**
```json
{
  "play_ids": ["string"],        // Playbook for training
  "scenario_ids": ["string"],    // Scenarios to train against
  "offensive_players": {},       // Role -> player_id mapping
  "defensive_players": {},       // Role -> player_id mapping
  "steps": 1000,                 // Training steps
  "algo": "BANDIT",              // BANDIT, UCB, or PPO
  "epsilon": 0.1,                // Exploration rate
  "learning_rate": 0.1,          // Learning rate
  "seed": 42
}
```

### List Entities

- **GET /plays** - List all plays
- **GET /scenarios** - List all scenarios
- **GET /players** - List all players
- **GET /routes** - List all routes
- **GET /formations** - List all formations

### Create Entities

- **POST /plays** - Create a new play
- **POST /scenarios** - Create a new scenario
- **POST /players** - Create a new player
- **POST /routes** - Create a new route
- **POST /formations** - Create a new formation

---

## Summary

1. **To run a single play**: Use `POST /simulate` with a play_id and scenario_ids
2. **To get a full report**: Run all plays against all scenarios, compare metrics
3. **To train the system**: Use `POST /train` with your playbook and scenarios
4. **To get recommendations**: Load the trained policy and call `policy.predict()`

The key insight is that **the RL system doesn't play football** - it learns which plays work best in which situations by running thousands of simulations and tracking what works.
