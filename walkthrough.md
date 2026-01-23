# FieldOS Engine Fixes & Enhancements

I have applied a series of critical, major, and minor fixes identified during the code audit, and added a **Visualization** feature.

## Changes Applied

### 1. Visualization (New Feature)

**Files**: `fieldos_engine/utils/viz.py`, `fieldos_engine/api/routes.py`, `pyproject.toml`

- **Feature**: The `/evaluate` endpoint now generates a **PNG bar chart** comparing play performance across different game situations.
- **Implementation**: Added `matplotlib` dependency, created a plotting module, and integrated it into the evaluation pipeline.
- **Output**: The API response now includes a `plot_url` field pointing to the generated image.

### 2. Metrics & Simulation Loop Fixes

**Files**: `fieldos_engine/api/routes.py`, `fieldos_engine/sim/metrics.py`

- **Issue**: The simulation loop was ignoring game situations, leading to empty "By Bucket" reports.
- **Fix**: Updated the loop in `simulate` endpoint to sample `GameSituation` properly and pass it to the metrics collector.
- **Fix**: Implemented `get_best_plays_by_bucket` in `MetricsCollector` to actually compute best plays based on mean yards gained.

### 3. RL Policy Persistence

**Files**: `fieldos_engine/rl/policy.py`, `fieldos_engine/api/routes.py`, `fieldos_engine/rl/train.py`

- **Issue**: Trained policies were not being saved, making training useless for inference.
- **Fix**: Added `save()` and `load()` methods to `BasePolicy` and its implementations (`EpsilonGreedyBandit`, `UpperConfidenceBound`).
- **Fix**: Updated `train_bandit` and `train_ppo` to return the trained policy object.
- **Fix**: Updated `/train` endpoint to save the policy to `fieldos_engine/data/policies/` and return the filename.

### 4. Validation Tightening

**Files**: `fieldos_engine/core/validation.py`, `fieldos_engine/sim/engine.py`

- **Issue**: Validation allowed for inexact rusher counts, and the engine executed rushes even if the defense call specified zero rushers.
- **Fix**: Made `validate_scenario` strict (rushers must exactly match the call).
- **Fix**: Updated `SimulationEngine` to check `defense_call.rushers_count > 0` before initiating a pass rush.

### 5. Logic Refactoring (DRY)

**Files**: `fieldos_engine/rl/evaluate.py`

- **Issue**: Logic for determining "Down & Distance Buckets" was duplicated, creating maintenance risk.
- **Fix**: Removed duplicate `_compute_bucket_from_situation` and reused the canonical `GameSituation.bucket` logic from the core models.

## How to View Plots

Run an evaluation request:

```bash
curl -X POST http://localhost:8000/evaluate ...
```

The JSON response will contain:

```json
{
  "policy_id": "baseline",
  "report": {...},
  "plot_url": "/static/plots/eval_plot_123.png"
}
```

You can find the image in `fieldos_engine/data/plots/`.
