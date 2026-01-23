# FieldOS Engine Code Audit

## What I Read

- `README.md`, `pyproject.toml`
- `fieldos_engine/core/`: `models.py`, `validation.py`, `registry.py`, `ids.py`
- `fieldos_engine/sim/`: `engine.py`, `field.py`, `motion.py`, `coverage.py`, `outcome.py`, `metrics.py`, `trace.py`
- `fieldos_engine/rl/`: `env.py`, `policy.py`, `train.py`, `evaluate.py`
- `fieldos_engine/api/`: `main.py`, `routes.py`, `schemas.py`
- `fieldos_engine/data/demo/`: `plays.json`, `scenarios.json`, `players.json`

## Findings

### Critical

1.  **Broken Metrics Bucketing in `/simulate`**:
    - The `simulate` endpoint in `routes.py` runs the simulation loop but **does not generate or pass a `GameSituation`** to the `MetricsCollector`.
    - As a result, `collector.record(outcome)` receives no situation, so `SlicedMetrics.by_bucket` is never populated. The "By Bucket" report in the API response will always be empty.
    - The `SimulateRequest` accepts `situation_distribution`, but it is completely ignored in the loop.

2.  **Missing RL Policy Persistence**:
    - The user requirement "Policy artifacts are persisted in a clear, deterministic format" is not met.
    - `api/routes.py` returns `training_id` as `policy_artifact_id`, but no file is actually saved.
    - `rl/policy.py` classes have no `save()` or `load()` methods.

### Major

3.  **Duplicated Logic (DRY Violation)**:
    - `rl/evaluate.py` contains `_compute_bucket_from_situation` which duplicates logic from `fieldos_engine.core.models.GameSituation.bucket`. This poses a risk of drift where RL evaluation buckets differ from Core model buckets.

4.  **Incomplete Best Plays Report**:
    - `fieldos_engine/sim/metrics.py`: `get_best_plays_by_bucket` is a placeholder returning `{}`. This means the engine cannot currently recommend plays based on simulation results.

### Minor

5.  **Validation Looseness**:
    - `fieldos_engine/core/validation.py`: `validate_scenario` allows `scenario.defense_call.rushers_count + 1` rushers. This "flexibility" is undocumented and likely fragile. It should be exact.

6.  **Production Logging**:
    - `api/routes.py` uses `print()` for logging errors and startup events. Should use Python's `logging` module.

7.  **Down/Distance Logic**:
    - 4th Down is implicitly hashed to "3rd Down" buckets in `GameSituation.bucket`. While acceptable for MVP, it should be explicit or have its own bucket.

## Fix Plan

### 1. Fix Metrics & Simulation Loop

- Modify `SimulationEngine.simulate_play` or the caller (`routes.py`) to handle `GameSituation`.
- In `routes.py`, inside the `simulate` loop:
  - Sample a `GameSituation` from `request.situation_distribution` (or default).
  - Pass this situation to `collector.record(outcome, situation)`.
- Update `MetricsCollector.get_best_plays_by_bucket` to actually compute the best plays (e.g. highest mean yards or completion rate).

### 2. Implement RL Persistence

- Add `save(path)` and `load(path)` to `BasePolicy` and implementations in `rl/policy.py`.
- Use `pickle` or `json` (for weights) to save to a `data/policies/` directory.
- Update `api/routes.py` to save the trained policy and return the actual filename.

### 3. Refactor Bucket Logic

- Remove `_compute_bucket_from_situation` from `rl/evaluate.py`.
- Use `GameSituation(**situation_dict).bucket` instead.

### 4. Tighten Validation

- Remove the `+ 1` allowance in `validate_scenario` for rushers.

### 5. Logging

- Configure basic logging in `api/main.py`.
- Replace `print` with `logger.info` / `logger.error`.

## Remaining Limitations (Post-Fix)

- 4th Down will still map to 3rd Down buckets (keeping as known limitation for MVP).
- Physics model remains "point-mass" (no momentum/turning radius beyond speed clamping).
- YAC (Yards After Catch) is a simple probabilistic model, not fully simulated pursuit.
