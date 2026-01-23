"""API route handlers."""

import traceback
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status

from . import schemas
from ..core.models import Role
from ..core.registry import registry
from ..core.ids import generate_id
from ..sim.engine import SimulationEngine
from ..sim.metrics import MetricsCollector
from ..sim.trace import TraceSampler, serialize_trace
from ..rl.env import PlayCallingEnv
from ..rl.train import TrainingConfig, train_bandit, train_ppo
from ..rl.evaluate import evaluate_policy_detailed
from ..rl.policy import RandomPolicy

router = APIRouter()


# ============================================================================
# Health
# ============================================================================

@router.get("/health", response_model=schemas.HealthResponse)
async def health_check():
    """Health check endpoint."""
    return schemas.HealthResponse(ok=True, version="0.1.0")


# ============================================================================
# Plays CRUD
# ============================================================================

@router.get("/plays")
async def list_plays():
    """List all plays."""
    plays = registry.plays.list()
    return [play.model_dump() for play in plays]


@router.post("/plays", status_code=status.HTTP_201_CREATED)
async def create_play(request: schemas.CreatePlayRequest):
    """Create a new play."""
    try:
        play = registry.plays.create(request.play.id, request.play)
        return play.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/plays/{play_id}")
async def get_play(play_id: str):
    """Get a specific play."""
    play = registry.plays.get(play_id)
    if play is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Play {play_id} not found"
        )
    return play.model_dump()


@router.put("/plays/{play_id}")
async def update_play(play_id: str, request: schemas.UpdatePlayRequest):
    """Update a play."""
    try:
        play = registry.plays.update(play_id, request.play)
        return play.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ============================================================================
# Players CRUD
# ============================================================================

@router.get("/players")
async def list_players():
    """List all players."""
    players = registry.players.list()
    return [player.model_dump() for player in players]


@router.post("/players", status_code=status.HTTP_201_CREATED)
async def create_player(request: schemas.CreatePlayerRequest):
    """Create a new player."""
    try:
        player = registry.players.create(request.player.id, request.player)
        return player.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ============================================================================
# Scenarios CRUD
# ============================================================================

@router.get("/scenarios")
async def list_scenarios():
    """List all scenarios."""
    scenarios = registry.scenarios.list()
    return [scenario.model_dump() for scenario in scenarios]


@router.post("/scenarios", status_code=status.HTTP_201_CREATED)
async def create_scenario(request: schemas.CreateScenarioRequest):
    """Create a new scenario."""
    try:
        scenario = registry.scenarios.create(request.scenario.id, request.scenario)
        return scenario.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ============================================================================
# Rulesets CRUD
# ============================================================================

@router.get("/rules")
async def list_rulesets():
    """List all rulesets."""
    rulesets = registry.rulesets.list()
    return [ruleset.model_dump() for ruleset in rulesets]


@router.post("/rules", status_code=status.HTTP_201_CREATED)
async def create_ruleset(request: schemas.CreateRulesetRequest):
    """Create a new ruleset."""
    try:
        ruleset = registry.rulesets.create(request.ruleset.id, request.ruleset)
        return ruleset.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ============================================================================
# Routes CRUD
# ============================================================================

@router.get("/routes")
async def list_routes():
    """List all routes."""
    routes = registry.routes.list()
    return [route.model_dump() for route in routes]


@router.post("/routes", status_code=status.HTTP_201_CREATED)
async def create_route(request: schemas.CreateRouteRequest):
    """Create a new route."""
    try:
        route = registry.routes.create(request.route.id, request.route)
        return route.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ============================================================================
# Formations CRUD
# ============================================================================

@router.get("/formations")
async def list_formations():
    """List all formations."""
    formations = registry.formations.list()
    return [formation.model_dump() for formation in formations]


@router.post("/formations", status_code=status.HTTP_201_CREATED)
async def create_formation(request: schemas.CreateFormationRequest):
    """Create a new formation."""
    try:
        formation = registry.formations.create(request.formation.id, request.formation)
        return formation.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ============================================================================
# Simulation
# ============================================================================


# Setup logging
import logging
logger = logging.getLogger("fieldos_engine")

@router.post("/simulate", response_model=schemas.SimulateResponse)
async def simulate(request: schemas.SimulateRequest):
    """Run play simulation."""
    try:
        # Get play
        play = registry.plays.get(request.play_id)
        if play is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Play {request.play_id} not found"
            )

        # Get scenarios
        scenarios = []
        for scenario_id in request.scenario_ids:
            scenario = registry.scenarios.get(scenario_id)
            if scenario is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scenario {scenario_id} not found"
                )
            scenarios.append(scenario)

        # Get players (use defaults if not provided)
        if request.offensive_players:
            off_players = {
                role: registry.players.get(player_id)
                for role, player_id in request.offensive_players.items()
            }
        else:
            # Get default offensive players
            off_players = _get_default_offensive_players(play)

        if request.defensive_players:
            def_players = {
                role: registry.players.get(player_id)
                for role, player_id in request.defensive_players.items()
            }
        else:
            # Get default defensive players
            def_players = _get_default_defensive_players()

        # Check all players found
        for role, player in off_players.items():
            if player is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Offensive player for {role} not found"
                )

        for role, player in def_players.items():
            if player is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Defensive player for {role} not found"
                )
        
        # Prepare situations
        from ..core.models import GameSituation
        import numpy as np
        
        situations = request.situation_distribution
        if not situations:
            # Default set of situations
            situations = [
                GameSituation(down=1, yards_to_gain=25.0, yardline_to_goal=40.0),
                GameSituation(down=2, yards_to_gain=7.0, yardline_to_goal=33.0),
                GameSituation(down=3, yards_to_gain=3.0, yardline_to_goal=29.0),
                GameSituation(down=1, yards_to_gain=15.0, yardline_to_goal=15.0), # Redzone
            ]

        # Run simulation
        run_id = generate_id("run")
        engine = SimulationEngine(seed=request.seed)
        
        # Helper RNG for situation selection
        rng = np.random.Generator(np.random.PCG64(request.seed))
        
        collector = MetricsCollector()

        # Trace sampler
        sampler = TraceSampler(
            mode=request.trace_policy.mode,
            top_n=request.trace_policy.top_n,
            sample_rate=request.trace_policy.sample_rate,
            seed=request.seed
        )

        # Run episodes
        for episode in range(request.num_episodes):
            # Select scenario (round-robin)
            scenario = scenarios[episode % len(scenarios)]
            
            # Select situation (random from dist)
            situation_idx = rng.integers(0, len(situations))
            situation = situations[situation_idx]

            # Simulate
            should_trace = sampler.should_record(episode)
            outcome, trace = engine.simulate_play(
                play, scenario, off_players, def_players,
                mode=request.mode,
                record_trace=should_trace
            )

            # Record with context
            collector.record(request.play_id, outcome, situation)

            if trace:
                sampler.add_trace(trace)

        # Finalize
        metrics = collector.get_metrics().to_dict()
        
        # Add best plays (not really useful for single-play simulation, but strict compliance)
        # Note: best_plays only makes sense if simulating MULTIPLE plays.
        # But this endpoint takes ONE play_id.
        # So best_plays_by_bucket will just map bucket -> this_play_id (if valid).
        metrics["best_plays"] = collector.get_best_plays_by_bucket()
        
        traces = sampler.finalize()

        response = schemas.SimulateResponse(
            run_id=run_id,
            play_id=request.play_id,
            num_episodes=request.num_episodes,
            metrics=metrics,
            artifacts={
                "traces": [serialize_trace(t) for t in traces] if traces else []
            }
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        # Log full traceback server-side
        logger.error(f"Simulation error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )


# ============================================================================
# RL Training
# ============================================================================

@router.post("/train", response_model=schemas.TrainResponse)
async def train(request: schemas.TrainRequest):
    """Train RL policy."""
    try:
        # Validate plays and scenarios exist
        for play_id in request.play_ids:
            if registry.plays.get(play_id) is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Play {play_id} not found"
                )

        for scenario_id in request.scenario_ids:
            if registry.scenarios.get(scenario_id) is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scenario {scenario_id} not found"
                )

        # Create environment
        env = PlayCallingEnv(
            playbook=request.play_ids,
            scenarios=request.scenario_ids,
            offensive_players=request.offensive_players,
            defensive_players=request.defensive_players,
            seed=request.seed
        )

        # Training config
        config = TrainingConfig(
            total_steps=request.steps,
            algorithm=request.algo,
            epsilon=request.epsilon,
            learning_rate=request.learning_rate,
            ucb_c=request.ucb_c,
            seed=request.seed
        )

        # Policies dir
        from pathlib import Path
        policy_dir = Path(__file__).parent.parent / "data" / "policies"
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Train
        if request.algo == "PPO":
            result, policy = train_ppo(env, config)
        else:
            result, policy = train_bandit(env, config)

        training_id = generate_id("train")
        policy_filename = f"{training_id}.pkl"
        policy_path = policy_dir / policy_filename
        
        # Save policy
        policy.save(str(policy_path))

        response = schemas.TrainResponse(
            training_id=training_id,
            summary=result.to_dict(),
            policy_artifact_id=policy_filename
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


# ============================================================================
# RL Evaluation
# ============================================================================

@router.post("/evaluate", response_model=schemas.EvaluateResponse)
async def evaluate(request: schemas.EvaluateRequest):
    """Evaluate RL policy."""
    try:
        # Create environment
        env = PlayCallingEnv(
            playbook=request.play_ids,
            scenarios=request.scenario_ids,
            offensive_players=request.offensive_players,
            defensive_players=request.defensive_players,
            seed=request.seed
        )

        # Create baseline policy (random)
        policy = RandomPolicy(len(request.play_ids), seed=request.seed)

        # Evaluate
        report = evaluate_policy_detailed(
            env, policy, request.play_ids,
            num_episodes=request.num_episodes,
            seed=request.seed
        )

        response = schemas.EvaluateResponse(
            policy_id=request.policy_id,
            report=report.to_dict()
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


# ============================================================================
# Utilities
# ============================================================================

def _get_default_offensive_players(play) -> Dict[Role, Any]:
    """Get default offensive players for a play."""
    # Find first players with matching roles
    all_players = registry.players.list()
    players = {}

    for slot in play.formation.slots:
        role = slot.role
        # Find first player with this role
        player = next((p for p in all_players if p.role == role), None)
        if player:
            players[role] = player

    return players


def _get_default_defensive_players() -> Dict[Role, Any]:
    """Get default defensive players."""
    all_players = registry.players.list()
    players = {}

    def_roles = [Role.RUSHER, Role.CB1, Role.CB2, Role.SAFETY, Role.LB]
    for role in def_roles:
        player = next((p for p in all_players if p.role == role), None)
        if player:
            players[role] = player

    return players


# ============================================================================
# Seed Data
# ============================================================================

@router.post("/seed-demo-data", response_model=schemas.SeedDataResponse)
async def seed_demo_data_endpoint():
    """Load demo data into registry."""
    import json
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data" / "demo"

    # Clear existing
    registry.clear_all()

    # Load all data files
    counts = {
        "routes": 0,
        "formations": 0,
        "players": 0,
        "rulesets": 0,
        "plays": 0,
        "scenarios": 0
    }

    try:
        # Routes
        with open(data_dir / "routes.json") as f:
            routes_data = json.load(f)
            for route_dict in routes_data:
                from ..core.models import Route
                route = Route(**route_dict)
                registry.routes.create(route.id, route)
            counts["routes"] = len(routes_data)

        # Formations
        with open(data_dir / "formations.json") as f:
            formations_data = json.load(f)
            for formation_dict in formations_data:
                from ..core.models import Formation
                formation = Formation(**formation_dict)
                registry.formations.create(formation.id, formation)
            counts["formations"] = len(formations_data)

        # Players
        with open(data_dir / "players.json") as f:
            players_data = json.load(f)
            for player_dict in players_data:
                from ..core.models import Player
                player = Player(**player_dict)
                registry.players.create(player.id, player)
            counts["players"] = len(players_data)

        # Rulesets
        with open(data_dir / "rules.json") as f:
            rules_data = json.load(f)
            for ruleset_dict in rules_data:
                from ..core.models import Ruleset
                ruleset = Ruleset(**ruleset_dict)
                registry.rulesets.create(ruleset.id, ruleset)
            counts["rulesets"] = len(rules_data)

        # Plays
        with open(data_dir / "plays.json") as f:
            plays_data = json.load(f)
            for play_dict in plays_data:
                from ..core.models import Play
                play = Play(**play_dict)
                registry.plays.create(play.id, play)
            counts["plays"] = len(plays_data)

        # Scenarios
        with open(data_dir / "scenarios.json") as f:
            scenarios_data = json.load(f)
            for scenario_dict in scenarios_data:
                from ..core.models import Scenario
                scenario = Scenario(**scenario_dict)
                registry.scenarios.create(scenario.id, scenario)
            counts["scenarios"] = len(scenarios_data)

        return schemas.SeedDataResponse(
            plays_loaded=counts["plays"],
            players_loaded=counts["players"],
            routes_loaded=counts["routes"],
            formations_loaded=counts["formations"],
            scenarios_loaded=counts["scenarios"],
            rulesets_loaded=counts["rulesets"],
            message=f"Successfully loaded demo data: {sum(counts.values())} total entities"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load demo data: {str(e)}"
        )
