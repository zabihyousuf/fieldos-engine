"""FastAPI application entry point."""

import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from ..core.registry import registry
from ..core.models import Route, Formation, Player, Ruleset, Play, Scenario

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fieldos_engine")

# Create FastAPI app
app = FastAPI(
    title="FieldOS Engine",
    description="5v5 flag football simulation and RL optimization engine",
    version="0.1.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


def load_demo_data():
    """Load demo data into registry on startup."""
    data_dir = Path(__file__).parent.parent / "data" / "demo"

    if not data_dir.exists():
        logger.warning(f"Demo data directory not found: {data_dir}")
        return

    try:
        # Load routes
        with open(data_dir / "routes.json") as f:
            for item in json.load(f):
                route = Route(**item)
                registry.routes.create(route.id, route)

        # Load formations
        with open(data_dir / "formations.json") as f:
            for item in json.load(f):
                formation = Formation(**item)
                registry.formations.create(formation.id, formation)

        # Load players
        with open(data_dir / "players.json") as f:
            for item in json.load(f):
                player = Player(**item)
                registry.players.create(player.id, player)

        # Load rulesets
        with open(data_dir / "rules.json") as f:
            for item in json.load(f):
                ruleset = Ruleset(**item)
                registry.rulesets.create(ruleset.id, ruleset)

        # Load plays
        with open(data_dir / "plays.json") as f:
            for item in json.load(f):
                play = Play(**item)
                registry.plays.create(play.id, play)

        # Load motion plays (if exists)
        motion_plays_path = data_dir / "motion_plays.json"
        if motion_plays_path.exists():
            with open(motion_plays_path) as f:
                for item in json.load(f):
                    play = Play(**item)
                    registry.plays.create(play.id, play)

        # Load trick plays (if exists)
        trick_plays_path = data_dir / "trick_plays.json"
        if trick_plays_path.exists():
            with open(trick_plays_path) as f:
                for item in json.load(f):
                    play = Play(**item)
                    registry.plays.create(play.id, play)

        # Load scenarios
        with open(data_dir / "scenarios.json") as f:
            for item in json.load(f):
                scenario = Scenario(**item)
                registry.scenarios.create(scenario.id, scenario)

        logger.info(f"Demo data loaded: {registry.plays.count()} plays, "
                    f"{registry.players.count()} players, "
                    f"{registry.routes.count()} routes, "
                    f"{registry.formations.count()} formations, "
                    f"{registry.scenarios.count()} scenarios, "
                    f"{registry.rulesets.count()} rulesets")
    except Exception as e:
        logger.error(f"Failed to load demo data: {e}")


@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    logger.info("FieldOS Engine starting up...")
    load_demo_data()
    logger.info("API documentation available at http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks."""
    logger.info("FieldOS Engine shutting down...")
