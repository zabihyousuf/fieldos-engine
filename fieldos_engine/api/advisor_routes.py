"""API routes for the AI play advisor and chatbot."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..core.models import (
    Team, GamePlayer, DualRolePlayerAttributes, GameConfig, PlayerSpecialty
)
from ..ai.opponent_model import OpponentModel, OpponentModelTrainer
from ..ai.play_advisor import PlayAdvisor
from ..ai.chat import PlayCallingChat

logger = logging.getLogger("fieldos_engine.advisor_api")

advisor_router = APIRouter(prefix="/api/advisor", tags=["advisor"])

# Directories
MODELS_DIR = Path(__file__).parent.parent.parent / "results" / "models"

# Global state - teams and chat instances loaded at startup
_my_team: Optional[Team] = None
_opponents: Dict[str, Team] = {}
_chat_instance: Optional[PlayCallingChat] = None


# ============================================================================
# Request / Response schemas
# ============================================================================

class RecommendRequest(BaseModel):
    """Play recommendation request."""
    down: int = Field(ge=1, le=3, description="Current down (1, 2, or 3)")
    yards_to_go: float = Field(gt=0, description="Yards needed for first down")
    field_position: float = Field(default=23.0, ge=0, le=46,
                                   description="Yards from own goal (0=own endzone, 46=opponent endzone)")
    score_diff: int = Field(default=0, description="Score differential (positive = winning)")
    opponent_id: str = Field(description="Opponent team ID")


class PlayRecommendationResponse(BaseModel):
    """A single play recommendation."""
    rank: int
    play_id: str
    play_name: str
    confidence: float
    reasoning: str
    success_rate: float
    avg_yards: float
    touchdowns: int
    turnovers: int
    attempts: int


class RecommendResponse(BaseModel):
    """Full recommendation response."""
    situation: str
    model_confidence: str
    top_pick: PlayRecommendationResponse
    alternatives: List[PlayRecommendationResponse]
    strategic_notes: List[str]
    model_games_trained: int


class TrainRequest(BaseModel):
    """Train opponent model request."""
    opponent_id: str = Field(description="Opponent team ID to train against")
    num_games: int = Field(default=100, ge=1, le=1000, description="Number of games to simulate")


class TrainResponse(BaseModel):
    """Train response."""
    opponent_id: str
    opponent_name: str
    games_trained: int
    record: str
    win_rate: float
    avg_points_for: float
    avg_points_against: float
    plays_learned: int
    model_path: str


class ModelInfoResponse(BaseModel):
    """Opponent model info response."""
    opponent_id: str
    opponent_name: str
    games_played: int
    record: str
    win_rate: float
    avg_points_for: float
    avg_points_against: float
    plays_learned: int
    total_plays_recorded: int
    last_trained: Optional[str]
    top_plays: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    """Chat message request."""
    message: str = Field(description="Your message (e.g. 'it's 3rd and 5 what should I run?')")
    opponent_id: Optional[str] = Field(default=None, description="Opponent team ID (optional if already set)")


class ChatResponse(BaseModel):
    """Chat response."""
    response: str
    opponent: Optional[str] = None


class OpponentListResponse(BaseModel):
    """List of available opponents."""
    opponents: List[Dict[str, str]]
    models_trained: List[str]


# ============================================================================
# Team setup - call this from startup
# ============================================================================

def setup_teams(my_team: Team, opponents: Dict[str, Team]):
    """Initialize the advisor with team data. Called at app startup."""
    global _my_team, _opponents, _chat_instance
    _my_team = my_team
    _opponents = opponents
    _chat_instance = PlayCallingChat(my_team, opponents, MODELS_DIR)
    logger.info(f"Advisor initialized: {my_team.name} vs {list(opponents.keys())}")


def _get_team():
    if _my_team is None:
        raise HTTPException(status_code=500, detail="Teams not initialized. Call setup_teams first.")
    return _my_team


def _get_opponent(opponent_id: str) -> Team:
    if opponent_id not in _opponents:
        available = list(_opponents.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Opponent '{opponent_id}' not found. Available: {available}"
        )
    return _opponents[opponent_id]


# ============================================================================
# Endpoints
# ============================================================================

@advisor_router.get("/opponents", response_model=OpponentListResponse)
async def list_opponents():
    """List available opponents and which have trained models."""
    _get_team()
    opponents_list = [{"id": t.id, "name": t.name} for t in _opponents.values()]

    # Check which have models
    trained = []
    for oid in _opponents:
        model_path = MODELS_DIR / f"opponent_{oid}.json"
        if model_path.exists():
            trained.append(oid)

    return OpponentListResponse(opponents=opponents_list, models_trained=trained)


@advisor_router.post("/recommend", response_model=RecommendResponse)
async def recommend_play(request: RecommendRequest):
    """
    Get a play recommendation for a specific game situation.

    Uses the trained opponent model to recommend the best play based on
    learned effectiveness data from simulated games.
    """
    my_team = _get_team()
    opponent = _get_opponent(request.opponent_id)

    advisor = PlayAdvisor(my_team, MODELS_DIR)

    # Check if model exists
    model_path = MODELS_DIR / f"opponent_{opponent.id}.json"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {opponent.name}. Train one first: POST /api/advisor/train"
        )

    response = advisor.recommend(
        opponent=opponent,
        down=request.down,
        yards_to_go=request.yards_to_go,
        field_position=request.field_position,
        score_diff=request.score_diff,
        auto_train=False
    )

    if not response:
        raise HTTPException(status_code=404, detail="No recommendation available for this situation.")

    model = OpponentModel.load(model_path)

    def _to_play_response(rec) -> PlayRecommendationResponse:
        return PlayRecommendationResponse(
            rank=rec.rank,
            play_id=rec.play_id,
            play_name=rec.play_name,
            confidence=rec.confidence,
            reasoning=rec.reasoning,
            success_rate=rec.stats.get("success_rate", 0),
            avg_yards=rec.stats.get("avg_yards", 0),
            touchdowns=rec.stats.get("touchdowns", 0),
            turnovers=rec.stats.get("turnovers", 0),
            attempts=rec.stats.get("attempts", 0),
        )

    return RecommendResponse(
        situation=response.situation_summary,
        model_confidence=response.model_confidence,
        top_pick=_to_play_response(response.top_recommendation),
        alternatives=[_to_play_response(a) for a in response.alternatives],
        strategic_notes=response.strategic_notes,
        model_games_trained=model.games_played,
    )


@advisor_router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train an opponent model by simulating games.

    This runs N simulated games against the opponent and learns
    what plays work in what situations. The model is saved to disk
    and used for future recommendations.
    """
    my_team = _get_team()
    opponent = _get_opponent(request.opponent_id)

    trainer = OpponentModelTrainer(my_team, MODELS_DIR)
    model = trainer.train(opponent, num_games=request.num_games, verbose=False)

    summary = model.get_model_summary()

    return TrainResponse(
        opponent_id=opponent.id,
        opponent_name=opponent.name,
        games_trained=summary["games_played"],
        record=summary["record"],
        win_rate=summary["win_rate"],
        avg_points_for=summary["avg_points_for"],
        avg_points_against=summary["avg_points_against"],
        plays_learned=summary["plays_in_model"],
        model_path=str(trainer.get_model_path(opponent.id)),
    )


@advisor_router.get("/model/{opponent_id}", response_model=ModelInfoResponse)
async def get_model_info(opponent_id: str):
    """Get info about a trained opponent model."""
    _get_team()
    opponent = _get_opponent(opponent_id)

    model_path = MODELS_DIR / f"opponent_{opponent.id}.json"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {opponent.name}. Train one first."
        )

    model = OpponentModel.load(model_path)
    summary = model.get_model_summary()

    # Top plays
    top_plays = []
    for play_id, data in model.overall_play_stats.items():
        stats = data["stats"]
        if stats["attempts"] >= 5:
            top_plays.append({
                "play_name": data["play_name"],
                "success_rate": round(stats["success_rate"], 1),
                "avg_yards": round(stats["avg_yards"], 1),
                "touchdowns": stats["touchdowns"],
                "turnovers": stats["turnovers"],
                "attempts": stats["attempts"],
            })
    top_plays.sort(key=lambda x: (x["success_rate"], x["avg_yards"]), reverse=True)

    return ModelInfoResponse(
        opponent_id=opponent.id,
        opponent_name=opponent.name,
        games_played=summary["games_played"],
        record=summary["record"],
        win_rate=summary["win_rate"],
        avg_points_for=summary["avg_points_for"],
        avg_points_against=summary["avg_points_against"],
        plays_learned=summary["plays_in_model"],
        total_plays_recorded=summary["total_plays_recorded"],
        last_trained=summary["last_trained"],
        top_plays=top_plays[:5],
    )


@advisor_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the AI play calling assistant.

    Send natural language messages like:
    - "it's 3rd and 5, what should I run?"
    - "we're in the redzone down by 7, what play?"
    - "what's our best play against their defense?"
    - "tell me about Maddox's strengths"

    Set the ANTHROPIC_API_KEY environment variable for Claude-powered responses.
    Without it, falls back to rule-based analysis from the trained model.
    """
    global _chat_instance

    if _chat_instance is None:
        raise HTTPException(status_code=500, detail="Chat not initialized.")

    opponent_id = request.opponent_id or _chat_instance.current_opponent_id
    if not opponent_id:
        raise HTTPException(status_code=400, detail="No opponent specified. Provide opponent_id.")

    response_text = _chat_instance.chat(
        user_message=request.message,
        opponent_id=opponent_id,
    )

    return ChatResponse(
        response=response_text,
        opponent=opponent_id,
    )


@advisor_router.post("/chat/clear")
async def clear_chat():
    """Clear the chat conversation history."""
    global _chat_instance
    if _chat_instance:
        _chat_instance.clear_history()
    return {"status": "ok", "message": "Conversation history cleared."}
