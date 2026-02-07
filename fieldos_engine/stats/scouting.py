"""
Scouting Report Generator - Creates detailed team analysis and play recommendations.

This module provides:
- Team scouting reports (player analysis, tendencies, strengths/weaknesses)
- Situational play recommendations ("It's 3rd and 5, what play should I run?")
- Historical performance analysis
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

from ..core.models import (
    Team, GamePlayer, GameResult, PlayGameStats, DriveResult,
    PlayerSpecialty, DualRolePlayerAttributes
)


@dataclass
class PlayerScoutingReport:
    """Detailed scouting report for a single player."""
    player_id: str
    name: str
    number: int
    height: str  # e.g., "6'2\""
    weight: int

    # Role suitability scores
    qb_score: float
    wr_outside_score: float
    wr_slot_score: float
    center_score: float
    defender_man_score: float
    defender_zone_score: float
    rusher_score: float

    # Best positions
    best_offensive_role: str
    best_defensive_role: str
    specialty: str  # "OFFENSE_ONLY", "DEFENSE_ONLY", "TWO_WAY"

    # Key attributes
    top_strengths: List[str]
    weaknesses: List[str]

    # Game stats (if available)
    games_played: int = 0
    pass_attempts: int = 0
    completions: int = 0
    passing_yards: float = 0.0
    touchdowns_thrown: int = 0
    targets: int = 0
    receptions: int = 0
    receiving_yards: float = 0.0
    touchdowns_receiving: int = 0


@dataclass
class TeamScoutingReport:
    """Complete scouting report for a team."""
    team_id: str
    team_name: str

    # Roster breakdown
    total_players: int
    offense_only_players: int
    defense_only_players: int
    two_way_players: int

    # Top players by position
    best_qb: str
    best_wr1: str
    best_wr2: str
    best_wr3_slot: str
    best_center: str
    best_defenders: List[str]
    best_rusher: str

    # Team strengths/weaknesses
    offensive_strengths: List[str]
    offensive_weaknesses: List[str]
    defensive_strengths: List[str]
    defensive_weaknesses: List[str]

    # Player reports
    player_reports: List[PlayerScoutingReport]

    # Playbook analysis
    playbook_size: int
    play_ids: List[str]


@dataclass
class PlayRecommendation:
    """A recommended play for a given situation."""
    play_id: str
    play_name: str
    confidence: float  # 0-100
    reasoning: str
    expected_yards: float
    success_rate: float  # Historical or estimated
    risk_level: str  # "LOW", "MEDIUM", "HIGH"


@dataclass
class SituationalRecommendation:
    """Play recommendations for a specific game situation."""
    down: int
    yards_to_go: float
    field_zone: str  # "OWN_TERRITORY", "MIDFIELD", "REDZONE", "GOALLINE"
    score_diff: int  # Positive = winning, negative = losing

    # Recommendations ranked by confidence
    recommendations: List[PlayRecommendation]
    situation_summary: str


class ScoutingReportGenerator:
    """Generates scouting reports and play recommendations."""

    def __init__(
        self,
        team: Team,
        game_history: Optional[List[GameResult]] = None
    ):
        self.team = team
        self.game_history = game_history or []

        # Aggregate historical stats
        self._aggregate_game_stats()

    def _aggregate_game_stats(self):
        """Aggregate statistics from game history."""
        self.player_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "games": 0, "pass_attempts": 0, "completions": 0, "passing_yards": 0.0,
            "touchdowns_thrown": 0, "targets": 0, "receptions": 0, "receiving_yards": 0.0,
            "touchdowns_receiving": 0
        })
        self.play_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "attempts": 0, "completions": 0, "yards": 0.0, "touchdowns": 0, "turnovers": 0,
            "by_down": {1: {"attempts": 0, "success": 0}, 2: {"attempts": 0, "success": 0}, 3: {"attempts": 0, "success": 0}},
            "by_zone": defaultdict(lambda: {"attempts": 0, "success": 0})
        })

        for game in self.game_history:
            # Find our team's stats
            if game.home_team_id == self.team.id:
                team_stats = game.home_stats
            elif game.away_team_id == self.team.id:
                team_stats = game.away_stats
            else:
                continue

            # Aggregate player stats
            for player_id, pstats in team_stats.player_stats.items():
                ps = self.player_stats[player_id]
                ps["games"] += 1
                ps["pass_attempts"] += pstats.pass_attempts
                ps["completions"] += pstats.completions
                ps["passing_yards"] += pstats.passing_yards
                ps["touchdowns_thrown"] += pstats.touchdowns_thrown
                ps["targets"] += pstats.targets
                ps["receptions"] += pstats.receptions
                ps["receiving_yards"] += pstats.receiving_yards
                ps["touchdowns_receiving"] += pstats.touchdowns_receiving

            # Aggregate play stats
            for play_id, play_stats in team_stats.play_stats.items():
                ps = self.play_stats[play_id]
                ps["attempts"] += play_stats.attempts
                ps["completions"] += play_stats.completions
                ps["yards"] += play_stats.total_yards
                ps["touchdowns"] += play_stats.touchdowns
                ps["turnovers"] += play_stats.turnovers

                for down in [1, 2, 3]:
                    ps["by_down"][down]["attempts"] += play_stats.times_called_by_down.get(down, 0)
                    ps["by_down"][down]["success"] += play_stats.success_by_down.get(down, 0)

                for zone, count in play_stats.times_called_by_zone.items():
                    ps["by_zone"][zone]["attempts"] += count

    def _analyze_player(self, player: GamePlayer) -> PlayerScoutingReport:
        """Create detailed scouting report for a player."""
        attrs = player.attributes

        # Calculate position scores
        qb_score = attrs.qb_score()
        wr_outside = attrs.wr_score("outside")
        wr_slot = attrs.wr_score("slot")
        center_score = attrs.center_score()
        def_man = attrs.defender_score("man")
        def_zone = attrs.defender_score("zone")
        rusher = attrs.rusher_score()

        # Determine best roles
        offense_scores = [("QB", qb_score), ("WR (Outside)", wr_outside), ("WR (Slot)", wr_slot), ("Center", center_score)]
        defense_scores = [("Man Coverage", def_man), ("Zone Coverage", def_zone), ("Pass Rusher", rusher)]

        best_offense = max(offense_scores, key=lambda x: x[1])[0] if attrs.can_play_offense() else "N/A"
        best_defense = max(defense_scores, key=lambda x: x[1])[0] if attrs.can_play_defense() else "N/A"

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        if attrs.speed >= 90:
            strengths.append("Elite speed")
        elif attrs.speed < 70:
            weaknesses.append("Below average speed")

        if attrs.hands >= 90:
            strengths.append("Outstanding hands")
        elif attrs.hands < 65:
            weaknesses.append("Inconsistent hands")

        if attrs.route_running >= 90:
            strengths.append("Excellent route runner")
        elif attrs.route_running < 65:
            weaknesses.append("Raw route running")

        if attrs.throw_power >= 85 and attrs.short_acc >= 80:
            strengths.append("Strong arm, accurate")
        elif attrs.throw_power < 50:
            weaknesses.append("Weak arm")

        if attrs.man_coverage >= 90:
            strengths.append("Lockdown man coverage")
        elif attrs.man_coverage < 65 and attrs.can_play_defense():
            weaknesses.append("Struggles in man coverage")

        if attrs.pass_rush >= 90:
            strengths.append("Elite pass rusher")

        if attrs.height_inches >= 76:
            strengths.append(f"Great size ({attrs.height_formatted()})")
        elif attrs.height_inches <= 68:
            strengths.append(f"Quick and agile ({attrs.height_formatted()})")

        # Get historical stats
        pstats = self.player_stats.get(player.id, {})

        return PlayerScoutingReport(
            player_id=player.id,
            name=player.name,
            number=player.number,
            height=attrs.height_formatted(),
            weight=int(attrs.weight_lbs),
            qb_score=round(qb_score, 1),
            wr_outside_score=round(wr_outside, 1),
            wr_slot_score=round(wr_slot, 1),
            center_score=round(center_score, 1),
            defender_man_score=round(def_man, 1),
            defender_zone_score=round(def_zone, 1),
            rusher_score=round(rusher, 1),
            best_offensive_role=best_offense,
            best_defensive_role=best_defense,
            specialty=attrs.specialty.value,
            top_strengths=strengths[:4],
            weaknesses=weaknesses[:3],
            games_played=pstats.get("games", 0),
            pass_attempts=pstats.get("pass_attempts", 0),
            completions=pstats.get("completions", 0),
            passing_yards=pstats.get("passing_yards", 0.0),
            touchdowns_thrown=pstats.get("touchdowns_thrown", 0),
            targets=pstats.get("targets", 0),
            receptions=pstats.get("receptions", 0),
            receiving_yards=pstats.get("receiving_yards", 0.0),
            touchdowns_receiving=pstats.get("touchdowns_receiving", 0),
        )

    def generate_team_report(self) -> TeamScoutingReport:
        """Generate complete team scouting report."""
        # Count player types
        offense_only = sum(1 for p in self.team.players if p.attributes.specialty == PlayerSpecialty.OFFENSE_ONLY)
        defense_only = sum(1 for p in self.team.players if p.attributes.specialty == PlayerSpecialty.DEFENSE_ONLY)
        two_way = sum(1 for p in self.team.players if p.attributes.specialty == PlayerSpecialty.TWO_WAY)

        # Find best players by position
        offense_players = [p for p in self.team.players if p.attributes.can_play_offense()]
        defense_players = [p for p in self.team.players if p.attributes.can_play_defense()]

        best_qb = max(offense_players, key=lambda p: p.attributes.qb_score()).name if offense_players else "N/A"
        wr_outside = sorted(offense_players, key=lambda p: p.attributes.wr_score("outside"), reverse=True)
        best_wr1 = wr_outside[0].name if len(wr_outside) > 0 else "N/A"
        best_wr2 = wr_outside[1].name if len(wr_outside) > 1 else "N/A"
        best_wr3 = max(offense_players, key=lambda p: p.attributes.wr_score("slot")).name if offense_players else "N/A"
        best_center = max(offense_players, key=lambda p: p.attributes.center_score()).name if offense_players else "N/A"

        best_defenders = sorted(defense_players, key=lambda p: p.attributes.defender_score("man"), reverse=True)[:3]
        best_rusher = max(defense_players, key=lambda p: p.attributes.rusher_score()).name if defense_players else "N/A"

        # Identify team strengths/weaknesses
        off_strengths = []
        off_weaknesses = []
        def_strengths = []
        def_weaknesses = []

        # Analyze offense
        avg_speed = sum(p.attributes.speed for p in offense_players) / len(offense_players) if offense_players else 0
        avg_hands = sum(p.attributes.hands for p in offense_players) / len(offense_players) if offense_players else 0

        if avg_speed >= 88:
            off_strengths.append("Fast receiving corps")
        elif avg_speed < 80:
            off_weaknesses.append("Lack of speed in receiving corps")

        if avg_hands >= 85:
            off_strengths.append("Reliable hands")
        elif avg_hands < 75:
            off_weaknesses.append("Drop issues")

        # Check QB quality
        top_qb = max(offense_players, key=lambda p: p.attributes.qb_score()) if offense_players else None
        if top_qb and top_qb.attributes.qb_score() >= 75:
            off_strengths.append(f"Elite QB ({top_qb.name})")
        elif top_qb and top_qb.attributes.qb_score() < 60:
            off_weaknesses.append("QB limitations")

        # Analyze defense
        avg_coverage = sum(p.attributes.defender_score("man") for p in defense_players) / len(defense_players) if defense_players else 0

        if avg_coverage >= 85:
            def_strengths.append("Strong coverage unit")
        elif avg_coverage < 75:
            def_weaknesses.append("Coverage concerns")

        top_rusher = max(defense_players, key=lambda p: p.attributes.rusher_score()) if defense_players else None
        if top_rusher and top_rusher.attributes.rusher_score() >= 85:
            def_strengths.append(f"Elite pass rusher ({top_rusher.name})")

        # Generate player reports
        player_reports = [self._analyze_player(p) for p in self.team.players]

        return TeamScoutingReport(
            team_id=self.team.id,
            team_name=self.team.name,
            total_players=len(self.team.players),
            offense_only_players=offense_only,
            defense_only_players=defense_only,
            two_way_players=two_way,
            best_qb=best_qb,
            best_wr1=best_wr1,
            best_wr2=best_wr2,
            best_wr3_slot=best_wr3,
            best_center=best_center,
            best_defenders=[p.name for p in best_defenders],
            best_rusher=best_rusher,
            offensive_strengths=off_strengths,
            offensive_weaknesses=off_weaknesses,
            defensive_strengths=def_strengths,
            defensive_weaknesses=def_weaknesses,
            player_reports=player_reports,
            playbook_size=len(self.team.playbook),
            play_ids=list(self.team.playbook),
        )

    def _analyze_opponent(self, opponent: Team) -> Dict[str, Any]:
        """Analyze opponent's defensive strengths and weaknesses."""
        defense_players = [p for p in opponent.players if p.attributes.can_play_defense()]

        if not defense_players:
            return {"weakness": None, "strength": None}

        # Calculate average defensive stats
        avg_man = sum(p.attributes.man_coverage for p in defense_players) / len(defense_players)
        avg_zone = sum(p.attributes.zone_coverage for p in defense_players) / len(defense_players)
        avg_speed = sum(p.attributes.speed for p in defense_players) / len(defense_players)
        best_rusher_score = max(p.attributes.rusher_score() for p in defense_players)

        # Find the weakest defender
        weakest = min(defense_players, key=lambda p: p.attributes.defender_score("man"))
        fastest_wr_needed = weakest.attributes.speed < 85

        analysis = {
            "avg_man_coverage": avg_man,
            "avg_zone_coverage": avg_zone,
            "avg_speed": avg_speed,
            "best_rusher_score": best_rusher_score,
            "weakest_defender": weakest.name,
            "weakest_defender_speed": weakest.attributes.speed,
            "weakness": None,
            "strength": None,
            "recommendations": []
        }

        # Identify weaknesses
        if avg_man < avg_zone - 5:
            analysis["weakness"] = "man_coverage"
            analysis["recommendations"].append("Attack with quick routes vs man coverage")
        elif avg_zone < avg_man - 5:
            analysis["weakness"] = "zone_coverage"
            analysis["recommendations"].append("Use crossing routes to exploit zone gaps")

        if avg_speed < 82:
            analysis["weakness"] = "speed"
            analysis["recommendations"].append(f"Use speed - their defense averages {avg_speed:.0f} speed")

        if best_rusher_score >= 90:
            analysis["strength"] = "pass_rush"
            analysis["recommendations"].append(f"Quick passing game - they have elite pass rush")

        return analysis

    def recommend_play(
        self,
        down: int,
        yards_to_go: float,
        field_zone: str = "MIDFIELD",
        score_diff: int = 0,
        opponent_team: Optional[Team] = None
    ) -> SituationalRecommendation:
        """
        Recommend plays for a specific game situation.

        Args:
            down: Current down (1, 2, or 3)
            yards_to_go: Yards needed for first down/touchdown
            field_zone: "OWN_TERRITORY", "MIDFIELD", "OPPONENT_TERRITORY", "REDZONE", "GOALLINE"
            score_diff: Point differential (positive = winning)
            opponent_team: Optional opponent for matchup analysis

        Returns:
            SituationalRecommendation with ranked play suggestions
        """
        recommendations = []

        # Analyze opponent if provided
        opponent_analysis = None
        if opponent_team:
            opponent_analysis = self._analyze_opponent(opponent_team)

        # Analyze each play in playbook
        for play_id in self.team.playbook:
            play_stats = self.play_stats.get(play_id, {})

            # Calculate base success rate and expected yards
            if play_stats.get("attempts", 0) > 0:
                success_rate = (play_stats["completions"] / play_stats["attempts"]) * 100
                avg_yards = play_stats["yards"] / play_stats["attempts"]

                # Get down-specific stats
                down_stats = play_stats.get("by_down", {}).get(down, {"attempts": 0, "success": 0})
                if down_stats["attempts"] > 0:
                    down_success = (down_stats["success"] / down_stats["attempts"]) * 100
                else:
                    down_success = success_rate
            else:
                # No historical data - use estimates based on play type
                success_rate = 55.0  # Default estimate
                avg_yards = 5.0
                down_success = success_rate

            # Adjust for situation
            confidence = down_success
            reasoning_parts = []

            # === OPPONENT-SPECIFIC ADJUSTMENTS ===
            if opponent_analysis:
                # Exploit man coverage weakness
                if opponent_analysis.get("weakness") == "man_coverage":
                    if "slant" in play_id.lower() or "mesh" in play_id.lower() or "pick" in play_id.lower():
                        confidence += 12
                        reasoning_parts.append(f"Exploits {opponent_team.name}'s weak man coverage")

                # Exploit zone coverage weakness
                if opponent_analysis.get("weakness") == "zone_coverage":
                    if "flood" in play_id.lower() or "levels" in play_id.lower() or "cross" in play_id.lower():
                        confidence += 12
                        reasoning_parts.append(f"Attacks {opponent_team.name}'s zone coverage gaps")

                # Exploit speed weakness
                if opponent_analysis.get("weakness") == "speed":
                    if "vertical" in play_id.lower() or "go" in play_id.lower() or "streak" in play_id.lower():
                        confidence += 15
                        reasoning_parts.append(f"Use speed vs their slow secondary ({opponent_analysis['avg_speed']:.0f} avg)")

                # Counter elite pass rush with quick passes
                if opponent_analysis.get("strength") == "pass_rush":
                    if "screen" in play_id.lower() or "slant" in play_id.lower() or "quick" in play_id.lower():
                        confidence += 10
                        reasoning_parts.append(f"Quick release beats {opponent_team.name}'s pass rush")
                    elif "vertical" in play_id.lower() or "deep" in play_id.lower():
                        confidence -= 8
                        reasoning_parts.append(f"Risk: {opponent_team.name} has elite pass rush")

            # Down adjustments
            if down == 3:
                if yards_to_go <= 5:
                    reasoning_parts.append("Short yardage situation")
                    # Favor high-percentage plays
                    if "slant" in play_id.lower() or "stick" in play_id.lower():
                        confidence += 10
                        reasoning_parts.append("Quick-hitting route concept")
                else:
                    reasoning_parts.append("Must-convert 3rd and long")
                    # Favor plays that can get big yards
                    if avg_yards >= yards_to_go:
                        confidence += 15
                        reasoning_parts.append(f"Averages {avg_yards:.1f} yds (need {yards_to_go:.1f})")
            elif down == 1:
                reasoning_parts.append("First down - aggressive opportunity")
                if "vertical" in play_id.lower() or "flood" in play_id.lower():
                    confidence += 5
                    reasoning_parts.append("Take a shot downfield")

            # Zone adjustments
            if field_zone == "REDZONE" or field_zone == "GOALLINE":
                reasoning_parts.append("Redzone situation")
                if "fade" in play_id.lower() or play_stats.get("touchdowns", 0) > 0:
                    confidence += 10
                    reasoning_parts.append("Scoring play")

            # Score adjustments
            if score_diff < -7:  # Losing by more than a TD
                reasoning_parts.append("Need to catch up")
                if avg_yards >= 8:
                    confidence += 5
                    reasoning_parts.append("Explosive potential")
            elif score_diff > 7:  # Winning big
                reasoning_parts.append("Protect the lead")
                if success_rate >= 60:
                    confidence += 5
                    reasoning_parts.append("High percentage")

            # Risk assessment
            if play_stats.get("turnovers", 0) > 2:
                risk_level = "HIGH"
                confidence -= 10
                reasoning_parts.append("Turnover risk")
            elif success_rate >= 65:
                risk_level = "LOW"
            else:
                risk_level = "MEDIUM"

            # Clean up play name for display
            play_name = play_id.replace("play_", "").replace("_", " ").title()

            recommendations.append(PlayRecommendation(
                play_id=play_id,
                play_name=play_name,
                confidence=min(100, max(0, confidence)),
                reasoning=". ".join(reasoning_parts) if reasoning_parts else "Standard play call",
                expected_yards=avg_yards,
                success_rate=success_rate,
                risk_level=risk_level
            ))

        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        # Generate situation summary
        down_str = f"{down}{'st' if down == 1 else 'nd' if down == 2 else 'rd'}"
        if yards_to_go <= 3:
            distance_str = "short"
        elif yards_to_go <= 7:
            distance_str = "medium"
        else:
            distance_str = "long"

        summary = f"{down_str} and {distance_str} ({yards_to_go:.0f} yards) in {field_zone.lower().replace('_', ' ')}"
        if score_diff > 0:
            summary += f", up by {score_diff}"
        elif score_diff < 0:
            summary += f", down by {abs(score_diff)}"

        if opponent_team:
            summary += f" vs {opponent_team.name}"

        return SituationalRecommendation(
            down=down,
            yards_to_go=yards_to_go,
            field_zone=field_zone,
            score_diff=score_diff,
            recommendations=recommendations[:5],  # Top 5
            situation_summary=summary
        )

    def to_text_report(self) -> str:
        """Generate text-based scouting report."""
        report = self.generate_team_report()

        lines = [
            "=" * 70,
            f"SCOUTING REPORT: {report.team_name.upper()}",
            "=" * 70,
            "",
            "ROSTER OVERVIEW",
            "-" * 40,
            f"Total Players: {report.total_players}",
            f"  Two-Way Players: {report.two_way_players}",
            f"  Offense Only: {report.offense_only_players}",
            f"  Defense Only: {report.defense_only_players}",
            "",
            "PROJECTED STARTERS",
            "-" * 40,
            f"QB: {report.best_qb}",
            f"WR1 (X): {report.best_wr1}",
            f"WR2 (Z): {report.best_wr2}",
            f"WR3 (Slot): {report.best_wr3_slot}",
            f"Center: {report.best_center}",
            "",
            "Top Defenders: " + ", ".join(report.best_defenders),
            f"Pass Rusher: {report.best_rusher}",
            "",
        ]

        # Strengths and weaknesses
        lines.extend([
            "OFFENSIVE ANALYSIS",
            "-" * 40,
        ])
        if report.offensive_strengths:
            lines.append("Strengths: " + ", ".join(report.offensive_strengths))
        if report.offensive_weaknesses:
            lines.append("Weaknesses: " + ", ".join(report.offensive_weaknesses))

        lines.extend([
            "",
            "DEFENSIVE ANALYSIS",
            "-" * 40,
        ])
        if report.defensive_strengths:
            lines.append("Strengths: " + ", ".join(report.defensive_strengths))
        if report.defensive_weaknesses:
            lines.append("Weaknesses: " + ", ".join(report.defensive_weaknesses))

        # Player breakdowns
        lines.extend([
            "",
            "=" * 70,
            "PLAYER BREAKDOWN",
            "=" * 70,
        ])

        for player in report.player_reports:
            lines.extend([
                "",
                f"#{player.number} {player.name} ({player.height}, {player.weight} lbs) - {player.specialty}",
                "-" * 40,
                f"  Best Offensive Role: {player.best_offensive_role} (Score: {player.qb_score if 'QB' in player.best_offensive_role else player.wr_outside_score})",
                f"  Best Defensive Role: {player.best_defensive_role}",
            ])
            if player.top_strengths:
                lines.append(f"  Strengths: {', '.join(player.top_strengths)}")
            if player.weaknesses:
                lines.append(f"  Areas to Improve: {', '.join(player.weaknesses)}")
            if player.games_played > 0:
                lines.append(f"  Season Stats ({player.games_played} games):")
                if player.pass_attempts > 0:
                    comp_pct = (player.completions / player.pass_attempts) * 100 if player.pass_attempts else 0
                    lines.append(f"    Passing: {player.completions}/{player.pass_attempts} ({comp_pct:.0f}%), {player.passing_yards:.0f} yds, {player.touchdowns_thrown} TD")
                if player.targets > 0:
                    catch_pct = (player.receptions / player.targets) * 100 if player.targets else 0
                    lines.append(f"    Receiving: {player.receptions}/{player.targets} ({catch_pct:.0f}%), {player.receiving_yards:.0f} yds, {player.touchdowns_receiving} TD")

        # Playbook
        lines.extend([
            "",
            "=" * 70,
            "PLAYBOOK",
            "=" * 70,
            f"Total Plays: {report.playbook_size}",
        ])
        for play_id in report.play_ids:
            play_name = play_id.replace("play_", "").replace("_", " ").title()
            lines.append(f"  - {play_name}")

        lines.append("=" * 70)

        return "\n".join(lines)


def generate_scouting_report(
    team: Team,
    game_history: Optional[List[GameResult]] = None,
    format: str = "text"
) -> str:
    """
    Generate a scouting report for a team.

    Args:
        team: The Team to scout
        game_history: Optional list of past game results
        format: "text" or "json"

    Returns:
        The formatted scouting report
    """
    generator = ScoutingReportGenerator(team, game_history)

    if format == "json":
        report = generator.generate_team_report()
        # Convert to dict
        import dataclasses
        return json.dumps(dataclasses.asdict(report), indent=2)
    else:
        return generator.to_text_report()


def recommend_play_for_situation(
    team: Team,
    down: int,
    yards_to_go: float,
    field_zone: str = "MIDFIELD",
    score_diff: int = 0,
    game_history: Optional[List[GameResult]] = None,
    opponent_team: Optional[Team] = None
) -> str:
    """
    Get play recommendation for a specific situation.

    Args:
        team: Your team
        down: Current down (1, 2, or 3)
        yards_to_go: Yards needed
        field_zone: Field position
        score_diff: Point differential
        game_history: Historical games for analysis
        opponent_team: Optional opponent team for matchup analysis

    Returns:
        Text recommendation
    """
    generator = ScoutingReportGenerator(team, game_history)
    rec = generator.recommend_play(down, yards_to_go, field_zone, score_diff, opponent_team)

    lines = [
        "=" * 60,
        f"SITUATION: {rec.situation_summary.upper()}",
        "=" * 60,
    ]

    # Show opponent analysis if available
    if opponent_team:
        opp_analysis = generator._analyze_opponent(opponent_team)
        lines.extend([
            "",
            f"OPPONENT ANALYSIS: {opponent_team.name}",
            "-" * 40,
        ])
        if opp_analysis.get("weakness"):
            lines.append(f"  Weakness: {opp_analysis['weakness'].replace('_', ' ').title()}")
        if opp_analysis.get("strength"):
            lines.append(f"  Strength: {opp_analysis['strength'].replace('_', ' ').title()}")
        lines.append(f"  Avg Defense Speed: {opp_analysis['avg_speed']:.0f}")
        lines.append(f"  Weakest Defender: {opp_analysis['weakest_defender']} (Speed: {opp_analysis['weakest_defender_speed']:.0f})")
        if opp_analysis.get("recommendations"):
            lines.append(f"  Strategy: {opp_analysis['recommendations'][0]}")

    lines.extend([
        "",
        "RECOMMENDED PLAYS:",
        "-" * 40,
    ])

    for i, play in enumerate(rec.recommendations, 1):
        lines.extend([
            f"",
            f"{i}. {play.play_name}",
            f"   Confidence: {play.confidence:.0f}% | Expected: {play.expected_yards:.1f} yds | Risk: {play.risk_level}",
            f"   Reasoning: {play.reasoning}",
        ])

    if rec.recommendations:
        top = rec.recommendations[0]
        lines.extend([
            "",
            "-" * 40,
            f"TOP PICK: {top.play_name}",
            f">> {top.reasoning}",
        ])

    lines.append("=" * 60)

    return "\n".join(lines)
