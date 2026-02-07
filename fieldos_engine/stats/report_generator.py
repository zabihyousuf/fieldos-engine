"""
Game Report Generator - Creates detailed post-game analysis.

Generates comprehensive reports including:
- Game summary
- Player statistics
- Play effectiveness analysis
- Situational breakdowns
- Key moments
- Strategic recommendations
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from ..core.models import (
    GameResult, TeamGameStats, PlayerGameStats, PlayGameStats,
    DriveRecord, DriveResult, OutcomeType, FieldZone
)


@dataclass
class PlayEffectivenessReport:
    """Report on a single play's effectiveness."""
    play_id: str
    play_name: str
    times_called: int
    success_rate: float
    avg_yards: float
    touchdowns: int
    turnovers: int
    best_down: Optional[int]
    best_zone: Optional[str]
    worst_down: Optional[int]
    worst_zone: Optional[str]
    recommendation: str


@dataclass
class SituationalBreakdown:
    """Breakdown of performance by situation."""
    situation: str
    plays_run: int
    success_rate: float
    avg_yards: float
    best_play: Optional[str]
    worst_play: Optional[str]


@dataclass
class KeyMoment:
    """A key moment in the game."""
    drive: int
    play_number: int
    description: str
    impact: str
    yards: float


@dataclass
class GameReport:
    """Complete game analysis report."""
    game_id: str
    timestamp: str

    # Summary
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    winner: Optional[str]

    # Stats
    home_stats_summary: Dict[str, Any]
    away_stats_summary: Dict[str, Any]

    # Analysis
    play_effectiveness: Dict[str, PlayEffectivenessReport]
    situational_analysis: List[SituationalBreakdown]
    key_moments: List[KeyMoment]

    # Recommendations
    recommendations: List[str]


class GameReportGenerator:
    """Generates comprehensive game analysis reports."""

    def __init__(self, game_result: GameResult):
        self.game = game_result

    def generate(self) -> GameReport:
        """Generate complete game report."""
        return GameReport(
            game_id=self.game.game_id,
            timestamp=datetime.now().isoformat(),
            home_team=self.game.home_team_id,
            away_team=self.game.away_team_id,
            home_score=self.game.home_score,
            away_score=self.game.away_score,
            winner=self.game.winner,
            home_stats_summary=self._summarize_team_stats(self.game.home_stats),
            away_stats_summary=self._summarize_team_stats(self.game.away_stats),
            play_effectiveness=self._analyze_play_effectiveness(),
            situational_analysis=self._analyze_situations(),
            key_moments=self._identify_key_moments(),
            recommendations=self._generate_recommendations()
        )

    def _summarize_team_stats(self, stats: TeamGameStats) -> Dict[str, Any]:
        """Create summary of team statistics."""
        return {
            "total_points": stats.total_points,
            "touchdowns": stats.touchdowns,
            "total_plays": stats.total_plays,
            "total_yards": round(stats.total_yards, 1),
            "yards_per_play": round(stats.yards_per_play, 2),
            "completion_pct": round(stats.completion_pct, 1),
            "completions": stats.completions,
            "attempts": stats.attempts,
            "first_downs": stats.first_downs,
            "third_down_pct": round(stats.third_down_pct, 1),
            "third_down_conversions": stats.third_down_conversions,
            "third_down_attempts": stats.third_down_attempts,
            "turnovers": stats.turnovers,
            "drives": stats.drives,
            "scoring_drives": stats.scoring_drives,
        }

    def _analyze_play_effectiveness(self) -> Dict[str, PlayEffectivenessReport]:
        """Analyze effectiveness of each play called."""
        all_plays: Dict[str, PlayGameStats] = {}

        # Combine stats from both teams
        for stats in [self.game.home_stats, self.game.away_stats]:
            for play_id, play_stats in stats.play_stats.items():
                if play_id not in all_plays:
                    all_plays[play_id] = play_stats
                else:
                    # Merge stats
                    existing = all_plays[play_id]
                    existing.times_called += play_stats.times_called
                    existing.completions += play_stats.completions
                    existing.attempts += play_stats.attempts
                    existing.total_yards += play_stats.total_yards
                    existing.touchdowns += play_stats.touchdowns
                    existing.turnovers += play_stats.turnovers
                    existing.first_down_conversions += play_stats.first_down_conversions

                    for down, count in play_stats.times_called_by_down.items():
                        existing.times_called_by_down[down] = existing.times_called_by_down.get(down, 0) + count
                        existing.success_by_down[down] = existing.success_by_down.get(down, 0) + play_stats.success_by_down.get(down, 0)

                    for zone, count in play_stats.times_called_by_zone.items():
                        existing.times_called_by_zone[zone] = existing.times_called_by_zone.get(zone, 0) + count

        reports = {}
        for play_id, ps in all_plays.items():
            # Find best/worst down
            best_down, worst_down = self._find_best_worst_down(ps)

            # Find best/worst zone
            best_zone = max(ps.times_called_by_zone.items(), key=lambda x: x[1])[0] if ps.times_called_by_zone else None
            worst_zone = min(ps.times_called_by_zone.items(), key=lambda x: x[1])[0] if len(ps.times_called_by_zone) > 1 else None

            # Generate recommendation
            recommendation = self._generate_play_recommendation(ps, best_down)

            reports[play_id] = PlayEffectivenessReport(
                play_id=play_id,
                play_name=ps.play_name,
                times_called=ps.times_called,
                success_rate=ps.success_rate,
                avg_yards=ps.avg_yards,
                touchdowns=ps.touchdowns,
                turnovers=ps.turnovers,
                best_down=best_down,
                best_zone=best_zone,
                worst_down=worst_down,
                worst_zone=worst_zone,
                recommendation=recommendation
            )

        return reports

    def _find_best_worst_down(self, ps: PlayGameStats):
        """Find best and worst down for a play."""
        best_down = None
        worst_down = None
        best_rate = -1.0
        worst_rate = 2.0

        for down in [1, 2, 3]:
            calls = ps.times_called_by_down.get(down, 0)
            success = ps.success_by_down.get(down, 0)
            if calls > 0:
                rate = success / calls
                if rate > best_rate:
                    best_rate = rate
                    best_down = down
                if rate < worst_rate:
                    worst_rate = rate
                    worst_down = down

        return best_down, worst_down

    def _generate_play_recommendation(self, ps: PlayGameStats, best_down: Optional[int]) -> str:
        """Generate recommendation for a play."""
        if ps.times_called < 2:
            return "Insufficient data - needs more attempts"

        if ps.success_rate >= 70:
            if best_down:
                return f"High performer - use especially on {best_down}{'st' if best_down == 1 else 'nd' if best_down == 2 else 'rd'} down"
            return "High performer - use frequently"
        elif ps.success_rate >= 50:
            if ps.avg_yards >= 6:
                return "Good play - solid yards per attempt"
            return "Average performer - use situationally"
        elif ps.success_rate >= 30:
            if ps.touchdowns > 0:
                return "Low success but scores - use in redzone"
            return "Underperforming - consider reducing usage"
        else:
            if ps.turnovers > 0:
                return "Poor performer with turnovers - avoid"
            return "Poor performer - remove from gameplan"

    def _analyze_situations(self) -> List[SituationalBreakdown]:
        """Analyze performance by game situation."""
        situations = []

        # Analyze by down
        for down in [1, 2, 3]:
            down_data = self._get_down_stats(down)
            if down_data["plays"] > 0:
                situations.append(SituationalBreakdown(
                    situation=f"{down}{'st' if down == 1 else 'nd' if down == 2 else 'rd'} Down",
                    plays_run=down_data["plays"],
                    success_rate=down_data["success_rate"],
                    avg_yards=down_data["avg_yards"],
                    best_play=down_data.get("best_play"),
                    worst_play=down_data.get("worst_play")
                ))

        # Analyze by zone (from drive records)
        zone_stats = self._get_zone_stats()
        for zone, data in zone_stats.items():
            if data["plays"] > 0:
                situations.append(SituationalBreakdown(
                    situation=zone.replace("_", " ").title(),
                    plays_run=data["plays"],
                    success_rate=data["success_rate"],
                    avg_yards=data["avg_yards"],
                    best_play=data.get("best_play"),
                    worst_play=data.get("worst_play")
                ))

        return situations

    def _get_down_stats(self, down: int) -> Dict[str, Any]:
        """Get aggregated stats for a specific down."""
        total_plays = 0
        total_success = 0
        total_yards = 0.0
        play_counts = {}
        play_success = {}

        for stats in [self.game.home_stats, self.game.away_stats]:
            for play_id, ps in stats.play_stats.items():
                calls = ps.times_called_by_down.get(down, 0)
                success = ps.success_by_down.get(down, 0)

                total_plays += calls
                total_success += success

                if play_id not in play_counts:
                    play_counts[play_id] = 0
                    play_success[play_id] = 0
                play_counts[play_id] += calls
                play_success[play_id] += success

        # Find best/worst play for this down
        best_play = None
        worst_play = None
        best_rate = -1
        worst_rate = 2

        for play_id, calls in play_counts.items():
            if calls >= 2:  # Need at least 2 attempts
                rate = play_success[play_id] / calls if calls > 0 else 0
                if rate > best_rate:
                    best_rate = rate
                    best_play = play_id
                if rate < worst_rate:
                    worst_rate = rate
                    worst_play = play_id

        return {
            "plays": total_plays,
            "success_rate": (total_success / total_plays * 100) if total_plays > 0 else 0,
            "avg_yards": (total_yards / total_plays) if total_plays > 0 else 0,
            "best_play": best_play,
            "worst_play": worst_play
        }

    def _get_zone_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated stats by field zone."""
        zones = {}

        for stats in [self.game.home_stats, self.game.away_stats]:
            for play_id, ps in stats.play_stats.items():
                for zone, count in ps.times_called_by_zone.items():
                    if zone not in zones:
                        zones[zone] = {"plays": 0, "success": 0, "yards": 0}
                    zones[zone]["plays"] += count

        # Calculate rates
        for zone in zones:
            if zones[zone]["plays"] > 0:
                zones[zone]["success_rate"] = 50.0  # Placeholder - would need more granular tracking
                zones[zone]["avg_yards"] = 5.0  # Placeholder
            else:
                zones[zone]["success_rate"] = 0
                zones[zone]["avg_yards"] = 0

        return zones

    def _identify_key_moments(self) -> List[KeyMoment]:
        """Identify key moments from the game."""
        moments = []

        for drive in self.game.drive_records:
            for i, play in enumerate(drive.plays):
                # Format player names for description
                if play.passer_name and play.receiver_name:
                    player_desc = f"{play.passer_name} to {play.receiver_name}"
                elif play.passer_name:
                    player_desc = f"{play.passer_name}"
                else:
                    player_desc = play.play_name

                # Touchdown
                if play.resulted_in_touchdown:
                    moments.append(KeyMoment(
                        drive=drive.drive_number,
                        play_number=i + 1,
                        description=f"{player_desc} - TOUCHDOWN!",
                        impact="Scoring play",
                        yards=play.yards_gained
                    ))

                # Big play (15+ yards)
                elif play.yards_gained >= 15:
                    moments.append(KeyMoment(
                        drive=drive.drive_number,
                        play_number=i + 1,
                        description=f"{player_desc} - Big gain!",
                        impact="Explosive play",
                        yards=play.yards_gained
                    ))

                # Turnover
                elif play.resulted_in_turnover:
                    moments.append(KeyMoment(
                        drive=drive.drive_number,
                        play_number=i + 1,
                        description=f"{player_desc} - INTERCEPTION",
                        impact="Momentum shift",
                        yards=play.yards_gained
                    ))

                # Critical 3rd down conversion
                elif play.down == 3 and play.resulted_in_first_down:
                    moments.append(KeyMoment(
                        drive=drive.drive_number,
                        play_number=i + 1,
                        description=f"{player_desc} - 3rd down conversion",
                        impact="Drive sustained",
                        yards=play.yards_gained
                    ))

        # Sort by drive and play number
        moments.sort(key=lambda m: (m.drive, m.play_number))

        return moments[:10]  # Top 10 key moments

    def _generate_recommendations(self) -> List[str]:
        """Generate strategic recommendations based on game data."""
        recommendations = []

        # Analyze both teams
        for team_name, stats in [
            ("Home", self.game.home_stats),
            ("Away", self.game.away_stats)
        ]:
            # Third down efficiency
            if stats.third_down_attempts > 3:
                if stats.third_down_pct < 40:
                    recommendations.append(
                        f"{team_name}: Third down conversion rate ({stats.third_down_pct:.0f}%) "
                        f"is low - consider more high-percentage plays on 3rd down"
                    )
                elif stats.third_down_pct > 60:
                    recommendations.append(
                        f"{team_name}: Excellent third down conversion ({stats.third_down_pct:.0f}%) - "
                        f"maintain aggressive approach"
                    )

            # Turnover differential
            if stats.turnovers > 2:
                recommendations.append(
                    f"{team_name}: {stats.turnovers} turnovers - "
                    f"reduce risky play calls or improve ball security"
                )

            # Play variety
            if len(stats.play_stats) < 3:
                recommendations.append(
                    f"{team_name}: Limited play variety ({len(stats.play_stats)} plays used) - "
                    f"consider expanding playbook"
                )

            # Identify ineffective plays
            for play_id, ps in stats.play_stats.items():
                if ps.times_called >= 3 and ps.success_rate < 30:
                    recommendations.append(
                        f"{team_name}: {ps.play_name} has low success rate ({ps.success_rate:.0f}%) - "
                        f"consider removing or using situationally"
                    )

                if ps.turnovers >= 2:
                    recommendations.append(
                        f"{team_name}: {ps.play_name} has multiple turnovers - "
                        f"high-risk play, use carefully"
                    )

        return recommendations[:10]  # Limit to 10 recommendations

    def to_text(self) -> str:
        """Generate text report."""
        report = self.generate()

        lines = [
            "=" * 60,
            "GAME ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Game ID: {report.game_id}",
            f"Date: {report.timestamp}",
            "",
            "FINAL SCORE",
            "-" * 30,
            f"{report.home_team}: {report.home_score}",
            f"{report.away_team}: {report.away_score}",
            f"Winner: {report.winner or 'TIE'}",
            "",
        ]

        # Team summaries
        for team, summary in [
            (report.home_team, report.home_stats_summary),
            (report.away_team, report.away_stats_summary)
        ]:
            lines.extend([
                f"\n{team.upper()} STATS",
                "-" * 30,
                f"Total Yards: {summary['total_yards']}",
                f"Yards/Play: {summary['yards_per_play']}",
                f"Completion %: {summary['completion_pct']}%",
                f"3rd Down %: {summary['third_down_pct']}%",
                f"First Downs: {summary['first_downs']}",
                f"Touchdowns: {summary['touchdowns']}",
                f"Turnovers: {summary['turnovers']}",
            ])

        # Play effectiveness
        lines.extend([
            "\n" + "=" * 60,
            "PLAY EFFECTIVENESS",
            "=" * 60,
        ])

        for play_id, pe in sorted(
            report.play_effectiveness.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        ):
            lines.extend([
                f"\n{pe.play_name}",
                f"  Called: {pe.times_called}x | Success: {pe.success_rate:.0f}% | Avg: {pe.avg_yards:.1f} yds",
                f"  TDs: {pe.touchdowns} | TOs: {pe.turnovers}",
                f"  Best on: {pe.best_down}{'st' if pe.best_down == 1 else 'nd' if pe.best_down == 2 else 'rd'} down" if pe.best_down else "",
                f"  > {pe.recommendation}",
            ])

        # Touchdown Drives
        lines.extend([
            "\n" + "=" * 60,
            "TOUCHDOWN DRIVES",
            "=" * 60,
        ])

        td_drives = [d for d in self.game.drive_records if d.result.value == "TOUCHDOWN"]
        if td_drives:
            for drive in td_drives:
                team_name = "Home" if drive.team_id == self.game.home_team_id else "Away"
                lines.append(f"\n--- Drive {drive.drive_number} ({team_name}) ---")
                lines.append(f"Total Yards: {drive.total_yards:.1f}")

                # Show lineup for this drive (from first play)
                if drive.plays and drive.plays[0].offensive_lineup:
                    lineup = drive.plays[0].offensive_lineup
                    lineup_str = ", ".join([f"{role}: {name}" for role, name in sorted(lineup.items())])
                    lines.append(f"Lineup: {lineup_str}")

                # Show each play in the drive
                for i, play in enumerate(drive.plays, 1):
                    outcome_str = "COMPLETE" if play.outcome.value == "COMPLETE" else play.outcome.value
                    # Format with player names if available
                    if play.passer_name and play.receiver_name and outcome_str == "COMPLETE":
                        play_desc = f"{play.passer_name} to {play.receiver_name}"
                    elif play.passer_name:
                        play_desc = f"{play.passer_name} pass ({play.play_name})"
                    else:
                        play_desc = play.play_name
                    lines.append(
                        f"  {i}. {play_desc}: {outcome_str}, "
                        f"{play.yards_gained:.1f} yds "
                        f"({'TD!' if play.resulted_in_touchdown else '1st Down' if play.resulted_in_first_down else f'D{play.down}'})"
                    )

                # Show extra point result
                if drive.extra_point_choice:
                    xp_type = "1-pt (5 yd)" if drive.extra_point_choice == "ONE_POINT" else "2-pt (12 yd)"
                    xp_result = "GOOD!" if drive.extra_point_success else "NO GOOD"
                    lines.append(f"  XP: {xp_type} - {xp_result}")

                lines.append(f"  Points: {drive.points_scored}")
        else:
            lines.append("No touchdown drives")

        # Key moments
        lines.extend([
            "\n" + "=" * 60,
            "KEY MOMENTS",
            "=" * 60,
        ])

        for km in report.key_moments:
            lines.append(
                f"Drive {km.drive}, Play {km.play_number}: {km.description} "
                f"({km.yards:.1f} yds) - {km.impact}"
            )

        # Recommendations
        lines.extend([
            "\n" + "=" * 60,
            "RECOMMENDATIONS",
            "=" * 60,
        ])

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def to_json(self) -> str:
        """Generate JSON report."""
        report = self.generate()

        # Convert dataclasses to dicts
        def to_dict(obj):
            if hasattr(obj, "__dict__"):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_dict(v) for v in obj]
            else:
                return obj

        return json.dumps(to_dict(report), indent=2)


def generate_game_report(game_result: GameResult, format: str = "text") -> str:
    """
    Generate a game report in the specified format.

    Args:
        game_result: The GameResult to analyze
        format: "text" or "json"

    Returns:
        The formatted report string
    """
    generator = GameReportGenerator(game_result)

    if format == "json":
        return generator.to_json()
    else:
        return generator.to_text()
