"""
Game Simulator - Simulates complete games between two teams.

Handles drives, field position, downs, scoring, and comprehensive statistics.
"""

import uuid
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.models import (
    Team, GamePlayer, GameConfig, GameState, GameResult,
    DriveResult, DriveRecord, PlayResult, PlayGameStats,
    TeamGameStats, PlayerGameStats, Player, Play, Scenario,
    Role, Side, OutcomeType, FieldZone, FieldConfig, Ruleset,
    DefenseCall, CoverageType, CoverageShell, RandomnessConfig, Point2D,
    ExtraPointChoice
)
from ..core.registry import registry
from .engine import SimulationEngine

logger = logging.getLogger("fieldos_engine.game_simulator")


class GameSimulator:
    """Simulates complete games between two teams."""

    def __init__(
        self,
        home_team: Team,
        away_team: Team,
        config: Optional[GameConfig] = None,
        seed: int = 42
    ):
        self.home_team = home_team
        self.away_team = away_team
        self.config = config or GameConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.play_engine = SimulationEngine(seed=seed)

        # Load plays from registry
        self.plays: Dict[str, Play] = {}
        self._load_playbooks()

        # Default defensive scenarios
        self.defense_scenarios = self._create_default_scenarios()

        # State
        self.state = self._init_game_state()

        # Statistics
        self.home_stats = TeamGameStats(
            team_id=home_team.id,
            team_name=home_team.name
        )
        self.away_stats = TeamGameStats(
            team_id=away_team.id,
            team_name=away_team.name
        )
        self.drive_records: List[DriveRecord] = []

    def _load_playbooks(self):
        """Load plays from registry based on team playbooks."""
        all_plays = registry.plays.list()
        play_lookup = {p.id: p for p in all_plays}

        for play_id in self.home_team.playbook + self.away_team.playbook:
            if play_id in play_lookup:
                self.plays[play_id] = play_lookup[play_id]
            else:
                logger.warning(f"Play {play_id} not found in registry")

    def _create_default_scenarios(self) -> List[Scenario]:
        """Create default defensive scenarios for game simulation."""
        field = FieldConfig(
            width_yards=self.config.field_width,
            total_length_yards=self.config.field_length,
            endzone_depth_yards=self.config.endzone_depth
        )

        rules = Ruleset(
            id="game_rules",
            name="Game Rules",
            field=field,
        )

        # Default defender positions for D1-D5
        base_positions = {
            Role.D1: Point2D(x=7.0, y=-10.0),   # Left corner
            Role.D2: Point2D(x=6.0, y=-4.0),    # Left linebacker
            Role.D3: Point2D(x=7.0, y=0.0),     # Middle/rusher
            Role.D4: Point2D(x=6.0, y=4.0),     # Right linebacker
            Role.D5: Point2D(x=7.0, y=10.0),    # Right corner
        }

        scenarios = [
            Scenario(
                id="game_cover2_no_rush",
                name="Cover 2 No Rush",
                field=field,
                rules=rules,
                defense_call=DefenseCall(
                    type=CoverageType.ZONE,
                    shell=CoverageShell.COVER2,
                    rushers_count=0
                ),
                defender_start_positions={
                    Role.D1: Point2D(x=6.0, y=-10.0),
                    Role.D2: Point2D(x=12.0, y=-6.0),  # Safety
                    Role.D3: Point2D(x=5.0, y=0.0),
                    Role.D4: Point2D(x=12.0, y=6.0),   # Safety
                    Role.D5: Point2D(x=6.0, y=10.0),
                },
                randomness=RandomnessConfig()
            ),
            Scenario(
                id="game_cover1_rush",
                name="Cover 1 Rush",
                field=field,
                rules=rules,
                defense_call=DefenseCall(
                    type=CoverageType.MAN,
                    shell=CoverageShell.COVER1,
                    rushers_count=1
                ),
                defender_start_positions={
                    Role.D1: Point2D(x=5.0, y=-10.0),
                    Role.D2: Point2D(x=5.0, y=-4.0),
                    Role.D3: Point2D(x=7.0, y=0.0),  # Rusher
                    Role.D4: Point2D(x=5.0, y=4.0),
                    Role.D5: Point2D(x=12.0, y=0.0), # Deep safety
                },
                randomness=RandomnessConfig()
            ),
            Scenario(
                id="game_man_no_rush",
                name="Man No Rush",
                field=field,
                rules=rules,
                defense_call=DefenseCall(
                    type=CoverageType.MAN,
                    shell=CoverageShell.COVER0,
                    rushers_count=0
                ),
                defender_start_positions=base_positions,
                randomness=RandomnessConfig()
            ),
        ]

        return scenarios

    def _init_game_state(self) -> GameState:
        """Initialize game state."""
        return GameState(
            home_team_id=self.home_team.id,
            away_team_id=self.away_team.id,
            field_position=self.config.start_position,
            yards_to_first=self.config.first_down_yards,
            total_drives=self.config.drives_per_team * 2,
        )

    def _get_current_team(self) -> Team:
        """Get the team currently on offense."""
        return self.home_team if self.state.possession == "home" else self.away_team

    def _get_defending_team(self) -> Team:
        """Get the team currently on defense."""
        return self.away_team if self.state.possession == "home" else self.home_team

    def _get_current_stats(self) -> TeamGameStats:
        """Get stats for the team on offense."""
        return self.home_stats if self.state.possession == "home" else self.away_stats

    def _get_offensive_players(self, team: Team) -> Dict[Role, Player]:
        """Create offensive player roster from team using dynamic position selection."""
        return self._select_offensive_lineup(team)

    def _get_defensive_players(self, team: Team) -> Dict[Role, Player]:
        """Create defensive player roster from team using dynamic position selection."""
        return self._select_defensive_lineup(team)

    def _select_offensive_lineup(self, team: Team) -> Dict[Role, Player]:
        """
        Select optimal offensive lineup based on player attribute scores.
        Returns dict mapping Role -> Player for the 5 offensive positions.

        Only considers players who can play offense (specialty = OFFENSE_ONLY or TWO_WAY).
        """
        # Filter to only players who can play offense
        available = [p for p in team.players if p.attributes.can_play_offense()]
        if len(available) < 5:
            # Fallback: use all players if not enough offense-eligible
            available = list(team.players)

        selected: Dict[Role, Player] = {}
        self._current_offensive_lineup: Dict[Role, GamePlayer] = {}  # Track for names

        # 1. Select QB - highest qb_score
        qb_candidates = sorted(available, key=lambda p: p.attributes.qb_score(), reverse=True)
        qb = qb_candidates[0]
        selected[Role.QB] = qb.as_offense_player(Role.QB)
        self._current_offensive_lineup[Role.QB] = qb
        available.remove(qb)

        # 2. Select WR1 (outside) - highest wr_score("outside")
        wr_candidates = sorted(available, key=lambda p: p.attributes.wr_score("outside"), reverse=True)
        wr1 = wr_candidates[0]
        selected[Role.WR1] = wr1.as_offense_player(Role.WR1)
        self._current_offensive_lineup[Role.WR1] = wr1
        available.remove(wr1)

        # 3. Select WR2 (outside) - next highest wr_score("outside")
        wr_candidates = sorted(available, key=lambda p: p.attributes.wr_score("outside"), reverse=True)
        wr2 = wr_candidates[0]
        selected[Role.WR2] = wr2.as_offense_player(Role.WR2)
        self._current_offensive_lineup[Role.WR2] = wr2
        available.remove(wr2)

        # 4. Select WR3 (slot) - highest wr_score("slot")
        wr_candidates = sorted(available, key=lambda p: p.attributes.wr_score("slot"), reverse=True)
        wr3 = wr_candidates[0]
        selected[Role.WR3] = wr3.as_offense_player(Role.WR3)
        self._current_offensive_lineup[Role.WR3] = wr3
        available.remove(wr3)

        # 5. Select Center - highest center_score from remaining
        center_candidates = sorted(available, key=lambda p: p.attributes.center_score(), reverse=True)
        center = center_candidates[0] if center_candidates else team.players[-1]
        selected[Role.CENTER] = center.as_offense_player(Role.CENTER)
        self._current_offensive_lineup[Role.CENTER] = center

        return selected

    def _select_defensive_lineup(self, team: Team, rusher_needed: bool = False) -> Dict[Role, Player]:
        """
        Select optimal defensive lineup based on player attribute scores.
        Returns dict mapping Role -> Player for the 5 defensive positions.

        Only considers players who can play defense (specialty = DEFENSE_ONLY or TWO_WAY).
        """
        # Filter to only players who can play defense
        available = [p for p in team.players if p.attributes.can_play_defense()]
        if len(available) < 5:
            # Fallback: use all players if not enough defense-eligible
            available = list(team.players)

        selected: Dict[Role, Player] = {}
        self._current_defensive_lineup: Dict[Role, GamePlayer] = {}  # Track for names

        # Determine coverage type from current scenario (default to man)
        coverage_type = "man"

        # If rusher needed, select best rusher for D3 first
        if rusher_needed:
            rusher_candidates = sorted(available, key=lambda p: p.attributes.rusher_score(), reverse=True)
            rusher = rusher_candidates[0]
            selected[Role.D3] = rusher.as_defense_player(Role.D3)
            self._current_defensive_lineup[Role.D3] = rusher
            available.remove(rusher)

        # Select remaining defenders by defender_score
        roles_to_fill = [Role.D1, Role.D2, Role.D4, Role.D5]
        if not rusher_needed:
            roles_to_fill.append(Role.D3)

        defender_candidates = sorted(available, key=lambda p: p.attributes.defender_score(coverage_type), reverse=True)

        for i, role in enumerate(roles_to_fill):
            if i < len(defender_candidates):
                defender = defender_candidates[i]
                selected[role] = defender.as_defense_player(role)
                self._current_defensive_lineup[role] = defender

        return selected

    def get_current_offensive_lineup_names(self) -> Dict[str, str]:
        """Get a mapping of role -> player name for current offensive lineup."""
        if not hasattr(self, '_current_offensive_lineup'):
            return {}
        return {role.value: player.name for role, player in self._current_offensive_lineup.items()}

    def get_current_defensive_lineup_names(self) -> Dict[str, str]:
        """Get a mapping of role -> player name for current defensive lineup."""
        if not hasattr(self, '_current_defensive_lineup'):
            return {}
        return {role.value: player.name for role, player in self._current_defensive_lineup.items()}

    def _select_play(self, team: Team) -> Optional[Play]:
        """Select a play from team's playbook (random for now, RL can override)."""
        if not team.playbook:
            # Use a random play from all available plays
            if self.plays:
                play_id = self.rng.choice(list(self.plays.keys()))
                return self.plays[play_id]
            return None

        play_id = self.rng.choice(team.playbook)
        return self.plays.get(play_id)

    def _select_defense(self) -> Scenario:
        """Select a defensive scenario (random for now)."""
        return self.rng.choice(self.defense_scenarios)

    def simulate_play(
        self,
        play: Play,
        offensive_team: Team,
        defensive_team: Team,
        scenario: Optional[Scenario] = None
    ) -> PlayResult:
        """Simulate a single play."""
        if scenario is None:
            scenario = self._select_defense()

        off_players = self._get_offensive_players(offensive_team)
        def_players = self._get_defensive_players(defensive_team)

        # Run simulation
        outcome, trace = self.play_engine.simulate_play(
            play,
            scenario,
            off_players,
            def_players,
            situation=self.state.to_game_situation(),
            record_trace=False
        )

        # Calculate new field position
        field_before = self.state.field_position
        yards_gained = outcome.yards_gained

        # Handle outcomes
        if outcome.outcome == OutcomeType.INTERCEPT:
            field_after = field_before  # Turnover at spot
            resulted_in_turnover = True
        elif outcome.outcome == OutcomeType.SACK:
            yards_gained = -3.0  # Sack loses yards
            field_after = max(0, field_before + yards_gained)
            resulted_in_turnover = False
        else:
            field_after = min(self.config.playing_field, field_before + yards_gained)
            resulted_in_turnover = False

        # Check for touchdown
        resulted_in_td = field_after >= self.config.playing_field

        # Check for first down
        resulted_in_first = (
            not resulted_in_td and
            yards_gained >= self.state.yards_to_first and
            not self.state.first_down_achieved
        )

        # Get player names from current lineup
        passer_name = None
        passer_id = None
        receiver_name = None
        receiver_id = None
        offensive_lineup = {}
        defensive_lineup = {}

        if hasattr(self, '_current_offensive_lineup'):
            # QB is always the passer
            qb_player = self._current_offensive_lineup.get(Role.QB)
            if qb_player:
                passer_name = qb_player.name
                passer_id = qb_player.id

            # Get receiver from target role
            if outcome.target_role and outcome.target_role in self._current_offensive_lineup:
                receiver_player = self._current_offensive_lineup.get(outcome.target_role)
                if receiver_player:
                    receiver_name = receiver_player.name
                    receiver_id = receiver_player.id

            # Build full lineup mapping
            offensive_lineup = {role.value: player.name for role, player in self._current_offensive_lineup.items()}

        if hasattr(self, '_current_defensive_lineup'):
            defensive_lineup = {role.value: player.name for role, player in self._current_defensive_lineup.items()}

        result = PlayResult(
            play_id=play.id,
            play_name=play.name,
            outcome=outcome.outcome,
            yards_gained=yards_gained,
            down=self.state.down,
            yards_to_first=self.state.yards_to_first,
            field_position_before=field_before,
            field_position_after=field_after,
            target_role=outcome.target_role,
            passer_id=passer_id,
            passer_name=passer_name,
            receiver_id=receiver_id,
            receiver_name=receiver_name,
            offensive_lineup=offensive_lineup,
            defensive_lineup=defensive_lineup,
            resulted_in_first_down=resulted_in_first,
            resulted_in_touchdown=resulted_in_td,
            resulted_in_turnover=resulted_in_turnover,
            time_to_throw_ms=outcome.time_to_throw_ms,
            completion_probability=outcome.completion_probability,
        )

        return result

    def _choose_extra_point(self, team: Team) -> ExtraPointChoice:
        """
        Decide whether to go for 1 or 2 points after touchdown.

        Strategy: Go for 2 when behind or tied late in game.
        """
        our_score = self.state.home_score if self.state.possession == "home" else self.state.away_score
        their_score = self.state.away_score if self.state.possession == "home" else self.state.home_score

        # Late game (last 4 drives) and behind/tied - go for 2
        drives_remaining = self.state.total_drives - self.state.current_drive
        if drives_remaining <= 4 and our_score <= their_score:
            return ExtraPointChoice.TWO_POINT

        # Down by 2+ scores - go for 2
        if their_score - our_score >= 8:
            return ExtraPointChoice.TWO_POINT

        # Random chance (20%) to go for 2 to keep it interesting
        if self.rng.random() < 0.2:
            return ExtraPointChoice.TWO_POINT

        return ExtraPointChoice.ONE_POINT

    def simulate_extra_point(
        self,
        team: Team,
        def_team: Team,
        choice: ExtraPointChoice
    ) -> Tuple[bool, Optional[PlayResult]]:
        """
        Simulate an extra point attempt.

        Returns (success, play_result)
        """
        # Set field position based on choice
        if choice == ExtraPointChoice.ONE_POINT:
            distance = self.config.xp1_distance  # 5 yards
            points = self.config.xp1_points
        else:
            distance = self.config.xp2_distance  # 12 yards
            points = self.config.xp2_points

        # Temporarily set field position for the conversion
        original_position = self.state.field_position
        self.state.field_position = self.config.playing_field - distance
        self.state.down = 1
        self.state.yards_to_first = distance

        # Select a play and simulate
        play = self._select_play(team)
        if play is None:
            self.state.field_position = original_position
            return False, None

        result = self.simulate_play(play, team, def_team)

        # Check if successful (completed pass in endzone)
        success = result.resulted_in_touchdown or (
            result.outcome == OutcomeType.COMPLETE and
            result.yards_gained >= distance
        )

        # Reset field position
        self.state.field_position = original_position

        return success, result

    def _update_game_state(self, play_result: PlayResult):
        """Update game state after a play."""
        self.state.field_position = play_result.field_position_after

        if play_result.resulted_in_touchdown:
            # Score touchdown (extra points handled separately in simulate_drive)
            if self.state.possession == "home":
                self.state.home_score += self.config.td_points
            else:
                self.state.away_score += self.config.td_points

        elif play_result.resulted_in_first_down:
            # First down achieved - reset downs
            self.state.first_down_achieved = True
            self.state.down = 1
            self.state.yards_to_first = min(
                self.config.first_down_yards,
                self.state.yards_to_goal  # Can't need more than yards to goal
            )

        elif play_result.resulted_in_turnover:
            pass  # Drive will end

        else:
            # Normal play - update down
            self.state.down += 1
            self.state.yards_to_first = max(0, self.state.yards_to_first - play_result.yards_gained)

    def _update_stats(self, play_result: PlayResult, team: Team, play: Play):
        """Update statistics after a play."""
        stats = self._get_current_stats()

        # Update team stats
        stats.total_plays += 1
        stats.total_yards += play_result.yards_gained
        stats.attempts += 1

        if play_result.outcome == OutcomeType.COMPLETE:
            stats.completions += 1
        elif play_result.outcome == OutcomeType.INTERCEPT:
            stats.interceptions_thrown += 1
            stats.turnovers += 1

        if play_result.resulted_in_first_down:
            stats.first_downs += 1

        if play_result.resulted_in_touchdown:
            stats.touchdowns += 1

        if play_result.down == 3:
            stats.third_down_attempts += 1
            if play_result.resulted_in_first_down or play_result.resulted_in_touchdown:
                stats.third_down_conversions += 1

        # Update play stats
        if play.id not in stats.play_stats:
            stats.play_stats[play.id] = PlayGameStats(
                play_id=play.id,
                play_name=play.name
            )

        play_stats = stats.play_stats[play.id]
        play_stats.times_called += 1
        play_stats.attempts += 1
        play_stats.total_yards += play_result.yards_gained

        if play_result.outcome == OutcomeType.COMPLETE:
            play_stats.completions += 1
        if play_result.resulted_in_touchdown:
            play_stats.touchdowns += 1
        if play_result.resulted_in_turnover:
            play_stats.turnovers += 1
        if play_result.resulted_in_first_down:
            play_stats.first_down_conversions += 1

        # Update by down
        down = play_result.down
        if down in play_stats.times_called_by_down:
            play_stats.times_called_by_down[down] += 1
            if play_result.outcome == OutcomeType.COMPLETE:
                play_stats.success_by_down[down] += 1

        if down == 3:
            play_stats.third_down_attempts += 1
            if play_result.resulted_in_first_down or play_result.resulted_in_touchdown:
                play_stats.third_down_conversions += 1

        # Update by zone
        zone = self.state.field_zone.value
        if zone not in play_stats.times_called_by_zone:
            play_stats.times_called_by_zone[zone] = 0
        play_stats.times_called_by_zone[zone] += 1

    def simulate_drive(self) -> DriveRecord:
        """Simulate a complete drive."""
        team = self._get_current_team()
        def_team = self._get_defending_team()
        stats = self._get_current_stats()

        drive = DriveRecord(
            drive_number=self.state.current_drive,
            team_id=team.id,
            starting_field_position=self.state.field_position,
            ending_field_position=self.state.field_position,
        )

        stats.drives += 1

        # Reset drive state
        self.state.down = 1
        self.state.first_down_achieved = False
        self.state.yards_to_first = min(
            self.config.first_down_yards,
            self.state.yards_to_goal
        )

        max_downs = self.config.downs_to_first_down
        if self.state.first_down_achieved:
            max_downs = self.config.downs_to_score

        while self.state.down <= max_downs:
            # Select and run play
            play = self._select_play(team)
            if play is None:
                logger.error("No play available")
                break

            result = self.simulate_play(play, team, def_team)
            drive.plays.append(result)
            drive.total_yards += result.yards_gained

            # Update stats
            self._update_stats(result, team, play)

            # Update game state
            self._update_game_state(result)

            # Check drive end conditions
            if result.resulted_in_touchdown:
                drive.result = DriveResult.TOUCHDOWN
                drive.points_scored = self.config.td_points
                stats.scoring_drives += 1

                # Attempt extra point
                xp_choice = self._choose_extra_point(team)
                xp_success, xp_result = self.simulate_extra_point(team, def_team, xp_choice)

                drive.extra_point_choice = xp_choice.value
                drive.extra_point_success = xp_success
                drive.extra_point_play = xp_result

                if xp_success:
                    xp_points = self.config.xp1_points if xp_choice == ExtraPointChoice.ONE_POINT else self.config.xp2_points
                    drive.points_scored += xp_points
                    if self.state.possession == "home":
                        self.state.home_score += xp_points
                    else:
                        self.state.away_score += xp_points

                break

            if result.resulted_in_turnover:
                drive.result = DriveResult.INTERCEPTION
                break

            if result.resulted_in_first_down:
                # Reset downs for second half of drive
                max_downs = self.state.down + self.config.downs_to_score - 1

        # Check if drive ended on downs
        if drive.result == DriveResult.IN_PROGRESS:
            if self.state.down > max_downs:
                drive.result = DriveResult.TURNOVER_ON_DOWNS

        drive.ending_field_position = self.state.field_position
        self.drive_records.append(drive)

        return drive

    def _switch_possession(self):
        """Switch possession to other team."""
        if self.state.possession == "home":
            self.state.possession = "away"
        else:
            self.state.possession = "home"

        # Reset field position
        self.state.field_position = self.config.start_position
        self.state.down = 1
        self.state.first_down_achieved = False
        self.state.yards_to_first = self.config.first_down_yards
        self.state.current_drive += 1

    def simulate_game(self) -> GameResult:
        """Simulate a complete game."""
        logger.info(f"Starting game: {self.home_team.name} vs {self.away_team.name}")

        for drive_num in range(self.config.drives_per_team * 2):
            if self.state.game_over:
                break

            team_name = self._get_current_team().name
            logger.debug(f"Drive {drive_num + 1}: {team_name}")

            drive = self.simulate_drive()

            logger.debug(
                f"  Result: {drive.result.value}, "
                f"Yards: {drive.total_yards:.1f}, "
                f"Plays: {drive.num_plays}"
            )

            self._switch_possession()

        self.state.game_over = True

        # Determine winner
        winner = None
        if self.state.home_score > self.state.away_score:
            winner = self.home_team.id
        elif self.state.away_score > self.state.home_score:
            winner = self.away_team.id

        # Update final stats
        self.home_stats.total_points = self.state.home_score
        self.away_stats.total_points = self.state.away_score

        result = GameResult(
            game_id=str(uuid.uuid4()),
            home_team_id=self.home_team.id,
            away_team_id=self.away_team.id,
            home_score=self.state.home_score,
            away_score=self.state.away_score,
            winner=winner,
            total_plays=self.home_stats.total_plays + self.away_stats.total_plays,
            total_drives=len(self.drive_records),
            home_stats=self.home_stats,
            away_stats=self.away_stats,
            drive_records=self.drive_records,
        )

        logger.info(
            f"Game complete: {self.home_team.name} {self.state.home_score} - "
            f"{self.state.away_score} {self.away_team.name}"
        )

        return result


def create_sample_teams() -> Tuple[Team, Team]:
    """Create two sample teams for testing."""

    # Team 1: Balanced team
    team1_players = [
        GamePlayer(id="t1_p1", name="Alex QB", number=7, attributes=DualRolePlayerAttributes(
            off_speed=75, off_hands=70, throw_power=85, short_acc=82, mid_acc=78, deep_acc=72,
            def_speed=70, man_coverage=65, zone_coverage=65, ball_skills=70
        )),
        GamePlayer(id="t1_p2", name="Ben Center", number=52, attributes=DualRolePlayerAttributes(
            off_speed=68, off_hands=75, throw_power=55, short_acc=60, mid_acc=50, deep_acc=45,
            def_speed=68, man_coverage=70, zone_coverage=72, ball_skills=65
        )),
        GamePlayer(id="t1_p3", name="Chris WR1", number=11, attributes=DualRolePlayerAttributes(
            off_speed=88, off_hands=85, off_route_running=82, throw_power=50, short_acc=55,
            def_speed=85, man_coverage=60, ball_skills=75
        )),
        GamePlayer(id="t1_p4", name="Dan WR2", number=82, attributes=DualRolePlayerAttributes(
            off_speed=82, off_hands=80, off_route_running=78, throw_power=48,
            def_speed=80, man_coverage=62, ball_skills=72
        )),
        GamePlayer(id="t1_p5", name="Eric WR3", number=15, attributes=DualRolePlayerAttributes(
            off_speed=85, off_hands=78, off_route_running=75, throw_power=52,
            def_speed=82, man_coverage=65, ball_skills=68
        )),
    ]

    # Team 2: Speed-focused team
    team2_players = [
        GamePlayer(id="t2_p1", name="Frank QB", number=12, attributes=DualRolePlayerAttributes(
            off_speed=78, off_hands=72, throw_power=82, short_acc=80, mid_acc=75, deep_acc=78,
            def_speed=75, man_coverage=62, zone_coverage=60, ball_skills=68
        )),
        GamePlayer(id="t2_p2", name="Gary Center", number=55, attributes=DualRolePlayerAttributes(
            off_speed=70, off_hands=72, throw_power=58, short_acc=62, mid_acc=52, deep_acc=48,
            def_speed=70, man_coverage=68, zone_coverage=70, ball_skills=62
        )),
        GamePlayer(id="t2_p3", name="Henry WR1", number=1, attributes=DualRolePlayerAttributes(
            off_speed=92, off_hands=82, off_route_running=80, throw_power=55,
            def_speed=90, man_coverage=58, ball_skills=70
        )),
        GamePlayer(id="t2_p4", name="Ivan WR2", number=88, attributes=DualRolePlayerAttributes(
            off_speed=88, off_hands=78, off_route_running=76, throw_power=50,
            def_speed=86, man_coverage=60, ball_skills=68
        )),
        GamePlayer(id="t2_p5", name="Jake WR3", number=17, attributes=DualRolePlayerAttributes(
            off_speed=86, off_hands=80, off_route_running=78, throw_power=48,
            def_speed=84, man_coverage=64, ball_skills=72
        )),
    ]

    team1 = Team(
        id="team_balanced",
        name="Balanced Blazers",
        players=team1_players,
        playbook=[
            "play_trips_flood",
            "play_bunch_slants",
            "play_bunch_wheel",
            "play_bunch_stick",
            "play_twins_smash",
        ]
    )

    team2 = Team(
        id="team_speed",
        name="Speed Demons",
        players=team2_players,
        playbook=[
            "play_spread_vertical",
            "play_bunch_mesh",
            "play_bunch_scissors",
            "play_jet_motion_sweep",
            "play_orbit_screen",
        ]
    )

    return team1, team2


# Need to import this for the create_sample_teams function
from ..core.models import DualRolePlayerAttributes
