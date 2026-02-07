"""
Team definitions for the FieldOS engine.

Defines Zabih's team (Butterfingers) and opponent teams.
These are used by both the CLI scripts and the API server.
"""

from typing import Dict, Tuple
from .core.models import Team, GamePlayer, DualRolePlayerAttributes, PlayerSpecialty


def create_butterfingers() -> Team:
    """Create the Butterfingers team (Zabih's team)."""
    players = [
        GamePlayer(
            id="bb_p1", name="Zabih Yousuf", number=1,
            attributes=DualRolePlayerAttributes(
                speed=82, acceleration=84, agility=83,
                height_inches=74, weight_lbs=195,
                hands=80, route_running=85,
                throw_power=89, short_acc=88, mid_acc=80, deep_acc=83,
                release_time_ms=380, decision_latency_ms=320,
                man_coverage=82, zone_coverage=80, ball_skills=85,
                closing_speed=84, pass_rush=72,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p2", name="Jaylen", number=23,
            attributes=DualRolePlayerAttributes(
                speed=84, acceleration=86, agility=82,
                height_inches=70, weight_lbs=175,
                hands=82, route_running=85,
                throw_power=45, short_acc=48, mid_acc=40, deep_acc=35,
                man_coverage=85, zone_coverage=82, ball_skills=85,
                closing_speed=86, pass_rush=70,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p3", name="Soh", number=11,
            attributes=DualRolePlayerAttributes(
                speed=87, acceleration=88, agility=85,
                height_inches=75, weight_lbs=200,
                hands=99, route_running=90,
                throw_power=30, short_acc=32, mid_acc=25, deep_acc=20,
                man_coverage=82, zone_coverage=80, ball_skills=85,
                closing_speed=86, pass_rush=65,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p4", name="Maddox", number=82,
            attributes=DualRolePlayerAttributes(
                speed=92, acceleration=90, agility=89,
                height_inches=76, weight_lbs=210,
                hands=99, route_running=95,
                throw_power=30, short_acc=32, mid_acc=25, deep_acc=20,
                man_coverage=92, zone_coverage=90, ball_skills=95,
                closing_speed=91, pass_rush=68,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p5", name="Mikail", number=2,
            attributes=DualRolePlayerAttributes(
                speed=85, acceleration=83, agility=82,
                height_inches=69, weight_lbs=165,
                hands=78, route_running=82,
                throw_power=25, short_acc=48, mid_acc=40, deep_acc=35,
                man_coverage=82, zone_coverage=80, ball_skills=83,
                closing_speed=84, pass_rush=60,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p6", name="DeAndre", number=8,
            attributes=DualRolePlayerAttributes(
                speed=88, acceleration=86, agility=84,
                height_inches=72, weight_lbs=185,
                hands=88, route_running=85,
                throw_power=28, short_acc=30, mid_acc=22, deep_acc=18,
                man_coverage=65, zone_coverage=62, ball_skills=70,
                closing_speed=75, pass_rush=55,
                specialty=PlayerSpecialty.OFFENSE_ONLY
            )
        ),
        GamePlayer(
            id="bb_p7", name="Trey", number=4,
            attributes=DualRolePlayerAttributes(
                speed=90, acceleration=92, agility=88,
                height_inches=68, weight_lbs=160,
                hands=85, route_running=88,
                throw_power=35, short_acc=40, mid_acc=32, deep_acc=25,
                man_coverage=60, zone_coverage=58, ball_skills=65,
                closing_speed=70, pass_rush=50,
                specialty=PlayerSpecialty.OFFENSE_ONLY
            )
        ),
        GamePlayer(
            id="bb_p8", name="Marcus", number=24,
            attributes=DualRolePlayerAttributes(
                speed=90, acceleration=92, agility=88,
                height_inches=71, weight_lbs=180,
                hands=68, route_running=62,
                throw_power=35, short_acc=38, mid_acc=30, deep_acc=25,
                man_coverage=94, zone_coverage=90, ball_skills=92,
                closing_speed=93, pass_rush=65,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
        GamePlayer(
            id="bb_p9", name="Terrell", number=33,
            attributes=DualRolePlayerAttributes(
                speed=86, acceleration=88, agility=82,
                height_inches=73, weight_lbs=215,
                hands=62, route_running=55,
                throw_power=40, short_acc=42, mid_acc=35, deep_acc=30,
                man_coverage=78, zone_coverage=82, ball_skills=80,
                closing_speed=88, pass_rush=95,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
        GamePlayer(
            id="bb_p10", name="Darius", number=31,
            attributes=DualRolePlayerAttributes(
                speed=88, acceleration=90, agility=85,
                height_inches=74, weight_lbs=200,
                hands=70, route_running=60,
                throw_power=38, short_acc=40, mid_acc=32, deep_acc=28,
                man_coverage=88, zone_coverage=94, ball_skills=90,
                closing_speed=92, pass_rush=58,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
    ]

    return Team(
        id="butterfingers",
        name="Butterfingers",
        players=players,
        playbook=[
            "play_trips_flood",
            "play_bunch_slants",
            "play_bunch_wheel",
            "play_bunch_stick",
            "play_twins_smash",
            "play_jet_motion_sweep",
            "play_shovel_option",
        ]
    )


def create_godbods() -> Team:
    """Create the Godbods opponent team."""
    players = [
        GamePlayer(
            id="gb_p1", name="Danny", number=22,
            attributes=DualRolePlayerAttributes(
                speed=82, acceleration=84, agility=89,
                height_inches=73, weight_lbs=190,
                hands=85, route_running=85,
                throw_power=89, short_acc=90, mid_acc=85, deep_acc=85,
                release_time_ms=310, decision_latency_ms=280,
                man_coverage=80, zone_coverage=82, ball_skills=85,
                closing_speed=83, pass_rush=70,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p2", name="Lincoln", number=55,
            attributes=DualRolePlayerAttributes(
                speed=65, acceleration=68, agility=62,
                height_inches=72, weight_lbs=200,
                hands=68, route_running=65,
                throw_power=92, short_acc=88, mid_acc=85, deep_acc=90,
                man_coverage=62, zone_coverage=60, ball_skills=58,
                closing_speed=64, pass_rush=55,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p3", name="Nick", number=1,
            attributes=DualRolePlayerAttributes(
                speed=94, acceleration=92, agility=88,
                height_inches=77, weight_lbs=215,
                hands=99, route_running=95,
                throw_power=72, short_acc=75, mid_acc=68, deep_acc=62,
                man_coverage=82, zone_coverage=80, ball_skills=85,
                closing_speed=90, pass_rush=62,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p4", name="Terrance", number=88,
            attributes=DualRolePlayerAttributes(
                speed=92, acceleration=90, agility=88,
                height_inches=74, weight_lbs=195,
                hands=92, route_running=95,
                throw_power=32, short_acc=25, mid_acc=15, deep_acc=10,
                man_coverage=82, zone_coverage=80, ball_skills=85,
                closing_speed=88, pass_rush=65,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p5", name="Mike", number=17,
            attributes=DualRolePlayerAttributes(
                speed=88, acceleration=85, agility=82,
                height_inches=68, weight_lbs=160,
                hands=72, route_running=70,
                throw_power=18, short_acc=15, mid_acc=10, deep_acc=8,
                man_coverage=88, zone_coverage=85, ball_skills=88,
                closing_speed=90, pass_rush=55,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p6", name="Tyreek", number=10,
            attributes=DualRolePlayerAttributes(
                speed=99, acceleration=98, agility=95,
                height_inches=68, weight_lbs=165,
                hands=82, route_running=88,
                throw_power=25, short_acc=28, mid_acc=20, deep_acc=15,
                man_coverage=65, zone_coverage=62, ball_skills=70,
                closing_speed=75, pass_rush=50,
                specialty=PlayerSpecialty.OFFENSE_ONLY
            )
        ),
        GamePlayer(
            id="gb_p7", name="Cooper", number=12,
            attributes=DualRolePlayerAttributes(
                speed=86, acceleration=84, agility=88,
                height_inches=71, weight_lbs=180,
                hands=90, route_running=92,
                throw_power=40, short_acc=45, mid_acc=38, deep_acc=30,
                man_coverage=60, zone_coverage=58, ball_skills=65,
                closing_speed=70, pass_rush=48,
                specialty=PlayerSpecialty.OFFENSE_ONLY
            )
        ),
        GamePlayer(
            id="gb_p8", name="Jalen", number=21,
            attributes=DualRolePlayerAttributes(
                speed=92, acceleration=94, agility=90,
                height_inches=70, weight_lbs=175,
                hands=70, route_running=65,
                throw_power=38, short_acc=40, mid_acc=32, deep_acc=28,
                man_coverage=96, zone_coverage=92, ball_skills=90,
                closing_speed=95, pass_rush=60,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
        GamePlayer(
            id="gb_p9", name="Khalil", number=99,
            attributes=DualRolePlayerAttributes(
                speed=84, acceleration=86, agility=78,
                height_inches=75, weight_lbs=235,
                hands=58, route_running=50,
                throw_power=45, short_acc=48, mid_acc=40, deep_acc=35,
                man_coverage=72, zone_coverage=75, ball_skills=70,
                closing_speed=85, pass_rush=98,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
        GamePlayer(
            id="gb_p10", name="Derwin", number=3,
            attributes=DualRolePlayerAttributes(
                speed=90, acceleration=92, agility=88,
                height_inches=73, weight_lbs=205,
                hands=72, route_running=60,
                throw_power=42, short_acc=45, mid_acc=38, deep_acc=32,
                man_coverage=90, zone_coverage=95, ball_skills=92,
                closing_speed=94, pass_rush=72,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
    ]

    return Team(
        id="godbods",
        name="Godbods",
        players=players,
        playbook=[
            "play_spread_vertical",
            "play_bunch_mesh",
            "play_bunch_scissors",
            "play_orbit_screen",
            "play_reverse_pass",
            "play_halfback_pass",
            "play_center_throwback",
        ]
    )


def get_all_teams() -> Tuple[Team, Dict[str, Team]]:
    """Get my team and all opponents."""
    my_team = create_butterfingers()
    opponents = {
        "godbods": create_godbods(),
    }
    return my_team, opponents
