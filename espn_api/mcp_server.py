"""MCP server exposing espn_api data sources.

This module provides a small Model Context Protocol (MCP) server that wraps
the ``espn_api`` package.  The server exposes a collection of tools that allow
MCP compliant clients to query ESPN fantasy league information using the same
parameters that the regular python API expects (``league_id``, ``year``,
``swid``, ``espn_s2`` …).

The implementation uses :class:`mcp.server.fastmcp.FastMCP` so that it can be
run directly via ``python -m espn_api.mcp_server`` when the optional ``mcp``
package is installed::

    pip install mcp
    python -m espn_api.mcp_server

By default the server communicates over stdio, which makes it compatible with
the reference MCP client implementations.  Individual tool invocations carry
the credentials that are needed to access a private league, keeping long-lived
credentials out of the transport layer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Type

from .base_league import BaseLeague

# ``mcp`` is an optional dependency.  The import is intentionally delayed until
# runtime so that simply importing espn_api does not require the extra package.
try:  # pragma: no cover - exercised when users opt into MCP support.
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - handled to give a friendly message.
    raise ImportError(
        "The optional 'mcp' package is required to use espn_api.mcp_server. "
        "Install it with 'pip install mcp'."
    ) from exc


SportName = Literal["football", "basketball", "baseball", "hockey", "wbasketball"]


@dataclass(frozen=True)
class LeagueKey:
    """Hashable cache key describing a league instance."""

    sport: SportName
    league_id: int
    year: int
    swid: Optional[str]
    espn_s2: Optional[str]


def _import_league_class(sport: SportName) -> Tuple[str, Type[BaseLeague]]:
    """Resolve the league class and sport code for a given sport name."""

    if sport == "football":
        from .football import League as LeagueClass

        return "nfl", LeagueClass
    if sport == "basketball":
        from .basketball import League as LeagueClass

        return "nba", LeagueClass
    if sport == "baseball":
        from .baseball import League as LeagueClass

        return "mlb", LeagueClass
    if sport == "hockey":
        from .hockey import League as LeagueClass

        return "nhl", LeagueClass
    if sport == "wbasketball":
        from .wbasketball import League as LeagueClass

        return "wnba", LeagueClass

    raise ValueError(f"Unsupported sport '{sport}'.")


def _normalise_sport(sport: str) -> SportName:
    sport_lower = sport.lower()
    aliases = {
        "nfl": "football",
        "ffl": "football",
        "football": "football",
        "nba": "basketball",
        "basketball": "basketball",
        "mlb": "baseball",
        "baseball": "baseball",
        "nhl": "hockey",
        "hockey": "hockey",
        "wnba": "wbasketball",
        "wbasketball": "wbasketball",
    }

    if sport_lower not in aliases:
        raise ValueError(
            f"Unknown sport '{sport}'. Expected one of: {', '.join(sorted(set(aliases.values())))}"
        )
    return aliases[sport_lower]  # type: ignore[return-value]


@lru_cache(maxsize=8)
def _load_league_cached(key: LeagueKey) -> BaseLeague:
    """Load and cache a league instance for the provided parameters."""

    _, league_class = _import_league_class(key.sport)
    return league_class(
        league_id=key.league_id,
        year=key.year,
        swid=key.swid,
        espn_s2=key.espn_s2,
    )


def _load_league(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str],
    espn_s2: Optional[str],
) -> BaseLeague:
    normalised = _normalise_sport(sport)
    return _load_league_cached(LeagueKey(normalised, league_id, year, swid, espn_s2))


def _owner_to_dict(owner: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": owner.get("id"),
        "display_name": owner.get("displayName"),
        "first_name": owner.get("firstName"),
        "last_name": owner.get("lastName"),
        "is_manager": owner.get("isLeagueManager"),
    }


def _team_summary(team: Any) -> Dict[str, Any]:
    if team is None or not hasattr(team, "team_id"):
        return {}

    owners = getattr(team, "owners", []) or []
    owner_dicts = [_owner_to_dict(owner) for owner in owners]

    return {
        "id": team.team_id,
        "abbreviation": team.team_abbrev,
        "name": team.team_name,
        "division_id": getattr(team, "division_id", None),
        "division_name": getattr(team, "division_name", None),
        "wins": getattr(team, "wins", None),
        "losses": getattr(team, "losses", None),
        "ties": getattr(team, "ties", None),
        "points_for": getattr(team, "points_for", None),
        "points_against": getattr(team, "points_against", None),
        "standing": getattr(team, "standing", None),
        "final_standing": getattr(team, "final_standing", None),
        "owners": owner_dicts,
        "logo_url": getattr(team, "logo_url", None),
    }


def _team_details(team: Any) -> Dict[str, Any]:
    base = _team_summary(team)
    base.update(
        {
            "streak": {
                "length": getattr(team, "streak_length", None),
                "type": getattr(team, "streak_type", None),
            },
            "acquisitions": getattr(team, "acquisitions", None),
            "drops": getattr(team, "drops", None),
            "trades": getattr(team, "trades", None),
            "move_to_ir": getattr(team, "move_to_ir", None),
            "playoff_pct": getattr(team, "playoff_pct", None),
            "waiver_rank": getattr(team, "waiver_rank", None),
        }
    )
    return base


def _matchup_to_dict(matchup: Any) -> Dict[str, Any]:
    return {
        "matchup_type": getattr(matchup, "matchup_type", None),
        "is_playoff": getattr(matchup, "is_playoff", None),
        "home_team": _team_summary(getattr(matchup, "home_team", None)),
        "away_team": _team_summary(getattr(matchup, "away_team", None)),
        "home_score": getattr(matchup, "home_score", None),
        "away_score": getattr(matchup, "away_score", None),
    }


def _player_to_dict(player: Any) -> Dict[str, Any]:
    return {
        "id": getattr(player, "playerId", None),
        "name": getattr(player, "name", None),
        "position": getattr(player, "position", None),
        "pro_team": getattr(player, "proTeam", None),
        "ownership": getattr(player, "ownership", None),
        "injury_status": getattr(player, "injuryStatus", None),
        "percent_owned": getattr(player, "percent_owned", None),
        "projected_points": getattr(player, "projected_total_points", None),
        "total_points": getattr(player, "total_points", None),
    }


def _recent_activity_to_dict(activity: Any) -> Dict[str, Any]:
    actions_payload: List[Dict[str, Any]] = []
    for entry in getattr(activity, "actions", []) or []:
        team, action, player, bid_amount = (list(entry) + [None] * 4)[:4]
        actions_payload.append(
            {
                "team": _team_summary(team),
                "action": action,
                "player": getattr(player, "name", player),
                "player_id": getattr(player, "playerId", None),
                "bid_amount": bid_amount,
            }
        )

    return {
        "date": getattr(activity, "date", None),
        "actions": actions_payload,
    }


def _league_metadata(league: BaseLeague, sport: SportName) -> Dict[str, Any]:
    return {
        "sport": sport,
        "league_id": league.league_id,
        "year": league.year,
        "current_week": getattr(league, "current_week", None),
        "current_matchup_period": getattr(league, "currentMatchupPeriod", None),
        "scoring_period_id": getattr(league, "scoringPeriodId", None),
        "first_scoring_period": getattr(league, "firstScoringPeriod", None),
        "final_scoring_period": getattr(league, "finalScoringPeriod", None),
        "previous_seasons": getattr(league, "previousSeasons", None),
    }


mcp = FastMCP("espn-api")


@mcp.tool()
def get_league_overview(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
) -> Dict[str, Any]:
    """Return core metadata and standings for a league."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    sport_name = _normalise_sport(sport)
    data = _league_metadata(league, sport_name)
    data["standings"] = [_team_summary(team) for team in league.standings()]
    return data


@mcp.tool()
def list_teams(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return detailed team information for the given league."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    return [_team_details(team) for team in league.teams]


@mcp.tool()
def get_scoreboard(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
    week: Optional[int] = None,
    matchup_period: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return the current (or requested) scoreboard for the league."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    sport_name = _normalise_sport(sport)

    if sport_name == "football":
        matchups = league.scoreboard(week=week)
    else:
        kwargs: Dict[str, Any] = {}
        if matchup_period is not None:
            kwargs["matchupPeriod"] = matchup_period
        elif week is not None:
            kwargs["matchupPeriod"] = week
        matchups = league.scoreboard(**kwargs)

    return [_matchup_to_dict(matchup) for matchup in matchups]


@mcp.tool()
def list_free_agents(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
    week: Optional[int] = None,
    size: int = 50,
    position: Optional[str] = None,
    position_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return free agent information for the given league.

    The parameters mirror the espn_api ``League.free_agents`` method.
    """

    league = _load_league(sport, league_id, year, swid, espn_s2)
    if not hasattr(league, "free_agents"):
        raise ValueError(f"Free agent queries are not supported for the {sport} API.")

    players = league.free_agents(week=week, size=size, position=position, position_id=position_id)
    return [_player_to_dict(player) for player in players]


@mcp.tool()
def recent_activity(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
    size: int = 25,
    msg_type: Optional[str] = None,
    offset: int = 0,
    include_moved: bool = False,
) -> List[Dict[str, Any]]:
    """Return recent league activity (add/drop/trade events)."""

    league = _load_league(sport, league_id, year, swid, espn_s2)

    kwargs: Dict[str, Any] = {
        "size": size,
        "offset": offset,
        "include_moved": include_moved,
    }
    if msg_type is not None:
        kwargs["msg_type"] = msg_type

    if not hasattr(league, "recent_activity"):
        raise ValueError(f"Recent activity is not supported for the {sport} API.")

    activities = league.recent_activity(**kwargs)
    return [_recent_activity_to_dict(activity) for activity in activities]


@mcp.tool()
def transactions(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
    scoring_period: Optional[int] = None,
    types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Return transactions for the requested scoring period."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    kwargs: Dict[str, Any] = {}
    if scoring_period is not None:
        kwargs["scoring_period"] = scoring_period
    if types is not None:
        kwargs["types"] = set(types)

    if not hasattr(league, "transactions"):
        raise ValueError(f"Transactions are not supported for the {sport} API.")

    txns = league.transactions(**kwargs)
    results: List[Dict[str, Any]] = []
    for txn in txns:
        results.append(
            {
                "id": getattr(txn, "id", None),
                "type": getattr(txn, "type", None),
                "status": getattr(txn, "status", None),
                "team": _team_summary(getattr(txn, "team", None)),
                "items": [str(item) for item in getattr(txn, "items", [])],
                "date": getattr(txn, "date", None),
            }
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the espn_api MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport to use (currently only stdio is supported).",
    )
    return parser.parse_args()


def main() -> None:
    _parse_args()  # Currently only used to validate input. Future transports can hook in here.
    mcp.run()


if __name__ == "__main__":  # pragma: no cover - convenience entry point.
    main()
