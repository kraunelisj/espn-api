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
import inspect
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Type, Union

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


def _team_details(team: Any, include_roster: bool = False) -> Dict[str, Any]:
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
    if include_roster and hasattr(team, "roster"):
        base["roster"] = [_player_to_dict(player) for player in getattr(team, "roster", [])]
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
    if player is None:
        return {}

    eligible_slots = getattr(player, "eligibleSlots", None)
    if eligible_slots is None:
        eligible_slots = getattr(player, "eligible_slots", None)

    payload: Dict[str, Any] = {
        "id": getattr(player, "playerId", getattr(player, "id", None)),
        "name": getattr(player, "name", None),
        "position": getattr(player, "position", None),
        "eligible_slots": eligible_slots,
        "pro_team": getattr(player, "proTeam", getattr(player, "proTeamId", None)),
        "injury_status": getattr(player, "injuryStatus", None),
        "is_injured": getattr(player, "injured", None),
        "on_team_id": getattr(player, "onTeamId", None),
        "lineup_slot": getattr(player, "lineupSlot", None),
        "acquisition_type": getattr(player, "acquisitionType", None),
        "ownership": getattr(player, "ownership", None),
        "percent_owned": getattr(player, "percent_owned", None),
        "percent_started": getattr(player, "percent_started", None),
        "total_points": getattr(player, "total_points", None),
        "projected_points": getattr(player, "projected_total_points", None),
        "average_points": getattr(player, "avg_points", None),
        "projected_average_points": getattr(player, "projected_avg_points", None),
        "active_status": getattr(player, "active_status", None),
        "stats": getattr(player, "stats", None),
        "schedule": getattr(player, "schedule", None),
        "news": getattr(player, "news", None),
    }

    # Remove keys with value ``None`` to keep the payload compact.
    return {key: value for key, value in payload.items() if value is not None}


def _box_player_to_dict(player: Any) -> Dict[str, Any]:
    payload = _player_to_dict(player)
    payload.update(
        {
            "slot_position": getattr(player, "slot_position", None),
            "pro_opponent": getattr(player, "pro_opponent", None),
            "pro_positional_rank": getattr(player, "pro_pos_rank", None),
            "game_played": getattr(player, "game_played", None),
            "on_bye_week": getattr(player, "on_bye_week", None),
            "points": getattr(player, "points", None),
            "projected_points": getattr(player, "projected_points", payload.get("projected_points")),
            "points_breakdown": getattr(player, "points_breakdown", None),
            "projected_breakdown": getattr(player, "projected_breakdown", None),
        }
    )
    return {key: value for key, value in payload.items() if value is not None}


def _recent_activity_to_dict(activity: Any) -> Dict[str, Any]:
    actions_payload: List[Dict[str, Any]] = []
    for entry in getattr(activity, "actions", []) or []:
        team: Any = None
        action: Any = None
        player: Any = None
        extra: Any = None
        expanded = list(entry)
        if expanded:
            team = expanded[0]
        if len(expanded) > 1:
            action = expanded[1]
        if len(expanded) > 2:
            player = expanded[2]
        if len(expanded) > 3:
            extra = expanded[3]

        payload: Dict[str, Any] = {
            "team": _team_summary(team),
            "action": action,
        }

        if player is not None:
            payload["player"] = getattr(player, "name", player)
            payload["player_id"] = getattr(player, "playerId", None)

        if isinstance(extra, (int, float)):
            payload["bid_amount"] = extra
        elif isinstance(extra, str):
            payload["position"] = extra
        elif extra is not None:
            payload["details"] = extra

        actions_payload.append(payload)

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


def _settings_to_dict(settings: Any) -> Dict[str, Any]:
    if settings is None:
        return {}

    payload = {key: getattr(settings, key) for key in dir(settings) if not key.startswith("_")}
    # Filter out callables and private attributes introduced by ``BaseSettings`` subclasses.
    return {
        key: value
        for key, value in payload.items()
        if not callable(value) and key not in {"logger"}
    }


def _draft_pick_to_dict(pick: Any) -> Dict[str, Any]:
    return {
        "team": _team_summary(getattr(pick, "team", None)),
        "player_id": getattr(pick, "playerId", None),
        "player_name": getattr(pick, "playerName", None),
        "round": getattr(pick, "round_num", None),
        "pick": getattr(pick, "round_pick", None),
        "bid_amount": getattr(pick, "bid_amount", None),
        "is_keeper": getattr(pick, "keeper_status", None),
        "nominating_team": _team_summary(getattr(pick, "nominatingTeam", None)),
    }


def _power_ranking_to_dict(entry: Tuple[Any, Any]) -> Dict[str, Any]:
    score, team = entry
    return {
        "score": float(score) if score is not None else None,
        "team": _team_summary(team),
    }


def _box_score_to_dict(box_score: Any) -> Dict[str, Any]:
    if box_score is None:
        return {}

    payload: Dict[str, Any] = {
        "winner": getattr(box_score, "winner", None),
        "matchup_type": getattr(box_score, "matchup_type", None),
        "is_playoff": getattr(box_score, "is_playoff", None),
        "home_team": _team_summary(getattr(box_score, "home_team", None)),
        "away_team": _team_summary(getattr(box_score, "away_team", None)),
    }

    for prefix in ("home", "away"):
        score_key = f"{prefix}_score"
        projected_key = f"{prefix}_projected"
        wins_key = f"{prefix}_wins"
        losses_key = f"{prefix}_losses"
        ties_key = f"{prefix}_ties"
        stats_key = f"{prefix}_stats"
        lineup_key = f"{prefix}_lineup"

        score = getattr(box_score, score_key, None)
        projected = getattr(box_score, projected_key, None)
        wins = getattr(box_score, wins_key, None)
        losses = getattr(box_score, losses_key, None)
        ties = getattr(box_score, ties_key, None)
        stats = getattr(box_score, stats_key, None)
        lineup = getattr(box_score, lineup_key, None)

        if score is not None:
            payload[score_key] = score
        if projected is not None:
            payload[projected_key] = projected
        if wins is not None:
            payload[wins_key] = wins
        if losses is not None:
            payload[losses_key] = losses
        if ties is not None:
            payload[ties_key] = ties
        if stats is not None:
            payload[stats_key] = stats
        if lineup is not None:
            payload[lineup_key] = [_box_player_to_dict(player) for player in lineup]

    return payload


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
    include_roster: bool = False,
) -> List[Dict[str, Any]]:
    """Return detailed team information for the given league."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    return [_team_details(team, include_roster=include_roster) for team in league.teams]


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
def get_box_scores(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
    week: Optional[int] = None,
    matchup_period: Optional[int] = None,
    scoring_period: Optional[int] = None,
    matchup_total: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Return box score details for the requested period."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    sport_name = _normalise_sport(sport)

    if sport_name == "football":
        box_scores = league.box_scores(week=week)
    elif sport_name in {"basketball", "hockey"}:
        kwargs: Dict[str, Any] = {}
        if matchup_period is not None:
            kwargs["matchup_period"] = matchup_period
        if scoring_period is not None:
            kwargs["scoring_period"] = scoring_period
        if matchup_total is not None:
            kwargs["matchup_total"] = matchup_total
        box_scores = league.box_scores(**kwargs)
    elif sport_name in {"baseball", "wbasketball"}:
        kwargs = {}
        if matchup_period is not None:
            kwargs["matchup_period"] = matchup_period
        if scoring_period is not None:
            kwargs["scoring_period"] = scoring_period
        box_scores = league.box_scores(**kwargs)
    else:  # pragma: no cover - defensive programming for future sports.
        raise ValueError(f"Box scores are not supported for the {sport} API.")

    return [_box_score_to_dict(box_score) for box_score in box_scores]


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
    }
    if msg_type is not None:
        kwargs["msg_type"] = msg_type

    if not hasattr(league, "recent_activity"):
        raise ValueError(f"Recent activity is not supported for the {sport} API.")

    signature = inspect.signature(league.recent_activity)
    if "include_moved" in signature.parameters:
        kwargs["include_moved"] = include_moved

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
        normalised_types = {str(item).upper() for item in types}
        kwargs["types"] = normalised_types

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
                "scoring_period": getattr(txn, "scoring_period", None),
                "bid_amount": getattr(txn, "bid_amount", None),
            }
        )
    return results


@mcp.tool()
def get_power_rankings(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
    week: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return power rankings (football only)."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    sport_name = _normalise_sport(sport)
    if not hasattr(league, "power_rankings") or sport_name != "football":
        raise ValueError("Power rankings are only supported for football leagues.")

    rankings = league.power_rankings(week=week)
    return [_power_ranking_to_dict(entry) for entry in rankings]


@mcp.tool()
def get_player_info(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
    name: Optional[str] = None,
    player_id: Optional[Union[int, List[int]]] = None,
    include_news: bool = False,
) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """Return player information for the given identifiers."""

    league = _load_league(sport, league_id, year, swid, espn_s2)

    kwargs: Dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if player_id is not None:
        kwargs["playerId"] = player_id

    signature = inspect.signature(league.player_info)
    if "include_news" in signature.parameters:
        kwargs["include_news"] = include_news

    result = league.player_info(**kwargs)
    if result is None:
        return None
    if isinstance(result, list):
        return [_player_to_dict(player) for player in result]
    return _player_to_dict(result)


@mcp.tool()
def get_league_settings(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
) -> Dict[str, Any]:
    """Return league settings information."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    return _settings_to_dict(getattr(league, "settings", None))


@mcp.tool()
def get_draft_results(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return draft picks for the league."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    if not hasattr(league, "draft"):
        return []
    return [_draft_pick_to_dict(pick) for pick in getattr(league, "draft", [])]


@mcp.tool()
def get_message_board(
    sport: str,
    league_id: int,
    year: int,
    swid: Optional[str] = None,
    espn_s2: Optional[str] = None,
    message_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return message board posts (football only)."""

    league = _load_league(sport, league_id, year, swid, espn_s2)
    sport_name = _normalise_sport(sport)
    if not hasattr(league, "message_board") or sport_name != "football":
        raise ValueError("Message board access is only supported for football leagues.")

    messages = league.message_board(msg_types=message_types)
    results: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            results.append({key: value for key, value in message.items()})
        else:
            # Fallback for message board entries returned as custom objects.
            payload = {
                key: getattr(message, key)
                for key in dir(message)
                if not key.startswith("_") and not callable(getattr(message, key))
            }
            results.append(payload)
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
