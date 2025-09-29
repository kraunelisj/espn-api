from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Optional

import pytest


def _ensure_fake_mcp() -> None:
    import sys

    if "mcp.server.fastmcp" in sys.modules:
        return

    fake_fastmcp = ModuleType("mcp.server.fastmcp")

    class _FakeFastMCP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def tool(self, *_args: Any, **_kwargs: Any):  # noqa: ANN401 - mimic decorator signature
            def decorator(func):
                return func

            return decorator

        def run(self) -> None:
            pass

    fake_fastmcp.FastMCP = _FakeFastMCP  # type: ignore[attr-defined]

    server_module = ModuleType("mcp.server")
    server_module.fastmcp = fake_fastmcp  # type: ignore[attr-defined]

    mcp_module = ModuleType("mcp")
    mcp_module.server = server_module  # type: ignore[attr-defined]

    sys.modules.setdefault("mcp", mcp_module)
    sys.modules.setdefault("mcp.server", server_module)
    sys.modules.setdefault("mcp.server.fastmcp", fake_fastmcp)


_ensure_fake_mcp()

from espn_api import mcp_server


@dataclass
class _StubTeam:
    team_id: int
    team_abbrev: str = "ABC"
    team_name: str = "Stub Team"
    division_id: int = 1
    division_name: str = "Division"
    wins: int = 1
    losses: int = 0
    ties: int = 0
    points_for: float = 10.0
    points_against: float = 5.0
    standing: int = 1
    final_standing: int = 1
    streak_length: int = 1
    streak_type: str = "WIN"
    acquisitions: int = 0
    drops: int = 0
    trades: int = 0
    move_to_ir: int = 0
    playoff_pct: float = 0.0
    waiver_rank: int = 1
    logo_url: Optional[str] = None
    owners: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        self.roster: List[Any] = []


class _StubPlayer:
    def __init__(self, player_id: int, name: str) -> None:
        self.playerId = player_id
        self.name = name
        self.position = "QB"
        self.proTeam = "Team"
        self.stats = {}


def test_list_teams_includes_roster(monkeypatch: pytest.MonkeyPatch) -> None:
    team = _StubTeam(team_id=1)
    team.roster = [_StubPlayer(1, "Player One")]

    class _League:
        teams = [team]

    monkeypatch.setattr(mcp_server, "_load_league", lambda *a, **k: _League())

    data = mcp_server.list_teams("football", 1, 2023, include_roster=True)
    assert data[0]["roster"][0]["name"] == "Player One"


def test_recent_activity_include_moved_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: Dict[str, Any] = {}

    class _BaseballLeague:
        def recent_activity(self, **kwargs: Any) -> List[Any]:  # noqa: ANN401 - match dynamic usage
            nonlocal captured_kwargs
            captured_kwargs = dict(kwargs)
            return []

    monkeypatch.setattr(mcp_server, "_load_league", lambda *a, **k: _BaseballLeague())

    mcp_server.recent_activity("baseball", 1, 2023, include_moved=True)
    assert "include_moved" not in captured_kwargs


def test_get_box_scores_returns_serialisable_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FootballLeague:
        def box_scores(self, week: Optional[int] = None) -> List[Any]:
            class _BoxScore:
                def __init__(self) -> None:
                    self.home_team = _StubTeam(team_id=1)
                    self.away_team = _StubTeam(team_id=2)
                    self.home_score = 100
                    self.away_score = 90
                    self.home_lineup = [_StubPlayer(1, "Player One")]
                    self.away_lineup = [_StubPlayer(2, "Player Two")]

            return [_BoxScore()]

    monkeypatch.setattr(mcp_server, "_load_league", lambda *a, **k: _FootballLeague())

    payload = mcp_server.get_box_scores("football", 1, 2023)
    assert payload[0]["home_score"] == 100
    assert payload[0]["home_lineup"][0]["name"] == "Player One"


def test_get_player_info_detects_include_news(monkeypatch: pytest.MonkeyPatch) -> None:
    received_kwargs: Dict[str, Any] = {}

    class _BasketballLeague:
        def player_info(self, *, playerId: int, include_news: bool = False) -> Any:  # noqa: ANN401
            nonlocal received_kwargs
            received_kwargs = {"playerId": playerId, "include_news": include_news}
            return _StubPlayer(playerId, "Player")

    monkeypatch.setattr(mcp_server, "_load_league", lambda *a, **k: _BasketballLeague())

    payload = mcp_server.get_player_info("basketball", 1, 2023, player_id=7, include_news=True)
    assert payload["id"] == 7
    assert received_kwargs == {"playerId": 7, "include_news": True}
