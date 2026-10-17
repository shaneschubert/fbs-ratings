"""CollegeFootballData.com API client.

Uses the REST API directly (no SDK) so the only dependency is requests.
Responses are cached to disk. Past seasons are reused from cache; the
current season is always refetched.
"""
import json
import re
import time
from pathlib import Path

import requests

BASE_URL = "https://api.collegefootballdata.com"


def _snake(key: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()


def normalize(record: dict) -> dict:
    """CFBD v2 returns camelCase, older versions snake_case. Accept both."""
    return {_snake(k): v for k, v in record.items()}


class CFBDClient:
    def __init__(self, api_key: str, cache_dir: Path):
        if not api_key:
            raise SystemExit(
                "No CFBD API key found. Set the CFBD_API_KEY environment variable "
                "(GitHub: Settings > Secrets and variables > Actions > New repository secret)."
            )
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def _get(self, path: str, params: dict):
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        for attempt in range(5):
            resp = requests.get(BASE_URL + path, params=params, headers=headers, timeout=90)
            self.calls += 1
            if resp.status_code == 401:
                raise SystemExit("CFBD rejected the API key (401). Check the CFBD_API_KEY secret.")
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        raise SystemExit("CFBD rate limit hit repeatedly (429). Try again later.")

    def _cached(self, name: str, path: str, params: dict, refresh: bool):
        file = self.cache_dir / f"{name}.json"
        if file.exists() and not refresh:
            return json.loads(file.read_text())
        data = self._get(path, params)
        file.write_text(json.dumps(data))
        return data

    def games(self, year: int, season_type: str, refresh: bool):
        raw = self._cached(
            f"games_{year}_{season_type}",
            "/games",
            {"year": year, "seasonType": season_type, "classification": "fbs"},
            refresh,
        )
        return [normalize(g) for g in raw]

    def teams(self, year: int, refresh: bool):
        raw = self._cached(f"teams_fbs_{year}", "/teams/fbs", {"year": year}, refresh)
        return [normalize(t) for t in raw]


    def talent(self, year: int, refresh: bool):
        """247 Team Talent Composite. Returns [] for seasons without data."""
        try:
            raw = self._cached(f"talent_{year}", "/talent", {"year": year}, refresh)
        except requests.HTTPError:
            return []
        out = []
        for t in raw:
            t = normalize(t)
            team = t.get("team") or t.get("school")
            try:
                val = float(t.get("talent"))
            except (TypeError, ValueError):
                continue
            if team:
                out.append({"season": year, "team": team, "talent": val})
        return out


def load_all(client: CFBDClient, season: int, history_start: int):
    """Return (current teams, games for every season, talent rows for every season)."""
    games, talent = [], []
    for year in range(history_start, season + 1):
        refresh = year == season
        for stype in ("regular", "postseason"):
            games.extend(client.games(year, stype, refresh))
        talent.extend(client.talent(year, refresh))
    teams = client.teams(season, refresh=True)
    return teams, games, talent
