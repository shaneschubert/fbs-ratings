"""Fetch data, run the model, and write the Quarto site.

    python build.py          # real data (needs CFBD_API_KEY)
    python build.py --demo   # fictional league, no key needed

Then: quarto render site   (output in site/_site)
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from pipeline import cfbd, demo, model, render, simulate

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


def resolve_season(cfg):
    y = cfg["season"]["year"]
    if str(y).lower() == "auto":
        today = dt.date.today()
        return today.year if today.month >= 8 else today.year - 1
    return int(y)


def quarto_yml(cfg):
    title = cfg["site"]["title"].replace('"', "'")
    return f"""project:
  type: website
  output-dir: _site
  resources:
    - "downloads/*"

website:
  title: "{title}"
  navbar:
    left:
      - text: Ratings
        href: ratings.qmd
      - text: Schedule
        href: schedule.qmd
      - text: Standings
        href: standings.qmd
      - text: Playoff
        href: playoff.qmd
      - text: Stats
        href: stats.qmd
      - text: Teams
        href: teams.qmd
    right:
      - text: Methodology
        href: methodology.qmd
  page-footer:
    center: "{title}. Data from CollegeFootballData.com."

format:
  html:
    theme:
      light: cosmo
      dark: darkly
    respect-user-color-scheme: true
    css: assets/site.css
    include-after-body: assets/site-js.html
    page-layout: full
    toc: false
    anchor-sections: false
    html-table-processing: none
"""


def games_of_week(games, tt, lev, engine, cfg, season, next_week, avg):
    if next_week is None:
        return games.iloc[0:0]
    wk = games[(games["season"] == season) & (games["season_type"] == "regular")
               & (games["week"] == next_week) & ~games["completed"]].copy()
    if wk.empty:
        return wk
    ovr = dict(zip(tt["team"], tt["ovr"]))
    fcs_ovr = float(100 * model.win_prob(engine.fcs - avg))
    wk["q"] = [(ovr.get(h, fcs_ovr) + ovr.get(a, fcs_ovr)) / 2 for h, a in zip(wk["home"], wk["away"])]
    wk["c"] = (1 - 2 * (wk["home_wp"] - 0.5).abs()) * 100
    wk = wk.merge(lev, on="game_id", how="left")
    stake = ((wk["home_cfp_if_win"] - wk["home_cfp_if_loss"]).fillna(0)
             + (wk["away_cfp_if_win"] - wk["away_cfp_if_loss"]).fillna(0))
    wk["s"] = np.minimum(100, stake * 100)
    w = cfg["games_of_week"]["weights"]
    wk["gamescore"] = w["quality"] * wk["q"] + w["closeness"] * wk["c"] + w["stakes"] * wk["s"]
    return wk.sort_values("gamescore", ascending=False).head(int(cfg["games_of_week"]["count"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Use a fictional league instead of the API")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    season = resolve_season(cfg)
    print(f"Season {season}")

    if args.demo:
        raw_teams, raw_games = demo.generate(cfg, season)
        raw_teams = [cfbd.normalize(t) for t in raw_teams]
        raw_games = [cfbd.normalize(g) for g in raw_games]
    else:
        client = cfbd.CFBDClient(os.environ.get("CFBD_API_KEY", ""), ROOT / "cache")
        raw_teams, raw_games = cfbd.load_all(client, season, int(cfg["season"]["history_start"]))
        print(f"API calls this run: {client.calls}")

    teams = model.build_teams(raw_teams, cfg)
    known = set(cfg["conferences"]["power"] + cfg["conferences"]["group_of"] + [cfg["conferences"]["independent"]])
    unknown = sorted(set(teams["conference"]) - known)
    if unknown:
        print(f"WARNING: conferences not listed in config.yaml (treated as non-power, no auto bid): {unknown}")

    games = model.build_games(raw_games, set(teams["team"]))
    print(f"{len(teams)} FBS teams, {len(games)} games loaded")

    engine = model.EloEngine(cfg)
    games, hist = engine.run(games)
    last_week, next_week = model.week_status(games, season)
    games = model.refresh_upcoming(games, engine, season)
    tt, avg = model.team_table(teams, games, engine, season, last_week)
    adj = model.adjusted_scoring(games, teams, season, engine.hfa / engine.per_point,
                                 float(cfg["adjusted_scoring"]["ridge_lambda"]))
    print(f"Last completed week {last_week}, next week {next_week}. Simulating...")
    proj, lev, runs = simulate.simulate(tt, games, engine, cfg, season, next_week)
    gotw = games_of_week(games, tt, lev, engine, cfg, season, next_week, avg)

    # Downloads: full model output
    dl = SITE / "downloads"
    dl.mkdir(parents=True, exist_ok=True)
    for f in dl.glob("*.csv"):
        f.unlink()
    full = tt.drop(columns=["last_game", "next_game"]).merge(proj, on="team", how="left").merge(adj, on="team", how="left")
    files = {
        f"ratings_projections_{season}.csv": full,
        f"games_{season}.csv": games[games["season"] == season],
        f"games_all_seasons.csv": games,
        f"elo_history.csv": hist,
        f"adjusted_scoring_{season}.csv": adj,
        f"game_leverage_week_{next_week or 'none'}.csv": lev,
        f"team_index_{season}.csv": teams[["team_id", "team", "slug", "abbr", "mascot", "conference", "division",
                                           "color", "logo"]],
    }
    for name, df in files.items():
        df.to_csv(dl / name, index=False)

    tz = ZoneInfo(cfg["site"]["timezone"])
    updated = dt.datetime.now(tz).strftime("%a %b %-d, %-I:%M %p %Z")
    fcs_ovr = float(100 * model.win_prob(engine.fcs - avg))
    ctx = render.Ctx(cfg, tt, games, proj, adj, season, last_week, next_week, updated, fcs_ovr)
    (SITE / "_quarto.yml").write_text(quarto_yml(cfg))
    render.write_site(ctx, SITE, gotw, lev, hist, list(files))
    print(f"Site pages written to {SITE}. Next: quarto render site")


if __name__ == "__main__":
    sys.exit(main())
