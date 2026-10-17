"""Elo ratings, game predictions, team summary table, opponent-adjusted scoring."""
import math
import re

import numpy as np
import pandas as pd


def win_prob(diff):
    return 1.0 / (1.0 + 10 ** (-np.asarray(diff, dtype=float) / 400.0))


def slugify(name: str) -> str:
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "team"


# --------------------------------------------------------------------------- data prep

def build_teams(raw_teams, cfg):
    rows = []
    for t in raw_teams:
        logos = t.get("logos") or []
        logo = logos[0].replace("http://", "https://") if logos else ""
        rows.append(
            {
                "team_id": t.get("id"),
                "team": t.get("school"),
                "mascot": t.get("mascot") or "",
                "abbr": t.get("abbreviation") or (t.get("school") or "")[:4].upper(),
                "conference": t.get("conference") or cfg["conferences"]["independent"],
                "division": t.get("division") or "",
                "color": "#" + (t.get("color") or "555555").lstrip("#"),
                "alt_color": "#" + (t.get("alternate_color") or "dddddd").lstrip("#"),
                "logo": logo if cfg["site"]["show_logos"] else "",
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("team").reset_index(drop=True)
    df["slug"] = df["team"].map(slugify)
    # Guarantee unique slugs
    dup = df["slug"].duplicated(keep=False)
    df.loc[dup, "slug"] = df.loc[dup, "slug"] + "-" + df.loc[dup, "team_id"].astype(str)
    return df


def build_games(raw_games, current_fbs: set):
    rows = []
    for g in raw_games:
        home, away = g.get("home_team"), g.get("away_team")
        if not home or not away:
            continue
        hc, ac = g.get("home_classification"), g.get("away_classification")
        home_fbs = (hc == "fbs") if hc else (home in current_fbs)
        away_fbs = (ac == "fbs") if ac else (away in current_fbs)
        if not (home_fbs or away_fbs):
            continue
        hp, ap = g.get("home_points"), g.get("away_points")
        completed = bool(g.get("completed")) and hp is not None and ap is not None
        notes = g.get("notes") or ""
        rows.append(
            {
                "game_id": g.get("id"),
                "season": int(g.get("season")),
                "week": int(g.get("week") or 0),
                "season_type": g.get("season_type") or "regular",
                "date": g.get("start_date"),
                "completed": completed,
                "neutral": bool(g.get("neutral_site")),
                "conf_game": bool(g.get("conference_game")),
                "home": home,
                "away": away,
                "home_fbs": home_fbs,
                "away_fbs": away_fbs,
                "home_conf": g.get("home_conference") or "",
                "away_conf": g.get("away_conference") or "",
                "home_pts": hp if completed else None,
                "away_pts": ap if completed else None,
                "venue": g.get("venue") or "",
                "notes": notes,
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["is_ccg"] = (
        df["conf_game"]
        & df["notes"].str.contains("championship", case=False, na=False)
        & (df["date"].dt.month == 12)
    )
    stype_order = df["season_type"].map({"regular": 0, "postseason": 1}).fillna(1)
    df = df.assign(_st=stype_order).sort_values(["season", "_st", "week", "date", "game_id"])
    return df.drop(columns="_st").reset_index(drop=True)


# --------------------------------------------------------------------------- Elo

class EloEngine:
    def __init__(self, cfg):
        e = cfg["elo"]
        self.mean, self.k, self.hfa = e["mean"], e["k"], e["home_field"]
        self.regress, self.fcs = e["offseason_regression"], e["fcs_rating"]
        self.new_team, self.per_point = e["new_team_rating"], e["elo_per_point"]
        self.ratings = {}
        self.snapshots = {}  # (season, season_type, week) -> ratings at start of that week

    def rating(self, team, fbs):
        if not fbs:
            return self.fcs
        return self.ratings.get(team, self.new_team)

    def run(self, games: pd.DataFrame):
        pre_h, pre_a, wp_h, post_h, post_a = [], [], [], [], []
        history = []
        season = None
        week_key = None
        for g in games.itertuples(index=False):
            if g.season != season:
                if season is not None:
                    self.ratings = {
                        t: self.mean + (r - self.mean) * (1 - self.regress) for t, r in self.ratings.items()
                    }
                season = g.season
            key = (g.season, g.season_type, g.week)
            if key != week_key:
                self.snapshots[key] = dict(self.ratings)
                week_key = key
            rh, ra = self.rating(g.home, g.home_fbs), self.rating(g.away, g.away_fbs)
            diff = rh - ra + (0 if g.neutral else self.hfa)
            p = float(win_prob(diff))
            pre_h.append(rh)
            pre_a.append(ra)
            wp_h.append(p)
            if not g.completed:
                post_h.append(np.nan)
                post_a.append(np.nan)
                continue
            margin = g.home_pts - g.away_pts
            result = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            winner_diff = diff if margin > 0 else -diff
            mult = math.log(abs(margin) + 1) * (2.2 / (winner_diff * 0.001 + 2.2)) if margin != 0 else 1.0
            shift = self.k * mult * (result - p)
            nh, na = rh + shift, ra - shift
            if g.home_fbs:
                self.ratings[g.home] = nh
                history.append((g.season, g.season_type, g.week, g.date, g.home, rh, nh, g.game_id))
            if g.away_fbs:
                self.ratings[g.away] = na
                history.append((g.season, g.season_type, g.week, g.date, g.away, ra, na, g.game_id))
            post_h.append(nh if g.home_fbs else np.nan)
            post_a.append(na if g.away_fbs else np.nan)
        games = games.copy()
        games["home_elo_pre"], games["away_elo_pre"] = pre_h, pre_a
        games["home_wp"] = wp_h
        games["home_elo_post"], games["away_elo_post"] = post_h, post_a
        hist = pd.DataFrame(
            history, columns=["season", "season_type", "week", "date", "team", "elo_pre", "elo_post", "game_id"]
        )
        return games, hist

    def predict(self, home, away, home_fbs, away_fbs, neutral, ratings=None):
        r = ratings if ratings is not None else self.ratings
        rh = r.get(home, self.new_team) if home_fbs else self.fcs
        ra = r.get(away, self.new_team) if away_fbs else self.fcs
        diff = rh - ra + (0 if neutral else self.hfa)
        return float(win_prob(diff)), diff / self.per_point


def refresh_upcoming(games, engine, season):
    """Upcoming games get predictions from the latest ratings, not the rating at fetch time."""
    games = games.copy()
    mask = (games["season"] == season) & ~games["completed"]
    for i in games.index[mask]:
        g = games.loc[i]
        p, _ = engine.predict(g.home, g.away, g.home_fbs, g.away_fbs, g.neutral)
        games.at[i, "home_elo_pre"] = engine.rating(g.home, g.home_fbs)
        games.at[i, "away_elo_pre"] = engine.rating(g.away, g.away_fbs)
        games.at[i, "home_wp"] = p
    elo_diff = games["home_elo_pre"] - games["away_elo_pre"] + np.where(games["neutral"], 0, engine.hfa)
    games["home_spread"] = -(elo_diff / engine.per_point)  # negative = home favored, sportsbook style
    return games


# --------------------------------------------------------------------------- weeks

def week_status(games, season):
    cur = games[(games["season"] == season) & (games["season_type"] == "regular")]
    done = cur[cur["completed"]]
    todo = cur[~cur["completed"]]
    last_week = int(done["week"].max()) if len(done) else 0
    next_week = int(todo["week"].min()) if len(todo) else None
    return last_week, next_week


# --------------------------------------------------------------------------- team table

def team_table(teams, games, engine, season, last_week):
    cur = games[games["season"] == season]
    played = cur[cur["completed"]]
    avg = np.mean([engine.ratings.get(t, engine.new_team) for t in teams["team"]])

    prev_snap = engine.snapshots.get((season, "regular", last_week), engine.ratings) if last_week else engine.ratings
    pre_keys = [k for k in engine.snapshots if k[0] == season]
    pre_snap = engine.snapshots[min(pre_keys, key=lambda k: (k[1] != "regular", k[2]))] if pre_keys else engine.ratings

    out = teams.copy()
    out["elo"] = out["team"].map(lambda t: engine.ratings.get(t, engine.new_team))
    out["elo_prev"] = out["team"].map(lambda t: prev_snap.get(t, engine.new_team))
    out["elo_preseason"] = out["team"].map(lambda t: pre_snap.get(t, engine.new_team))
    out["ovr"] = 100 * win_prob(out["elo"] - avg)
    out["rank"] = out["elo"].rank(ascending=False, method="min").astype(int)
    out["rank_prev"] = out["elo_prev"].rank(ascending=False, method="min").astype(int)
    out["rank_change"] = out["rank_prev"] - out["rank"]

    stats = {t: dict(w=0, l=0, cw=0, cl=0, pf=0, pa=0, gp=0, opp=[], rem=[], last=None, nxt=None)
             for t in out["team"]}

    for g in cur.itertuples(index=False):
        for side in ("home", "away"):
            team = getattr(g, side)
            if team not in stats:
                continue
            opp_side = "away" if side == "home" else "home"
            opp = getattr(g, opp_side)
            opp_fbs = getattr(g, f"{opp_side}_fbs")
            opp_elo = engine.ratings.get(opp, engine.new_team) if opp_fbs else engine.fcs
            s = stats[team]
            if g.completed:
                pf, pa = getattr(g, f"{side}_pts"), getattr(g, f"{opp_side}_pts")
                s["gp"] += 1
                s["pf"] += pf
                s["pa"] += pa
                s["opp"].append(opp_elo)
                won = pf > pa
                s["w" if won else "l"] += 1
                if g.conf_game and not g.is_ccg:
                    s["cw" if won else "cl"] += 1
                s["last"] = (g, side)
            elif g.season_type == "regular":
                s["rem"].append(opp_elo)
                if s["nxt"] is None:
                    s["nxt"] = (g, side)

    def f(t, key):
        return stats[t][key]

    out["wins"] = out["team"].map(lambda t: f(t, "w"))
    out["losses"] = out["team"].map(lambda t: f(t, "l"))
    out["conf_wins"] = out["team"].map(lambda t: f(t, "cw"))
    out["conf_losses"] = out["team"].map(lambda t: f(t, "cl"))
    out["games"] = out["team"].map(lambda t: f(t, "gp"))
    out["ppg"] = out["team"].map(lambda t: f(t, "pf") / f(t, "gp") if f(t, "gp") else np.nan)
    out["papg"] = out["team"].map(lambda t: f(t, "pa") / f(t, "gp") if f(t, "gp") else np.nan)
    out["sos"] = out["team"].map(lambda t: np.mean(f(t, "opp")) if f(t, "opp") else np.nan)
    out["sos_remaining"] = out["team"].map(lambda t: np.mean(f(t, "rem")) if f(t, "rem") else np.nan)
    out["sos_rank"] = out["sos"].rank(ascending=False, method="min")
    out["sos_remaining_rank"] = out["sos_remaining"].rank(ascending=False, method="min")
    out["last_game"] = out["team"].map(lambda t: f(t, "last"))
    out["next_game"] = out["team"].map(lambda t: f(t, "nxt"))
    return out.sort_values("rank").reset_index(drop=True), avg


# --------------------------------------------------------------------------- adjusted scoring

def adjusted_scoring(games, teams, season, hfa_points, lam):
    """Ridge regression: points = mu + home_adj + OFF[offense] - DEF[defense].

    FCS opponents are pooled into one "FCS" team. Positive OFF = scores more than an
    average offense would against the same defenses; positive DEF = allows fewer.
    """
    played = games[(games["season"] == season) & games["completed"]]
    names = list(teams["team"]) + ["FCS"]
    idx = {t: i for i, t in enumerate(names)}
    n = len(names)
    rows, y = [], []
    for g in played.itertuples(index=False):
        h = g.home if g.home_fbs else "FCS"
        a = g.away if g.away_fbs else "FCS"
        if h not in idx or a not in idx:
            continue
        for off, dfn, pts, loc in ((h, a, g.home_pts, 1), (a, h, g.away_pts, -1)):
            x = np.zeros(2 + 2 * n)
            x[0] = 1.0
            x[1] = 0.0 if g.neutral else loc
            x[2 + idx[off]] = 1.0
            x[2 + n + idx[dfn]] = -1.0
            rows.append(x)
            y.append(pts)
    if not rows:
        return pd.DataFrame({"team": teams["team"], "adj_off": np.nan, "adj_def": np.nan, "adj_net": np.nan})
    X, y = np.vstack(rows), np.array(y, dtype=float)
    pen = np.full(2 + 2 * n, lam)
    pen[:2] = 0.0
    beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
    mu = beta[0]
    off = beta[2 : 2 + n]
    dfn = beta[2 + n :]
    res = pd.DataFrame({"team": names, "off_rating": off, "def_rating": dfn})
    res["adj_off"] = mu + res["off_rating"]      # points per game vs an average defense
    res["adj_def"] = mu - res["def_rating"]      # points allowed per game vs an average offense
    res["adj_net"] = res["off_rating"] + res["def_rating"]
    res = res[res["team"] != "FCS"].copy()
    res["adj_off_rank"] = res["adj_off"].rank(ascending=False, method="min")
    res["adj_def_rank"] = res["adj_def"].rank(ascending=True, method="min")
    res["adj_net_rank"] = res["adj_net"].rank(ascending=False, method="min")
    res["league_avg_ppg"] = mu
    return res
