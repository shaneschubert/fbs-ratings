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

TUNED_KEYS = ["k", "home_field", "offseason_regression", "talent_weight", "mov_cap"]


def season_conferences(games, teams, season, independent):
    """season -> {team: conference} for FBS teams, from the games themselves (realignment-safe)."""
    out = {}
    cols = ["season", "home", "away", "home_conf", "away_conf", "home_fbs", "away_fbs"]
    for yr, h, a, hc, ac, hf, af in games[cols].itertuples(index=False, name=None):
        d = out.setdefault(yr, {})
        if hf and hc and h not in d:
            d[h] = hc
        if af and ac and a not in d:
            d[a] = ac
    out.setdefault(season, {}).update(dict(zip(teams["team"], teams["conference"])))
    return out


def talent_z(rows):
    """season -> {team: z-score of talent among teams that season}."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for yr, g in df.groupby("season"):
        sd = g["talent"].std() or 1.0
        out[int(yr)] = dict(zip(g["team"], (g["talent"] - g["talent"].mean()) / sd))
    return out


class EloEngine:
    def __init__(self, cfg, params=None, conf_map=None, talent=None):
        e = cfg["elo"]
        p = {k: e[k] for k in TUNED_KEYS}
        p["elo_per_point"] = e["elo_per_point"]
        p.update(params or {})
        self.params = p
        self.mean, self.fcs, self.new_team = e["mean"], e["fcs_rating"], e["new_team_rating"]
        self.k, self.hfa = float(p["k"]), float(p["home_field"])
        self.regress, self.tw = float(p["offseason_regression"]), float(p["talent_weight"])
        self.mov_cap, self.per_point = float(p["mov_cap"]), float(p["elo_per_point"])
        self.independent = cfg["conferences"]["independent"]
        self.conf_map = conf_map or {}
        self.talent = talent or {}
        self.ratings = {}
        self.snapshots = {}

    def rating(self, team, fbs):
        if not fbs:
            return self.fcs
        return self.ratings.get(team, self.new_team)

    def _new_season(self, season):
        prev = self.ratings
        confs = self.conf_map.get(season, {})
        tal = self.talent.get(season, {})
        sd = float(np.std(list(prev.values()))) if len(prev) > 10 else 200.0
        groups = {}
        for t, c in confs.items():
            if t in prev and c and c != self.independent:
                groups.setdefault(c, []).append(prev[t])
        cmean = {c: float(np.mean(v)) for c, v in groups.items() if len(v) >= 4}
        new = {}
        for t in set(prev) | set(confs):
            if t in prev:
                target = cmean.get(confs.get(t), self.mean)
                base = target + (prev[t] - target) * (1 - self.regress)
            elif not prev:
                base = self.mean
            else:
                base = self.new_team
            z = tal.get(t)
            if z is not None and self.tw > 0:
                base = (1 - self.tw) * base + self.tw * (self.mean + sd * z)
            new[t] = base
        self.ratings = new

    def run(self, games: pd.DataFrame, record=True, eval_seasons=()):
        """Play every game in order. Returns (games with pregame Elo, history, metrics by season)."""
        cols = ["season", "season_type", "week", "date", "game_id", "home", "away", "home_fbs", "away_fbs",
                "neutral", "completed", "home_pts", "away_pts"]
        rows = games[cols].itertuples(index=False, name=None)
        pre_h, pre_a, wp_h, post_h, post_a, history = [], [], [], [], [], []
        metrics = {}
        season, week_key = None, None
        hfa, k, cap, ratings_get = self.hfa, self.k, self.mov_cap, None
        for yr, stype, wk, date, gid, home, away, hf, af, neutral, done, hp, ap in rows:
            if yr != season:
                if season is not None or self.conf_map or self.talent:
                    self._new_season(yr)
                season = yr
            if record:
                key = (yr, stype, wk)
                if key != week_key:
                    self.snapshots[key] = dict(self.ratings)
                    week_key = key
            R = self.ratings
            rh = R.get(home, self.new_team) if hf else self.fcs
            ra = R.get(away, self.new_team) if af else self.fcs
            diff = rh - ra + (0.0 if neutral else hfa)
            p = 1.0 / (1.0 + 10 ** (-diff / 400.0))
            if record:
                pre_h.append(rh); pre_a.append(ra); wp_h.append(p)
            if not done:
                if record:
                    post_h.append(np.nan); post_a.append(np.nan)
                continue
            margin = hp - ap
            if yr in eval_seasons and hf and af and margin != 0:
                m = metrics.setdefault(yr, [0, 0.0, 0.0, 0, 0.0, 0.0])
                y = 1.0 if margin > 0 else 0.0
                pc = min(max(p, 1e-6), 1 - 1e-6)
                m[0] += 1
                m[1] -= y * math.log(pc) + (1 - y) * math.log(1 - pc)
                m[2] += (p - y) ** 2
                m[3] += (p >= 0.5) == (y == 1.0)
                m[4] += diff * margin
                m[5] += diff * diff
            result = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            winner_diff = diff if margin > 0 else -diff
            mult = (math.log(min(abs(margin), cap) + 1) * (2.2 / (winner_diff * 0.001 + 2.2))) if margin else 1.0
            shift = k * mult * (result - p)
            if hf:
                R[home] = rh + shift
            if af:
                R[away] = ra - shift
            if record:
                post_h.append(rh + shift if hf else np.nan)
                post_a.append(ra - shift if af else np.nan)
                if hf:
                    history.append((yr, stype, wk, date, home, rh, rh + shift, gid))
                if af:
                    history.append((yr, stype, wk, date, away, ra, ra - shift, gid))
        if not record:
            return None, None, metrics
        games = games.copy()
        games["home_elo_pre"], games["away_elo_pre"] = pre_h, pre_a
        games["home_wp"] = wp_h
        games["home_elo_post"], games["away_elo_post"] = post_h, post_a
        hist = pd.DataFrame(history, columns=["season", "season_type", "week", "date", "team",
                                              "elo_pre", "elo_post", "game_id"])
        return games, hist, metrics

    def predict(self, home, away, home_fbs, away_fbs, neutral, ratings=None):
        r = ratings if ratings is not None else self.ratings
        rh = r.get(home, self.new_team) if home_fbs else self.fcs
        ra = r.get(away, self.new_team) if away_fbs else self.fcs
        diff = rh - ra + (0 if neutral else self.hfa)
        return float(win_prob(diff)), diff / self.per_point


def summarize(metrics, seasons=None):
    rows = [m for yr, m in metrics.items() if seasons is None or yr in seasons]
    n = sum(m[0] for m in rows)
    if not n:
        return {"games": 0, "log_loss": np.nan, "brier": np.nan, "accuracy": np.nan, "elo_per_point": np.nan}
    sxy, sxx = sum(m[4] for m in rows), sum(m[5] for m in rows)
    return {"games": n, "log_loss": sum(m[1] for m in rows) / n, "brier": sum(m[2] for m in rows) / n,
            "accuracy": sum(m[3] for m in rows) / n, "elo_per_point": sxx / sxy if sxy > 0 else np.nan}


def tune(cfg, games, conf_map, talent, eval_seasons, log=print):
    """Coordinate descent over the grid in config.yaml, minimizing log loss on past seasons."""
    grid = cfg["tuning"]["grid"]
    params = {k: cfg["elo"][k] for k in TUNED_KEYS}
    cache = {}

    def score(p):
        key = tuple(p[k] for k in TUNED_KEYS)
        if key not in cache:
            eng = EloEngine(cfg, p, conf_map, talent)
            _, _, m = eng.run(games, record=False, eval_seasons=eval_seasons)
            cache[key] = summarize(m)["log_loss"]
        return cache[key]

    best = score(params)
    start = best
    for _ in range(int(cfg["tuning"]["passes"])):
        improved = False
        for key in TUNED_KEYS:
            for v in grid.get(key, []):
                trial = {**params, key: v}
                s = score(trial)
                if s < best - 1e-6:
                    best, params, improved = s, trial, True
        if not improved:
            break
    log(f"Tuning: {len(cache)} settings tried, log loss {start:.4f} -> {best:.4f}")
    return params


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
