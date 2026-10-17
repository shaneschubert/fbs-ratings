"""Monte Carlo simulation of the rest of the season, conference title games, and the 12-team CFP."""
import numpy as np
import pandas as pd

from .model import win_prob


def simulate(teams, games, engine, cfg, season, next_week):
    sc = cfg["simulation"]
    S = int(sc["runs"])
    rng = np.random.default_rng(sc["seed"])
    k_sim = engine.k * float(sc["hot_k_multiplier"])
    hfa = engine.hfa

    names = list(teams["team"])
    n = len(names)
    idx = {t: i for i, t in enumerate(names)}
    conf_of = dict(zip(teams["team"], teams["conference"]))
    div_of = dict(zip(teams["team"], teams["division"]))

    base_elo = np.array([engine.ratings.get(t, engine.new_team) for t in names])
    elo = np.tile(base_elo, (S, 1))
    wins = np.zeros((S, n), dtype=np.int16)
    losses = np.zeros((S, n), dtype=np.int16)
    cw = np.zeros((S, n), dtype=np.int16)
    cl = np.zeros((S, n), dtype=np.int16)

    cur = games[(games["season"] == season)]
    reg = cur[cur["season_type"] == "regular"]

    # Completed results (same in every simulation)
    for g in reg[reg["completed"] & ~reg["is_ccg"]].itertuples(index=False):
        hw = g.home_pts > g.away_pts
        for team, won in ((g.home, hw), (g.away, not hw)):
            if team in idx:
                i = idx[team]
                if won:
                    wins[:, i] += 1
                else:
                    losses[:, i] += 1
                if g.conf_game:
                    if won:
                        cw[:, i] += 1
                    else:
                        cl[:, i] += 1

    # Remaining regular season games
    remaining = reg[~reg["completed"] & ~reg["is_ccg"]]
    week_games = []  # (game_id, home_idx_or_None, away_idx_or_None, home_win_array)
    for g in remaining.itertuples(index=False):
        hi, ai = idx.get(g.home), idx.get(g.away)
        eh = elo[:, hi] if hi is not None else np.full(S, engine.fcs)
        ea = elo[:, ai] if ai is not None else np.full(S, engine.fcs)
        p = win_prob(eh - ea + (0 if g.neutral else hfa))
        hw = rng.random(S) < p
        shift = k_sim * (hw.astype(float) - p)
        if hi is not None:
            elo[:, hi] += shift
            wins[:, hi] += hw
            losses[:, hi] += ~hw
            if g.conf_game:
                cw[:, hi] += hw
                cl[:, hi] += ~hw
        if ai is not None:
            elo[:, ai] -= shift
            wins[:, ai] += ~hw
            losses[:, ai] += hw
            if g.conf_game:
                cw[:, ai] += ~hw
                cl[:, ai] += hw
        if next_week is not None and g.week == next_week:
            week_games.append((g.game_id, hi, ai, hw.copy()))

    reg_wins, reg_losses = wins.copy(), losses.copy()

    # Conference title games
    conf_cfg = cfg["conferences"]
    all_confs = conf_cfg["power"] + conf_cfg["group_of"]
    ccg_rows = cur[cur["is_ccg"]]
    champ = {}
    in_ccg = np.zeros((S, n), dtype=bool)
    tiebreak = rng.random((S, n)) * 1e-9

    for conf in all_confs:
        members = [idx[t] for t in names if conf_of[t] == conf]
        if not members:
            continue
        m = np.array(members)
        actual = ccg_rows[(ccg_rows["home_conf"] == conf) | (ccg_rows["away_conf"] == conf)]
        if len(actual):
            g = actual.iloc[0]
            hi, ai = idx.get(g.home), idx.get(g.away)
            if hi is None or ai is None:
                continue
            in_ccg[:, hi] = in_ccg[:, ai] = True
            if g.completed:
                winner = hi if g.home_pts > g.away_pts else ai
                champ[conf] = np.full(S, winner)
                wins[:, winner] += 1
                losses[:, ai if winner == hi else hi] += 1
                continue
            a, b = np.full(S, hi), np.full(S, ai)
            neutral = bool(g.neutral)
        else:
            played = cw[:, m] + cl[:, m]
            pct = np.where(played > 0, cw[:, m] / np.maximum(played, 1), 0.0)
            key = pct + elo[:, m] * 1e-7 + tiebreak[:, m]
            if conf in conf_cfg.get("divisions", []):
                divs = sorted({div_of[names[i]] for i in m})
                picks = []
                for d in divs[:2]:
                    mask = np.array([div_of[names[i]] == d for i in m])
                    k2 = np.where(mask, key, -np.inf)
                    picks.append(m[np.argmax(k2, axis=1)])
                a, b = picks[0], picks[1]
            else:
                order = np.argsort(-key, axis=1)
                a, b = m[order[:, 0]], m[order[:, 1]]
            if conf in conf_cfg.get("no_title_game", []):
                champ[conf] = a
                continue
            neutral = True
            in_ccg[np.arange(S), a] = True
            in_ccg[np.arange(S), b] = True
        ea = elo[np.arange(S), a]
        eb = elo[np.arange(S), b]
        p = win_prob(ea - eb + (0 if neutral else hfa))
        aw = rng.random(S) < p
        w = np.where(aw, a, b)
        lo = np.where(aw, b, a)
        np.add.at(wins, (np.arange(S), w), 1)
        np.add.at(losses, (np.arange(S), lo), 1)
        champ[conf] = w

    is_champ = np.zeros((S, n), dtype=bool)
    for conf, w in champ.items():
        is_champ[np.arange(S), w] = True

    # CFP selection (committee proxy) and bracket
    po = cfg["playoff"]
    field = int(po["field_size"])
    score = elo - po["loss_penalty"] * losses + po["champion_bonus"] * is_champ + tiebreak
    order = np.argsort(-score, axis=1)
    nd = idx.get(po.get("notre_dame_rule") or "", None)
    power = [c for c in conf_cfg["power"] if c in champ]
    gof = [c for c in conf_cfg["group_of"] if c in champ]
    seeds = np.zeros((S, field), dtype=int)
    rank_pos = np.empty((S, n), dtype=int)
    rank_pos[np.arange(S)[:, None], order] = np.arange(n)[None, :]

    for s in range(S):
        auto = set()
        if po["power_auto_bids"]:
            auto.update(int(champ[c][s]) for c in power)
        if po["group_of_auto_bids"] and gof:
            g_champs = sorted((int(champ[c][s]) for c in gof), key=lambda t: rank_pos[s, t])
            auto.update(g_champs[: int(po["group_of_auto_bids"])])
        if nd is not None and rank_pos[s, nd] < field:
            auto.add(nd)
        chosen = list(auto)
        for t in order[s]:
            if len(chosen) >= field:
                break
            if t not in auto:
                chosen.append(int(t))
        chosen.sort(key=lambda t: rank_pos[s, t])
        seeds[s] = chosen[:field]

    rows = np.arange(S)

    def play(a, b, a_home):
        ea, eb = elo[rows, a], elo[rows, b]
        p = win_prob(ea - eb + np.where(a_home, hfa, 0))
        aw = rng.random(S) < p
        return np.where(aw, a, b)

    made = np.zeros((S, n), dtype=bool)
    made[rows[:, None], seeds] = True
    seed_num = np.zeros((S, n), dtype=np.int16)
    seed_num[rows[:, None], seeds] = np.arange(1, field + 1)[None, :]
    bye = np.zeros((S, n), dtype=bool)
    bye[rows[:, None], seeds[:, : po["byes"]]] = True
    host = np.zeros((S, n), dtype=bool)
    host[rows[:, None], seeds[:, 4:8]] = True

    # Seeds are 1-based in comments; columns are 0-based.
    r1 = {
        "5v12": play(seeds[:, 4], seeds[:, 11], np.ones(S, bool)),
        "6v11": play(seeds[:, 5], seeds[:, 10], np.ones(S, bool)),
        "7v10": play(seeds[:, 6], seeds[:, 9], np.ones(S, bool)),
        "8v9": play(seeds[:, 7], seeds[:, 8], np.ones(S, bool)),
    }
    no_home = np.zeros(S, bool)
    qf = [
        play(seeds[:, 0], r1["8v9"], no_home),
        play(seeds[:, 3], r1["5v12"], no_home),
        play(seeds[:, 2], r1["6v11"], no_home),
        play(seeds[:, 1], r1["7v10"], no_home),
    ]
    sf = [play(qf[0], qf[1], no_home), play(qf[2], qf[3], no_home)]
    title = play(sf[0], sf[1], no_home)

    def mark(arrs):
        out = np.zeros((S, n), dtype=bool)
        for a in arrs:
            out[rows, a] = True
        return out

    reach_qf = bye | mark(r1.values())
    reach_sf = mark(qf)
    reach_final = mark(sf)
    won_title = mark([title])

    # Team projection table
    def mean(a):
        return a.mean(axis=0)

    proj = pd.DataFrame({
        "team": names,
        "proj_wins": mean(reg_wins), "proj_losses": mean(reg_losses),
        "proj_conf_wins": mean(cw), "proj_conf_losses": mean(cl),
        "p_ccg": mean(in_ccg), "p_conf_champ": mean(is_champ),
        "p_cfp": mean(made), "p_bye": mean(bye), "p_host_r1": mean(host),
        "p_qf": mean(reach_qf), "p_sf": mean(reach_sf), "p_final": mean(reach_final),
        "p_title": mean(won_title),
        "p_unbeaten": mean(reg_losses == 0),
    })
    with np.errstate(invalid="ignore"):
        proj["avg_seed"] = np.where(made.any(axis=0), seed_num.sum(axis=0) / np.maximum(made.sum(axis=0), 1), np.nan)
    for k in range(1, field + 1):
        proj[f"p_seed_{k}"] = (seed_num == k).mean(axis=0)
    win_hist = {w: (reg_wins == w).mean(axis=0) for w in range(0, 16)}
    for w, v in win_hist.items():
        proj[f"p_{w}_reg_wins"] = v

    # Leverage of this week's games on CFP odds
    lev = []
    for gid, hi, ai, hw in week_games:
        row = {"game_id": gid}
        for side, ti, won in (("home", hi, hw), ("away", ai, ~hw)):
            if ti is None or won.all() or (~won).all():
                row[f"{side}_cfp_if_win"] = row[f"{side}_cfp_if_loss"] = np.nan
                continue
            row[f"{side}_cfp_if_win"] = made[won, ti].mean()
            row[f"{side}_cfp_if_loss"] = made[~won, ti].mean()
        lev.append(row)
    lev = pd.DataFrame(lev, columns=["game_id", "home_cfp_if_win", "home_cfp_if_loss",
                                     "away_cfp_if_win", "away_cfp_if_loss"])

    return proj, lev, S
