"""Writes the Quarto pages (site/*.qmd) from the model outputs.

Tables are pre-rendered HTML inside ```{=html} blocks, so Quarto needs no
Python/Jupyter at render time.
"""
import html
from pathlib import Path

import numpy as np
import pandas as pd

E = html.escape


# --------------------------------------------------------------------------- formatting

def num(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{d}f}"


def pct(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "-"
    if p <= 0:
        return "0%"
    if p >= 1:
        return "100%"
    if p < 0.005:
        return "<1%"
    if p > 0.995:
        return ">99%"
    return f"{p * 100:.0f}%"


def rec(w, l):
    return f"{int(w)}-{int(l)}"


def raw_html(s):
    return "\n```{=html}\n" + s + "\n```\n"


def page(title, body, extra_yaml=""):
    return f'---\ntitle: "{title}"\n{extra_yaml}---\n' + body


class Ctx:
    """Shared lookups for rendering."""

    def __init__(self, cfg, teams, games, proj, adj, season, last_week, next_week, updated, fcs_ovr):
        self.cfg = cfg
        self.t = teams.merge(proj, on="team", how="left").merge(adj, on="team", how="left")
        self.by_team = self.t.set_index("team")
        self.games = games
        self.season = season
        self.last_week = last_week
        self.next_week = next_week
        self.updated = updated
        self.fcs_ovr = fcs_ovr
        self.footer = cfg["site"]["title"] + f" {season}" + (
            f" | {cfg['site']['handle']}" if cfg["site"].get("handle") else ""
        )

    # ---- cells
    def logo(self, team):
        if team not in self.by_team.index:
            return '<span class="logo logo-fcs" aria-hidden="true">FCS</span>'
        r = self.by_team.loc[team]
        if r["logo"]:
            return f'<img class="logo" src="{E(r["logo"])}" alt="" loading="lazy">'
        return (f'<span class="logo logo-fallback" style="background:{E(r["color"])}" aria-hidden="true">'
                f'{E(str(r["abbr"])[:4])}</span>')

    def team_link(self, team, prefix="", mascot=False, rank=False):
        if team not in self.by_team.index:
            return f'<span class="team-name fcs">{E(team)}</span>'
        r = self.by_team.loc[team]
        rk = f'<span class="rk">{int(r["rank"])}</span>' if rank else ""
        m = f' <span class="mascot">{E(r["mascot"])}</span>' if mascot and r["mascot"] else ""
        return f'{rk}<a class="team-name" href="{prefix}teams/{r["slug"]}.html">{E(team)}</a>{m}'

    def team_cell(self, team, prefix="", mascot=True, rank=False):
        return f'<span class="team-cell">{self.logo(team)}<span>{self.team_link(team, prefix, mascot, rank)}</span></span>'

    def ovr_elo(self, team):
        r = self.by_team.loc[team]
        return f'<span class="ovr">{num(r["ovr"])}</span><span class="sub">{r["elo"]:.0f}</span>'

    def change(self, v):
        v = int(v)
        if v > 0:
            return f'<span class="up">+{v}</span>'
        if v < 0:
            return f'<span class="down">{v}</span>'
        return '<span class="flat">0</span>'

    def bar(self, p):
        if p is None or np.isnan(p):
            return "-"
        w = max(0.0, min(1.0, p)) * 100
        return f'<span class="pbar"><span style="width:{w:.0f}%"></span></span><span class="pval">{pct(p)}</span>'

    def last_str(self, item, prefix=""):
        if not item:
            return "-"
        g, side = item
        opp_side = "away" if side == "home" else "home"
        pf, pa = getattr(g, f"{side}_pts"), getattr(g, f"{opp_side}_pts")
        res = "W" if pf > pa else "L"
        where = "vs" if g.neutral or side == "home" else "@"
        opp = getattr(g, opp_side)
        cls = "win" if res == "W" else "loss"
        return f'<span class="{cls}">{res} {int(pf)}-{int(pa)}</span> {where} {self.team_link(opp, prefix)}'

    def next_str(self, item, prefix=""):
        if not item:
            return "-"
        g, side = item
        opp_side = "away" if side == "home" else "home"
        opp = getattr(g, opp_side)
        where = "vs" if g.neutral or side == "home" else "@"
        rk = ""
        if opp in self.by_team.index:
            rk = f'<span class="rk">{int(self.by_team.loc[opp, "rank"])}</span>'
        return f'{where} {rk}{self.team_link(opp, prefix)}'


def table(cols, rows, cls="", tid=None, note=None, footer=None):
    """cols: list of (label, align, title). rows: list of lists of cells; a cell is str or (html, sort_value)."""
    tid_attr = f' id="{tid}"' if tid else ""
    head = "".join(
        f'<th class="{a}" title="{E(t or l)}" data-sort>{l}</th>' for l, a, t in cols
    )
    body = []
    for r in rows:
        tds = []
        for (l, a, t), c in zip(cols, r):
            if isinstance(c, tuple):
                h, sv = c
                tds.append(f'<td class="{a}" data-v="{E(str(sv))}">{h}</td>')
            else:
                tds.append(f'<td class="{a}">{c}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    foot = ""
    if note or footer:
        foot = f'<caption class="tfoot">{E(footer or "")}{"<br>" if note and footer else ""}{E(note or "")}</caption>'
    return (f'<div class="tbl-wrap"><table class="data sortable {cls}"{tid_attr}>{foot}'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


# --------------------------------------------------------------------------- page sections

def ratings_rows(ctx, df, prefix=""):
    rows = []
    for r in df.itertuples(index=False):
        conf = E(r.conference)
        rows.append([
            (str(int(r.rank)), int(r.rank)),
            (ctx.change(r.rank_change), int(r.rank_change)),
            (ctx.team_cell(r.team, prefix), r.team),
            (ctx.ovr_elo(r.team), r.elo),
            (f'{conf}', r.conference),
            (rec(r.wins, r.losses), r.wins - r.losses),
            (rec(r.conf_wins, r.conf_losses), r.conf_wins - r.conf_losses),
            (f'{num(r.sos, 0)}<span class="sub">#{int(r.sos_rank) if not np.isnan(r.sos_rank) else "-"}</span>',
             r.sos if not np.isnan(r.sos) else 0),
            (num(r.ppg), r.ppg if not np.isnan(r.ppg) else 0),
            (num(r.papg), r.papg if not np.isnan(r.papg) else 0),
            (f'{num(r.proj_wins)}-{num(r.proj_losses)}', r.proj_wins),
            (pct(r.p_cfp), r.p_cfp),
            ctx.last_str(r.last_game, prefix),
            ctx.next_str(r.next_game, prefix),
        ])
    return rows


RATING_COLS = [
    ("Rk", "c", "Rank by Elo"), ("+/-", "c", "Rank change since last week"), ("Team", "l", None),
    ("OVR / Elo", "r", "OVR = chance to beat an average FBS team on a neutral field"),
    ("Conf", "l", "Conference"), ("Record", "c", None), ("Conf Rec", "c", "Conference record"),
    ("SOS", "r", "Average current Elo of opponents played"), ("PPG", "r", None), ("PAPG", "r", None),
    ("Proj", "c", "Projected regular season record"), ("CFP%", "r", "Chance to make the CFP"),
    ("Last week", "l", None), ("Next", "l", None),
]


def game_rows(ctx, gdf, lev=None, prefix="", show_score=True):
    rows = []
    lev_map = lev.set_index("game_id") if lev is not None and len(lev) else None
    for g in gdf.itertuples(index=False):
        def side(team):
            if team in ctx.by_team.index:
                r = ctx.by_team.loc[team]
                return (f'{ctx.team_cell(team, prefix, mascot=False, rank=True)}'
                        f'<span class="sub">{rec(r.wins, r.losses)}</span>')
            return ctx.team_cell(team, prefix, mascot=False)
        sep = "vs" if g.neutral else "@"
        matchup = (f'<div class="matchup"><div><span class="at"></span>{side(g.away)}</div>'
                   f'<div><span class="at">{sep}</span>{side(g.home)}</div></div>')
        hp = g.home_wp
        fav, p = (g.home, hp) if hp >= 0.5 else (g.away, 1 - hp)
        spread = abs(g.home_spread)
        pick = f'{ctx.logo(fav)} {E(fav)} <span class="pval">{pct(p)}</span> <span class="sub">by {spread:.1f}</span>'
        date = g.date.tz_convert(ctx.cfg["site"]["timezone"]).strftime("%a %-m/%-d") if pd.notna(g.date) else "TBD"
        row = [(date, g.date.value if pd.notna(g.date) else 0), matchup, (pick, p)]
        if show_score:
            if g.completed:
                hw = g.home_pts > g.away_pts
                correct = (hp >= 0.5) == hw
                score = f'{int(g.away_pts)}-{int(g.home_pts)}'
                mark = '<span class="hit" title="Pick correct">&#10003;</span>' if correct else '<span class="miss" title="Pick missed">&#10007;</span>'
                row.append(f'{score} {mark}')
            else:
                row.append('<span class="sub">-</span>')
        if lev_map is not None:
            gs = getattr(g, "gamescore", np.nan)
            row.append((num(gs), gs if not np.isnan(gs) else 0))
        rows.append(row)
    return rows


def write_site(ctx, site_dir: Path, gotw, lev, elo_hist, downloads):
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "teams").mkdir(exist_ok=True)
    for f in (site_dir / "teams").glob("*.qmd"):
        f.unlink()
    cfg = ctx.cfg
    t = ctx.t.sort_values("rank")
    wk_label = f"Week {ctx.next_week}" if ctx.next_week else "Postseason"
    updated = ctx.updated

    # ---------------------------------------------------------------- home
    ratings_top = table(RATING_COLS, ratings_rows(ctx, t.head(10)), footer=ctx.footer)
    gotw_cols = [("Date", "c", None), ("Matchup", "l", None), ("Pick", "l", "Favorite, win chance, projected margin"),
                 ("Final", "c", None), ("GameScore", "r", "Quality, closeness, and CFP stakes")]
    gotw_html = (table(gotw_cols, game_rows(ctx, gotw, lev), footer=ctx.footer)
                 if len(gotw) else '<p class="empty">No upcoming games. Check back when the next week is scheduled.</p>')

    off = t.sort_values("adj_off", ascending=False).head(10)
    dfn = t.sort_values("adj_def", ascending=True).head(10)
    lead_cols = [("Rk", "c", None), ("Team", "l", None), ("Pts", "r", None)]
    off_tbl = table(lead_cols, [[str(i + 1), ctx.team_cell(r.team), num(r.adj_off)] for i, r in enumerate(off.itertuples())])
    def_tbl = table(lead_cols, [[str(i + 1), ctx.team_cell(r.team), num(r.adj_def)] for i, r in enumerate(dfn.itertuples())])

    field = projected_field(ctx)
    cfp_cols = [("Seed", "c", None), ("Team", "l", None), ("Record", "c", None), ("CFP%", "r", None), ("Title%", "r", None)]
    cfp_rows = [[str(s), ctx.team_cell(tm), rec(ctx.by_team.loc[tm, "wins"], ctx.by_team.loc[tm, "losses"]),
                 ctx.bar(ctx.by_team.loc[tm, "p_cfp"]), pct(ctx.by_team.loc[tm, "p_title"])] for s, tm in field]
    cfp_tbl = table(cfp_cols, cfp_rows, footer=ctx.footer)

    home = f"""
<section class="hero">
  <div class="hero-field" aria-hidden="true"></div>
  <div class="hero-text">
    <div class="hero-title" role="heading" aria-level="1">{E(cfg['site']['title'])}</div>
    <p class="hero-tag">{E(cfg['site']['tagline'])}</p>
    <p class="hero-meta"><strong>{wk_label}</strong> <span>Updated {E(updated)}</span></p>
  </div>
</section>
<h2>Games of the week</h2>
<p class="lede">Top {cfg['games_of_week']['count']} by GameScore. <a href="schedule.html">Full schedule and scores</a></p>
{gotw_html}
<h2>Ratings</h2>
<p class="lede">Top 10 of {len(t)}. <a href="ratings.html">All FBS teams</a></p>
{ratings_top}
<h2>Projected playoff field</h2>
<p class="lede">{ctx.cfg['simulation']['runs']:,} simulations of the rest of the season. <a href="playoff.html">Bracket and full odds</a></p>
{cfp_tbl}
<h2>Adjusted scoring leaders</h2>
<p class="lede">Points per game against an average opponent. <a href="stats.html">Full stats</a></p>
<div class="two-up"><div><h3>Offense</h3>{off_tbl}</div><div><h3>Defense</h3>{def_tbl}</div></div>
"""
    (site_dir / "index.qmd").write_text(page("", raw_html(home), 'pagetitle: "' + cfg["site"]["title"] + '"\n'))

    # ---------------------------------------------------------------- ratings
    confs = sorted(t["conference"].unique())
    opts = "".join(f'<option value="{E(c)}">{E(c)}</option>' for c in confs)
    filt = (f'<div class="filters"><label>Conference <select data-filter-col="4" data-filter-table="ratingstable">'
            f'<option value="">All</option>{opts}</select></label>'
            f'<label>Search <input type="search" data-search-table="ratingstable" placeholder="Team name"></label></div>')
    body = (f'<p class="lede">{wk_label}. Updated {E(updated)}. Click any column to sort.</p>' + filt
            + table(RATING_COLS, ratings_rows(ctx, t), tid="ratingstable", footer=ctx.footer))
    (site_dir / "ratings.qmd").write_text(page("Ratings", raw_html(body)))

    # ---------------------------------------------------------------- schedule
    cur = ctx.games[(ctx.games["season"] == ctx.season)].copy()
    cur["wk_label"] = np.where(cur["season_type"] == "postseason", "Bowls and CFP",
                               "Week " + cur["week"].astype(str))
    done = cur[cur["completed"]]
    correct = ((done["home_wp"] >= 0.5) == (done["home_pts"] > done["away_pts"])).sum()
    sched_cols = [("Date", "c", None), ("Matchup", "l", None), ("Pick", "l", "Favorite, win chance, projected margin"),
                  ("Final", "c", "Away-Home")]
    cur["_st"] = (cur["season_type"] == "postseason").astype(int)
    labels = list(dict.fromkeys(cur.sort_values(["_st", "week"])["wk_label"]))
    default = f"Week {ctx.next_week}" if ctx.next_week else labels[-1] if labels else ""
    opts = "".join(f'<option value="{E(l)}"{" selected" if l == default else ""}>{E(l)}</option>' for l in labels)
    sections = []
    for l in labels:
        gw = cur[cur["wk_label"] == l].sort_values("date")
        sections.append(f'<section class="week" data-week="{E(l)}">{table(sched_cols, game_rows(ctx, gw), footer=ctx.footer)}</section>')
    acc = f"{correct}-{len(done) - correct} ({correct / len(done) * 100:.1f}%)" if len(done) else "no games yet"
    body = (f'<p class="lede">Pick record this season: <strong>{acc}</strong>. Every pick is made before kickoff from the ratings at that time.</p>'
            f'<div class="filters"><label>Week <select id="week-select">{opts}</select></label></div>' + "".join(sections))
    (site_dir / "schedule.qmd").write_text(page("Schedule and scores", raw_html(body)))

    # ---------------------------------------------------------------- standings
    st_cols = [("Rk", "c", "National rank"), ("Team", "l", None), ("OVR / Elo", "r", None), ("Overall", "c", None),
               ("Conf", "c", None), ("PPG", "r", None), ("PAPG", "r", None),
               ("Proj Conf", "c", "Projected conference record"), ("Title game%", "r", None),
               ("Champ%", "r", None), ("CFP%", "r", None)]
    cc = cfg["conferences"]
    order = [c for c in cc["power"] + cc["group_of"] + [cc["independent"]] if c in set(t["conference"])]
    order += [c for c in confs if c not in order]
    parts = ["\n::: {.panel-tabset}\n"]
    for c in order:
        ct = t[t["conference"] == c].copy()
        ct["cpct"] = ct["conf_wins"] / (ct["conf_wins"] + ct["conf_losses"]).replace(0, np.nan)
        ct = ct.sort_values(["cpct", "proj_conf_wins", "elo"], ascending=False, na_position="last")
        rows = [[(str(int(r.rank)), r.rank), ctx.team_cell(r.team), (ctx.ovr_elo(r.team), r.elo),
                 rec(r.wins, r.losses), rec(r.conf_wins, r.conf_losses), num(r.ppg), num(r.papg),
                 f"{num(r.proj_conf_wins)}-{num(r.proj_conf_losses)}", (pct(r.p_ccg), r.p_ccg),
                 (pct(r.p_conf_champ), r.p_conf_champ), (pct(r.p_cfp), r.p_cfp)] for r in ct.itertuples()]
        parts.append(f"\n## {c}\n" + raw_html(table(st_cols, rows, footer=ctx.footer,
                     note=f"Projections from {cfg['simulation']['runs']:,} simulations based on current Elo")))
    parts.append("\n:::\n")
    (site_dir / "standings.qmd").write_text(page("Conference standings", "".join(parts)))

    # ---------------------------------------------------------------- playoff
    bracket = bracket_html(ctx, field)
    odds = t[t["p_cfp"] >= 0.001].sort_values("p_cfp", ascending=False)
    po_cols = [("Team", "l", None), ("Record", "c", None), ("Conf champ%", "r", None), ("CFP%", "r", None),
               ("Bye%", "r", "Top-4 seed"), ("Host%", "r", "Seeds 5-8 host a first round game"),
               ("Avg seed", "r", "Average seed when in the field"), ("QF%", "r", None), ("Semis%", "r", None),
               ("Final%", "r", None), ("Title%", "r", None)]
    po_rows = [[ctx.team_cell(r.team), rec(r.wins, r.losses), (pct(r.p_conf_champ), r.p_conf_champ),
                (ctx.bar(r.p_cfp), r.p_cfp), (pct(r.p_bye), r.p_bye), (pct(r.p_host_r1), r.p_host_r1),
                (num(r.avg_seed), r.avg_seed if not np.isnan(r.avg_seed) else 99), (pct(r.p_qf), r.p_qf),
                (pct(r.p_sf), r.p_sf), (pct(r.p_final), r.p_final), (pct(r.p_title), r.p_title)]
               for r in odds.itertuples()]
    po = cfg["playoff"]
    body = (f'<p class="lede">{po["field_size"]} teams, straight seeding, top {po["byes"]} seeds get a bye. '
            f'Field: four power conference champions, the top group of five champion, and the next highest ranked teams. '
            f'Projected field below is the most likely outcome; odds come from {cfg["simulation"]["runs"]:,} simulations.</p>'
            + bracket + "<h2>Playoff odds</h2>" + table(po_cols, po_rows, footer=ctx.footer))
    (site_dir / "playoff.qmd").write_text(page("College Football Playoff projection", raw_html(body)))

    # ---------------------------------------------------------------- stats
    s_cols = [("Team", "l", None), ("Conf", "l", None), ("Adj OFF", "r", "Points per game vs an average defense"),
              ("Rk", "c", None), ("Adj DEF", "r", "Points allowed per game vs an average offense"), ("Rk", "c", None),
              ("Adj Net", "r", "Adjusted points per game margin"), ("Rk", "c", None), ("PPG", "r", None),
              ("PAPG", "r", None), ("SOS", "r", "Opponents played"), ("Remaining SOS", "r", "Opponents still to play")]
    s_rows = [[ctx.team_cell(r.team), E(r.conference), num(r.adj_off), (num(r.adj_off_rank, 0), r.adj_off_rank),
               num(r.adj_def), (num(r.adj_def_rank, 0), r.adj_def_rank), num(r.adj_net),
               (num(r.adj_net_rank, 0), r.adj_net_rank), num(r.ppg), num(r.papg), num(r.sos, 0),
               num(r.sos_remaining, 0)] for r in t.sort_values("adj_net", ascending=False).itertuples()]
    body = ('<p class="lede">Opponent-adjusted scoring from a ridge regression on every completed game this season. '
            'FCS opponents are pooled as one team.</p>' + table(s_cols, s_rows, footer=ctx.footer))
    (site_dir / "stats.qmd").write_text(page("Stats", raw_html(body)))

    # ---------------------------------------------------------------- teams index + pages
    blocks = []
    for c in order:
        ct = t[t["conference"] == c].sort_values("team")
        items = "".join(f'<li>{ctx.team_cell(r.team, mascot=False)}</li>' for r in ct.itertuples())
        blocks.append(f'<div class="conf-block"><h3>{E(c)}</h3><ul class="team-list">{items}</ul></div>')
    (site_dir / "teams.qmd").write_text(page("Teams", raw_html(f'<div class="conf-grid">{"".join(blocks)}</div>')))

    for r in t.itertuples():
        (site_dir / "teams" / f"{r.slug}.qmd").write_text(team_page(ctx, r, elo_hist))

    # ---------------------------------------------------------------- downloads list
    links = "".join(f'<li><a href="downloads/{E(f)}">{E(f)}</a></li>' for f in downloads)
    (site_dir / "methodology.qmd").write_text(methodology(ctx, links))


def projected_field(ctx):
    cfg = ctx.cfg
    t = ctx.t
    po = cfg["playoff"]
    field = int(po["field_size"])
    auto = []
    for c in cfg["conferences"]["power"]:
        ct = t[t["conference"] == c]
        if len(ct) and po["power_auto_bids"]:
            auto.append(ct.sort_values("p_conf_champ", ascending=False).iloc[0]["team"])
    g_ch = []
    for c in cfg["conferences"]["group_of"]:
        ct = t[t["conference"] == c]
        if len(ct):
            g_ch.append(ct.sort_values("p_conf_champ", ascending=False).iloc[0])
    if g_ch and po["group_of_auto_bids"]:
        g_ch = sorted(g_ch, key=lambda r: -r["p_cfp"])
        auto += [r["team"] for r in g_ch[: int(po["group_of_auto_bids"])]]
    chosen = list(dict.fromkeys(auto))
    for tm in t.sort_values("p_cfp", ascending=False)["team"]:
        if len(chosen) >= field:
            break
        if tm not in chosen:
            chosen.append(tm)
    chosen.sort(key=lambda tm: (ctx.by_team.loc[tm, "avg_seed"] if not np.isnan(ctx.by_team.loc[tm, "avg_seed"]) else 99))
    return list(enumerate(chosen, start=1))


def bracket_html(ctx, field):
    seed = {s: tm for s, tm in field}

    def slot(s):
        tm = seed.get(s)
        if not tm:
            return '<div class="slot">TBD</div>'
        return f'<div class="slot"><span class="seed">{s}</span>{ctx.team_cell(tm, mascot=False)}</div>'

    pods = [(1, 8, 9), (4, 5, 12), (3, 6, 11), (2, 7, 10)]
    out = ['<div class="bracket">']
    for i, (top, a, b) in enumerate(pods):
        out.append(
            f'<div class="pod"><div class="pod-label">Quarterfinal {i + 1}</div>{slot(top)}'
            f'<div class="r1"><div class="r1-label">First round at #{a}</div>{slot(a)}{slot(b)}</div></div>'
        )
    out.append('</div><p class="sub bracket-note">Semifinals: Quarterfinal 1 vs 2, Quarterfinal 3 vs 4. No reseeding.</p>')
    return "".join(out)


def sparkline(values, w=360, h=70):
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * (w - 8) / (len(values) - 1) + 4:.1f},{h - 6 - (v - lo) / rng * (h - 12):.1f}"
                   for i, v in enumerate(values))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" role="img" aria-label="Elo this season, {lo:.0f} to {hi:.0f}">'
            f'<polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linejoin="round"/></svg>')


def team_page(ctx, r, elo_hist):
    p = "../"
    cfg = ctx.cfg
    g = ctx.games[(ctx.games["season"] == ctx.season) & ((ctx.games["home"] == r.team) | (ctx.games["away"] == r.team))]
    g = g.sort_values("date")
    rows = []
    for x in g.itertuples(index=False):
        home = x.home == r.team
        opp = x.away if home else x.home
        where = "vs" if x.neutral or home else "@"
        wp = x.home_wp if home else 1 - x.home_wp
        date = x.date.tz_convert(cfg["site"]["timezone"]).strftime("%a %-m/%-d") if pd.notna(x.date) else "TBD"
        if x.completed:
            pf, pa = (x.home_pts, x.away_pts) if home else (x.away_pts, x.home_pts)
            res = f'<span class="{"win" if pf > pa else "loss"}">{"W" if pf > pa else "L"} {int(pf)}-{int(pa)}</span>'
            pre = x.home_elo_pre if home else x.away_elo_pre
            post = x.home_elo_post if home else x.away_elo_post
            delta = (post - pre) if not np.isnan(post) else np.nan
            dtxt = f'<span class="{"up" if delta >= 0 else "down"}">{delta:+.0f}</span>' if not np.isnan(delta) else "-"
        else:
            res, dtxt = '<span class="sub">-</span>', "-"
        tag = " <span class=\"sub\">title game</span>" if x.is_ccg else (" <span class=\"sub\">bowl/CFP</span>" if x.season_type == "postseason" else "")
        rows.append([date, f'{where} {ctx.team_cell(opp, p, mascot=False, rank=True)}{tag}', ctx.bar(wp), res, dtxt])
    sched = table([("Date", "c", None), ("Opponent", "l", None), ("Win chance", "l", "Pregame for played games, current for upcoming"),
                   ("Result", "c", None), ("Elo +/-", "r", None)], rows)

    h = elo_hist[(elo_hist["team"] == r.team) & (elo_hist["season"] == ctx.season)]
    vals = [r.elo_preseason] + list(h["elo_post"])
    spark = sparkline(vals)

    dist = [(w, getattr(r, f"p_{w}_reg_wins")) for w in range(0, 16) if getattr(r, f"p_{w}_reg_wins") > 0.0005]
    mx = max([v for _, v in dist] or [1])
    bars = "".join(f'<div class="wbar"><span class="wlab">{w}</span><span class="wfill" style="width:{v / mx * 100:.0f}%"></span>'
                   f'<span class="wval">{pct(v)}</span></div>' for w, v in dist)

    tiles = [("National rank", f"#{int(r.rank)}"), ("OVR", num(r.ovr)), ("Elo", f"{r.elo:.0f}"),
             ("Record", rec(r.wins, r.losses)), ("Conf", rec(r.conf_wins, r.conf_losses)),
             ("CFP chance", pct(r.p_cfp)), ("Conf title", pct(r.p_conf_champ)), ("Natl title", pct(r.p_title))]
    tiles_html = "".join(f'<div class="tile"><div class="tv">{v}</div><div class="tl">{E(k)}</div></div>' for k, v in tiles)
    adj = (f'<p>Adjusted offense {num(r.adj_off)} (#{num(r.adj_off_rank, 0)}), adjusted defense {num(r.adj_def)} '
           f'(#{num(r.adj_def_rank, 0)}), net {num(r.adj_net)} (#{num(r.adj_net_rank, 0)}). '
           f'Strength of schedule so far {num(r.sos, 0)} (#{num(r.sos_rank, 0)}), remaining {num(r.sos_remaining, 0)}.</p>')

    body = f"""
<div class="team-head" style="--team:{E(r.color)}">
  {ctx.logo(r.team).replace('class="logo', 'class="logo big')}
  <div><div class="team-title">{E(r.team)}</div><div class="sub">{E(r.mascot)} | {E(r.conference)}</div></div>
</div>
<div class="tiles">{tiles_html}</div>
<div class="two-up">
  <div><h3>Elo this season</h3><div class="spark-wrap" style="color:{E(r.color)}">{spark}</div>
  <p class="sub">Preseason {r.elo_preseason:.0f}, now {r.elo:.0f}</p></div>
  <div><h3>Projected regular season wins</h3><div class="wdist">{bars}</div>
  <p class="sub">Average {num(r.proj_wins)}-{num(r.proj_losses)}, {pct(r.p_unbeaten)} to finish unbeaten</p></div>
</div>
<h3>Schedule</h3>
{sched}
<h3>Scoring</h3>
{adj}
"""
    return page("", raw_html(body), f'pagetitle: "{r.team} | {cfg["site"]["title"]}"\n')


def methodology(ctx, links):
    e = ctx.cfg["elo"]
    po = ctx.cfg["playoff"]
    sim = ctx.cfg["simulation"]
    hist = ctx.cfg["season"]["history_start"]
    acc = getattr(ctx, "accuracy", None) or {}
    p = acc.get("params", {k: e[k] for k in ("k", "home_field", "offseason_regression", "talent_weight", "mov_cap")})
    epp = p.get("elo_per_point", e["elo_per_point"])
    bt, lv = acc.get("backtest", {}), acc.get("live", {})

    def row(label, m):
        if not m or not m.get("games"):
            return f"| {label} | 0 | - | - | - |"
        return (f"| {label} | {m['games']:,} | {m['accuracy'] * 100:.1f}% | {m['log_loss']:.3f} | {m['brier']:.3f} |")

    tuned = ("These settings were chosen automatically: the model replayed every season from "
             f"{acc.get('seasons', '-')} under many combinations and kept the one whose pregame win chances best matched "
             "what actually happened (lowest log loss)." if ctx.cfg["tuning"]["enabled"] else
             "These settings come straight from config.yaml (tuning is turned off).")
    cap = "no cap" if float(p["mov_cap"]) >= 99 else f"capped at {float(p['mov_cap']):.0f} points"
    return page("Methodology", f"""
## Data

Scores, schedules, and roster talent come from the CollegeFootballData.com API. Ratings are built from every FBS
game since {hist}.

## Elo ratings

Every FBS team has an Elo rating centered on {e['mean']}. After each game the winner takes points from the loser.
How many points depends on how surprising the result was and on the margin of victory ({cap}), with the margin
effect reduced when a heavy favorite wins big.

| Setting | Value |
|---|---|
| Rating change speed (K) | {float(p['k']):g} |
| Home field | {float(p['home_field']):g} Elo points ({float(p['home_field']) / epp:.1f} points on the scoreboard) |
| Offseason pull toward conference average | {float(p['offseason_regression']) * 100:.0f}% |
| Weight on roster talent in preseason rating | {float(p['talent_weight']) * 100:.0f}% |
| Elo points per point of margin | {epp:g} |

{tuned}

**Preseason ratings.** Between seasons each team moves part of the way toward the average of its own conference
(using the conference it plays in that season), not toward one league-wide average. That keeps a dominant team in a
weaker conference from starting the year rated like a playoff team. The result is then blended with the 247 Team
Talent Composite, scaled to the spread of the ratings.

FCS opponents use a fixed rating of {e['fcs_rating']}. Programs new to FBS start at {e['new_team_rating']}.

**OVR** is the chance a team would beat an average FBS team on a neutral field, times 100.

## How accurate is it

FBS vs FBS games only, every pick made before kickoff with the ratings at that moment.

| Games | Count | Picked winner | Log loss | Brier |
|---|---|---|---|---|
{row(f"Past seasons ({acc.get('seasons', '-')})", bt)}
{row(f"This season ({ctx.season})", lv)}

Lower log loss and Brier scores are better. A coin flip scores 0.693 and 0.250.

## Adjusted offense and defense

A ridge regression on every completed game this season estimates how many points each team would score and allow
against an average opponent, with a home field term. Shrinkage (lambda {ctx.cfg['adjusted_scoring']['ridge_lambda']})
keeps early season numbers from swinging on one or two games.

## Season simulations

The rest of the season is played out {sim['runs']:,} times. Each game is decided by a random draw against the Elo win
chance, and ratings update inside each simulated season, so a team that gets hot keeps winning more often.

Conference title games match the top two teams by conference win percentage (ties broken by rating). Once a title game
is set or played, the real matchup or result is used.

## Playoff selection

The {po['field_size']}-team field is the four power conference champions, the highest ranked champion from the other
conferences, and the next highest ranked teams. Notre Dame gets in if ranked inside the top {po['field_size']}.
Seeds follow the ranking and the top {po['byes']} get byes. Seeds 5 through 8 host first round games.

The committee is approximated in each simulation by a ranking score: simulated Elo, minus {po['loss_penalty']} per loss,
plus {po['champion_bonus']} for a conference title.

## GameScore

GameScore ranks the week's games by quality (average OVR), closeness (how near the win chance is to 50%), and stakes
(how much the result moves both teams' playoff odds in the simulations).

## Downloads

Full model output for anyone who wants to dig in:

```{{=html}}
<ul>{links}</ul>
```
""")
