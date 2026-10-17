"""Fictional league in the same shape as CFBD API responses.

Run `python build.py --demo` to preview the full site without an API key.
Team names are invented; none of the numbers are real.
"""
import datetime as dt

import numpy as np

CONF_SIZES = {
    "ACC": 17, "Big Ten": 18, "Big 12": 16, "SEC": 16, "American Athletic": 13,
    "Conference USA": 10, "Mid-American": 12, "Mountain West": 10, "Pac-12": 8, "Sun Belt": 14,
    "FBS Independents": 2,
}
PRE = ["Ash", "Briar", "Cedar", "Dun", "Elm", "Fox", "Glen", "Hart", "Iron", "Kings", "Lake", "Mill",
       "North", "Oak", "Pine", "Quarry", "Red", "Stone", "Thorn", "Vale", "West", "Wolf"]
SUF = ["ford", "field", "haven", "port", "ridge", "ton", "wood", "brook", "dale", "mont", "view", "crest"]
KIND = ["State", "Tech", "University", "A&M", "College", "Poly"]
MASCOTS = ["Hawks", "Bison", "Rams", "Cougars", "Knights", "Owls", "Wolves", "Mustangs", "Bears", "Falcons",
           "Raiders", "Tigers", "Pilots", "Comets", "Foxes", "Gators"]
COLORS = ["8a1c2b", "1d3f8f", "0f6b3a", "c25a00", "5b2a86", "222222", "b8860b", "006d77", "9d0208", "33415c"]


def generate(cfg, season):
    rng = np.random.default_rng(7)
    teams, names = [], set()
    tid = 1
    for conf, size in CONF_SIZES.items():
        for _ in range(size):
            while True:
                nm = f"{rng.choice(PRE)}{rng.choice(SUF)} {rng.choice(KIND)}"
                if nm not in names:
                    names.add(nm)
                    break
            teams.append({
                "id": tid, "school": nm, "mascot": str(rng.choice(MASCOTS)), "abbreviation": nm[:3].upper(),
                "conference": conf, "classification": "fbs", "color": "#" + str(rng.choice(COLORS)),
                "alternateColor": "#ffffff", "logos": [], "division": "",
            })
            tid += 1
    strength = {t["school"]: rng.normal(0, 1) * (1.3 if t["conference"] in cfg["conferences"]["power"] else 0.9)
                + (0.6 if t["conference"] in cfg["conferences"]["power"] else -0.3) for t in teams}
    fcs = [f"FCS Opponent {i}" for i in range(1, 31)]
    for f in fcs:
        strength[f] = rng.normal(-1.8, 0.5)
    by_conf = {}
    for t in teams:
        by_conf.setdefault(t["conference"], []).append(t["school"])

    games, gid = [], 1
    hist = cfg["season"]["history_start"]
    for year in range(max(hist, season - 4), season + 1):
        drift = {k: v + rng.normal(0, 0.35) for k, v in strength.items()}
        start = dt.datetime(year, 8, 29, 19, 0)
        current = year == season
        done_through = 7 if current else 99
        for week in range(1, 14):
            day = start + dt.timedelta(days=7 * (week - 1))
            pairs = []
            if week <= 4:
                pool = [t["school"] for t in teams]
                rng.shuffle(pool)
                while len(pool) >= 2:
                    a = pool.pop()
                    if rng.random() < 0.18:
                        pairs.append((a, str(rng.choice(fcs)), False))
                    else:
                        pairs.append((a, pool.pop(), False))
            else:
                for conf, mem in by_conf.items():
                    if conf == "FBS Independents":
                        continue
                    m = list(mem)
                    rng.shuffle(m)
                    while len(m) >= 2:
                        pairs.append((m.pop(), m.pop(), True))
            for a, b, cg in pairs:
                completed = week <= done_through
                home, away = (a, b) if rng.random() < 0.5 or b.startswith("FCS") else (b, a)
                hp = ap = None
                if completed:
                    margin = (drift[home] - drift[away]) * 10 + 2.5 + rng.normal(0, 13)
                    total = max(20, rng.normal(52, 11))
                    hp = int(max(0, round(total / 2 + margin / 2)))
                    ap = int(max(0, round(total / 2 - margin / 2)))
                    if hp == ap:
                        hp += 3
                games.append({
                    "id": gid, "season": year, "week": week, "seasonType": "regular",
                    "startDate": day.isoformat() + "Z", "completed": completed, "neutralSite": False,
                    "conferenceGame": cg, "homeTeam": home, "awayTeam": away,
                    "homeClassification": "fcs" if home.startswith("FCS") else "fbs",
                    "awayClassification": "fcs" if away.startswith("FCS") else "fbs",
                    "homeConference": "", "awayConference": "", "homePoints": hp, "awayPoints": ap,
                    "venue": "", "notes": "",
                })
                gid += 1
        strength = drift
    return teams, games
