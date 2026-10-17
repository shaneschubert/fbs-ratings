# FBS Ratings site

An FBS college football version of texastng.com: Elo ratings, games of the week, full ratings table,
schedule and scores with pick record, conference standings with projections, a 12-team CFP projection,
opponent-adjusted stats, and a page for every team.

GitHub rebuilds and republishes the site automatically every Sunday and Wednesday morning. Nothing has to run on your computer.

## What's in here

| Path | What it does |
|---|---|
| `config.yaml` | Site name, model settings, conference lists, playoff rules. The only file you should need to edit. |
| `build.py` | Pulls data, runs the model, writes the site pages. |
| `pipeline/cfbd.py` | CollegeFootballData.com API client with caching. |
| `pipeline/model.py` | Elo ratings, predictions, team table, adjusted offense/defense. |
| `pipeline/simulate.py` | 10,000-run season simulation, conference title games, CFP selection and bracket. |
| `pipeline/render.py` | Builds every page (home, ratings, schedule, standings, playoff, stats, teams, methodology). |
| `pipeline/demo.py` | Fictional league for previewing without an API key. |
| `site/assets/` | Stylesheet and the script for sorting, filtering, and the week picker. |
| `.github/workflows/update.yml` | The scheduled job that builds and publishes the site. |

## Setup (one time, about 15 minutes)

1. **Get a free API key** at https://collegefootballdata.com/key. It arrives by email.
2. **Create a GitHub repository** (public is required for free GitHub Pages). Upload every file and folder from this kit.
   The `.github` folder is hidden on Mac and sometimes skipped by drag and drop. If it does not show up in the repo,
   use **Add file > Create new file**, name it `.github/workflows/update.yml`, and paste in the contents of that file.
3. **Add the key as a secret:** repo **Settings > Secrets and variables > Actions > New repository secret**.
   Name `CFBD_API_KEY`, value is your key.
4. **Turn on Pages:** **Settings > Pages > Build and deployment > Source: GitHub Actions**.
5. **Run it:** **Actions > Update site > Run workflow**. The first run takes a few minutes because it pulls every season
   back to 2014. When it finishes, the site address appears on the run summary (usually `https://<username>.github.io/<repo>/`).

After that it updates itself. Any commit to `main` (for example a config change) also triggers a rebuild.

## Preview on your own computer (optional)

Requires Python 3.10+ and Quarto (https://quarto.org/docs/get-started/).

```
pip install -r requirements.txt
python build.py --demo        # fictional league, no key needed
quarto preview site
```

For real data, set the key first (`set CFBD_API_KEY=yourkey` on Windows, `export CFBD_API_KEY=yourkey` on Mac) and run `python build.py`.

## Things you will likely want to change

- `site.title`, `site.tagline`, `site.handle` in `config.yaml`.
- `conferences`: names must match CFBD exactly. If a conference is missing, the build prints a warning naming it.
- `playoff.loss_penalty` and `playoff.champion_bonus`: how the simulated committee ranks teams.
- `tuning.enabled`: when on (default), the build replays past seasons and picks the Elo settings that predicted
  them best, so the `elo` values in the config are only starting points. Turn it off to use your own values exactly.

## Data and downloads

Every run publishes CSVs on the Methodology page: model accuracy by season, preseason talent scores, ratings with all projection columns, every game with pregame
Elo and win chance, full Elo history since 2014, adjusted scoring, this week's playoff leverage by game,
and a team index (school name, slug, conference, colors, logo URL) as the record of every name mapping.

## Known limits

- Conference title game qualifiers use conference win percentage with rating as the tiebreaker, not each conference's
  full tiebreaker rules. Once a title game is set, the real matchup is used.
- The CFP field is simulated from a committee proxy, not the actual rankings. After Selection Sunday the playoff page
  still shows the model's field until actual-bracket support is added.
- Bowl and CFP results update Elo but are not simulated in advance.
- The CFBD free tier allows 1,000 calls a month. A normal run uses 3 calls after the first.
- GitHub pauses scheduled jobs in a repo with no commits for 60 days. If updates stop in the offseason, use Run workflow or make any commit.
