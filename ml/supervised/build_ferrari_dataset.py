import time
import requests
import pandas as pd
import numpy as np


URL = "https://api.jolpi.ca/ergast/f1/constructors/ferrari/results/"
HEADERS = {"User-Agent": "FerrariMLDataset/1.0 student-project"}

rows = []
limit = 100
offset = 0

print("Загруpзка данных Ferrari...")


while True:
    response = requests.get(
        URL,
        params={"limit": limit, "offset": offset},
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()["MRData"]
    total = int(data["total"])
    races = data["RaceTable"]["Races"]

    if not races:
        break

    for race in races:
        season = int(race["season"])

        if season > 2025:
            continue

        for result in race["Results"]:
            grid = int(result["grid"])

            if grid == 0:
                grid = np.nan

            status = result.get("status", "")

            dnf = int(
                status != "Finished"
                and not status.startswith("+")
            )

            rows.append({
                "season": season,
                "round": int(race["round"]),
                "date": race["date"],
                "circuit": race["Circuit"]["circuitId"],
                "country": race["Circuit"]["Location"]["country"],
                "grid": grid,
                "finish": int(result["position"]),
                "dnf": dnf
            })

    offset += limit

    print(f"Загружено {min(offset, total)} / {total}")

    if offset >= total:
        break

    time.sleep(0.2)


drivers = pd.DataFrame(rows)

races = (
    drivers
    .groupby(["season", "round"], as_index=False)
    .agg(
        date=("date", "first"),
        circuit=("circuit", "first"),
        country=("country", "first"),

        best_grid=("grid", "min"),
        avg_grid=("grid", "mean"),

        best_finish=("finish", "min"),
        race_dnfs=("dnf", "sum")
    )
)


races["ferrari_podium"] = (races["best_finish"] <= 3).astype(int)


races["date"] = pd.to_datetime(races["date"])

races = (races.sort_values(["date", "round"]) .reset_index(drop=True))


season_group = races.groupby("season")


races["prev_best_finish"] = (
    season_group["best_finish"]
    .shift(1)
)


races["avg_finish_last_3"] = (
    season_group["best_finish"]
    .transform(
        lambda x:
        x.shift(1).rolling(3, min_periods=1).mean()
    )
)


races["avg_grid_last_3"] = (
    season_group["best_grid"]
    .transform(
        lambda x:
        x.shift(1).rolling(3, min_periods=1).mean()
    )
)


races["podiums_last_5"] = (
    season_group["ferrari_podium"]
    .transform(
        lambda x:
        x.shift(1).rolling(5, min_periods=1).sum()
    )
)


races["dnfs_last_5"] = (
    season_group["race_dnfs"]
    .transform(
        lambda x:
        x.shift(1).rolling(5, min_periods=1).sum()
    )
)


races["circuit_podium_rate_prev"] = (
    races
    .groupby("circuit")["ferrari_podium"]
    .transform(
        lambda x:
        x.shift(1).expanding().mean()
    )
)


dataset = races[
    [
        "season",
        "round",

        "circuit",
        "country",

        "best_grid",
        "avg_grid",

        "prev_best_finish",
        "avg_finish_last_3",
        "avg_grid_last_3",

        "podiums_last_5",
        "dnfs_last_5",

        "circuit_podium_rate_prev",

        "ferrari_podium"
    ]
].copy()


dataset.to_csv(
    "ferrari_f1_dataset.csv",
    index=False
)


print()
print("DATASET ГОТОВ")
print()
print("Файл сохранён: ferrari_f1_dataset.csv")