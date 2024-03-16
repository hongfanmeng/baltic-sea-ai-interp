from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

base_dir = Path(__file__).resolve().parent.parent.parent
data_dir = base_dir / "data"


def download_data(year: int) -> None:
    date_begin = date(year, 1, 1).strftime("%Y-%m-%d")
    date_end = date(year, 12, 31).strftime("%Y-%m-%d")

    url = (
        "http://nest.su.se//dataPortal/getStations?"
        + "latBegin=53.6667&latEnd=66.0000&lonBegin=9.0000&lonEnd=30.3322"
        + f"&dateBegin={date_begin}&dateEnd={date_end}&timeWindow=365"
        + "&noRestriction&removeDuplicates=1,1"
        + "&database=1&database=2&database=3&database=4&database=5&database=6&database=7"
    )

    r = requests.get(url)

    with open(data_dir / f"nest/nest_{year}.csv", "w") as f:
        f.write(r.text)


def download_all_data() -> None:
    for year in tqdm(range(1900, 2024)):
        download_data(year)


def merge_data() -> pd.DataFrame:
    results = []
    for year in tqdm(range(1900, 2024)):
        df = pd.read_csv(
            data_dir / f"nest/nest_{year}.csv",
            dtype={"SHIP": str},
        )
        results.append(df)
    return pd.concat(results)


if __name__ == "__main__":
    download_all_data()

    df = merge_data()
    df.to_parquet(data_dir / "nest_baltic_sea.parquet")
