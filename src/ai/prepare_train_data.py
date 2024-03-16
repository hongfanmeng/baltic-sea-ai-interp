import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

base_dir = Path(__file__).resolve().parent.parent.parent

# depth grid
# 0 - 100m, every 5m, 100 - 250 m, every 10m
dep_lb = np.concatenate((np.arange(0, 100, 5), np.arange(100, 250, 10)))
dep_ub = np.concatenate((np.arange(0, 100, 5) + 5, np.arange(100, 250, 10) + 10))


if __name__ == "__main__":
    df = pd.read_parquet(base_dir / "data/nest_baltic_sea.parquet")

    # filter qc flag
    df = df[df["QTOTOXY"] == 1]
    df = df[df["QSALIN"] == 1]
    df = df[df["QTEMP"] == 1]

    # convert date to year mon
    df["OBSDATE"] = pd.to_datetime(df["OBSDATE"])
    df["YEAR"] = df["OBSDATE"].dt.year
    df["MON"] = df["OBSDATE"].dt.month

    # round lat, lon to 2 decimal
    df["LATITUDE"] = df["LATITUDE"].round(2)
    df["LONGITUDE"] = df["LONGITUDE"].round(2)

    # rename columns
    df = df[["YEAR", "MON", "LATITUDE", "LONGITUDE", "OBSDEP", "TOTOXY", "TEMP", "SALIN"]]
    df.columns = ["year", "mon", "lat", "lon", "dep", "oxy", "tmp", "sal"]

    # calc mean, std for normalization
    mean = df.mean(numeric_only=True)[["oxy", "tmp", "sal"]]
    std = df.std(numeric_only=True)[["oxy", "tmp", "sal"]]

    # grid dep
    for lb, ub in tqdm(zip(dep_lb, dep_ub)):
        mid = (lb + ub) / 2
        df.loc[(lb <= df["dep"]) & (df["dep"] < ub), "dep"] = mid

    df.drop(df[df["dep"] > dep_ub[-1]].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    groups = df.groupby(["year", "mon", "lat", "lon", "dep"]).mean()
    groups = groups.reset_index()

    # save mean, std, data
    output_data = (mean, std, groups)
    with open(base_dir / "data/train_data.pkl", "wb") as f:
        pickle.dump(output_data, f)
