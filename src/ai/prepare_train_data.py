from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

base_dir = Path(__file__).resolve().parent.parent.parent

# depth grid
# 0 - 100m, every 5m, 100 - 200 m, every 10m
dep_lb = np.concatenate((np.arange(0, 100, 5), np.arange(100, 200, 10)))
dep_ub = np.concatenate((np.arange(0, 100, 5) + 5, np.arange(100, 200, 10) + 10))
dep_mid = (dep_lb + dep_ub) / 2

df_depth = pd.read_parquet(base_dir / "data/baltic_sea_depth.parquet")
df_depth = df_depth.set_index(["lon", "lat"])


def prepare_train_data():
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
    df = df.reset_index(drop=True)
    df = df.drop(df[df["dep"] < dep_lb[0]].index).reset_index(drop=True)
    df = df.drop(df[df["dep"] >= dep_ub[-1]].index).reset_index(drop=True)
    for lb, ub in tqdm(zip(dep_lb, dep_ub)):
        mid = (lb + ub) / 2
        df.loc[(lb <= df["dep"]) & (df["dep"] < ub), "dep"] = mid
    df.reset_index(drop=True, inplace=True)

    df = df.groupby(["year", "mon", "lat", "lon", "dep"]).mean().reset_index()

    return mean, std, df


def convert_vae_data(data_entry: tuple[tuple, pd.DataFrame]):
    meta, data = data_entry
    year, mon, lat, lon = meta
    max_dep = df_depth.loc[(lon, lat), "dep"] if (lon, lat) in df_depth.index else 0

    data_vae = pd.DataFrame(columns=["dep", "oxy", "tmp", "sal"])
    data_vae["dep"] = dep_mid

    data_vae = data_vae.set_index("dep")
    data_vae.loc[data["dep"], ["oxy", "tmp", "sal"]] = data[["oxy", "tmp", "sal"]].values
    data_vae.reset_index(inplace=True)

    meta = pd.Series({"year": year, "mon": mon, "lat": lat, "lon": lon, "max_dep": max_dep})

    return meta, data_vae


def prepare_vae_data(train_data: pd.DataFrame):
    data_list = train_data.groupby(["year", "mon", "lat", "lon"])
    vae_train_data = process_map(convert_vae_data, data_list, max_workers=12, chunksize=10000)
    return vae_train_data


if __name__ == "__main__":
    mean, std, train_data = prepare_train_data()
    vae_train_data = prepare_vae_data(train_data)

    pd.to_pickle(mean, base_dir / "data/mean.pkl")
    train_data.to_parquet(base_dir / "data/train_data.parquet")

    pd.to_pickle(vae_train_data, base_dir / "data/vae_train_data.parquet")
