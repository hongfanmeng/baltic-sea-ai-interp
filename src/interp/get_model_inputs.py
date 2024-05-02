from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

base_dir = Path(__file__).resolve().parent.parent.parent

YEAR_START = 1960
YEAR_END = 2023
NEIGHBOR_SIZE = 20
YEAR_STEPS = 3


@cache
def read_vae_data():
    df = pd.read_parquet(base_dir / "data/vae_data.parquet")
    df["lon"] = df["lon"].round(2)
    df["lat"] = df["lat"].round(2)
    df = df.groupby(["year", "lat", "lon", "dep"]).mean().reset_index()
    df = df.drop("mon", axis=1)
    df["year"] = df["year"].astype(int)
    df.set_index(["year", "lat", "lon"], inplace=True)

    return df


@cache
def read_mean_std():
    mean, std = pd.read_pickle(base_dir / "data/train_mean_std.pkl")
    return mean, std


@cache
def get_data_list():
    print("Reading profile data...", end=" ")
    vae_data = read_vae_data().reset_index()

    # normalize data
    mean, std = read_mean_std()
    vae_data[["oxy", "tmp", "sal"]] = (vae_data[["oxy", "tmp", "sal"]] - mean) / std

    data_list = list(vae_data.groupby(["year", "lat", "lon"]))
    print("done")
    return data_list


def get_fill_rate(data: tuple[dict, pd.DataFrame]) -> float:
    meta, df = data
    max_dep = df["max_dep"].iloc[0]
    total = sum(df["dep"] <= max_dep)
    return len(df[df["dep"] <= max_dep].dropna()) / total if total > 0 else 0


@cache
def get_fill_rate_list():

    data_list = get_data_list()
    print("Calculating fill rate for each profile...")
    fill_rate_list = process_map(get_fill_rate, data_list, max_workers=12, chunksize=5000)
    return fill_rate_list


@cache
def filter_data_list(fill_rate: float | None = None):
    vae_data_list = get_data_list()
    fill_rate_list = get_fill_rate_list()
    data_list = [data for data, rate in zip(vae_data_list, fill_rate_list) if not fill_rate or rate >= fill_rate]
    return data_list


@cache
def get_raw_input():
    data_list = filter_data_list(fill_rate=0.3)
    df = pd.concat([df for meta, df in data_list])
    df = df[df["max_dep"] >= 10]
    df = df[df["year"].between(YEAR_START - YEAR_STEPS + 1, YEAR_END)]
    df = df.set_index(["year", "lat", "lon"])
    return df


@cache
def build_kd_tree():
    df_raw = get_raw_input()

    print("Building kdtree for each year...")
    years = list(range(YEAR_START - YEAR_STEPS + 1, YEAR_END + 1))
    data_mapping = {"trees": {}, "coords": {}}
    for year in tqdm(years):
        # find all unique coordinates and build kdtree for each year
        df_year = df_raw.loc[year].reset_index()
        coords = df_year[["lat", "lon"]].drop_duplicates().values
        tree = KDTree(coords)
        data_mapping["trees"][year] = tree
        data_mapping["coords"][year] = coords

    return data_mapping


def get_model_input(lat: float, lon: float, max_dep: float):
    data_mapping = build_kd_tree()
    df_raw = get_raw_input()

    years = list(range(YEAR_START - YEAR_STEPS + 1, YEAR_END + 1))

    x = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = np.sin(np.deg2rad(lat))

    neighbors_input = np.zeros((len(years), NEIGHBOR_SIZE, 2 + 30))

    for y_idx, year in enumerate(years):
        tree: KDTree = data_mapping["trees"][year]
        coords = data_mapping["coords"][year]
        neighbors = tree.query([[lat, lon]], k=NEIGHBOR_SIZE, return_distance=False)
        neighbors_coords = coords[neighbors[0]]

        for n_idx, (lat_n, lon_n) in enumerate(neighbors_coords):
            neighbor_data: pd.DataFrame = df_raw.loc[(year, lat_n, lon_n)]
            neighbor_data = neighbor_data.copy()

            # interpolate for VAE input
            neighbor_data["oxy"] = neighbor_data["oxy"].interpolate(method="linear", limit_direction="both")

            # set depth > max_dep to 0
            dep = neighbor_data["max_dep"].iloc[0]
            neighbor_data[neighbor_data["dep"] > dep] = 0

            dlon = lon_n - lon
            dlat = lat_n - lat

            neighbors_input[y_idx, n_idx, :2] = [dlon, dlat]
            neighbors_input[y_idx, n_idx, 2:] = neighbor_data["oxy"].values

    model_input = []
    for y_idx in range(len(years) - YEAR_STEPS + 1):
        model_input.append(neighbors_input[y_idx : y_idx + YEAR_STEPS])

    years = years[YEAR_STEPS - 1 :]

    label = {}
    label["years"] = years
    label["max_dep"] = [max_dep] * len(years)
    label["lon"] = [lon] * len(years)
    label["lat"] = [lat] * len(years)
    label["x"] = [x] * len(years)
    label["y"] = [y] * len(years)
    label["z"] = [z] * len(years)

    return np.array(model_input), label


def get_model_input_wrapper(args):
    return get_model_input(*args)


if __name__ == "__main__":
    df = pd.read_parquet(base_dir / "data/baltic_sea_depth.parquet")

    df_dep = df.copy()
    df_dep = df_dep[df_dep["dep"] >= 10]
    df_dep["lon"] = df_dep["lon"].round(1)
    df_dep["lat"] = df_dep["lat"].round(1)
    df_dep = df_dep.groupby(["lat", "lon"]).mean().reset_index()

    coords = df_dep.values.tolist()

    # pre cache data
    build_kd_tree()

    model_inputs = process_map(get_model_input_wrapper, coords, max_workers=8, chunksize=10)

    pd.to_pickle(model_inputs, base_dir / "data/model_inputs.pkl")
