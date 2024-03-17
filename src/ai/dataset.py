import pickle
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map

base_dir = Path(__file__).resolve().parent.parent.parent


def read_depth():
    df = pd.read_parquet(base_dir / "data/baltic_sea_depth.parquet")
    df = df.set_index(["lon", "lat"])
    return df


def read_mean_std():
    with open(base_dir / "data/train_mean_std.pkl", "rb") as f:
        mean, std = pickle.load(f)
    return mean, std


class BalticSeaDataset(Dataset):
    def __init__(self):
        self.data_list = pd.read_parquet(base_dir / "data/train_data.parquet")
        self.data_list = list(self.data_list.groupby(["year", "mon", "lat", "lon"]))

        self.df_dep = read_depth()
        self.mean, self.std = read_mean_std()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> tuple[pd.Series, pd.DataFrame]:
        meta, data = self.data_list[idx]
        year, mon, lat, lon = meta
        max_dep = self.df_dep.loc[(lon, lat), "dep"] if (lon, lat) in self.df_dep.index else 0

        meta = pd.Series({"year": year, "mon": mon, "lat": lat, "lon": lon, "max_dep": max_dep})

        return meta, data.reset_index(drop=True)


class VaeDataset(Dataset):

    def __init__(self):
        print("reading data...")
        self.df_dep = read_depth()
        self.mean, self.std = read_mean_std()
        vae_train_data = pd.read_pickle("data/vae_train_data.pkl")

        print("filtering data for training...")
        # use data with max_dep >= 50 to train
        data_list = [(meta, data) for (meta, data) in vae_train_data if meta["max_dep"] >= 50]
        fill_rate = process_map(self.get_fill_rate, data_list, max_workers=16, chunksize=1000)

        # use data with fill rate >= 1 to train
        self.data_list = [data for data, rate in zip(data_list, fill_rate) if rate >= 1]

    @staticmethod
    def get_fill_rate(data: tuple[pd.Series, pd.DataFrame]):
        meta, data = data
        total = sum(data["dep"] <= meta["max_dep"])
        return len(data[data["dep"] <= meta["max_dep"]].dropna()) / total if total > 0 else 0

    def preprocess(self, group: tuple[tuple, pd.DataFrame]):
        meta, data = group
        year, mon, lat, lon = meta
        max_dep = self.df_dep.loc[(lon, lat), "dep"] if (lon, lat) in self.df_dep.index else 0
        meta = pd.Series({"year": year, "mon": mon, "lat": lat, "lon": lon, "max_dep": max_dep})

        return meta, data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> tuple[pd.Series, pd.DataFrame]:
        meta, data = self.data_list[idx]
        data = data.copy()
        data[["oxy", "tmp", "sal"]] = (data[["oxy", "tmp", "sal"]] - self.mean) / self.std
        return meta, data.reset_index(drop=True)
