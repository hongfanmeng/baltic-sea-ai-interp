import pickle
from functools import cache
from multiprocessing import Manager
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

base_dir = Path(__file__).resolve().parent.parent.parent

manager = Manager()


def read_depth():
    df = pd.read_parquet(base_dir / "data/baltic_sea_depth.parquet")
    df = df.set_index(["lat", "lon"])
    return df


def read_mean_std():
    with open(base_dir / "data/train_mean_std.pkl", "rb") as f:
        mean, std = pickle.load(f)
    return mean, std


def read_vae_data():
    df = pd.read_parquet(base_dir / "data/vae_data.parquet")
    df["lon"] = df["lon"].round(2)
    df["lat"] = df["lat"].round(2)
    df = df.groupby(["year", "lat", "lon", "dep"]).mean().reset_index()
    df = df.drop("mon", axis=1)
    df["year"] = df["year"].astype(int)
    df.set_index(["year", "lat", "lon"], inplace=True)

    return df


class MLPDataset(Dataset):
    YEAR_START = 1960
    YEAR_END = 2020

    def __init__(self, split: Literal["train", "test"] = "train", test_size=0.2, neighbor_size=5):
        self.mean, self.std = read_mean_std()
        self.neighbor_size = neighbor_size
        self.data_mapping = self.get_data_mapping()

        # shared memory cache of dataset
        self.dataset_cache = manager.dict()

        _, mlp_train_data_out = self.get_mlp_train_data()
        train_data, test_data = train_test_split(mlp_train_data_out, test_size=test_size)
        self.data_list = train_data if split == "train" else test_data

    @staticmethod
    @cache
    def get_data_mapping():
        mlp_train_data_in, _ = MLPDataset.get_mlp_train_data()

        print("Building kdtree for each year...")
        years = list(range(MLPDataset.YEAR_START, MLPDataset.YEAR_END + 1))
        data_mapping = {"trees": {}, "coords": {}}
        for year in tqdm(years):
            # find all unique coordinates and build kdtree for each year
            df_year = mlp_train_data_in.loc[year].reset_index()
            coords = df_year[["lat", "lon"]].drop_duplicates().values
            tree = KDTree(coords)
            data_mapping["trees"][year] = tree
            data_mapping["coords"][year] = coords

        return data_mapping

    @staticmethod
    @cache
    def get_mlp_train_data():
        data_list = MLPDataset.filter_data_list(fill_rate=1)
        data_list = [(meta, df) for meta, df in data_list if df["max_dep"].iloc[0] >= 20]
        mlp_train_data_out = data_list

        data_list = MLPDataset.filter_data_list(fill_rate=0.3)
        data_list = [(meta, df) for meta, df in data_list if df["max_dep"].iloc[0] >= 20]
        df = pd.concat([df for meta, df in data_list])
        df = df.set_index(["year", "lat", "lon"])
        mlp_train_data_in = df

        return mlp_train_data_in, mlp_train_data_out

    @staticmethod
    @cache
    def get_data_list():
        print("Reading profile data...", end=" ")
        vae_data = read_vae_data().reset_index()
        vae_data = vae_data[vae_data["year"].between(MLPDataset.YEAR_START, MLPDataset.YEAR_END)]
        data_list = list(vae_data.groupby(["year", "lat", "lon"]))
        print("done")
        return data_list

    @staticmethod
    @cache
    def get_fill_rate_list():
        data_list = MLPDataset.get_data_list()
        print("Calculating fill rate for each profile...")
        fill_rate_list = process_map(MLPDataset.get_fill_rate, data_list, max_workers=12, chunksize=5000)
        return fill_rate_list

    @staticmethod
    @cache
    def filter_data_list(fill_rate: float | None = None):
        vae_data_list = MLPDataset.get_data_list()
        fill_rate_list = MLPDataset.get_fill_rate_list()

        data_list = [data for data, rate in zip(vae_data_list, fill_rate_list) if not fill_rate or rate >= fill_rate]

        return data_list

    @staticmethod
    def get_fill_rate(data: tuple[dict, pd.DataFrame]) -> float:
        meta, df = data
        max_dep = df["max_dep"].iloc[0]
        total = sum(df["dep"] <= max_dep)
        return len(df[df["dep"] <= max_dep].dropna()) / total if total > 0 else 0

    def get_neighbors(self, year: int, lat: float, lon: float, k: int):
        tree = self.data_mapping["trees"][year]
        coords = self.data_mapping["coords"][year]
        neighbors = tree.query([[lat, lon]], k=k + 1, return_distance=False)
        neighbors_coords = coords[neighbors[0]]
        return neighbors_coords[1:]

    def __len__(self):
        return len(self.data_list)

    @cache
    def __getitem__(self, idx) -> tuple[np.ndarray, dict]:
        # return cached data if available
        if idx in self.dataset_cache:
            return self.dataset_cache[idx]

        meta, data = self.data_list[idx]
        year, lat, lon = meta
        max_dep = data["max_dep"].iloc[0]
        label = {"year": year, "lat": lat, "lon": lon, "max_dep": max_dep}

        data: pd.DataFrame = data.copy()
        data[["oxy", "tmp", "sal"]] = (data[["oxy", "tmp", "sal"]] - self.mean) / self.std
        label["real"] = data["oxy"].values

        label["x"] = np.cos(lat) * np.cos(lon)
        label["y"] = np.cos(lat) * np.sin(lon)
        label["z"] = np.sin(lat)

        inputs = []
        mlp_train_data_in, _ = self.get_mlp_train_data()
        neighbors = self.get_neighbors(year, lat, lon, k=self.neighbor_size)
        for lat, lon in neighbors:
            neighbor_data: pd.DataFrame = mlp_train_data_in.loc[(year, lat, lon)]
            neighbor_data = neighbor_data.copy()

            # normalize
            neighbor_data[["oxy", "tmp", "sal"]] = (neighbor_data[["oxy", "tmp", "sal"]] - self.mean) / self.std

            # interpolate for VAE input
            neighbor_data = neighbor_data.interpolate(method="linear", limit_direction="both")

            # set depth > max_dep to 0
            max_dep = neighbor_data["max_dep"].iloc[0]
            neighbor_data[neighbor_data["dep"] > max_dep] = 0

            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)

            dx = x - label["x"]
            dy = y - label["y"]
            dz = z - label["z"]

            inputs.append([dx, dy, dz] + neighbor_data["oxy"].values.tolist())

        self.dataset_cache[idx] = (np.array(inputs), label)
        return np.array(inputs), label


class MLPDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        neighbor_size: int = 5,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.neighbor_size = neighbor_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MLPDataset(split="train", neighbor_size=self.neighbor_size)
        self.val_dataset = MLPDataset(split="test", neighbor_size=self.neighbor_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=128,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
