import pickle
from functools import cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
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


class VAEDataset(Dataset):
    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        test_size=0.2,
        nan_rate=0.7,
    ):
        self.df_dep = read_depth()
        self.mean, self.std = read_mean_std()
        self.nan_rate = nan_rate

        data_list = self.get_vae_train_data()
        train_data, test_data = train_test_split(data_list, test_size=test_size)
        self.data_list = train_data if split == "train" else test_data

    @staticmethod
    @cache
    def get_vae_train_data():
        print("reading data...")
        vae_train_data = pd.read_pickle("data/vae_train_data.pkl")

        print("filtering data for training...")
        # use data with max_dep >= 50 to train
        vae_data_list = [(meta, data) for (meta, data) in vae_train_data if meta["max_dep"] >= 50]
        fill_rate = process_map(VAEDataset.get_fill_rate, vae_data_list, max_workers=12, chunksize=2000)

        # use data with fill rate >= 1 to train
        vae_data_list = [data for data, rate in zip(vae_data_list, fill_rate) if rate >= 1]

        return vae_data_list

    @staticmethod
    def get_fill_rate(data: tuple[dict, pd.DataFrame]):
        meta, data = data
        total = sum(data["dep"] <= meta["max_dep"])
        return len(data[data["dep"] <= meta["max_dep"]].dropna()) / total if total > 0 else 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> tuple[np.ndarray, dict]:
        meta, data = self.data_list[idx]
        data: pd.DataFrame = data.copy()
        data[["oxy", "tmp", "sal"]] = (data[["oxy", "tmp", "sal"]] - self.mean) / self.std

        # origin value
        meta["real"] = data.values[:, 1].copy()

        # set value to nan and interpolate to simulate missing data
        non_nan_idx = data.dropna().index
        nan_cnt = round(len(non_nan_idx) * self.nan_rate)
        random_nan_idx = data.iloc[non_nan_idx].sample(nan_cnt, replace=False).index
        data.loc[random_nan_idx, ["oxy", "tmp", "sal"]] = np.nan
        data = data.interpolate(method="linear", limit_direction="both")

        # fill data which dep exceed max dep to 0
        data[~data.index.isin(non_nan_idx)] = 0

        # save nan mask for testing
        meta["nan_mask"] = np.zeros(len(data), dtype=bool)
        meta["nan_mask"][random_nan_idx] = True

        meta["x"] = np.cos(meta["lat"]) * np.cos(meta["lon"])
        meta["y"] = np.cos(meta["lat"]) * np.sin(meta["lon"])
        meta["z"] = np.sin(meta["lat"])

        return data.values[:, 1], meta


class VAEDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        nan_rate: float = 0.7,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.nan_rate = nan_rate

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = VAEDataset(nan_rate=self.nan_rate, split="train")
        self.val_dataset = VAEDataset(nan_rate=self.nan_rate, split="test")

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
            shuffle=False,
            pin_memory=self.pin_memory,
        )
