import pickle
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

base_dir = Path(__file__).resolve().parent.parent.parent


def read_bathymeetry():
    df = pd.read_parquet(base_dir / "data/baltic_sea_depth.parquet")
    df = df.set_index(["lon", "lat"])
    return df


class BalticSeaDataset(Dataset):
    def __init__(self, data_path=base_dir / "data/train_data.pkl"):
        with open(data_path, "rb") as f:
            self.mean, self.std, self.data_list = pickle.load(f)
        self.df_dep = read_bathymeetry()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> tuple[pd.Series, pd.DataFrame]:
        meta, data = self.data_list[idx]
        year, mon, lat, lon = meta
        max_dep = self.df_dep.loc[(lon, lat), "dep"] if (lon, lat) in self.df_dep.index else 0

        meta = pd.Series({"year": year, "mon": mon, "lat": lat, "lon": lon, "max_dep": max_dep})

        return meta, data.reset_index(drop=True)
