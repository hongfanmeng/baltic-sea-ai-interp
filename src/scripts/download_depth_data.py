import os
import sys
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
from tqdm import tqdm

base_dir = Path(__file__).resolve().parent.parent.parent
data_dir = base_dir / "data"


def download_data():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(data_dir / "bsbd-0.9.3_3035_clip.tif"):
        return

    url = "https://s3.eu-west-1.amazonaws.com/data.bshc.pro/Post_retirement/bsbd-0.9.3_3035_clip.tif"
    with open(data_dir / "bsbd-0.9.3_3035_clip.tif", "wb") as download_file:
        with httpx.stream("GET", url) as response:
            total = int(response.headers["Content-Length"])

            with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B") as progress:
                num_bytes_downloaded = response.num_bytes_downloaded
                for chunk in response.iter_bytes():
                    download_file.write(chunk)
                    progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                    num_bytes_downloaded = response.num_bytes_downloaded


def convert_to_parquet() -> pd.DataFrame:
    path = data_dir / "bsbd-0.9.3_3035_clip.tif"

    xds = rioxarray.open_rasterio(path, masked=True)
    xds = xds.rio.reproject("EPSG:4326")
    df = xds[0].to_pandas()
    df["y"] = df.index
    df = pd.melt(df, id_vars="y")

    df = df.rename(columns={"value": "dep", "x": "lon", "y": "lat"})
    df = df.dropna().reset_index(drop=True)

    df["lat"] = df["lat"].astype(float).round(2)
    df["lon"] = df["lon"].astype(float).round(2)
    df["dep"] = df["dep"] * -1

    df = df.groupby(["lon", "lat"])["dep"].mean().reset_index()
    df = df[df["dep"] > 0]

    df.to_parquet(data_dir / "baltic_sea_depth.parquet")

    return df


def plot_data():
    df_dep = pd.read_parquet(data_dir / "baltic_sea_depth.parquet")
    df_dep.plot.scatter(x="lon", y="lat", c="dep", cmap="viridis", s=0.01, marker="s", vmin=0, vmax=200)
    plt.show()


if __name__ == "__main__":
    download_data()
    convert_to_parquet()

    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_data()
