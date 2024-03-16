from pathlib import Path

import pandas as pd
import rioxarray

base_dir = Path(__file__).resolve().parent.parent.parent
path = base_dir / "data/Baltic_Sea_Bathymetry_Database_v091.tif"

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

df.to_parquet(base_dir / "data/baltic_sea_depth.parquet")
