from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap

base_dir = Path(__file__).resolve().parent.parent.parent

df = pd.read_parquet(base_dir / "data/baltic_sea_depth.parquet")

plt.subplots(figsize=(8, 6))

m = Basemap(projection="merc", resolution="i", llcrnrlat=53, urcrnrlat=66, llcrnrlon=9, urcrnrlon=32)
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.fillcontinents(color="lightgray", zorder=0)

x, y = m(df["lon"].values, df["lat"].values)
sc = m.scatter(x, y, c=df["dep"], cmap="Blues", s=0.15, marker="s", vmin=0, vmax=200)

cbar = plt.colorbar(sc, orientation="vertical")
cbar.set_label("Depth(m)")

plt.show()
