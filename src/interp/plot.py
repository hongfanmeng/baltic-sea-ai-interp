from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dep_lb = np.concatenate((np.arange(0, 100, 5), np.arange(100, 200, 10)))
dep_ub = np.concatenate((np.arange(0, 100, 5) + 5, np.arange(100, 200, 10) + 10))
dep_mid = (dep_lb + dep_ub) / 2

base_dir = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    df = pd.read_parquet(base_dir / "data/interp_results.parquet")
    df_year = df[df["year"] == 2000]
    df_year = df_year.copy().reset_index()

    mean, std = pd.read_pickle(base_dir / "data/train_mean_std.pkl")

    for idx, row in df_year.iterrows():
        row["oxy"] = row["output"] * std["oxy"] + mean["oxy"]
        try:
            dep_idx = np.where(dep_ub > row["max_dep"])[0][0]
            row["oxy"][dep_idx:] = np.nan
        # if max_dep over all dep_lb
        except IndexError:
            pass

        df_year.loc[idx, "oxy"] = row["max_dep"]

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw={"projection": projection})

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    df_year.plot.scatter(ax=ax, x="lat", y="lon", c="max_dep", s=2, marker="s", cmap="Blues", vmin=0, vmax=200)

    # df_train = pd.read_parquet(base_dir / "data/train_data.parquet")
    # df_train_year = df_train[(df_train["year"] == 1960) & (df_train["dep"] == 2.5)]
    # df_train_year.plot.scatter(ax=ax, x="lon", y="lat", c="oxy", cmap="rainbow", s=5, vmin=6, vmax=10)

    plt.xlabel("Longitude (°E)")
    plt.xlim(8, 32)
    plt.xticks(np.arange(8, 32 + 1, 2))

    plt.ylabel("Latitude (°N)")
    plt.ylim(52, 68)
    plt.yticks(np.arange(52, 68 + 1, 2))

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()
