from pathlib import Path

import pandas as pd
import torch
from get_model_inputs import YEAR_END, YEAR_START
from tqdm import tqdm

from rnn import InterpRNN
from vae import VanillaVAE

base_dir = Path(__file__).parent.parent.parent


if __name__ == "__main__":

    model_inputs = pd.read_pickle(base_dir / "data/model_inputs.pkl")

    vae_model = VanillaVAE.load_from_checkpoint(base_dir / "logs/VAE/version_0/checkpoints/last.ckpt")
    rnn_model = InterpRNN.load_from_checkpoint(
        base_dir / "logs/GRU/version_0/checkpoints/last.ckpt",
        vae_model=vae_model,
    )

    years = list(range(YEAR_START, YEAR_END + 1))

    results = []
    for input, label in tqdm(model_inputs):
        input = torch.Tensor(input).to("cuda:0")
        meta = {k: v[0] for k, v in label.items() if k in ["lon", "lat", "max_dep"]}
        label = {k: torch.Tensor(v).to("cuda:0") for k, v in label.items()}

        with torch.no_grad():
            output = rnn_model.forward(input, label)

        for idx, year in enumerate(years):
            results.append({**meta, "year": year, "output": output[0][idx].cpu().detach().numpy()})

    df = pd.DataFrame(results)
    df.to_parquet(base_dir / "data/interp_results.parquet")
