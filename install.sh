#!/bin/bash

conda create -n baltic python=3.11
conda activate baltic

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyyaml numpy pandas scikit-learn tqdm httpx lightning matplotlib rioxarray pyarrow tensorboard