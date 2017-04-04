#!/usr/bin/env bash

conda install -n sigv python=3
source activate sigv
conda install ipython jupyter matplotlib numpy h5py
pip install keras
source deactivate sigv
