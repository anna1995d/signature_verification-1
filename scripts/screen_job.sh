#!/usr/bin/env bash

source activate sigv
TF_CPP_MIN_LOG_LEVEL=3 screen -dmS sigv -L python main.py
source deactivate sigv
