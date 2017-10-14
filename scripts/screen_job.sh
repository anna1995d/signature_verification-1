#!/usr/bin/env bash

source activate sigv
PYTHONHASHSEED=1996 screen -dmS sigv -L python -W ignore main.py
source deactivate sigv
