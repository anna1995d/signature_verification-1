#!/usr/bin/env bash

source activate sigv
screen -dmS sigv -L python -W ignore main.py
source deactivate sigv
