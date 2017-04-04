#!/usr/bin/env bash

source activate sigv
screen -dmS sigv -L python main.py
source deactivate sigv
