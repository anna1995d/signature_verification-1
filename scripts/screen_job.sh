#!/usr/bin/env bash

source activate sigv
KERAS_BACKEND=theano screen -dmS sigv -L python main.py
source deactivate sigv
