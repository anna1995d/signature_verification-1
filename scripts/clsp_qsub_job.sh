#!/usr/bin/env bash

#$ -cwd
#$ -o ./models
#$ -e ./models
#$ -j y
#$ -m b
#$ -m e
#$ -m a
#$ -m s
#$ -M "kahrabian@gmail.com"
#$ -l gpu=1
#$ -l 'arch=*64*'
#$ -l mem_free=32G,ram_free=32G
#$ -l "hostname=b1[123456789]*|c*"
#$ -pe smp 8

source activate sigv
CUDA_VISIBLE_DEVICES=`free-gpu` TF_CPP_MIN_LOG_LEVEL=1 python -W ignore main.py
source deactivate sigv
