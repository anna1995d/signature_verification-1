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
#$ -l mem_free=16G,ram_free=16G
#$ -l "hostname=b1[123456789]*|c*"
#$ -pe smp 8

source activate sigv
CUDA_VISIBLE_DEVICES=`free-gpu` python -W ignore main.py
source deactivate sigv
