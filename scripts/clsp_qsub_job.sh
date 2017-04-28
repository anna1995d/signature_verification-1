#!/usr/bin/env bash

#$ -cwd
#$ -o ./logs
#$ -e ./logs
#$ -j y
#$ -m b
#$ -m e
#$ -m a
#$ -m s
#$ -M "kahrabian@gmail.com"
#$ -l gpu=1
#$ -l 'arch=*64*'
#$ -l mem_free=8G,ram_free=8G
#$ -l "hostname=b1[123456789]*"
#$ -pe smp 8

source activate sigv
CUDA_VISIBLE_DEVICES=`/home/gkumar/scripts/free-gpu` python main.py
source deactivate sigv
