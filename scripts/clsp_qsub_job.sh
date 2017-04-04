#!/usr/bin/env bash

#$ -N sigv
#$ -cwd
#$ -j yes
#$ -m b
#$ -m e
#$ -m a
#$ -m s
#$ -M "kahrabian@gmail.com"
#$ -l gpu=1
#$ -l 'arch=*64*'
#$ -l mem_free=4G,ram_free=4G
#$ -l "hostname=b1[123456789]*"
#$ -pe smp 8

source activate sigv
CUDA_VISIBLE_DEVICES=`/home/gkumar/scripts/free-gpu` python main.py
source deactivate sigv
