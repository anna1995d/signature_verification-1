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
#$ -l "hostname=b*|c*"
#$ -pe smp 16

source activate sigv
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=`free-gpu` python -W ignore main.py
source deactivate sigv
