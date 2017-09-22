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
#$ -l "hostname=b1[123456789]*"
#$ -pe smp 8

source activate sigv
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=`/home/gkumar/scripts/free-gpu` KERAS_BACKEND=theano python -W ignore main.py
source deactivate sigv
