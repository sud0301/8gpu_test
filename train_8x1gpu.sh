#!/bin/bash

echo "This script starts 8 single-gpu jobs at the same time."
echo "All jobs will print to the same CLI. Don't be confused by this."

for i in {0..7}
do
    echo "Starting job $i"
    export CUDA_VISIBLE_DEVICES=$i
    python train_full.py --batch-size 5 &
done
