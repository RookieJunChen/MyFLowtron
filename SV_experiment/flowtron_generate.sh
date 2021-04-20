#!/usr/bin/env bash

cd ..

for i in {1..10}
do
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='5' python inference.py \
            -c config.json \
            -f checkpoints/flowtron_ljs.pt \
            -w checkpoints/waveglow.pt \
            -t "How much variation is there?" \
            -s 0.0 \
            -o SV_experiment/Flowtron_sigma0.0 \
            -i 0 \
            -time $i
done
