#!/usr/bin/env bash

cd ..
cd tacotron2

for i in {7..10}
do
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='6' python inference.py \
            -c /ceph/home/qq295951960/cjcode/flowtron/checkpoints/tacotron2_statedict.pt \
            -w /ceph/home/qq295951960/cjcode/flowtron/checkpoints/waveglow.pt \
            -t "How much variation is there?" \
            -o /ceph/home/qq295951960/cjcode/flowtron/SV_experiment/Tacotron \
            -time $i
done