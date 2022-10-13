#!/usr/bin/env bash

export OMP_NUM_THREADS=1

mode=$1

if [ $mode == "dist" ]; then
        python -m trainer.distribute \
                --gpus "0, 1, 2, 3, 4, 5, 6, 7" \
                --script train_vits.py
else
        CUDA_VISIBLE_DEVICES="0" python train_vits.py
fi


