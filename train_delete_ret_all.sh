#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate py37-cuda10

python train.py --config configs/medseq_config_ret.json

python train.py --config configs/medseq_config_ret2.json

python train.py --config configs/medseq_config_ret3.json

python train.py --config configs/medseq_config_ret4.json