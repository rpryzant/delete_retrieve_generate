#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate py37-cuda10

python train.py --config configs/medseq_config.json

python train.py --config configs/medseq_config2.json

python train.py --config configs/medseq_config3.json

python train.py --config configs/medseq_config4.json