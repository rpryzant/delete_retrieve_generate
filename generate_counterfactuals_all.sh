#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate py37-cuda11

python train_main_model.py --dataset cardiovascular --databatch 5 --vocab-size 1400 --max-len 104


eval "$(conda shell.bash hook)"
conda activate py37-cuda10

python inference.py --config configs/medseq_config_eval.json --checkpoint experiment_cardiovascular5/working_dir_cardiovascular5_delete1/model.*.ckpt

python inference.py --config configs/medseq_config_eval2.json --checkpoint experiment_cardiovascular5/working_dir_cardiovascular5_delete2/model.*.ckpt

python inference.py --config configs/medseq_config_eval3.json --checkpoint experiment_cardiovascular5/working_dir_cardiovascular5_delete3/model.*.ckpt

python inference.py --config configs/medseq_config_eval4.json --checkpoint experiment_cardiovascular5/working_dir_cardiovascular5_delete4/model.*.ckpt


python inference.py --config configs/medseq_config_eval_ret.json --checkpoint experiment_cardiovascular5/working_dir_cardiovascular5_delete_ret1/model.*.ckpt

python inference.py --config configs/medseq_config_eval_ret2.json --checkpoint experiment_cardiovascular5/working_dir_cardiovascular5_delete_ret2/model.*.ckpt

python inference.py --config configs/medseq_config_eval_ret3.json --checkpoint experiment_cardiovascular5/working_dir_cardiovascular5_delete_ret3/model.*.ckpt

python inference.py --config configs/medseq_config_eval_ret4.json --checkpoint experiment_cardiovascular5/working_dir_cardiovascular5_delete_ret4/model.*.ckpt


eval "$(conda shell.bash hook)"
conda activate py37-cuda11

python cf_evaluate.py --dataset cardiovascular --databatch 5 --vocab-size 1400 --max-len 104 --output results.csv