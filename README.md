# Counterfactual Explanations for Survival Prediction of Cardiovascular Disease
We adopt the implementation of the Delete-Retrieve-Generate framework from [Reid Pryzant](https://github.com/rpryzant/delete_retrieve_generate) to address the proposed counterfactual explanation problem on medical event sequence data.

## MIMIC-III data 
[MIMIC-III](https://mimic.physionet.org/gettingstarted/overview/) dataset is originally collected from ICU patients' eletronic health records in the Beth Israel Deaconess Medical Center. 

The data pre-processing step is done in the [Jupyter notebook](./notebooks/1-data-preprocessing.ipynb). Coding environment is Python 3. Please note that one need to request the access from MIMIC III and install the postgres database as their instruction, in order to actually use the notebook for generating training/validation dataset.


## Installation

`pip install -r requirements.txt`

## Running the DRG method

Before acutally running the training script, we need to generate n-gram attribute vocabulary list. First, we need to concanate `train_pos.txt` and `train_neg.txt` into `train_all.txt`,

```
cat mimic_data/train_pos.txt mimic_data/train_neg.txt > mimic_data/train_all.txt
```

and then simply run:
```
python tools/make_vocab.py mimic_data/train_all.txt 3000 > mimic_data/vocab.txt

python tools/make_ngram_attribute_vocab.py mimic_data/vocab.txt mimic_data/train_neg.txt mimic_data/train_pos.txt 15 > mimic_data/ngram_attribute_vocab.txt
```

After that, we run the training script: 

```
python train.py --config medseq_config.json
```

The default [configuration file](./medseq_config.json) is for DeleteOnly (*Alg. 1* with *r=False*), which can generate a trained DeleteOnly model in the folder `working_dir_delete`. The folder also include checkpoints, logs, model outputs, and TensorBoard summaries.  


For the DeleteAndRetrieve model (*Alg. 1* with *r=True*), we simply need to change `model_type` parameter to `delete_retrieve` in `medseq_config.json`. And run the same command above for training. Note that other hyper-papermeters are also editable in the config file. 

## Inference
There is an inference script that we can apply the trained model to do extra inferences with a new test dataset. We can modify `src_test` parameters in the config file (we can ignore the `tgt_test` parameter since there is no target dataset at inference time).

```
python inference.py --config medseq_config.json --checkpoint working_dir_delete/$checkpoint_file$

python inference.py --config medseq_config.json --checkpoint working_dir_delete_retrieve/$checkpoint_file$
```

MIMIC-III demo data: https://physionet.org/content/mimiciii-demo/1.4/. 


## Running the 1-NN method
The 1-NN method and the LSTM model are implemented in [this Jupyter notebook](./notebooks). 
