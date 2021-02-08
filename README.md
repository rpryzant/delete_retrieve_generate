# Counterfactual Explanations for Survival Prediction of Cardiovascular ICU Patients
We adopt the implementation of the Delete-Retrieve-Generate framework from [Reid Pryzant](https://github.com/rpryzant/delete_retrieve_generate) to address the proposed counterfactual explanation problem, based on event sequence data which is originally generated from MIMIC-III dataset. Additionally, we implement a baseline 1-NN solution for the same problem that shows competitive performance.

## MIMIC-III data 
[MIMIC-III](https://mimic.physionet.org/gettingstarted/overview/) dataset is a collection of electronic health records (EHRs) from over 40,000 ICU patients at the Beth Israel Deaconess Medical Center, collected between 2001 and 2012. 

The data pre-processing step is done in the [Jupyter notebook](./notebooks/1-data-preprocessing.ipynb). Coding environment is Python 3. With this script, we pre-process the data as follows:
1. We select drug events, medical procedures, and diagnosis codes, and represent each patient as a historical sequence of events, limiting the selection to the last 12 months of patients visits. 
2. Moreover, to limit the scope our experiments, we focus on patients that have been diagnosed with cardiovascular diseases (corresponding to ICD-9 codes: 393-398, 410-414 and 420-429, representing chronic rheumatic, ischemic and other forms of heart disease separately).

Please note that one need to request the access from MIMIC III and install the postgres database as their instruction, in order to actually use the notebook for generating training/validation dataset.

## Running the DRG method

### Installation

`pip install -r requirements.txt`

### Train DRG models

Before actually running the training script, we need to generate n-gram attribute vocabulary list. First, we need to concatenate `train_pos.txt` and `train_neg.txt` into `train_all.txt`,

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


For the DeleteAndRetrieve model (*Alg. 1* with *r=True*), we simply need to change `model_type` parameter to `delete_retrieve` in `medseq_config.json`. And run the same command above for training. Note that other hyper-parameters are also editable in the config file. 

### Inference
There is an inference script that we can apply the trained model to do extra inferences with a new test dataset. We can modify `src_test` parameters in the config file (we can ignore the `tgt_test` parameter since there is no target dataset at inference time).

```
python inference.py --config medseq_config.json --checkpoint working_dir_delete/$checkpoint_file$

python inference.py --config medseq_config.json --checkpoint working_dir_delete_retrieve/$checkpoint_file$
```

MIMIC-III demo data: https://physionet.org/content/mimiciii-demo/1.4/. 


## Running the 1-NN method
The 1-NN method and the LSTM model (for survival prediction) are implemented in [this Jupyter notebook](./notebooks/2-LSTM-model-and-generate-counterfactuals.ipynb). All the evaluation results and plots are implemented in the same notebook as well.
