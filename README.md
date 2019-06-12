# Description

This is an implementation of the DeleteOnly and DeleteAndRetrieve models from [Delete, Retrieve, Generate:
A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/pdf/1804.06437.pdf)

# Installation

`pip install -r requirements.txt`

This code uses python 3. 

# Usage

### Training

`python train.py --config yelp_config.json --bleu`

This will reproduce the _delete_ model on a dataset of yelp reviews:

![curves](https://i.imgur.com/jfYaDBr.png)


Checkpoints, logs, model outputs, and TensorBoard summaries are written in the config's `working_dir`.

See `yelp_config.json` for all of the training options. The most important parameter is `model_type`, which can be set to `delete`, `delete_retrieve`, or `seq2seq` (which is a standard translation-style model).




### Data prep

Given two pre-tokenized corpus files, use the scripts in `tools/` to generate a vocabulary and attribute vocabulary:

```
python tools/make_vocab.py [entire corpus file (src + tgt cat'd)] [vocab size] > vocab.txt
python tools/make_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
```

# Questions, feedback, bugs

rpryzant@stanford.edu

