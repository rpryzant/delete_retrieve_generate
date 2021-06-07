# Description

This is an implementation of the DeleteOnly and DeleteAndRetrieve models from [Delete, Retrieve, Generate:
A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/pdf/1804.06437.pdf)

# Installation

`pip3 install -r requirements.txt`

This code uses python 3. 

# Usage

### Training (runs inference on the dev set after each epoch)

`python3 train.py --config yelp_config.json --bleu`

This will reproduce the _delete_ model on a dataset of yelp reviews:

![curves](https://i.imgur.com/jfYaDBr.png)


Checkpoints, logs, model outputs, and TensorBoard summaries are written in the config's `working_dir`.

See `yelp_config.json` for all of the training options. The most important parameter is `model_type`, which can be set to `delete`, `delete_retrieve`, or `seq2seq` (which is a standard translation-style model).

### Inference

`python inference.py --config yelp_config.json --checkpoint path/to/model.ckpt`

To run inference, you can point the `src_test` and `tgt_test` fields in your config to new data. 


### Data prep

Given two pre-tokenized corpus files, use the scripts in `tools/` to generate a vocabulary and attribute vocabulary:

```
python tools/make_vocab.py [entire corpus file (src + tgt cat'd)] [vocab size] > vocab.txt
python tools/make_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
python tools/make_ngram_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
```

# Citation

If you use this code as part of your own research can you please cite 

(1) the original paper:
```
@inproceedings{li2018transfer,
 author = {Juncen Li and Robin Jia and He He and Percy Liang},
 booktitle = {North American Association for Computational Linguistics (NAACL)},
 title = {Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer},
 url = {https://nlp.stanford.edu/pubs/li2018transfer.pdf},
 year = {2018}
}

```

(2) The paper that this implementation was developed for:
```
@inproceedings{pryzant2020bias,
 author = {Pryzant, Reid and Richard, Diehl Martinez and Dass, Nathan and Kurohashi, Sadao and Jurafsky, Dan and Yang, Diyi},
 booktitle = {Association for the Advancement of Artificial Intelligence (AAAI)},
 link = {https://nlp.stanford.edu/pubs/pryzant2020bias.pdf},
 title = {Automatically Neutralizing Subjective Bias in Text},
 url = {https://nlp.stanford.edu/pubs/pryzant2020bias.pdf},
 year = {2020}
}
```


# FAQ

### Why can't I get the same BLEU score as the original paper? 

- My script just runs in one direction (e.g. pos => neg). Maybe running the model in both directions (pos => neg, neg => pos) and then averaging the BLEU would get closer to their results
- The [implementation of BLEU that the original paper used](https://github.com/lijuncen/Sentiment-and-Style-Transfer/blob/250d22d39607bf697082861af935ab8e66e2160c/src/test_tool/BLEU/my_bleu_evaluate.py) has bugs in it and does not report correct BLEU scores. For example, it disagrees with [multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) which is a canonical implementation of BLEU. If you use their script on our outputs you get something more similar (I think ~7.6 ish) but again their script might not be producing correct BLEU scores. 

### Why does `delete_retrieve` run so slowly? 

- The system runs a similarity search over the entire dataset on each training step. Precomputing some of these similarities would definitely speed things up if people are interested in contributing!

### What does the salience ratio mean? How was your number chosen?

- Intuitively the salience ratio says how strongly associated with each class do you want the attribute ngrams to be. Higher numbers means that the attribute vocab will be more strongly associated with each class, but also that you will have fewer vocab items because the threshold is tighter.
- The example attributes in this repo use the ratios from the paper, which were selected manually using a dev set. 


### I keep getting `IndexError: list index out of range` errors! 

- There is a known bug where the size of the A and B datasets need to match each other (again a great place to contribute!). Since the corpora don't need to be in alignment you can just duplicate some examples or trim one of the datasets so that they match each other. 

### I keep getting `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` errors!
The pytorch version this has been tested on is 1.1.0, which is compatible with cudatoolkit=9.0/10.0. If your cuda version is newer than this you may get the above error. Possible fix: 
```
$ conda install pytorch==1.1.0 torchvison==0.3.0 cudatoolkit=10.0 -c pytorch
```


# Acknowledgements

Thanks lots to [Karishma Mandyam](https://github.com/kmandyam) for contributing! 
