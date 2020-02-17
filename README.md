# Description

This is an implementation of the DeleteOnly and DeleteAndRetrieve models from [Delete, Retrieve, Generate:
A Simple Approach to Sentiment and Style Transfer](https://arxiv.org/pdf/1804.06437.pdf)

# Installation

`pip3 install -r requirements.txt`

This code uses python 3. 

# Usage

### Training

`python3 train.py --config yelp_config.json --bleu`

This will reproduce the _delete_ model on a dataset of yelp reviews:

![curves](https://i.imgur.com/jfYaDBr.png)


Checkpoints, logs, model outputs, and TensorBoard summaries are written in the config's `working_dir`.

See `yelp_config.json` for all of the training options. The most important parameter is `model_type`, which can be set to `delete`, `delete_retrieve`, or `seq2seq` (which is a standard translation-style model).




### Data prep

Given two pre-tokenized corpus files, use the scripts in `tools/` to generate a vocabulary and attribute vocabulary:

```
python tools/make_vocab.py [entire corpus file (src + tgt cat'd)] [vocab size] > vocab.txt
python tools/make_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
python tools/make_ngram_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
```

# Questions, feedback, bugs

rpryzant@stanford.edu

# Acknowledgements

Thanks lots to [Karishma Mandyam](https://github.com/kmandyam) for contributing! 


# Citation

If you use this package in your own research can you please cite 

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
