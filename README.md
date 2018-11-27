
This is an implementation of the DeleteOnly and DeleteAndRetrieve models from this paper: https://arxiv.org/pdf/1804.06437.pdf

# Usage

### Training

`python train.py --config sample_config.json --bleu`

This will train a model using the parameters in `sample_config.json`. Checkpoints, logs, decodings, and TensorBoard summaries will go into config's `working_dir`.

Supported model types are `delete`, `delete_retrieve`, and `seq2seq`.

### Vocab generation

Given two corpus files, use the scripts in `tools/` to generate a vocabulary and attribute vocabulary:

```
python tools/make_vocab.py [entire corpus] [vocab size] > vocab.txt
python tools/make_attribute_vocab.py vocab.txt [corpus src file] [corpus tgt file] [salience ratio] > attribute_vocab.txt
```

# Questions, feedback

rpryzant@stanford.edu

