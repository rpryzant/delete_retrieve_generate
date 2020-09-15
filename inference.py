import sys

import json
import numpy as np
import logging
import argparse
import os
import time
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import src.evaluation as evaluation
from src.cuda import CUDA
import src.data as data
import src.models as models



parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config.",
    required=True
)
parser.add_argument(
    "--checkpoint",
    help="path to model checkpoint",
    required=True
)


args = parser.parse_args()
config = json.load(open(args.config, 'r'))

working_dir = config['data']['working_dir']

if not os.path.exists(working_dir):
    os.makedirs(working_dir)

config_path = os.path.join(working_dir, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='%s/train_log' % working_dir,
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info('Reading data ...')

src, tgt = data.read_nmt_data(
    src=config['data']['src'],
    config=config,
    tgt=config['data']['tgt'],
    attribute_vocab=config['data']['attribute_vocab'],
    ngram_attributes=config['data']['ngram_attributes']
)

src_test, tgt_test = data.read_nmt_data(
    src=config['data']['src_test'],
    config=config,
    tgt=config['data']['tgt_test'],
    attribute_vocab=config['data']['attribute_vocab'],
    ngram_attributes=config['data']['ngram_attributes'],
    train_src=src,
    train_tgt=tgt
)
logging.info('...done!')

src_vocab_size = len(src['tok2id'])
tgt_vocab_size = len(tgt['tok2id'])

torch.manual_seed(config['training']['random_seed'])
np.random.seed(config['training']['random_seed'])

model = models.SeqModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    pad_id_src=src['tok2id']['<pad>'],
    pad_id_tgt=tgt['tok2id']['<pad>'],
    config=config
)

logging.info('MODEL HAS %s params' %  model.count_params())
model, start_epoch = models.attempt_load_model(
    model=model,
    checkpoint_path=args.checkpoint)
if CUDA:
    model = model.cuda()

start = time.time()
model.eval()
dev_loss = evaluation.evaluate_lpp(
        model, src_test, tgt_test, config)

bleu, edit_distance, inputs, preds, golds, auxs = evaluation.inference_metrics(
    model, src_test, tgt_test, config)

with open(working_dir + '/auxs', 'w') as f:
    f.write('\n'.join(auxs) + '\n')
with open(working_dir + '/inputs', 'w') as f:
    f.write('\n'.join(inputs) + '\n')
with open(working_dir + '/preds', 'w') as f:
    f.write('\n'.join(preds) + '\n')
with open(working_dir + '/golds', 'w') as f:
    f.write('\n'.join(golds) + '\n')

logging.info('INFERENCE DONE. Outputs written to %s' %  (working_dir + "/preds"))
logging.info('\tEdit distance: ' + str(edit_distance))
logging.info('\tBLEU: ' + str(bleu))
logging.info('\tLPP: ' + str(dev_loss))


