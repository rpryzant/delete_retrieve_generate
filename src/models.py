"""Sequence to Sequence models."""
import glob
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import src.decoders as decoders
import src.encoders as encoders

from src.cuda import CUDA


def get_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    # nothing to load, continue with fresh params
    if len(ckpts) == 0:
        return -1, None
    ckpts = map(lambda ckpt: (
        int(ckpt.split('.')[1]),
        ckpt), ckpts)
    # get most recent checkpoint
    epoch, ckpt_path = sorted(ckpts)[-1]
    return epoch, ckpt_path


def attempt_load_model(model, checkpoint_dir=None, checkpoint_path=None):
    assert checkpoint_dir or checkpoint_path

    if checkpoint_dir:
        epoch, checkpoint_path = get_latest_ckpt(checkpoint_dir)
    else:
        epoch = int(checkpoint_path.split('.')[-2])

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print('Load from %s sucessful!' % checkpoint_path)
        return model, epoch + 1
    else:
        return model, 0


class SeqModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        pad_id_src,
        pad_id_tgt,
        config=None,
        diag_vocab_size=None,
        pad_id_diag=None,
    ):
        """Initialize model."""
        super(SeqModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_id_src = pad_id_src
        self.pad_id_tgt = pad_id_tgt
        self.batch_size = config['data']['batch_size']
        self.config = config
        self.options = config['model']
        self.model_type = config['model']['model_type']

        # update: add an embedding layer for diagnosis codes
        self.diag_vocab_size = diag_vocab_size
        self.pad_id_diag = pad_id_diag

        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.options['emb_dim'],
            self.pad_id_src)
        
        # update: add an embedding layer for diagnosis codes
        if self.config['model']['add_diagnosis_layer']:
            self.diag_embedding = nn.Embedding(
                self.diag_vocab_size,
                self.options['emb_dim'],
                self.pad_id_diag)

        # update: add an embedding layer for coexisting drugs
        if self.config['model']['add_coexist_layer']: # alternative 1: use a linear layer, transform from 1202 to 256
            self.coexist_bridge = nn.Linear(
                    in_features=self.src_vocab_size+1,
                    out_features=256)
        """
        # alternative 2: 
        # coexist_ht = self.coexist_embedding(coexist_attr); store all the files beforehand, get max_len_coexist_events
        # coexist_attr.shape: 128 (batch) x 300 (max_len_coexist_events)
        # coexist_ht.shape = 128 x 300 x 128, then do average pooling or max pooling
        # 
        # if self.config['model']['add_drug_coexist_layer']:
        #     self.coexist_embedding = nn.Embedding(
        #         self.src_vocab_size,
        #         self.options['emb_dim'])
        """
        
        if self.config['data']['share_vocab']:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(
                self.tgt_vocab_size,
                self.options['emb_dim'],
                self.pad_id_tgt)

        if self.options['encoder'] == 'lstm':
            self.encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'])
            
            # update: add an embedding layer for diagnosis codes; fixed hyperparameters
            if self.config['model']['add_diagnosis_layer']:
                self.diag_encoder = encoders.LSTMEncoder(
                    emb_dim=128,
                    hidden_dim=256,                 
                    layers=1,                    
                    bidirectional=False,                # not bidirectional since the patient sequence follows the single-direction order 
                    dropout=0.2,
                    pack=False)

            self.ctx_bridge = nn.Linear(
                self.options['src_hidden_dim'],
                self.options['tgt_hidden_dim'])

        else:
            raise NotImplementedError('unknown encoder type')

        # # # # # #  # # # # # #  # # # # #  NEW STUFF FROM STD SEQ2SEQ
        
        if self.model_type == 'delete':
            self.attribute_embedding = nn.Embedding(
                num_embeddings=2, 
                embedding_dim=self.options['emb_dim'])
            attr_size = self.options['emb_dim']

        elif self.model_type == 'delete_retrieve':
            self.attribute_encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'],
                pack=False)
            attr_size = self.options['src_hidden_dim']

        elif self.model_type == 'seq2seq':
            attr_size = 0

        else:
            raise NotImplementedError('unknown model type')

        # update: add an embedding layer for diagnosis codes AND coexisting events; concatenate with hidden/cell units here
        extra_dim_diag = 256 if self.config['model']['add_diagnosis_layer'] else 0 # 256 == extra dimension by extra diagnosis embeddings, fixed value
        extra_dim_coexist = 256 if self.config['model']['add_coexist_layer'] else 0 # 256 == the dimension of the extra embedding layer for coexisting drugs
        bridge_dim = attr_size + self.options['src_hidden_dim'] + extra_dim_diag + extra_dim_coexist
        
        self.c_bridge = nn.Linear(
            bridge_dim,  
            self.options['tgt_hidden_dim'])
        self.h_bridge = nn.Linear(
            bridge_dim,  
            self.options['tgt_hidden_dim'])

        # # # # # #  # # # # # #  # # # # # END NEW STUFF

        self.decoder = decoders.StackedAttentionLSTM(config=config)

        self.output_projection = nn.Linear(
            self.options['tgt_hidden_dim'],
            tgt_vocab_size)

        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.h_bridge.bias.data.fill_(0)
        self.c_bridge.bias.data.fill_(0)
        self.output_projection.bias.data.fill_(0)

    def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask, diag_input_lines=None, diag_lens=None, diag_mask=None, coexist_attr=None):
        src_emb = self.src_embedding(input_src)

        use_diagnosis_layer = (diag_input_lines is not None) and (diag_lens is not None) and (diag_mask is not None)
        if use_diagnosis_layer:
            diag_emb = self.diag_embedding(diag_input_lines)

        srcmask = (1-srcmask).byte()

        src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, srclens, srcmask)
        if use_diagnosis_layer:
            diag_outputs, (diag_h_t, diag_c_t) = self.diag_encoder(diag_emb, diag_lens, diag_mask)
            #diag_outputs.shape = 128x116x512; diag_h_t.shape = 2x128x512; diag_c_t.shape = 2x128x512

        if self.options['bidirectional']:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1) # src_h_t[-2] # src_h_t.shape = 2x128x256
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        src_outputs = self.ctx_bridge(src_outputs)

        # update: add an embedding layer for diagnosis codes; concatenate with hidden/cell units here
        if use_diagnosis_layer:
            h_t = torch.cat((h_t, diag_h_t[-1]), -1) # [-1] since no bidirectional, add for the extra embedding layer
            c_t = torch.cat((c_t, diag_c_t[-1]), -1)

        # update: add for coexisting drug events
        use_coexist_layer = (coexist_attr is not None)
        if use_coexist_layer:
            coexist_attr = self.coexist_bridge(coexist_attr)         # coexist_attr.shape = 128 x 1202 -> 128 x 256
            h_t = torch.cat((h_t, coexist_attr), -1)                 # ht.shape = 128 x 512
            c_t = torch.cat((c_t, coexist_attr), -1)

        # # # #  # # # #  # #  # # # # # # #  # # seq2seq diff
        # join attribute with h/c then bridge 'em
        # TODO -- put this stuff in a method, overlaps w/above

        if self.model_type == 'delete':
            # just do h i guess?
            a_ht = self.attribute_embedding(input_attr)              # input_attr.shape = 128x1
            c_t = torch.cat((c_t, a_ht), -1)                         # a_ht.shape = 128x128
            h_t = torch.cat((h_t, a_ht), -1)                         # c_t.shape = 128x512

        elif self.model_type == 'delete_retrieve':
            attr_emb = self.src_embedding(input_attr)
            _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attrlens, attrmask)
            if self.options['bidirectional']:
                a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
                a_ct = torch.cat((a_ct[-1], a_ct[-2]), 1)
            else:
                a_ht = a_ht[-1]
                a_ct = a_ct[-1]

            h_t = torch.cat((h_t, a_ht), -1)
            c_t = torch.cat((c_t, a_ct), -1)
            
        c_t = self.c_bridge(c_t)
        h_t = self.h_bridge(h_t)

        # # # #  # # # #  # #  # # # # # # #  # # end diff

        tgt_emb = self.tgt_embedding(input_tgt)
        # decoder.forward(self, input, hidden, ctx, srcmask, kb=None)
        tgt_outputs, (_, _) = self.decoder(
            tgt_emb,
            (h_t, c_t),
            src_outputs, # TODO: do we need to use `diag_outputs`?
            srcmask)

        tgt_outputs_reshape = tgt_outputs.contiguous().view(
            tgt_outputs.size()[0] * tgt_outputs.size()[1],
            tgt_outputs.size()[2])
        decoder_logit = self.output_projection(tgt_outputs_reshape)
        decoder_logit = decoder_logit.view(
            tgt_outputs.size()[0],
            tgt_outputs.size()[1],
            decoder_logit.size()[1])

        probs = self.softmax(decoder_logit)

        return decoder_logit, probs

    def count_params(self):
        n_params = 0
        for param in self.parameters():
            n_params += np.prod(param.data.cpu().numpy().shape)
        return n_params
        
        
        
