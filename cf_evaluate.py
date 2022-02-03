#!/usr/bin/env python
# coding: utf-8
import csv
import logging
import os
import random as python_random
from argparse import ArgumentParser

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboardX import writer
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

os.environ['TF_DETERMINISTIC_OPS'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# # Fix the random seeds to get consistent models
# ## ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# seed_value = 3
# os.environ['PYTHONHASHSEED']=str(seed_value)
# # The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
# np.random.seed(seed_value)
# # The below is necessary for starting core Python generated random numbers in a well-defined state.
# python_random.seed(seed_value)
# # The below set_seed() will make random number generation
# tf.random.set_seed(seed_value)
# # configure a new global `tensorflow` session
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)

def reset_seeds(seed_value=3):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value) 
    python_random.seed(seed_value)
    tf.random.set_seed(seed_value)

reset_seeds() 

def main():
    parser = ArgumentParser(description='Run this script to evaluate all the generated counterfactuals.')
    parser.add_argument('--dataset', type=str, help='Dataset of a specific disease, e.g. cardiovascular.') 
    parser.add_argument('--databatch', type=str, help='The batch number of random splits, used for determining the file path for loading data and stroing results/models.') 
    parser.add_argument('--vocab-size', type=int, help='The vocabulary size of the dataset, e.g. 1000.') 
    parser.add_argument('--max-len', type=int, help='The maximum sequence length of the dataset, e.g. 100.') 
    parser.add_argument('--output', type=str, help='Output file name.')
    A = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")

    # Define MIMIC data path 
    data_path = './mimic_data_' + A.dataset + A.databatch + '/' 
    result_path = './experiment_' + A.dataset + A.databatch + '/'

    result_writer = ResultWriter(file_name=result_path+A.output, dataset_name=A.dataset+A.databatch)
    result_writer.write_head()
    
    # ## 1. Load data    
    X_train, y_train, X_val, y_val = load_train_val_data(data_path)
    logger.info(f"Data loaded from {data_path} successfully.")
    
    # pre-processing
    vocab_size = A.vocab_size #(max vocab id~=1200 in the training data; ~670 for sepsis; ~870 for ards)
    max_seq_length = A.max_len

    tokenizer = Tokenizer(num_words = vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_val_sequences = tokenizer.texts_to_sequences(X_val)
    X_train_padded = sequence.pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post')
    X_val_padded = sequence.pad_sequences(X_val_sequences, maxlen=max_seq_length, padding='post')

    # Load trained model
    main_model = keras.models.load_model(result_path + 'main_model')
    logger.info(f"Model loaded from {result_path} successfully.")

    # Get negative predictions (test samples)
    y_pred = np.array([1 if pred > 0.5 else 0 for pred in main_model.predict(X_val_padded)])
    X_pred_negative = X_val_padded[y_pred == 0]
    original_event_sequences = tokenizer.sequences_to_texts(X_pred_negative)

    # ### Apply 1NN baseline method to modify the negatively predicted instances
    # Fit an unsupervised 1NN with all the positive seuquences, using 'hamming' distance
    nn_model = NearestNeighbors(1, metric='hamming')
    target_label = 1 
    X_target_label = X_train_padded[y_train == target_label] # training using the true labels
    nn_model.fit(X_target_label)

    # Find the closest neighbor (positive sequence) with the minimum 'hamming' distance, take it as a counterfactual
    closest = nn_model.kneighbors(X_pred_negative, return_distance=False)
    trans_results_nn = X_target_label[closest[:, 0]]
    # Rename 'trans_results_nn' to 'X_test_padded3' for result comparison
    X_test_padded3 = trans_results_nn
    
    # Fit the model for novelty detection (novelty=True), in order to get LOF score
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    # clf.fit(X_train_padded)
    clf.fit(X_target_label) # update: use the target class to train, instead of all

    # Get the LOF score for leave-out validation data
    y_pred_val = clf.predict(X_val_padded)

    n_error_val = y_pred_val[y_pred_val == -1].size

    validation_size = X_val_padded.shape[0]
    outlier_score_val = n_error_val/validation_size
    outlier_score_val

    # Here, we need to use the inference script from the DRG framework (instructions in the README file) to modify those 110 negative predictions into positive instances. After that, we import the transformed results as below.
    for model_idx in [1, 2, 3, 4]:
        model_idx = str(model_idx) # loop through all the folders of generated CFs

        # ### 3.1 DeleteOnly model results
        # Load the transformed data
        results = result_path + 'working_dir_' + A.dataset + A.databatch + '_delete' + model_idx + '/preds'
        trans_results_delete = pd.read_csv(results, header=None)

        X_test_sequences = tokenizer.texts_to_sequences(trans_results_delete[0])
        X_test_padded = sequence.pad_sequences(X_test_sequences, maxlen=max_seq_length, padding='post')

        # ### 3.2 DeleteAndRetrieve model results
        # Load the transformed data
        results2 = result_path + 'working_dir_' + A.dataset + A.databatch + '_delete_ret' + model_idx + '/preds'
        delete_generate_results = pd.read_csv(results2, header=None)

        X_test_sequences2 = tokenizer.texts_to_sequences(delete_generate_results[0])
        X_test_padded2 = sequence.pad_sequences(X_test_sequences2, maxlen=max_seq_length, padding='post')

        # ### 3.4 Convert transformed results to event sequence format
        # Convert transformed sequences back to the form of original event sequences
        trans_event_sequences1 = tokenizer.sequences_to_texts(X_test_padded)
        trans_event_sequences2 = tokenizer.sequences_to_texts(X_test_padded2)
        trans_event_sequences3 = tokenizer.sequences_to_texts(X_test_padded3)

        # ## 4. Results comparison
        # ### 4.1 Comparison between fraction of valid CFs (i.e. successfully generated counterfactuals)
        # Get the total counts 
        test_size = X_pred_negative.shape[0]

        # Fraction of valid transformed sequences, for DeleteOnly
        validity1 = np.sum(main_model.predict(X_test_padded) > 0.5)/test_size
        # For DeleteAndRetrieve
        validity2 = np.sum(main_model.predict(X_test_padded2) > 0.5)/test_size
        # For 1NN modification
        validity3 = np.sum(main_model.predict(X_test_padded3) > 0.5)/test_size

        # ### 4.2 Local outlier factor (LOF score)
        # Get the LOF score for DeleteOnly results
        y_pred_test = clf.predict(X_test_padded)
        n_error_test = y_pred_test[y_pred_test == -1].size
        lof_score1 = n_error_test / test_size

        # Get the outlier score for DeleteAndRetrieve results
        y_pred_test2 = clf.predict(X_test_padded2)
        n_error_test2 = y_pred_test2[y_pred_test2 == -1].size
        lof_score2 = n_error_test2 / test_size

        # Outlier score for 1NN baseline method
        y_pred_test3 = clf.predict(X_test_padded3)
        n_error_test3 = y_pred_test3[y_pred_test3 == -1].size
        lof_score3 = n_error_test3 / test_size

        # ### 4.3 BLEU-4 score (cumulative 4-gram BLEU score) 
        pairwise_bleu = get_pairwise_bleu(original_event_sequences, trans_event_sequences1)
        avg_bleu = sum(pairwise_bleu)/test_size

        pairwise_bleu2 = get_pairwise_bleu(original_event_sequences, trans_event_sequences2)
        avg_bleu2 = sum(pairwise_bleu2)/test_size

        pairwise_bleu3 = get_pairwise_bleu(original_event_sequences, trans_event_sequences3)
        avg_bleu3 = sum(pairwise_bleu3)/test_size

        # ### 4.5 Edit distance (Levenshtein)
        edit_dist1 = get_edit_distance(X_test_padded, X_pred_negative)
        edit_dist2 = get_edit_distance(X_test_padded2, X_pred_negative)
        edit_dist3 = get_edit_distance(X_test_padded3, X_pred_negative)

        if model_idx == '1':
            result_writer.write_result(method_name='1-NN', validity=validity3, lof_score=lof_score3, bleu_4=avg_bleu3, edit_distance=edit_dist3, lof_ref=outlier_score_val)
        
        result_writer.write_result(method_name='DeleteOnly'+model_idx, validity=validity1, lof_score=lof_score1, bleu_4=avg_bleu, edit_distance=edit_dist1)
        result_writer.write_result(method_name='DeleteRetrieve'+model_idx, validity=validity2, lof_score=lof_score2, bleu_4=avg_bleu2, edit_distance=edit_dist2)

    logger.info(f"Results written to {result_path}.")
    logger.info("Done.")

def load_train_val_data(data_path):
    # Load training data
    train_pos = pd.read_csv(data_path+'train_pos.txt', header=None)
    train_neg = pd.read_csv(data_path+'train_neg.txt', header=None)
    # Add target class label
    train_pos['survival'] = [1 for i in range(train_pos.shape[0])]
    train_neg['survival'] = [0 for i in range(train_neg.shape[0])]

    # Load diagnosis data
    train_pos['diag'] = pd.read_csv(data_path+'train_pos_diag.txt', header=None)
    train_neg['diag'] = pd.read_csv(data_path+'train_neg_diag.txt', header=None)

    # Concat into one data frame; and reorder it
    train = pd.concat([train_pos, train_neg]).reset_index()
    train_reordered = train.sample(frac=1, random_state=3)

    X_train, y_train = train_reordered[0], train_reordered['survival']

    # Load validation data
    validation_pos = pd.read_csv(data_path+'validation_pos.txt', header=None)
    validation_neg = pd.read_csv(data_path+'validation_neg.txt', header=None)
    # Add target class
    validation_pos['survival'] = [1 for i in range(validation_pos.shape[0])]
    validation_neg['survival'] = [0 for i in range(validation_neg.shape[0])]

    # Load diagnosis data
    validation_pos['diag'] = pd.read_csv(data_path+'validation_pos_diag.txt', header=None)
    validation_neg['diag'] = pd.read_csv(data_path+'validation_neg_diag.txt', header=None)

    validation = pd.concat([validation_pos, validation_neg]).reset_index()
    validation_reordered = validation.sample(frac=1, random_state=3)

    X_val, y_val = validation_reordered[0], validation_reordered['survival']
    return X_train, y_train, X_val, y_val

# BLEU-4 score
# Define smoothing function
chencherry = SmoothingFunction()
# Define a function to get pairwise BLEU scores
def get_pairwise_bleu(original, transformed):
    # 'weights=[0.25, 0.25, 0.25, 0.25]' means that calculate 4-gram BLEU scores cumulatively
    results = [sentence_bleu(
        references=[pair[0].split()], 
        hypothesis=pair[1].split(), 
        weights=[0.25, 0.25, 0.25, 0.25], 
        smoothing_function=chencherry.method1) 
        for pair in zip(original, transformed)]
    
    return results

# Edit distance
def get_edit_distance(original, transformed):
    edit_distance_pair = [editdistance.eval(o, t) for o, t in zip(original.tolist(), transformed.tolist())]
    edit_score = np.mean(edit_distance_pair)
    
    return round(edit_score, 4)

class ResultWriter:
    def __init__(self, file_name, dataset_name):
        self.file_name = file_name
        self.dataset_name = dataset_name
        
    def write_head(self):
        # write the head in csv file
        with open(self.file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "method",
                "validity",
                "lof",
                "bleu",
                "edit",
                "lof_ref"
            ])
        
    def write_result(self, method_name, validity, lof_score, bleu_4, edit_distance, lof_ref=0):        
        with open(self.file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.dataset_name,
                method_name,
                validity,
                lof_score,
                bleu_4,
                edit_distance,
                lof_ref
            ])

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    main()
