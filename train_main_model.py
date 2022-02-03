#!/usr/bin/env python
# coding: utf-8
import csv
import logging
import os
import random as python_random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix

os.environ['TF_DETERMINISTIC_OPS'] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Fix the random seeds to get consistent models
## ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
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
    parser = ArgumentParser(description='Run this script to train the main classification model.')
    parser.add_argument('--dataset', type=str, help='Dataset of a specific disease, e.g. cardiovascular.') 
    parser.add_argument('--databatch', type=str, help='The batch number of random splits, used for determining the file path for loading data and stroing results/models.') 
    parser.add_argument('--vocab-size', type=int, help='The vocabulary size of the dataset, e.g. 1000.') 
    parser.add_argument('--max-len', type=int, help='The maximum sequence length of the dataset, e.g. 100.') 
    A = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")

    # ## 1. Load data
    # Define MIMIC data path 
    data_path = './mimic_data_' + A.dataset + A.databatch + '/' 
    result_path = './experiment_' + A.dataset + A.databatch + '/'

    X_train, y_train, X_val, y_val, validation_reordered = load_train_val_data(data_path)
    logger.info(f"Data loaded from '{data_path}' successfully.")

    # ### Data preprocessing
    # 
    # #### 1.1 Conver all the events into sequence (token) ids

    # Set the vocab size and max sequence length
    vocab_size = A.vocab_size #(max vocab id~=1200 in the training data; ~670 for sepsis; ~870 for ards)
    max_seq_length = A.max_len

    logger.info(f"Data processing using vocab_size={vocab_size}; max_seq_length={max_seq_length}.")

    # Use a text tokenizer to convert events
    tokenizer = Tokenizer(num_words = vocab_size)
    tokenizer.fit_on_texts(X_train)

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_val_sequences = tokenizer.texts_to_sequences(X_val)

    # #### 1.2 Padding converted sequences
    # Pad X_train_sequences and X_val_sequences
    X_train_padded = sequence.pad_sequences(X_train_sequences, maxlen=max_seq_length, padding='post')
    X_val_padded = sequence.pad_sequences(X_val_sequences, maxlen=max_seq_length, padding='post')

    # Train if the model did not exist
    if os.path.exists(result_path+'main_model'):
        logger.info(f"Main classification model already exists, skip the training.")
        quit()

    # ## 2. Train the main LSTM model for survival prediction

    # Train the main classification model
    # Define the early stopping criteria
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    # Define the model structure
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(vocab_size, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    main_model = keras.Model(inputs, outputs)

    # main_model.summary()
    main_model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    reset_seeds()
    model_history = main_model.fit(
        X_train_padded, 
        y_train, 
        epochs=30, 
        batch_size=64, 
        validation_data=(X_val_padded, y_val), 
        callbacks=[early_stopping])

    # Get the predicted target class: if pred > 0.5, then y_pred = 1; else, y_pred = 0
    y_pred = np.array([1 if pred > 0.5 else 0 for pred in main_model.predict(X_val_padded)])

    # Calculate the validation accuracy
    validation_acc = sum(y_pred == y_val)/len(y_val)
    logger.info(f"Main classification trained, with validation accuracy: {validation_acc}.")

    # Get the confusion matrix
    confusion_matrix_df = pd.DataFrame(
            confusion_matrix(y_true=y_val, y_pred=y_pred, labels=[1, 0]),
            index=['True:pos', 'True:neg'], 
            columns=['Pred:pos', 'Pred:neg']
        )
    logger.info(f"Confusion matrix on validation data: \n{confusion_matrix_df}.")
    
    # Counts of positive and negative predictions
    logger.info(f"Value counts of predictions: \n{pd.value_counts(y_pred)}.")

    # ### 2.1 Save the trained classifier for evaluation
    # Create a saved model folder `main_model`.
    main_model.save(result_path + 'main_model')

    # ## 3. Get the negative predictions from LSTM, for counterfactual explanations

    # Get these instances of negative predictions
    X_pred_negative = X_val_padded[y_pred == 0]

    # Use the index of negative predictions to find the original row with the diagnosis codes
    # np.where(y_pred == 0)
    pred_neg_data = validation_reordered.iloc[np.where(y_pred == 0)]
    pred_pos_data = validation_reordered.iloc[np.where(y_pred == 1)]

    # ### Export as the desired input format of the DRG framework
    out_datapath = data_path

    pd.DataFrame(pred_neg_data[0]).to_csv(path_or_buf=out_datapath+'test_neg.txt', index=False, header=False, sep=' ', quoting=csv.QUOTE_NONE, escapechar=' ')
    pd.DataFrame(pred_neg_data['diag']).to_csv(path_or_buf=out_datapath+'test_neg_diag.txt', index=False, header=False, sep=' ', quoting=csv.QUOTE_NONE, escapechar=' ')
    logger.info(f"Test datasets written to '{out_datapath}'.")

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
    validation_reordered = validation.sample(frac=1, random_state=3) # for output the dianosis codes 

    X_val, y_val = validation_reordered[0], validation_reordered['survival']
    return X_train, y_train, X_val, y_val, validation_reordered

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    main()
