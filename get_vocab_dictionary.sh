#!/bin/bash
# replace for each cohort: mimic_data_cardiovascular OR mimic_data_sepsis OR mimic_data_ards

# idx:1 
# for patient representation
cat mimic_data_ards1/train_pos.txt mimic_data_ards1/train_neg.txt > mimic_data_ards1/train_all.txt
python3 tools/make_vocab.py mimic_data_ards1/train_all.txt 3000 > mimic_data_ards1/vocab.txt
python3 tools/make_ngram_attribute_vocab.py mimic_data_ards1/vocab.txt mimic_data_ards1/train_neg.txt mimic_data_ards1/train_pos.txt 15 > mimic_data_ards1/ngram_attribute_vocab.txt

# for diagnosis codes
cat mimic_data_ards1/train_pos_diag.txt mimic_data_ards1/train_neg_diag.txt > mimic_data_ards1/train_all_diag.txt
python3 tools/make_vocab.py mimic_data_ards1/train_all_diag.txt 5000 > mimic_data_ards1/vocab_diag.txt

# idx:2
# for patient representation
cat mimic_data_ards2/train_pos.txt mimic_data_ards2/train_neg.txt > mimic_data_ards2/train_all.txt
python3 tools/make_vocab.py mimic_data_ards2/train_all.txt 3000 > mimic_data_ards2/vocab.txt
python3 tools/make_ngram_attribute_vocab.py mimic_data_ards2/vocab.txt mimic_data_ards2/train_neg.txt mimic_data_ards2/train_pos.txt 15 > mimic_data_ards2/ngram_attribute_vocab.txt

# for diagnosis codes
cat mimic_data_ards2/train_pos_diag.txt mimic_data_ards2/train_neg_diag.txt > mimic_data_ards2/train_all_diag.txt
python3 tools/make_vocab.py mimic_data_ards2/train_all_diag.txt 5000 > mimic_data_ards2/vocab_diag.txt

# idx:3
# for patient representation
cat mimic_data_ards3/train_pos.txt mimic_data_ards3/train_neg.txt > mimic_data_ards3/train_all.txt
python3 tools/make_vocab.py mimic_data_ards3/train_all.txt 3000 > mimic_data_ards3/vocab.txt
python3 tools/make_ngram_attribute_vocab.py mimic_data_ards3/vocab.txt mimic_data_ards3/train_neg.txt mimic_data_ards3/train_pos.txt 15 > mimic_data_ards3/ngram_attribute_vocab.txt

# for diagnosis codes
cat mimic_data_ards3/train_pos_diag.txt mimic_data_ards3/train_neg_diag.txt > mimic_data_ards3/train_all_diag.txt
python3 tools/make_vocab.py mimic_data_ards3/train_all_diag.txt 5000 > mimic_data_ards3/vocab_diag.txt

# idx:4
# for patient representation
cat mimic_data_ards4/train_pos.txt mimic_data_ards4/train_neg.txt > mimic_data_ards4/train_all.txt
python3 tools/make_vocab.py mimic_data_ards4/train_all.txt 3000 > mimic_data_ards4/vocab.txt
python3 tools/make_ngram_attribute_vocab.py mimic_data_ards4/vocab.txt mimic_data_ards4/train_neg.txt mimic_data_ards4/train_pos.txt 15 > mimic_data_ards4/ngram_attribute_vocab.txt

# for diagnosis codes
cat mimic_data_ards4/train_pos_diag.txt mimic_data_ards4/train_neg_diag.txt > mimic_data_ards4/train_all_diag.txt
python3 tools/make_vocab.py mimic_data_ards4/train_all_diag.txt 5000 > mimic_data_ards4/vocab_diag.txt


# idx:5
# for patient representation
cat mimic_data_ards5/train_pos.txt mimic_data_ards5/train_neg.txt > mimic_data_ards5/train_all.txt
python3 tools/make_vocab.py mimic_data_ards5/train_all.txt 3000 > mimic_data_ards5/vocab.txt
python3 tools/make_ngram_attribute_vocab.py mimic_data_ards5/vocab.txt mimic_data_ards5/train_neg.txt mimic_data_ards5/train_pos.txt 15 > mimic_data_ards5/ngram_attribute_vocab.txt

# for diagnosis codes
cat mimic_data_ards5/train_pos_diag.txt mimic_data_ards5/train_neg_diag.txt > mimic_data_ards5/train_all_diag.txt
python3 tools/make_vocab.py mimic_data_ards5/train_all_diag.txt 5000 > mimic_data_ards5/vocab_diag.txt


