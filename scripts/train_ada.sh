#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python ../train.py \
--voc ../data/processed/vocab.json \
--train_defs ../data/train.json \
--eval_defs ../data/val.json \
--test_defs ../data/test.json \
--context_voc ../data/processed/context_vocab.json \
--emdim 300 \
--nlayers 2 \
--hidim 300 \
--lr 0.001 \
--epochs 16 \
--batch_size 30 \
--clip 5 \
--random_seed 1 \
--exp_dir ../checkpoints/ \
--cuda \
--w2v_weights ../data/processed/def_embedding \
--use_seed \
--use_input_adaptive \
--ac_re \
--rnn_type LSTM \
--input_adaptive_train ../data/processed/train_ada.npy \
--input_adaptive_eval ../data/processed/val_ada.npy \
--input_adaptive_test ../data/processed/test_ada.npy \

