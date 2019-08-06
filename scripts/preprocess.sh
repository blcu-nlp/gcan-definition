#!/usr/bin/env bash
dir="../data/processed"

if [ ! -d $dir  ];then
  mkdir $dir
fi

python ../pre_vocab.py \
--defs ../data/train.json ../data/val.json ../data/test.json \
--save ../data/processed/vocab.json \
--save_chars ../data/processed/char_vocab.json \
--save_context ../data/processed/context_vocab.json \

python ../pre_wordemb.py \
--vocab ../data/processed/vocab.json \
--w2v ../data/word2vec/GoogleNews-vectors-negative300.bin \
--save ../data/processed/def_embedding \

python ../pre_wordemb.py \
--vocab ../data/processed/context_vocab.json \
--w2v ../data/word2vec/GoogleNews-vectors-negative300.bin \
--save ../data/processed/context_embedding \

echo "Preparing vocab and embedding finished! "