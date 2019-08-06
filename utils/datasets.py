#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 上午10:16
import json
import math
import numpy as np
from . import constants
from torch.utils.data import Dataset


def pad(seq, size, value):
    """padding"""
    if len(seq) < size:
        seq.extend([value] * (size - len(seq)))
    return seq


class Vocabulary:
    """Word/char vocabulary"""

    def __init__(self):
        self.token2id = {
            constants.PAD: constants.PAD_IDX,
            constants.UNK: constants.UNK_IDX,
            constants.BOS: constants.BOS_IDX,
            constants.EOS: constants.EOS_IDX,
        }
        self.id2token = {
            constants.PAD_IDX: constants.PAD,
            constants.UNK_IDX: constants.UNK,
            constants.BOS_IDX: constants.BOS,
            constants.EOS_IDX: constants.EOS,
        }
        self.token_maxlen = -float("inf")

    def encode(self, tok):
        if tok in self.token2id:
            return self.token2id[tok]
        else:
            return constants.UNK_IDX

    def decode(self, idx):
        if idx in self.id2token:
            return self.id2token[idx]
        else:
            raise ValueError("No such idx: {0}".format(idx))

    def encode_seq(self, seq):
        e_seq = []
        for s in seq:
            e_seq.append(self.encode(s))
        return e_seq

    def decode_seq(self, seq):
        d_seq = []
        for i in seq:
            d_seq.append(self.decode(i))
        return d_seq

    def decode_char_seq(self, seq):
        c_seq = []
        for i in seq:
            c_seq.append(
                [constants.BOS_IDX] + \
                self.encode_seq(list(i)) + \
                [constants.EOS_IDX]
            )
        return c_seq

    def add_token(self, tok):
        if tok not in self.token2id:
            self.token2id[tok] = len(self.token2id)
            self.id2token[len(self.id2token)] = tok

    def save(self, path):
        with open(path, "w") as outfile:
            json.dump([self.id2token, self.token_maxlen], outfile, indent=4)
        outfile.close()

    def load(self, path):
        with open(path, 'r') as infile:
            self.id2token, self.token_maxlen = json.load(infile)
            self.id2token = {int(k): v for k, v in self.id2token.items()}
            self.token2id = {}
            for i in self.id2token.keys():
                self.token2id[self.id2token[i]] = i

    def __len__(self):
        return len(self.token2id)


class DefinitionModelingDataset(Dataset):
    def __init__(self, file, vocab_path, input_adaptive_vectors_path=None,
                 context_vocab_path=None, ch_vocab_path=None, use_seed=False, mode='train'):
        with open(file, "r") as infile:
            self.data = json.load(infile)
        self.voc = Vocabulary()
        self.voc.load(vocab_path)
        if context_vocab_path is not None:
            self.context_voc = Vocabulary()
            self.context_voc.load(context_vocab_path)
        self.use_seed = use_seed
        self.mode = mode
        assert self.mode == "train" or "gen", "mode only in train or gen"
        if input_adaptive_vectors_path is not None:
            self.input_adaptive_vectors = np.load(
                input_adaptive_vectors_path
            ).astype(np.float32)
        if ch_vocab_path is not None:
            self.ch_voc = Vocabulary()
            self.ch_voc.load(ch_vocab_path)

    def __getitem__(self, idx):
        inp = {
            "seq": self.voc.encode_seq([constants.BOS] + self.data[idx][1] + [constants.EOS]),
            "target": self.voc.encode_seq(self.data[idx][1] + [constants.EOS]),
            "word": self.voc.encode(self.data[idx][0][0])
        }
        if self.use_seed:
            inp['target'] = inp['seq'][1:]
            inp['seq'] = [self.voc.encode(self.data[idx][0][0])] + inp['seq'][1:]
        if self.mode == "gen":
            inp["seq"] = [inp["seq"][0]]
        if hasattr(self, "context_voc"):
            inp["context_word"] = self.context_voc.encode(self.data[idx][0][0])
            inp["context"] = self.context_voc.encode_seq(self.data[idx][2])
        if hasattr(self, "ch_voc"):
            inp['chars'] = [constants.BOS_IDX] + \
                           self.ch_voc.encode_seq(list(self.data[idx][0])) + \
                           [constants.EOS_IDX]
            # CH_maxlen: +2 because EOS + BOS
            inp["CH_maxlen"] = self.ch_voc.token_maxlen + 2
        if hasattr(self, "input_adaptive_vectors"):
            inp["input_adaptive"] = self.input_adaptive_vectors[idx]
        return inp

    def __len__(self):
        return len(self.data)


def DefinitionModelingCollate(batch):
    batch_word = []
    batch_x = []
    batch_y = []

    is_ch = "chars" in batch[0] and "CH_maxlen" in batch[0]
    is_ada = "input_adaptive" in batch[0]
    is_attn = "context_word" in batch[0] and "context" in batch[0]
    if is_ch:
        batch_ch = []
        CH_maxlen = batch[0]["CH_maxlen"]
    if is_ada:
        batch_input_adaptive = []
    if is_attn:
        batch_context_word = []
        batch_context = []
        context_maxlen = -float("inf")
    definition_lengths = []
    for i in range(len(batch)):
        batch_x.append(batch[i]["seq"])
        batch_y.append(batch[i]["target"])
        batch_word.append(batch[i]["word"])
        if is_ch:
            batch_ch.append(batch[i]["chars"])
        if is_ada:
            batch_input_adaptive.append(batch[i]["input_adaptive"])
        if is_attn:
            batch_context_word.append(batch[i]["context_word"])
            batch_context.append(batch[i]["context"])
            context_maxlen = max(context_maxlen, len(batch_context[-1]))
        definition_lengths.append(len(batch_x[-1]))

    definition_maxlen = max(definition_lengths)

    for i in range(len(batch)):
        batch_x[i] = pad(batch_x[i], definition_maxlen, constants.PAD_IDX)
        batch_y[i] = pad(batch_y[i], definition_maxlen, constants.PAD_IDX)
        if is_attn:
            batch_context[i] = pad(
                batch_context[i], context_maxlen, constants.PAD_IDX
            )
        if is_ch:
            batch_ch[i] = pad(batch_ch[i], CH_maxlen, constants.PAD_IDX)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    batch_word = np.array(batch_word)
    ret_batch = {
        "word": batch_word,
        "seq": batch_x,
        "target": batch_y,
    }
    if is_ch:
        batch_ch = np.array(batch_ch)
        ret_batch["chars"] = batch_ch
    if is_ada:
        batch_input_adaptive = np.array(
            batch_input_adaptive,
            dtype=np.float32
        )
        ret_batch["input_adaptive"] = batch_input_adaptive
    if is_attn:
        batch_context_word = np.array(batch_context_word)
        batch_context = np.array(batch_context)
        ret_batch["context_word"] = batch_context_word
        ret_batch["context"] = batch_context
    return ret_batch


class LanguageModelingDataset(Dataset):
    """LanguageModeling dataset."""

    def __init__(self, file, vocab_path, bptt):
        """
        Args:
            file (string): Path to the file
            vocab_path (string): path to word vocab to use
            bptt (int): length of one sentence
        """
        with open(file, "r") as infile:
            self.data = infile.read().lower().split()
        self.voc = Vocabulary()
        self.voc.load(vocab_path)
        self.bptt = bptt

    def __len__(self):
        return math.ceil(len(self.data) / (self.bptt + 1))

    def __getitem__(self, idx):
        i = idx + self.bptt * idx
        sample = {
            "seq": self.voc.encode_seq(self.data[i: i + self.bptt]),
            "target": self.voc.encode_seq(self.data[i + 1: i + self.bptt + 1]),
        }
        return sample


def LanguageModelingCollate(batch):
    batch_x = []
    batch_y = []
    maxlen = -float("inf")
    for i in range(len(batch)):
        batch_x.append(batch[i]["seq"])
        batch_y.append(batch[i]["target"])
        maxlen = max(maxlen, len(batch[i]["seq"]), len(batch[i]["target"]))

    for i in range(len(batch)):
        batch_x[i] = pad(batch_x[i], maxlen, constants.PAD_IDX)
        batch_y[i] = pad(batch_y[i], maxlen, constants.PAD_IDX)

    ret_batch = {
        "seq": np.array(batch_x),
        "target": np.array(batch_y),
    }
    return ret_batch
