#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-13 下午1:35

import argparse
import json
import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from utils.pipeline import train_epoch, test
from src.model import GCA
from utils import constants
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils.datasets import LanguageModelingDataset, LanguageModelingCollate
from utils.util import get_time_dif, get_logger

# Read all arguments and prepare all stuff for training
parser = argparse.ArgumentParser(description='Gated Context-Aware Network')
# Type of training
parser.add_argument(
    '--pretrain', dest='pretrain', action="store_true",
    help='whether to pretrain model on LM dataset or train on definitions'
)
# Common data arguments
parser.add_argument(
    "--voc", type=str, required=True, help="location of vocabulary file"
)
# Definitions data arguments
parser.add_argument(
    '--train_defs', type=str, required=False,
    help="location of txt file with train definitions."
)
parser.add_argument(
    '--eval_defs', type=str, required=False,
    help="location of txt file with metrics definitions."
)
parser.add_argument(
    '--test_defs', type=str, required=False,
    help="location of txt file with test definitions"
)
parser.add_argument(
    '--input_adaptive_train', type=str, required=False,
    help="location of train vectors for InputAdaptive conditioning"
)
parser.add_argument(
    '--input_adaptive_eval', type=str, required=False,
    help="location of eval vectors for InputAdaptive conditioning"
)
parser.add_argument(
    '--input_adaptive_test', type=str, required=False,
    help="location test vectors for InputAdaptive conditioning"
)
parser.add_argument(
    '--context_voc', type=str, required=False,
    help="location of context vocabulary file"
)
parser.add_argument(
    '--ch_voc', type=str, required=False,
    help="location of CH vocabulary file"
)
# LM data arguments
parser.add_argument(
    '--train_lm', type=str, required=False,
    help="location of txt file train LM data"
)
parser.add_argument(
    '--eval_lm', type=str, required=False,
    help="location of txt file eval LM data"
)
parser.add_argument(
    '--test_lm', type=str, required=False,
    help="location of txt file test LM data"
)
parser.add_argument(
    '--bptt', type=int, required=False,
    help="sequence length for BackPropThroughTime in LM pretraining"
)
# Model parameters arguments
parser.add_argument(
    '--rnn_type', type=str, default='GRU',
    help='type of recurrent neural network(LSTM,GRU)'
)
parser.add_argument(
    '--emdim', type=int, default=300,
    help='size of word embeddings'
)
parser.add_argument(
    '--hidim', type=int, default=300,
    help='numbers of hidden units per layer'
)
parser.add_argument(
    '--nlayers', type=int, default=2,
    help='number of recurrent neural network layers'
)
parser.add_argument(
    '--use_seed', action='store_true',
    help='whether to use Seed conditioning or not'
)
parser.add_argument(
    '--use_gated', action='store_true',
    help='whether to use Gated conditioning or not'
)
parser.add_argument(
    '--use_ch', action='store_true',
    help='use character level GRU'
)
parser.add_argument(
    '--ch_emb_size', type=int, required=False,
    help="size of embeddings in CH conditioning"
)
parser.add_argument(
    '--ch_gru_dim', type=int, required=False,
    help="character level GRU hidden size"
)
parser.add_argument(
    '--use_context_interaction', dest="use_context_interaction",
    action="store_true",
    help="whether to use context interaction or not"
)
parser.add_argument(
    '--init_state', action='store_true',
    help='whether to initial decoder hidden state'
)
parser.add_argument(
    '--use_input_adaptive', dest="use_input_adaptive", action="store_true",
    help="whether to use InputAdaptive conditioning or not"
)
parser.add_argument(
    '--use_input_attention', dest="use_input_attention",
    action="store_true",
    help="whether to use InputAttention conditioning or not"
)
parser.add_argument(
    '--n_attn_embsize', type=int, required=False,
    help="size of Attention embeddings"
)
parser.add_argument(
    '--n_attn_hid', type=int, required=False,
    help="size of Attention linear layer"
)
parser.add_argument(
    '--attn_dropout', type=float, required=False,
    help="probability of Attention dropout"
)
# Training and dropout arguments
parser.add_argument(
    '--lr', type=float, default=0.001,
    help='initial learning rate'
)
parser.add_argument(
    "--wdecay", type=float, default=1.2e-06,
    help="decay weight"
)
parser.add_argument(
    '--clip', type=int, default=5,
    help='value to clip norm of gradients to'
)
parser.add_argument(
    '--epochs', type=int, default=40,
    help='upper epoch limit'
)
parser.add_argument(
    '--batch_size', type=int, default=20,
    help='batch size'
)
parser.add_argument(
    '--random_seed', type=int, default=22222,
    help='random seed'
)
parser.add_argument(
    '--dropout', type=float, default=0,
    help='dropout applied to layers (0 = no dropout)'
)
parser.add_argument(
    '--alpha', type=float, default=2,
    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)'
)
parser.add_argument(
    '--beta', type=float, default=1,
    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)'
)
# Utility arguments
parser.add_argument(
    "--exp_dir", type=str, required=True,
    help="where to save all stuff about training"
)
parser.add_argument(
    "--w2v_weights", type=str, required=False,
    help="path to pretrained embeddings to init"
)
parser.add_argument(
    "--att_w2v", type=str, required=False,
    help="path to pretrained embeddings to init"
)
parser.add_argument(
    "--fix_embeddings", action="store_true",
    help="whether to update embedding matrix or not"
)
parser.add_argument(
    "--fix_attn_embeddings", dest="fix_attn_embeddings", action="store_true",
    help="whether to update attention embedding matrix or not"
)
parser.add_argument(
    "--lm_ckpt", type=str, required=False,
    help="path to pretrained language model weights"
)
parser.add_argument(
    '--cuda', action='store_true',
    help='use CUDA'
)
parser.add_argument(
    '--ac_re', action='store_true',
    help='use regularization'
)
# set default args
parser.set_defaults(fix_embeddings=True)
# read args
args = vars(parser.parse_args())

if args["pretrain"]:
    assert args["train_lm"] is not None, "--train_lm is required if --pretrain"
    assert args["eval_lm"] is not None, "--eval_lm is required if --pretrain"
    assert args["test_lm"] is not None, "--test_lm is required if --pretrain"
    assert args["bptt"] is not None, "--bptt is required if --pretrain"

    train_dataset = LanguageModelingDataset(
        file=args["train_lm"],
        vocab_path=args["voc"],
        bptt=args["bptt"],
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args["batch_size"],
        collate_fn=LanguageModelingCollate,
        shuffle=True,
        num_workers=2
    )
    valid_dataset = LanguageModelingDataset(
        file=args["eval_lm"],
        vocab_path=args["voc"],
        bptt=args["bptt"],
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args["batch_size"],
        collate_fn=LanguageModelingCollate,
        shuffle=True,
        num_workers=2
    )
else:
    assert args["train_defs"] is not None, ("--pretrain is False,"
                                            " --train_defs is required")
    assert args["eval_defs"] is not None, ("--pretrain is False,"
                                           " --eval_defs is required")
    assert args["test_defs"] is not None, ("--pretrain is False,"
                                           " --test_defs is required")
    train_dataset = DefinitionModelingDataset(
        file=args["train_defs"],
        vocab_path=args["voc"],
        input_adaptive_vectors_path=args["input_adaptive_train"],
        context_vocab_path=args["context_voc"],
        ch_vocab_path=args["ch_voc"],
        use_seed=args["use_seed"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        collate_fn=DefinitionModelingCollate,
        shuffle=True,
        num_workers=2
    )
    valid_dataset = DefinitionModelingDataset(
        file=args["eval_defs"],
        vocab_path=args["voc"],
        input_adaptive_vectors_path=args["input_adaptive_eval"],
        context_vocab_path=args["context_voc"],
        ch_vocab_path=args["ch_voc"],
        use_seed=args["use_seed"],
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args["batch_size"],
        collate_fn=DefinitionModelingCollate,
        shuffle=True,
        num_workers=2
    )
    if args["use_input_adaptive"]:
        assert args["input_adaptive_train"] is not None, ("--use_input_adaptive "
                                                          "--input_adaptive_train is required")
        assert args["input_adaptive_eval"] is not None, ("--use_input_adaptive "
                                                         "--input_adaptive_eval is required")
        assert args["input_adaptive_test"] is not None, ("--use_input_adaptive "
                                                         "--input_adaptive_test is required")
        args["input_adaptive_dim"] = train_dataset.input_adaptive_vectors.shape[1]

    if args["use_input_attention"] or args["use_context_interaction"]:
        assert args["context_voc"] is not None, ("--use_input_attention or --use_context_interaction"
                                                 "--context_voc is required")
        assert args["n_attn_embsize"] is not None, ("--use_input_attention or --use_context_interaction"
                                                    "--n_attn_embsize is required")
        assert args["n_attn_hid"] is not None, ("--use_input_attention  or --use_context_interaction-"
                                                "-n_attn_hid is required")
        assert args["attn_dropout"] is not None, ("--use_input_attention or --use_context_interaction"
                                                  "--attn_dropout is required")
        args["n_attn_tokens"] = len(train_dataset.context_voc.token2id)

    if args["use_ch"]:
        assert args["ch_voc"] is not None, ("--ch_voc is required "
                                            "if --use_ch")
        assert args["ch_emb_size"] is not None, ("--ch_emb_size is required "
                                                 "if --use_ch")
        assert args["ch_gru_dim"] is not None, ("--ch_gru_dim is "
                                                "required if --use_ch")

        args["n_ch_tokens"] = len(train_dataset.ch_voc.token2id)
        args["ch_maxlen"] = train_dataset.ch_voc.token_maxlen + 2

args["vocab_size"] = len(train_dataset.voc.token2id)
# Set the random seed manually for reproducibility
np.random.seed(args["random_seed"])
torch.manual_seed(args["random_seed"])
if torch.cuda.is_available():
    if not args["cuda"]:
        print('WARNING:You have a CUDA device,so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed(args["random_seed"])
device = torch.device('cuda' if args["cuda"] else 'cpu')


def train():
    print('=========model architecture==========')
    model = GCA(args).to(device)
    print(model)
    print('=============== end =================')
    loss_fn = nn.CrossEntropyLoss(ignore_index=constants.PAD_IDX)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, args["lr"], weight_decay=args["wdecay"])
    print('Training and evaluating...')
    logger.info(
        "Batch Size: %d, Dropout: %.2f" % (
            args["batch_size"], args["dropout"]
        )
    )
    if args["ac_re"]:
        logger.info(
            "Alpha L2 Regularization: %d, Beta Slowness Regularization: %d" % (
                args["alpha"], args["beta"],
            )
        )
    start_time = time.time()
    if not os.path.exists(args["exp_dir"]):
        os.makedirs(args["exp_dir"])
    best_ppl = 9999999
    last_improved = 0
    require_improvement = 5
    with open(args["exp_dir"] + "params.json", "w") as outfile:
        json.dump(args, outfile, indent=4)
    for epoch in range(args["epochs"]):
        print('=============== Epoch %d=================' % (epoch + 1))
        logger.info("Optimizer: %s" % (optimizer))
        train_loss, train_ppl = train_epoch(
            epoch, train_dataloader, model, loss_fn, optimizer, device, args["clip"], args["ac_re"], args["alpha"],
            args["beta"]
        )
        valid_loss, valid_ppl = test(
            model, valid_dataloader, device
        )
        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            last_improved = epoch
            torch.save(model.state_dict(), args["exp_dir"] +
                       'model_params_%s_min_ppl.pkl' % (epoch + 1)
                       )
            improved_str = '*'
        else:
            improved_str = ''
        time_dif = get_time_dif(start_time)
        msg = 'Epoch: {0:>6},Train Loss: {1:>6.6}, Train Ppl: {2:>6.6},' \
              + ' Val loss: {3:>6.6}, Val Ppl: {4:>6.6},Time:{5} {6}'
        print(msg.format(epoch + 1, train_loss, train_ppl, valid_loss, valid_ppl, time_dif, improved_str))
        if epoch - last_improved > require_improvement:
            print("No optimization for a long time, auto-stopping...")
            break
    return 1


if __name__ == "__main__":
    if args["pretrain"]:
        logger = get_logger("Pretrain Language Model")
    else:
        logger = get_logger("Gated Context-Aware Network")
        logger.info("Definiton Vocab Size: %d" % args["vocab_size"])
        logger.info("Use Seed: %s" % args["use_seed"])
        if args["use_gated"]:
            logger.info("Use Gated: True")
        if args["use_input_adaptive"]:
            logger.info("Use Input Adaptive: True")
        if args["use_input_attention"] or args["use_context_interaction"]:
            if args["use_input_attention"]:
                logger.info("Use Input Attention: True")
            elif args["use_context_interaction"]:
                logger.info("Use Context Interaction: True")
            logger.info("Context Vocab Size: %d" % args["n_attn_tokens"])
        if args["lm_ckpt"]:
            logger.info("Use Pretrained Language Model: True")
        logger.info("Use Char Embedding: %s" % args["use_ch"])

    train()
