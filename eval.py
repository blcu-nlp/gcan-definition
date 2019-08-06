#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 下午2:40
import argparse
import torch
import json
from src.model import GCA
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils.pipeline import test
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Script to evaluate model')
parser.add_argument(
    "--params", type=str, required=True,
    help="path to saved model params"
)
parser.add_argument(
    "--ckpt", type=str, required=True,
    help="path to saved model weights"
)
parser.add_argument(
    "--datasplit", type=str, required=True,
    help="train, val or test set to evaluate on"
)
args = parser.parse_args()
assert args.datasplit in ["train", "val", "test"], ("--datasplit must be "
                                                    "train, val or test")

with open(args.params, "r") as infile:
    model_params = json.load(infile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCA(model_params).to(device)
model.load_state_dict(torch.load(args.ckpt))

if args.datasplit == "train":
    dataset = DefinitionModelingDataset(
        file=model_params["train_defs"],
        vocab_path=model_params["voc"],
        input_adaptive_vectors_path=model_params["input_adaptive_train"],
        context_vocab_path=model_params["context_voc"],
        ch_vocab_path=model_params["ch_voc"],
        use_seed=model_params["use_seed"],
    )
elif args.datasplit == "val":
    dataset = DefinitionModelingDataset(
        file=model_params["eval_defs"],
        vocab_path=model_params["voc"],
        input_adaptive_vectors_path=model_params["input_adaptive_eval"],
        context_vocab_path=model_params["context_voc"],
        ch_vocab_path=model_params["ch_voc"],
        use_seed=model_params["use_seed"],
    )
elif args.datasplit == "test":
    dataset = DefinitionModelingDataset(
        file=model_params["test_defs"],
        vocab_path=model_params["voc"],
        input_adaptive_vectors_path=model_params["input_adaptive_test"],
        context_vocab_path=model_params["context_voc"],
        ch_vocab_path=model_params["ch_voc"],
        use_seed=model_params["use_seed"],
    )

dataloader = DataLoader(
    dataset,
    model_params["batch_size"],
    collate_fn=DefinitionModelingCollate,
    num_workers=2,
    shuffle=True
)

if __name__ == '__main__':
    print('=========model architecture==========')
    print(model)
    print('=============== end =================')
    loss, ppl = test(model, dataloader, device)
    print("The test set Loss:{0:>6.6},Ppl:{1:>6.6}".format(loss, ppl))
