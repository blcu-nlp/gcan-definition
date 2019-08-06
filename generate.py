#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 上午10:18
import torch
import json
import argparse
from src.model import GCA
from torch.utils.data import DataLoader
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils.pipeline import generate
from utils.datasets import Vocabulary

parser = argparse.ArgumentParser(description='Script to generate using model')
parser.add_argument(
    "--params", type=str, required=True,
    help="path to saved model params"
)
parser.add_argument(
    "--ckpt", type=str, required=True,
    help="path to saved model weights"
)
parser.add_argument(
    "--strategy", type=str, default="Greedy",
    help="generate strategy(Greedy,Multinomial or Beam)."
)
parser.add_argument(
    "--tau", type=float, required=False,
    help="temperature to use in sampling"
)
parser.add_argument(
    "--length", type=int, required=True,
    help="maximum length of generated samples"
)
parser.add_argument(
    "--gen_dir", type=str, default="gen/",
    help="where to save generate file"
)
parser.add_argument(
    "--gen_name", type=str, default="gen.txt",
    help="generate file name"
)
args = parser.parse_args()
assert args.strategy in ["Greedy", "Multinomial"], ("--type must be Greedy,Multinomial")
if args.strategy == "Multinomial":
    assert args.tau is not None, ("--strategy is Multinomial,"
                                  " --tau is required")
with open(args.params, "r") as infile:
    model_params = json.load(infile)
dataset = DefinitionModelingDataset(
    file=model_params["test_defs"],
    vocab_path=model_params["voc"],
    input_adaptive_vectors_path=model_params["input_adaptive_test"],
    context_vocab_path=model_params["context_voc"],
    ch_vocab_path=model_params["ch_voc"],
    use_seed=model_params["use_seed"],
    mode="gen"
)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=DefinitionModelingCollate,
    num_workers=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCA(model_params).to(device)
model.load_state_dict(torch.load(args.ckpt))
voc = Vocabulary()
voc.load(model_params["voc"])
context_voc = Vocabulary()
context_voc.load(model_params["context_voc"])

if __name__ == "__main__":
    print('=========model architecture==========')
    print(model)
    print('=============== end =================')
    generate(
        model, dataloader, voc, context_voc, tau=args.tau, length=args.length, device=device, save=args.gen_dir,
        strategy=args.strategy
    )
    print("Finished!")
