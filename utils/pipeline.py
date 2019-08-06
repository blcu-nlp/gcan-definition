#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-30 上午10:22

import torch
import numpy as np
import os
import torch.nn.functional as F
from utils import constants
from tqdm import tqdm
from torch import nn


def train_epoch(epoch, dataloader, model, loss_fn, optimizer, device, clip_to, ac_re=False, alpha=0, beta=0):
    """
    Function for training the model one epoch
        epoch - training epoch
        dataloader - DefinitionModeling dataloader
        model - DefinitionModelingModel
        loss_fn - loss function to use
        optimizer - optimizer to use (usually Adam)
        device - cuda/cpu
        clip_to - value to clip gradients
        ac_re - regularization
        alpha - regularization weight
        beta - regularization weight
    """
    # switch model to training mode
    model.train()
    # train
    loss_epoch = []
    for batch, inp in enumerate(tqdm(dataloader, desc='Epoch: %03d' % (epoch + 1), leave=False)):
        data = {
            'seq': torch.t(torch.from_numpy(
                inp['seq'])
            ).long().to(device),
        }
        if not model.params["pretrain"]:
            data["word"] = torch.from_numpy(
                inp['word']
            ).to(device)
            if model.is_ada:
                data["input"] = torch.from_numpy(
                    inp["input_adaptive"]
                ).to(device)
            if model.is_attn or model.use_ci:
                data["context_word"] = torch.from_numpy(
                    inp['context_word']
                ).to(device)
                data["context"] = torch.t(torch.from_numpy(
                    inp["context"]
                )).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
        targets = torch.t(torch.from_numpy(inp['target'])).to(device)
        if ac_re:
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, None, return_h=ac_re)
        else:
            output, hidden = model(data, None, return_h=ac_re)
        loss = loss_fn(output, targets.contiguous().view(-1))
        optimizer.zero_grad()
        if ac_re:
            # Activiation Regularization
            loss = loss + sum(
                alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()
        # `clip_grad_norm`
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_to)
        optimizer.step()
        loss_epoch.append(loss.item())
    train_loss = np.mean(loss_epoch)
    train_ppl = np.exp(train_loss)
    return train_loss, train_ppl


def test(model, dataloader, device):
    """
    Function for testing the model on dataloader
        dataloader - DefinitionModeling dataloader
        model - DefinitionModelingModel
        device - cuda/cpu
    """
    # switch model to evaluation mode
    model.eval()
    # eval
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        total_loss = []
        for inp in tqdm(dataloader, desc='Evaluate model in definitions.', leave=False):
            data = {
                'seq': torch.t(torch.from_numpy(
                    inp['seq'])
                ).long().to(device),
            }
            if not model.params["pretrain"]:
                data["word"] = torch.from_numpy(
                    inp['word']
                ).to(device)
                if model.is_ada:
                    data["input"] = torch.from_numpy(
                        inp["input_adaptive"]
                    ).to(device)
                if model.is_attn or model.use_ci:
                    data["context_word"] = torch.from_numpy(
                        inp['context_word']
                    ).to(device)
                    data["context"] = torch.t(torch.from_numpy(
                        inp["context"]
                    )).to(device)
                if model.use_ch:
                    data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
            targets = torch.t(torch.from_numpy(inp['target'])).to(device)
            output, hidden = model(data, None)
            loss = loss_fn(output, targets.contiguous().view(-1))
            total_loss.append(loss.item())
    return np.mean(total_loss), np.exp(np.mean(total_loss))


def generate(model, dataloader, voc, context_voc, tau, length, device, save, strategy="Greedy"):
    """
    model - DefinitionModelingModel
    dataloader - DefinitionModeling dataloader
    voc - model Vocabulary
    context_voc - Vocabulary for InputAttention conditioning
    tau - temperature to generate with
    length - length of the sample
    device - cuda/cpu
    save - save dir
    strategy - generate strategy
    """
    model.eval()
    if not os.path.exists(save):
        os.makedirs(save)
    defsave = open(
        save + "gen.txt",
        "w"
    )
    for inp in tqdm(dataloader, desc='Generate definitions.', leave=False):
        data = {
            'seq': torch.t(torch.from_numpy(
                inp['seq'])
            ).long().to(device),
        }
        if not model.params["pretrain"]:
            data["word"] = torch.from_numpy(
                inp['word']
            ).to(device)
            if model.is_ada:
                data["input"] = torch.from_numpy(
                    inp["input_adaptive"]
                ).to(device)
            if model.is_attn or model.use_ci:
                data["context_word"] = torch.from_numpy(
                    inp['context_word']
                ).to(device)
                data["context"] = torch.t(torch.from_numpy(
                    inp["context"]
                )).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
        def_word = voc.id2token[inp['word'][0]]
        context = context_voc.decode_seq(inp["context"][0])
        defsave.write("Word:" + def_word + "\n")
        defsave.write("Context:")
        defsave.write(" ".join(context) + "\n")
        defsave.write("Definition:")
        hidden = None
        ret = []
        with torch.no_grad():
            for i in range(length):
                output, hidden = model(data, hidden)
                if strategy == "Greedy":
                    word_weights = output.squeeze().div(1).exp().cpu()
                    word_idx = torch.argmax(word_weights)
                elif strategy == "Multinomial":
                    word_weights = F.softmax(
                        output / tau, dim=1
                    ).multinomial(num_samples=1)
                    word_idx = word_weights[0][0]
                if word_idx == constants.EOS_IDX:
                    break
                else:
                    data['seq'].fill_(word_idx)
                    word = word_idx.item()
                    ret.append(voc.decode(word))
            output = " ".join(map(str, ret))
            defsave.write(output + "\n")
