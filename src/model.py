#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:19-6-26 下午4:28

import torch
import torch.nn as nn
import pickle
import numpy as np
from src.layers import Gated
from src.layers import InputAttention, GCAInteraction, CHGRU


class GCA(nn.Module):
    def __init__(self, params):
        super(GCA, self).__init__()
        self.params = params

        self.embedding = nn.Embedding(self.params["vocab_size"], self.params["emdim"])
        self.embedding_dim = self.params["emdim"]
        self.drop = nn.Dropout(self.params["dropout"])

        if self.params["pretrain"]:
            assert self.params["use_input_adaptive"] + self.params["use_input_attention"] + \
                   self.params["use_gated"] + self.params["use_context_interaction"] + \
                   self.params["use_ch"] == 0, "Pretrain LM without any condition!"
            self.cond_size = None
        else:
            self.use_gated = self.params["use_gated"]
            self.is_ada = self.params["use_input_adaptive"]
            self.is_attn = self.params["use_input_attention"]
            self.use_ci = self.params["use_context_interaction"]
            self.use_ch = self.params["use_ch"]
            self.initial_state = self.params["init_state"]

            self.is_conditioned = self.use_gated
            self.is_conditioned += self.is_ada
            self.is_conditioned += self.is_attn
            self.is_conditioned += self.use_ci
            self.device = torch.device('cuda' if self.params["cuda"] else 'cpu')

            assert self.use_gated + self.is_ada + self.is_attn + self.use_ci <= 1, \
                "Too many conditionings used"
            if not self.is_conditioned and self.params["use_ch"]:
                raise ValueError("Don't use CH conditioning without others")

            self.cond_size = self.params["emdim"]
            # ch
            if self.use_ch:
                self.ch = CHGRU(
                    n_ch_tokens=self.params["n_ch_tokens"],
                    char_lstm_dim=self.params["ch_gru_dim"],
                    ch_emb_size=self.params["ch_emb_size"],
                    device=self.device

                )
                self.cond_size += 150
            # elmo
            self.hs_size = self.cond_size
            if self.is_ada:
                self.embedding_dim += self.params["input_adaptive_dim"]
                self.cond_size = self.params["input_adaptive_dim"]
            if self.is_attn:
                self.input_attention = InputAttention(
                    n_attn_tokens=self.params["n_attn_tokens"],
                    n_attn_embsize=self.params["n_attn_embsize"],
                    n_attn_hid=self.params["n_attn_hid"],
                    attn_dropout=self.params["attn_dropout"],
                    sparse=False
                )
                self.embedding_dim += self.params["n_attn_embsize"]
                self.cond_size = self.params["n_attn_embsize"]
            if self.use_ci:
                self.context_interaction = GCAInteraction(
                    n_attn_tokens=self.params["n_attn_tokens"],
                    n_attn_embsize=self.params["n_attn_embsize"],
                    n_attn_hid=self.params["n_attn_hid"],
                    attn_dropout=self.params["attn_dropout"],
                    freeze=self.params["fix_attn_embeddings"],
                    device=torch.device('cuda' if self.params["cuda"] else 'cpu'),
                    pretrain_w2v=self.params["att_w2v"],
                )
                self.embedding_dim += self.params["n_attn_hid"] * 2
                self.cond_size = self.params["n_attn_hid"] * 2

            if self.initial_state:
                self.hs = nn.Linear(
                    self.hs_size + self.params["n_attn_hid"] * 2, self.params["hidim"]
                )
            if self.params["rnn_type"] in ['LSTM', 'GRU']:
                self.rnn = getattr(nn, self.params["rnn_type"])(self.embedding_dim, self.params["hidim"],
                                                                self.params["nlayers"], dropout=0)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.params["rnn_type"]]
                except KeyError:
                    raise ValueError("""An invalid option for `--model` was supplied,
                                                   options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                self.rnn = nn.RNN(self.embedding_dim, self.params["hidim"], self.params["nlayers"],
                                  nonlinearity=nonlinearity, dropout=0)
            if self.use_gated:
                self.gated = Gated(
                    cond_size=self.cond_size,
                    hidden_size=self.params["hidim"]
                )
        self.decoder = nn.Linear(self.params["hidim"], self.params["vocab_size"])
        self.init_weights()

    def forward(self, inputs, init_hidden, return_h=False):
        seq = inputs["seq"]
        hidden = init_hidden
        seq_emb = self.embedding(seq)
        data = {
            'seq_emb': seq_emb
        }
        if not self.params["pretrain"]:
            word = inputs['word']
            word_emb = self.embedding(word)
            batch_size = seq.size(1)
            if self.is_ada:
                data["input_adaptive"] = inputs['input']
            if self.is_attn:
                context, att = self.input_attention(
                    inputs["context_word"], torch.transpose(inputs["context"], 0, 1)
                )
                data["att"] = att
            if self.use_ci:
                context_attention = self.context_interaction(
                    inputs["context_word"], inputs["context"]
                )
                att = context_attention[0]
                context = context_attention[1]
                data["att"] = att
            if self.use_ch:
                char_embeddings = self.ch(inputs['chars'])
                word_emb = torch.cat(
                    [word_emb, char_embeddings], dim=-1)
            data["word_emb"] = word_emb
            if init_hidden is not None:
                hidden = init_hidden
            elif self.initial_state:
                hidden = self.init_hidden(
                    word_emb, batch_size, self.params["nlayers"], self.params["hidim"], context
                )
        seq_emb = data["seq_emb"]
        batch_size = seq_emb.size(1)
        raw_outputs = []
        lock_outputs = []
        outputs = []
        for time_step in range(seq_emb.size(0)):
            if time_step != 0:
                raw_outputs = []
                lock_outputs = []
            inp_seq = seq_emb[time_step, :, :].view(1, batch_size, -1)
            if self.is_ada:
                inp_seq = torch.cat([torch.unsqueeze(data["input_adaptive"], 0), inp_seq], dim=-1)
            elif self.is_attn or self.use_ci:
                inp_seq = torch.cat([torch.unsqueeze(data["att"], 0), inp_seq], dim=-1)
            outs, hidden = self.rnn(inp_seq, hidden)
            if self.use_gated:
                outs = self.gated(self.params["rnn_type"], data["word_emb"], outs)
            raw_outputs.append(outs)
            lock_outputs.append(outs)
            if time_step == 0:
                rnn_hs = raw_outputs
                dropped_rnn_hs = lock_outputs
            else:
                for i in range(len(rnn_hs)):
                    rnn_hs[i] = torch.cat((rnn_hs[i], raw_outputs[i]), 0)
                    dropped_rnn_hs[i] = torch.cat((dropped_rnn_hs[i], lock_outputs[i]), 0)
            outputs.append(outs)
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
        decoded = self.decoder(self.drop(outputs))
        if return_h:
            return decoded, hidden, rnn_hs, dropped_rnn_hs
        return decoded, hidden

    def init_hidden(self, v, batch_size, num_layers, hidden_dim, feature=None):
        if feature is not None:
            v = torch.cat([v, feature], dim=-1)
        hidden = self.hs(v).view(-1, batch_size, hidden_dim)
        hidden = hidden.expand(num_layers, batch_size, hidden_dim).contiguous()
        if self.params["rnn_type"] == 'LSTM':
            h_c = hidden
            h_h = torch.zeros_like(h_c)
            hidden = (h_h, h_c)
        return hidden

    def init_weights(self):
        if self.params["pretrain"]:
            if self.params["w2v_weights"]:
                with open(self.params["w2v_weights"], 'rb') as infile:
                    pretrain_emb = pickle.load(infile)
                    infile.close()
                self.embedding.weight.data.copy_(
                    torch.from_numpy(pretrain_emb)
                )
            else:
                self.embedding.weight.data.copy_(
                    torch.from_numpy(
                        self.random_embedding(self.params["vocab_size"], self.embedding_dim)
                    )
                )
            self.embedding.weight.requires_grad = not self.params["fix_embeddings"]
            self.init_linear()
            self.init_rnn()
        else:
            if self.params["lm_ckpt"] is not None:
                lm_ckpt_weights = torch.load(self.params["lm_ckpt"])
                self.init_embeddings_from_pretrained(
                    lm_ckpt_weights["embedding.weight"],
                    self.params["fix_embeddings"]
                )
                self.init_linear_from_pretrained(lm_ckpt_weights)
                self.init_rnn_from_pretrained(lm_ckpt_weights)
            else:
                if self.params["w2v_weights"]:
                    with open(self.params["w2v_weights"], 'rb') as infile:
                        pretrain_emb = pickle.load(infile)
                        infile.close()
                    self.embedding.weight.data.copy_(
                        torch.from_numpy(pretrain_emb)
                    )
                else:
                    self.embedding.weight.data.copy_(
                        torch.from_numpy(
                            self.random_embedding(self.params["vocab_size"], self.embedding_dim)
                        )
                    )
                self.embedding.weight.requires_grad = not self.params["fix_embeddings"]
                self.init_linear()
                self.init_rnn()
            if self.use_gated:
                self.gated.init_gated()
            if self.use_ch:
                self.ch.init_ch()
            if self.initial_state:
                nn.init.constant_(self.hs.bias, 0.0)
                nn.init.xavier_normal_(self.hs.weight)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            pretrain_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def init_embeddings_from_pretrained(self, weights, freeze):
        self.embedding = self.embedding.from_pretrained(weights, freeze)

    def init_linear(self):
        nn.init.constant_(self.decoder.bias, 0.0)
        nn.init.xavier_normal_(self.decoder.weight)

    def init_linear_from_pretrained(self, weights):
        # k[8: ] because we need to remove prefix "decoder." because
        # self.decoder.state_dict() is without "decoder." prefix
        self.decoder.load_state_dict(
            {k[8:]: v for k, v in weights.items() if k[:8] == "decoder."}
        )

    def init_rnn(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def init_rnn_from_pretrained(self, weights):
        # k[4:] because we need to remove prefix "rnn." because
        # self.rnn.state_dict() is without "rnn." prefix
        # if use lstm pretrained use lstm.rnn
        correct_state_dict = {
            k[8:]: v for k, v in weights.items() if k[:8] == "gru.rnn."
        }
        # also we need to correctly initialize weight_ih_l0
        # with pretrained weights because it has different size with
        # self.rnn.state_dict(), other weights has correct shapes if
        # hidden sizes have same shape as in the LM pretraining
        if self.is_ada or self.is_attn or self.use_ci:
            w = torch.empty(3 * self.params["hidim"], self.embedding_dim)
            nn.init.xavier_uniform_(w)
            w[:, self.cond_size:] = correct_state_dict["weight_ih_l0"]
            correct_state_dict["weight_ih_l0"] = w
        self.rnn.load_state_dict(correct_state_dict)
