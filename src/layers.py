#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-20 下午2:41
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from torch import nn
from utils import constants

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class Gated(nn.Module):
    """
    Class for Gated conditioning
    """

    def __init__(self, cond_size, hidden_size):
        super(Gated, self).__init__()
        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.in_size = self.cond_size + self.hidden_size
        self.zt_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.hidden_size
        )
        self.zt_bn = nn.BatchNorm1d(self.hidden_size)
        self.rt_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.cond_size
        )
        self.rt_bn = nn.BatchNorm1d(self.cond_size)
        self.ht_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.hidden_size
        )
        self.ht_bn = nn.BatchNorm1d(self.hidden_size)

    def forward(self, rnn_type, v, hidden):
        inp_h = torch.cat(
            [torch.unsqueeze(v, 0), hidden], dim=-1)
        z_t = F.sigmoid(self.zt_linear(inp_h))
        r_t = F.sigmoid(self.rt_linear(inp_h))
        mul = torch.mul(r_t, v)
        hidden_ = torch.cat([mul, hidden], dim=-1)
        hidden_ = F.tanh(self.ht_linear(hidden_))
        hidden = torch.mul((1 - z_t), hidden) + torch.mul(z_t, hidden_)
        return hidden

    def init_gated(self):
        nn.init.constant_(self.zt_linear.bias, 0.0)
        nn.init.xavier_normal_(self.zt_linear.weight)
        nn.init.constant_(self.rt_linear.bias, 0.0)
        nn.init.xavier_normal_(self.rt_linear.weight)
        nn.init.constant_(self.ht_linear.bias, 0.0)
        nn.init.xavier_normal_(self.ht_linear.weight)


class InputAttention(nn.Module):
    """
    Class for Input Attention conditioning
    """

    def __init__(self, n_attn_tokens, n_attn_embsize,
                 n_attn_hid, attn_dropout, sparse=False):
        super(InputAttention, self).__init__()
        self.n_attn_tokens = n_attn_tokens
        self.n_attn_embsize = n_attn_embsize
        self.n_attn_hid = n_attn_hid
        self.attn_dropout = attn_dropout
        self.sparse = sparse

        self.embs = nn.Embedding(
            num_embeddings=self.n_attn_tokens,
            embedding_dim=self.n_attn_embsize,
            padding_idx=constants.PAD_IDX,
            sparse=self.sparse
        )

        self.ann = nn.Sequential(
            nn.Dropout(p=self.attn_dropout),
            nn.Linear(
                in_features=self.n_attn_embsize,
                out_features=self.n_attn_hid
            ),
            nn.Tanh()
        )  # maybe use ReLU or other?

        self.a_linear = nn.Linear(
            in_features=self.n_attn_hid,
            out_features=self.n_attn_embsize
        )

    def forward(self, word, context):
        x_embs = self.embs(word)
        mask = self.get_mask(context)
        att = mask * x_embs
        return mask, att

    def get_mask(self, context):
        context_embs = self.embs(context)
        lengths = (context != constants.PAD_IDX)
        for_sum_mask = lengths.unsqueeze(2).float()
        lengths = lengths.sum(1).float().view(-1, 1)
        logits = self.a_linear(
            (self.ann(context_embs) * for_sum_mask).sum(1) / lengths
        )
        return F.sigmoid(logits)

    def init_attn(self, freeze):
        initrange = 0.5 / self.n_attn_embsize
        with torch.no_grad():
            nn.init.uniform_(self.embs.weight, -initrange, initrange)
            nn.init.xavier_uniform_(self.a_linear.weight)
            nn.init.constant_(self.a_linear.bias, 0)
            nn.init.xavier_uniform_(self.ann[1].weight)
            nn.init.constant_(self.ann[1].bias, 0)
        self.embs.weight.requires_grad = not freeze

    def init_attn_from_pretrained(self, weights, freeze):
        self.load_state_dict(weights)
        self.embs.weight.requires_grad = not freeze


class CHGRU(nn.Module):
    """
    Class for character level GRU
    """

    def __init__(self, n_ch_tokens, char_lstm_dim, ch_emb_size, device, ch_drop=0.1):
        super(CHGRU, self).__init__()
        self.n_ch_tokens = n_ch_tokens
        self.ch_emb_size = ch_emb_size
        self.char_lstm_dim = char_lstm_dim
        self.ch_drop = nn.Dropout(ch_drop)
        self.device = device
        self.char_embedding = nn.Embedding(
            self.n_ch_tokens,
            self.ch_emb_size,
            padding_idx=constants.PAD_IDX
        )

        self.char_gru = nn.GRU(input_size=self.ch_emb_size,
                               hidden_size=self.char_lstm_dim,
                               dropout=0,
                               batch_first=True,
                               bidirectional=True)
        # (batch, seq_len, embed_dim)

        self.trans = nn.Linear(
            self.char_lstm_dim * 2, 300 // 2)

    def forward(self, x):
        x = x.permute(1, 0)
        x_embs = self.char_embedding(x)
        lengths = (x != constants.PAD_IDX).sum(dim=0).detach().cpu()
        char_len_sorted, idx_sort = np.sort(lengths.numpy())[::-1], np.argsort(-lengths.numpy())
        char_len_sorted = torch.from_numpy(char_len_sorted.copy())
        idx_unsort = np.argsort(idx_sort)
        x_embs = x_embs.index_select(1, torch.from_numpy(idx_sort).to(self.device))
        x_embs = pack(x_embs, char_len_sorted, batch_first=False)
        _, hn = self.char_gru(x_embs, None)
        emb = torch.cat((hn[0], hn[1]), 1)  # batch x 2*nhid
        emb = emb.index_select(0, torch.from_numpy(idx_unsort).to(self.device))
        emb = self.ch_drop(self.trans(emb))
        return emb

    def init_ch(self):
        initrange = 0.5 / self.ch_emb_size
        nn.init.uniform_(self.char_embedding.weight, -initrange, initrange)
        for name, param in self.char_gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.constant_(self.trans.bias, 0.0)
        nn.init.xavier_normal_(self.trans.weight)


class GCAInteraction(nn.Module):
    """
    Class for Word-Context Interaction
    """

    def __init__(self, n_attn_tokens, n_attn_embsize, n_attn_hid, attn_dropout,
                 freeze, device, pretrain_w2v=None):
        super(GCAInteraction, self).__init__()
        self.context_tokens = n_attn_tokens
        self.attn_embsize = n_attn_embsize
        self.attn_hid = n_attn_hid
        self.dropout = nn.Dropout(attn_dropout)
        self.device = device
        self.embedding = nn.Embedding(
            self.context_tokens,
            self.attn_embsize,
            padding_idx=constants.PAD_IDX
        )
        self.rnn = nn.GRU(
            input_size=self.attn_embsize,
            hidden_size=self.attn_hid,
            num_layers=1,
            dropout=0,
            bidirectional=True
        )
        self.softmax = nn.Softmax(dim=2)
        self.linear_q = nn.Linear(300, self.attn_embsize)
        self.z1 = nn.Linear(self.attn_hid * 8, self.attn_embsize)
        self.z2 = nn.Linear(self.attn_embsize, self.attn_embsize)
        self.f1 = nn.Linear(self.attn_hid * 8, self.attn_embsize)
        self.f2 = nn.Linear(self.attn_embsize, self.attn_embsize)

        self.init_weights(freeze=freeze, pretrain_w2v=pretrain_w2v)

    def forward(self, word, context, pool_type="max", gate_attention=True):
        word_emb = self.embedding(word)
        context_emb = self.embedding(context)
        lengths = (context != constants.PAD_IDX).sum(dim=0).detach().cpu()

        # Sort by length (keep idx)
        context_len_sorted, idx_sort = np.sort(lengths.numpy())[::-1], np.argsort(-lengths.numpy())
        context_len_sorted = torch.from_numpy(context_len_sorted.copy())
        idx_unsort = np.argsort(idx_sort)
        context_emb = context_emb.index_select(1, torch.from_numpy(idx_sort).to(self.device))
        context_emb = pack(context_emb, context_len_sorted, batch_first=False)
        context_vec, _ = self.rnn(context_emb, None)
        context_vec = unpack(context_vec, batch_first=False)[0]

        # Un-sort by length
        context_vec = context_vec.index_select(1, torch.from_numpy(idx_unsort).to(self.device))
        # Pooling
        if pool_type == "mean":
            lengths = torch.FloatTensor(lengths.numpy().copy()).unsqueeze(1)
            emb = torch.sum(context_vec, 0).squeeze(0)
            if emb.ndimension() == 1:
                emb = emb.unsqueeze(0)
            emb = emb / lengths.expand_as(emb).to(self.device)
        elif pool_type == "max":
            emb = torch.max(context_vec, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        elewise = word_emb * emb
        elediff = torch.abs(word_emb - emb)
        Hw = torch.cat(
            [word_emb, emb, elewise, elediff], dim=-1
        )
        V_ = self.linear_q(word_emb)
        Z_ = F.tanh(self.z1(Hw) + self.z2(V_))
        F_ = F.sigmoid(self.f1(Hw) + self.f2(V_))
        U_ = (1 - F_) * V_ + F_ * Z_
        if gate_attention:
            U_ = self.gated_attention(U_, context_vec)
        return U_, emb

    def init_weights(self, freeze, pretrain_w2v=None):
        if pretrain_w2v is not None:
            with open(pretrain_w2v, 'rb') as infile:
                emb = pickle.load(infile)
                infile.close()
            self.embedding.weight.data.copy_(
                torch.from_numpy(emb)
            )
        else:
            self.embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(self.context_tokens, self.attn_embsize)
                )
            )
        self.embedding.weight.requires_grad = not freeze
        nn.init.constant_(self.linear_q.bias, 0.0)
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.constant_(self.z1.bias, 0.0)
        nn.init.xavier_normal_(self.z1.weight)
        nn.init.constant_(self.z2.bias, 0.0)
        nn.init.xavier_normal_(self.z2.weight)
        nn.init.constant_(self.f1.bias, 0.0)
        nn.init.xavier_normal_(self.f1.weight)
        nn.init.constant_(self.f2.bias, 0.0)
        nn.init.xavier_normal_(self.f2.weight)
        self.init_rnn()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            pretrain_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def init_rnn(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def tmul(self, t1, t2):
        return torch.mul(t1, t2)

    def tcat(self, t1, t2):
        return torch.cat([t1, t2], axis=2)

    def tsum(self, t1, t2):
        return t1 + t2

    def gated_attention(self, w, c):

        word = w.unsqueeze(1)
        context = c.permute(1, 2, 0)
        inter = torch.bmm(word, context)
        alphas_r = F.softmax(inter, dim=2)
        q_rep = torch.bmm(alphas_r, context.permute(0, 2, 1))
        att = self.tmul(word, q_rep)
        return att.squeeze(1)
