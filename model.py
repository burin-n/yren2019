#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.multiprocessing as mp

# # Models

# model setup
d_model=256
nhead=8
num_encoder_layers=4
num_decoder_layers=4
dim_feedforward=1024
dropout=0.1
mel_channels = 80

# MultiHeadAttention dim convention
# - src: :math:`(S, N, E)`.
# - tgt: :math:`(T, N, E)`.
# - src_mask: :math:`(S, S)`.
# - tgt_mask: :math:`(T, T)`.
# - memory_mask: :math:`(T, S)`.
# - src_key_padding_mask: :math:`(N, S)`.
# - tgt_key_padding_mask: :math:`(N, T)`.
# - memory_key_padding_mask: :math:`(N, S)`.
#
# where S is the source sequence length, T is the target sequence length, N is the
#             batch size, E is the feature number

# ## Positional Encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model, device=torch.device('cpu')):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.FloatTensor(pos_encoding).to(device)


# ## In/Out Module

# spectrogram input (n x feats x times)
# out as (times x n x feats)
class SpeechInput(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim):
        super(SpeechInput, self).__init__()
        self.out_dim = out_dim
        self.Linear1 = nn.Linear(in_dim, h_dim)
        self.Linear2 = nn.Linear(h_dim, out_dim)

    def forward(self, src):
        t_max = src.shape[-1]
        out = torch.empty(src.shape[0], self.out_dim, src.shape[-1], device=src.device)
        for i in range(t_max):
            x = self.Linear1(src[:,:,i])
            x = self.Linear2(x)
            out[:,:,i] = x
        # MultiHead attention Compatibility
        pos_enc = positional_encoding(src.shape[2], self.out_dim, device=src.device).permute(1,0,2)
        return out.permute(2,0,1) + pos_enc


# logits input (n x feats x times)
# logits output (n x feats x times)
class PostNet(nn.Module):
    def __init__(self, in_channels, out_channels=80, n_hidden=256, kernel_size=3):
        super(PostNet, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels, n_hidden, kernel_size, padding=1)
        self.cnn2 = nn.Conv1d(n_hidden, n_hidden, kernel_size, padding=1)
        self.cnn3 = nn.Conv1d(n_hidden, n_hidden, kernel_size, padding=1)
        self.cnn4 = nn.Conv1d(n_hidden, n_hidden, kernel_size, padding=1)
        self.cnn5 = nn.Conv1d(n_hidden, out_channels, kernel_size, padding=1)

    def forward(self, src):
        x = self.cnn1(src)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        return self.cnn5(x)


# logits input (times x n x feats )
# out as (n x feats x times)
class SpeechOutput(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim):
        super(SpeechOutput, self).__init__()
        self.StopLinear = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.MelLinear = nn.Linear(in_dim, out_dim)
        self.PostNet = PostNet(out_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, src):
        # back from MultiHeadAttention
        # n x feats x times
        tgt = src.permute(1,2,0)

        stop = torch.empty(tgt.shape[0], tgt.shape[-1], device=src.device)
        mel_logit = torch.empty(tgt.shape[0], self.out_dim, tgt.shape[-1], device=src.device)

        for i in range(tgt.shape[-1]):
            stop_logit = self.StopLinear(tgt[:,:,i])
            stop[:,i] = self.sigmoid(stop_logit).flatten()
            mel_logit[:,:,i] = self.MelLinear(tgt[:,:,i])
        mel = self.PostNet(mel_logit)

        return mel, stop


# phoneme input (n x times)
# out as (times x n x feats)
class TextInput(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TextInput, self).__init__()
        self.out_dim = out_dim
        self.embed = nn.Embedding(num_embeddings=in_dim, embedding_dim=out_dim)

    def forward(self, src):
        # embed out as (n x times x feats)
        out = self.embed(src).permute(1,0,2)
        pos_enc = positional_encoding(src.shape[-1], self.out_dim, device=src.device).permute(1,0,2)
        return out + pos_enc


# phoneme input (times x n x feats)
# out as (n x feats x times)
class TextOutput(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TextOutput, self).__init__()
        self.out_dim = out_dim
        self.TextLinear = nn.Linear(in_dim, out_dim)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src):
        # Linear take (n x times x feats)
        out = self.TextLinear(src.permute(1,0,2))
        # out = self.softmax(out)
        # permute back to (n x feats x times)
        return out.permute(0,2,1)


# ## Transformer
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(Decoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)


    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        return output