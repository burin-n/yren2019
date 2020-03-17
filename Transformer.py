#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from torchvision import transforms
import librosa
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from g2p_en import G2p
import re
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.multiprocessing as mp

import copy
import time
import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datadir = 'LJSpeech-1.1'
wav_dirs = os.path.join(datadir, 'wavs')

n_val = 300
n_test = 300
n_training = len(os.listdir(wav_dirs)) - n_val - n_test

n_paired = 200
batch_size = 8

Constants = {
    "PAD" : 0,
    "UNK" : 1,
    "BOS" : 2,
    "EOS" : 3,

    "PAD_WORD" : '<pad>',
    "UNK_WORD" : '<unk>',
    "BOS_WORD" : '<s>',
    "EOS_WORD" : '</s>'
}

n_phones = 75

# model setup
d_model=256
nhead=8
num_encoder_layers=4
num_decoder_layers=4
dim_feedforward=1024
dropout=0.1
mel_channels = 80


class LJDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, n_mels=80, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transcripts = pd.read_csv(csv_file, sep='|', header=None,\
                                names=['ID', 'Transcription', 'Normalized Transcription']).fillna(-1)
        self.root_dir = root_dir
        self.transform = transform
        self.n_mels = n_mels


    def __len__(self):
        return len(self.transcripts)


    def __getitem__(self, idx):
        wav_path = os.path.join(self.root_dir,
                                self.transcripts.iloc[idx, 0]) + '.wav'

        # speech part
        y, sr = librosa.load(wav_path)

        # text part
        txt = self.transcripts.iloc[idx, 1]
        norm_txt = self.transcripts.iloc[idx, 2]
        transcript = txt if norm_txt == -1 else norm_txt

        sample = {'spec': (y, sr), 'text': transcript}

        if self.transform:
            sample = self.transform(sample)

        return sample


class speech_transform(object):
    def __init__(self, n_mels):
        self.n_mels = n_mels
        self.start_token = np.zeros((self.n_mels,1), dtype=np.float)

    def __call__(self, sample):
        y, sr = sample['spec']
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        S = np.log(S)

        sample['spec'] = np.concatenate((self.start_token, S), axis=1)
        return sample


class text_transform(object):
    # normalize -> g2p
    def __init__(self):
        self.g2p = G2p()

        extra_syms = [' ']
        self.ph2id = self.g2p.p2idx.copy()
        self.id2ph = self.g2p.idx2p.copy()
        for sym in extra_syms:
            self.ph2id[sym] = len(self.id2ph)
            self.id2ph[len(self.id2ph)] = sym

        self.start_token = '<s>'
        self.end_token = '</s>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'


    def __call__(self, sample):
        # text = self.normalize(sample['text'])
        # phones = [self.start_token] + self.g2p(text) + [self.end_token]
        # phones = self.g2p(text) + [self.end_token]
        # phones = self.encode(phones)
        # sample['text'] = phones
        sample['text'] = self.encode(sample['text'].split(' '))
        return sample


    def encode(self, inputs):
        return np.array([self.ph2id[x] for x in inputs])


    def decode(self, inputs):
        return np.array([self.id2ph[x] for x in inputs])


    def normalize(self, text):
        return re.sub('[^a-z0-9\ ]', '', text.lower())


class reverse_transform(object):
    def __init__(self):
        self.dim=-1

    def __call__(self, sample):
        return { **sample,
            'spec_r' : self.reverse(sample['spec']),
            'text_r' : self.reverse(sample['text'])
        }

    # def reverse(self, ndarray):
    #     return np.flip(ndarray, self.dim)


    def reverse(self, ndarray, pad_mask=None, batch=False):
        if(type(ndarray) == type(torch.rand(0))):
            pad = pad_mask.detach().cpu().numpy()
            x = ndarray.detach().cpu().numpy()
        else:
            pad = pad_mask
            x = ndarray

        out = np.zeros_like(x)

        if(not batch or type(pad) == type(None)):
            # not flip start and end token
            return np.flip(ndarray, -1)
            # return np.concatenate((np.concatenate((ndarray[:1], np.flip(ndarray[1:-1], -1))), ndarray[-1:]))


        for i in range(len(x)):
            where = np.argwhere(pad[i] == True)
            if(where.shape[0] == 0): til = len(pad[0])
            else: til = where[0][0]
            if(len(out.shape) == 3):
                out[i, :, :] = np.concatenate((x[i, :, :til][::-1], x[i, :, til:]), axis=-1)
            else:
                out[i, :] = np.concatenate((x[i, :til][::-1], x[i, til:]), axis=-1)

        if(type(ndarray) == type(torch.rand(0))):
            return torch.from_numpy(out).to(ndarray.device)
        else:
            return out


class add_noise_transform(object):
    # G. Lample 2018a, https://arxiv.org/abs/1711.00043
    def __init__(self, pwd=0.1, k=3):
        self.pwd = pwd # word drop probability, replace by zero vector
        self.k = k # shuffle distance
        self.alpha = k + 1


    def __call__(self, sample):
        return {**sample, 'spec_noised': self.adding_noise(sample['spec']), \
                        'text_noised': self.adding_noise(sample['text']), \
                        'spec_r_noised': self.adding_noise(sample['spec_r']), \
                        'text_r_noised': self.adding_noise(sample['text_r'])}


    def adding_noise(self, ndarray):
        if(len(ndarray.shape) > 1):
            ndarray = np.transpose(ndarray, (1, 0))
            drop_array = np.zeros(shape=ndarray.shape[1], dtype=ndarray.dtype)
        else:
            drop_array = np.zeros(shape=1, dtype=ndarray.dtype)

        # (old index, new index)
        shuffled = [(i, i+sample) for i, sample in enumerate(np.random.uniform(0, self.alpha, len(ndarray)))]
        shuffled_sort = sorted(shuffled, key=lambda x: x[-1])
        noised_ndarray = np.empty_like(ndarray)
        for i, (j, _) in enumerate(shuffled_sort):
            drop_sampling = np.random.uniform(0, 1)
            if(drop_sampling <= self.pwd):
                noised_ndarray[i] = drop_array
            else:
                noised_ndarray[i] = ndarray[j]

        if(len(ndarray.shape) > 1): return np.transpose(noised_ndarray, (1, 0))
        else: return noised_ndarray



# pad element in each batch
def pad_collate(batch):

    def f(batch, k):
        max_len = 0
        for b in batch:
            max_len = max(max_len, b[k].shape[-1])
        return max_len

    spec_max_len = f(batch, 'spec')
    text_max_len = f(batch, 'text')
    spec_dtype = batch[0]['spec'].dtype
    text_dtype = batch[0]['text'].dtype

    ret = {}
    ret['spec_len'] = []
    ret['text_len'] = []
    ret['spec_pad'] = []
    ret['text_pad'] = []

    for i in range(len(batch)):
        ret['spec_len'].append(batch[i]['spec'].shape[-1])
        pad_vec = np.zeros((batch[i]['spec'].shape[0] ,spec_max_len - batch[i]['spec'].shape[-1]), dtype=spec_dtype)
        batch[i]['spec'] = np.concatenate( (batch[i]['spec'], pad_vec), axis=1)
        batch[i]['spec_r'] = np.concatenate( (batch[i]['spec_r'], pad_vec), axis=1)
        batch[i]['spec_noised'] = np.concatenate( (batch[i]['spec_noised'], pad_vec), axis=-1 )
        batch[i]['spec_r_noised'] = np.concatenate( (batch[i]['spec_r_noised'], pad_vec), axis=-1 )
        ret['spec_pad'].append(np.concatenate((np.zeros(batch[i]['spec'].shape[-1] - pad_vec.shape[-1]), np.ones(pad_vec.shape[-1])), axis=-1))

        ret['text_len'].append(batch[i]['text'].shape[-1])
        pad_vec = np.zeros((text_max_len - batch[i]['text'].shape[-1]), dtype=text_dtype)
        batch[i]['text'] = np.concatenate( (batch[i]['text'], pad_vec), axis=-1)
        batch[i]['text_r'] = np.concatenate( (batch[i]['text_r'], pad_vec), axis=-1)
        batch[i]['text_noised'] = np.concatenate( (batch[i]['text_noised'], pad_vec), axis=-1)
        batch[i]['text_r_noised'] = np.concatenate( (batch[i]['text_r_noised'], pad_vec), axis=-1)
        ret['text_pad'].append(np.concatenate((np.zeros(batch[i]['text'].shape[-1] - pad_vec.shape[-1]), np.ones(pad_vec.shape[-1])), axis=-1))

    ret['spec'] = torch.FloatTensor([x['spec'] for x in batch])
    ret['spec_r'] = torch.FloatTensor([x['spec_r'] for x in batch])
    ret['spec_noised'] = torch.FloatTensor([x['spec_noised'] for x in batch])
    ret['spec_r_noised'] = torch.FloatTensor([x['spec_r_noised'] for x in batch])

    ret['text'] = torch.LongTensor([x['text'] for x in batch])
    ret['text_r'] = torch.LongTensor([x['text_r'] for x in batch])
    ret['text_noised'] = torch.LongTensor([x['text_noised'] for x in batch])
    ret['text_r_noised'] = torch.LongTensor([x['text_r_noised'] for x in batch])

    ret['spec_pad'] = torch.LongTensor(ret['spec_pad'])
    ret['text_pad'] = torch.LongTensor(ret['text_pad'])

    ret['spec_len'] = torch.LongTensor([x for x in ret['spec_len']])
    ret['text_len'] = torch.LongTensor([x for x in ret['text_len']])
    return ret



# # Models

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


'''A wrapper class for optimizer '''
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def cal_loss(criterion, prediction, target, pad, seq_len):
    if(len(prediction.shape) > 2):
        prediction = prediction.permute(0,2,1).reshape(-1, prediction.shape[1])
    else:
        prediction = prediction.flatten()

    if(len(target.shape) > 2):
        target = target.permute(0, 2 ,1)
        target = target.reshape(-1, target.shape[-1])
    else:
        target = target.flatten()

    pad = pad.flatten()
    loss = criterion(prediction, target)
    if(len(loss.shape) > 1): loss = loss.mean(dim=1)
    loss *= pad.float()
    return loss.sum() / seq_len.sum()


def wer(list1, list2):
    dist = np.ones((len(list1)+1, len(list2)+1)) * -1
    dist[0][0] = 0
    for i,c1 in enumerate(list1, 1):
        for j,c2 in enumerate(list2, 1):
            if( c1 == c2 ):
                dist[i][j] = dist[i-1][j-1]
            else:
                dist[i][j] = min(dist[i-1][j-1], min(dist[i-1][j], dist[i][j-1])) + 1
    return dist[i][j] / len(list1)


def metric_text(pred, tgt):
    return wer(pred[1:-1], tgt[1:-1])


def metric_spec(pred, tgt, pad, seq_len):
    with torch.no_grad():
        return cal_loss(spec_criterion, pred[:,:,1:], tgt[:,:,1:], pad[:,1:], seq_len-1)


def create_look_ahead_mask(size, device=torch.device('cpu')):
    mask = torch.triu(torch.ones(size, size)) - torch.eye(size)
    sel = mask.type(torch.BoolTensor)  # (seq_len, seq_len)
    return mask.masked_fill(sel, float('-inf')).to(device)


def forward(src_in_block, tgt_in_block, enc, dec, out_block, src, tgt, src_pad_mask, tgt_pad_mask, look_ahead_mask):
    # in block
    src_inp = src_in_block(src)
    tgt_inp = tgt_in_block(tgt)
    # enc-dec
    enc_out = enc(src_inp, src_key_padding_mask=src_pad_mask)
    dec_out = dec(tgt_inp, enc_out, tgt_mask=look_ahead_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask)
    # out block
    out = out_block(dec_out)
    return out


# src, (n x feat x times)
def inference_batch(in_block, enc, dec, out_block, src, pad_mask, output):
    if(output == 'spec'):
        src_inp = in_block(src)
        dec_inp = torch.zeros_like(src, device=src.device)
        seq_len = torch.zeros(src.shape[0], device=src.device)
        dec_inp[:,0] = Constants["BOS"]
        dec_out = in_block(dec_inp)
        enc_out = enc(src_inp, src_key_padding_mask=pad_mask)
        look_ahead_mask = create_look_ahead_mask(src.shape[-1], device=dec_out.device)

        active_mask = torch.ones((src.shape[0],src.shape[-1]), device=src.device)
        threshold = torch.Tensor([0.5]).to(dec_out.device)
        out = torch.zeros(src.shape[0], mel_channels, 1, device=src.device)

        for t in range(1, pad_mask.shape[-1]):
            dec_out_tmp = dec(dec_out, enc_out,
                              tgt_mask=look_ahead_mask,
                              tgt_key_padding_mask=None,
                              memory_key_padding_mask=None)

            dec_out[t,:,:] = dec_out_tmp[t,:,:]
            mel_out,stop_out = out_block(dec_out)
            tmp = mel_out[:,:,t:t+1] * active_mask[:,t:t+1].unsqueeze(1)
            out = torch.cat((out, tmp), dim=2)
            active_mask[:,t] = active_mask[:,t-1] * (stop_out[:,t] < threshold).float()

            for i, e in enumerate(seq_len):
                if(e == 0 and stop_out[i,t] >= threshold):
                    seq_len[i] = t+1


            if(active_mask[:,t].sum() == 0):
                active_mask[:,t:] = 0

            for i, e in enumerate(seq_len):
                if(e == 0):
                    seq_len[i] = pad_mask.shape[-1]

        return out, 1-active_mask, seq_len


    elif(output == 'text'):
        src_inp = in_block(src)
        dec_inp = torch.zeros_like(src, device=src.device)
        dec_out = in_block(dec_inp)
        seq_len = torch.zeros(src.shape[0], device=src.device)
        enc_out = enc(src_inp, src_key_padding_mask=pad_mask)
        look_ahead_mask = create_look_ahead_mask(src.shape[-1], device=dec_out.device)

        active_mask = torch.ones((src.shape[0],src.shape[-1]), device=src.device)
        threshold = torch.Tensor([0.5]).to(dec_out.device)
        out = torch.zeros(src.shape[0], n_phones, 1, device=src.device)

        for t in range(1, pad_mask.shape[-1]):
            dec_out_tmp = dec(dec_out, enc_out,
                              tgt_mask=look_ahead_mask,
                              tgt_key_padding_mask=None,
                              memory_key_padding_mask=None)

            dec_out[t,:,:] = dec_out_tmp[t,:,:]
            token_out = out_block(dec_out)
            tmp = token_out[:,:,t:t+1] * active_mask[:,t:t+1].unsqueeze(1)
            out = torch.cat((out, tmp), dim=2)
            for i, e in enumerate(seq_len):
                if(e == 0 and token_out.argmax(dim=1)[i,t].float() == Constants["EOS"]):
                    seq_len[i] = t+1

            active_mask[:,t] = active_mask[:,t-1] * (token_out.argmax(dim=1)[:,t].float() != Constants["EOS"]).float()
            if(active_mask[:,t].sum() == 0):
                active_mask[:,t:] = 0

            for i, e in enumerate(seq_len):
                if(e == 0):
                    seq_len[i] = src.shape[-1]

        # out block
        return out, 1 - active_mask, seq_len



if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)


    # # Data loader
    # Read text data
    # transcript = pd.read_csv(os.path.join(datadir, 'metadata-phones.csv'), sep='|',
    #                         header=None, names=['ID', 'Transcription', 'Normalized Transcription']).fillna(-1)
    # transcript.iloc[-300:].to_csv(os.path.join(datadir,'test_set.csv'), sep='|', index=False , header=None)
    # transcript.iloc[-600:-300].to_csv(os.path.join(datadir,'val_set.csv'), sep='|', index=False, header=None)
    # transcript.iloc[:-600].to_csv(os.path.join(datadir,'train_set.csv'), sep='|', index=False, header=None)


    dataset = LJDataset(os.path.join(datadir, 'metadata-phones.csv'),
                        os.path.join(datadir, 'wavs'),
                        transform = transforms.Compose([
                            speech_transform(mel_channels),
                            text_transform(),
                            reverse_transform(),
                            add_noise_transform()
                        ])
                    )


    train_indices = list(range(len(dataset)))[:-(n_val+n_test)]
    val_indices = list(range(len(dataset)))[-(n_val+n_test):-n_test]
    test_indices = list(range(len(dataset)))[-n_test:]

    np.random.seed(0)
    np.random.shuffle(train_indices)
    paired_indices, unpaired_indices = train_indices[:n_paired], train_indices[n_paired:]
    
    
    # Creating PT data samplers and loaders:
    unpaired_sampler = SubsetRandomSampler(unpaired_indices)
    paired_sampler = SubsetRandomSampler(paired_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    unpaired_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=batch_size,
                                            sampler=unpaired_sampler, collate_fn=pad_collate)
    paired_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=batch_size,
                                                    sampler=paired_sampler, collate_fn=pad_collate)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=batch_size,
                                                    sampler=val_sampler, collate_fn=pad_collate)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=batch_size,
                                                    sampler=test_sampler, collate_fn=pad_collate)

    speech_inp = SpeechInput(in_dim=mel_channels, out_dim=d_model, h_dim=d_model)
    speech_out = SpeechOutput(in_dim=d_model, out_dim=mel_channels, h_dim=d_model)
    speech_enc = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
    speech_dec = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)

    text_inp = TextInput(in_dim=n_phones, out_dim=d_model)
    text_out = TextOutput(in_dim=d_model, out_dim=n_phones)
    text_enc = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
    text_dec = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)


    # Criterion
    import torch.optim as optim
    spec_criterion = nn.MSELoss(reduction='none')
    stop_criterion = nn.BCELoss(reduction='none')
    # text_criterion = nn.NLLLoss(reduction='none')
    text_criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, list(speech_inp.parameters()) + list(speech_out.parameters())
                + list(speech_enc.parameters()) + list(speech_dec.parameters())),
            betas=(0.9, 0.98),
            eps=1e-09),
        d_model=d_model, n_warmup_steps=4000)


    _ = speech_inp.to(device)
    _ = speech_out.to(device)
    _ = speech_enc.to(device)
    _ = speech_dec.to(device)
    _ = text_inp.to(device)
    _ = text_out.to(device)
    _ = text_enc.to(device)
    _ = text_dec.to(device)

    total_params = 0
    print('=== parameters ===')
    tmp = sum(p.numel() for p in speech_inp.parameters() if p.requires_grad)
    total_params += tmp
    print('speech_inp:', tmp)
    tmp = sum(p.numel() for p in speech_out.parameters() if p.requires_grad)
    total_params += tmp
    print('speech_out:', tmp)
    tmp = sum(p.numel() for p in speech_enc.parameters() if p.requires_grad)
    total_params += tmp
    print('speech_enc:', tmp)
    tmp = sum(p.numel() for p in speech_dec.parameters() if p.requires_grad)
    total_params += tmp
    print('speech_dec:', tmp)

    tmp = sum(p.numel() for p in text_inp.parameters() if p.requires_grad)
    total_params += tmp
    print('text_inp:', tmp)
    tmp = sum(p.numel() for p in text_out.parameters() if p.requires_grad)
    total_params += tmp
    print('text_out:', tmp)
    tmp = sum(p.numel() for p in text_enc.parameters() if p.requires_grad)
    total_params += tmp
    print('text_enc:', tmp)
    tmp = sum(p.numel() for p in text_dec.parameters() if p.requires_grad)
    total_params += tmp
    print('text_dec:', tmp)

    print('=== total params:', total_params, '===')





    reverse = reverse_transform()
    # ### loop train
    # stop_token = 0 : stop, 1 : not-stop

    # del writer
    exp_dir = 'exp'
    model_name = 'Transformer_v1.5'
    checkpoints_dir= os.path.join(exp_dir, 'checkpoints', model_name)
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'logs', model_name))
    logfile = os.path.join(exp_dir, 'logs', model_name, 'train.log')


    load_weight_iter = 0
    if(load_weight_iter > 0 ):
        print('loading weight from iter', load_weight_iter)
        speech_inp.load_state_dict(torch.load(os.path.join(checkpoints_dir, str(load_weight_iter), "speech_inp.pt")))
        speech_out.load_state_dict(torch.load(os.path.join(checkpoints_dir, str(load_weight_iter), "speech_out.pt")))
        speech_enc.load_state_dict(torch.load(os.path.join(checkpoints_dir, str(load_weight_iter), "speech_enc.pt")))
        speech_dec.load_state_dict(torch.load(os.path.join(checkpoints_dir, str(load_weight_iter), "speech_dec.pt")))
        text_inp.load_state_dict(torch.load(os.path.join(checkpoints_dir, str(load_weight_iter), "text_inp.pt")))
        text_out.load_state_dict(torch.load(os.path.join(checkpoints_dir, str(load_weight_iter), "text_out.pt")))
        text_enc.load_state_dict(torch.load(os.path.join(checkpoints_dir, str(load_weight_iter), "text_enc.pt")))
        text_dec.load_state_dict(torch.load(os.path.join(checkpoints_dir, str(load_weight_iter), "text_dec.pt")))
    else:
        print('initial new weight')

    save_iter = 100
    log_iter = 10
    val_iter = 1500

    max_iter = 4500

    n_iter = 1
    is_train = True

    print('start iter', n_iter)
    print('batch_size', batch_size)

    start_time = time.time()
    running_time = time.time()

    while (n_iter <= max_iter):
        print('start iter ', n_iter)
        print('loading data...')

        if(is_train):
            load_timing = time.time()
            running_loss = 0.0

            unpaired_data = next(iter(unpaired_loader))
            # get the inputs; data is a list of [inputs, labels]
            unpaired_spec = unpaired_data['spec'].to(device)
            unpaired_spec_n = unpaired_data['spec_noised'].to(device)
            unpaired_stop = unpaired_data['spec_pad'].type(torch.FloatTensor).to(device)
            unpaired_text = unpaired_data['text'].to(device)
            unpaired_text_n = unpaired_data['text_noised'].to(device)

            unpaired_spec_r = unpaired_data['spec_r'].to(device)
            unpaired_spec_r_n = unpaired_data['spec_r_noised'].to(device)
            unpaired_stop_r = unpaired_data['spec_pad'].type(torch.FloatTensor).to(device)
            unpaired_text_r = unpaired_data['text_r'].to(device)
            unpaired_text_r_n = unpaired_data['text_r_noised'].to(device)

            unpaired_spec_len = unpaired_data['spec_len']
            unpaired_text_len = unpaired_data['text_len']
            unpaired_spec_pad_mask = unpaired_data['spec_pad'].type(torch.BoolTensor).to(device)
            unpaired_spec_look_ahead_mask = create_look_ahead_mask(unpaired_spec.size(-1)).to(device)
            unpaired_text_pad_mask = unpaired_data['text_pad'].type(torch.BoolTensor).to(device)
            unpaired_text_look_ahead_mask = create_look_ahead_mask(unpaired_text.size(-1)).to(device)

            # should predict stop at the last time step
            unpaired_stop[torch.arange(unpaired_stop.shape[0]), unpaired_spec_len-1] = 1.
            unpaired_stop_r[torch.arange(unpaired_stop_r.shape[0]), unpaired_spec_len-1] = 1.


            paired_data = next(iter(paired_loader))
            paired_spec = paired_data['spec'].to(device)
            paired_spec_n = paired_data['spec_noised'].to(device)
            paired_stop = paired_data['spec_pad'].type(torch.FloatTensor).to(device)
            paired_text = paired_data['text'].to(device)
            paired_text_n = paired_data['text_noised'].to(device)

            paired_spec_r = paired_data['spec_r'].to(device)
            paired_spec_r_n = paired_data['spec_r_noised'].to(device)
            paired_stop_r = paired_data['spec_pad'].type(torch.FloatTensor).to(device)
            paired_text_r = paired_data['text_r'].to(device)
            paired_text_r_n = paired_data['text_r_noised'].to(device)

            paired_spec_len = paired_data['spec_len']
            paired_text_len = paired_data['text_len']
            paired_spec_pad_mask = paired_data['spec_pad'].type(torch.BoolTensor).to(device)
            paired_spec_look_ahead_mask = create_look_ahead_mask(paired_spec.size(-1)).to(device)
            paired_text_pad_mask = paired_data['text_pad'].type(torch.BoolTensor).to(device)
            paired_text_look_ahead_mask = create_look_ahead_mask(paired_text.size(-1)).to(device)
            paired_stop[torch.arange(paired_stop.shape[0]), paired_spec_len-1] = 1.
            paired_stop_r[torch.arange(paired_stop_r.shape[0]), paired_spec_len-1] = 1.

            print('load time', time.time() - load_timing)

            print('forward...')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # ===== DAE =====
            # LTR - spec to spec
            sTs_spec_out, sTs_stop_out = forward(speech_inp, speech_inp, speech_enc, speech_dec, speech_out,
                                                unpaired_spec_n, unpaired_spec, unpaired_spec_pad_mask,
                                                unpaired_spec_pad_mask, unpaired_spec_look_ahead_mask)

            sTs_loss_mask = 1 - unpaired_spec_pad_mask.float()
            sTs_spec_loss = cal_loss(spec_criterion, sTs_spec_out, unpaired_spec, unpaired_spec_pad_mask, unpaired_spec_len)
            sTs_stop_loss = cal_loss(stop_criterion, sTs_stop_out, unpaired_stop, unpaired_spec_pad_mask, unpaired_spec_len)

            # LTR - text to text
            tTt_text_out = forward(text_inp, text_inp, text_enc, text_dec, text_out,
                            unpaired_text_n, unpaired_text, unpaired_text_pad_mask,
                            unpaired_text_pad_mask, unpaired_text_look_ahead_mask)

            tTt_loss_mask = 1 - unpaired_text_pad_mask.float()
            tTt_text_loss = cal_loss(text_criterion, tTt_text_out, unpaired_text, unpaired_text_pad_mask, unpaired_text_len)

            # RTL - spec to spec
            sTs_spec_out_r, sTs_stop_out_r = forward(speech_inp, speech_inp, speech_enc, speech_dec, speech_out,
                                                unpaired_spec_r_n, unpaired_spec_r, unpaired_spec_pad_mask,
                                                unpaired_spec_pad_mask, unpaired_spec_look_ahead_mask)

            sTs_loss_mask = 1 - unpaired_spec_pad_mask.float()
            sTs_spec_r_loss = cal_loss(spec_criterion, sTs_spec_out_r, unpaired_spec_r, unpaired_spec_pad_mask, unpaired_spec_len)
            sTs_stop_r_loss = cal_loss(stop_criterion, sTs_stop_out_r, unpaired_stop_r, unpaired_spec_pad_mask, unpaired_spec_len)

            # RTL - text to text
            tTt_out_text_r = forward(text_inp, text_inp, text_enc, text_dec, text_out,
                            unpaired_text_r_n, unpaired_text_r, unpaired_text_pad_mask,
                            unpaired_text_pad_mask, unpaired_text_look_ahead_mask)

            tTt_loss_mask = 1 - unpaired_text_pad_mask.float()
            tTt_text_r_loss = cal_loss(text_criterion, tTt_out_text_r, unpaired_text_r, unpaired_text_pad_mask, unpaired_text_len)

            loss_dae = sTs_spec_loss + sTs_stop_loss + sTs_spec_r_loss + sTs_stop_r_loss + tTt_text_loss + tTt_text_r_loss


            # ===== Back Translation =====
            with torch.no_grad():
                BT_text, BT_pad_mask, _ = inference_batch(speech_inp, speech_enc, text_dec, text_out,
                                            unpaired_spec_n, unpaired_spec_pad_mask, 'text')
                BT_text_r, BT_pad_mask_r, _ = inference_batch(speech_inp, speech_enc, text_dec, text_out,
                                            unpaired_spec_r_n, unpaired_spec_pad_mask, 'text')
                BT_spec, BT_stop, _ = inference_batch(text_inp, text_enc, speech_dec, speech_out,
                                            unpaired_text_n, unpaired_text_pad_mask, 'spec')
                BT_spec_r, BT_stop_r, _ = inference_batch(text_inp, text_enc, speech_dec, speech_out,
                                            unpaired_text_r_n, unpaired_text_pad_mask, 'spec')
                r_BT_text = reverse.reverse(BT_text.argmax(dim=1), pad_mask=BT_pad_mask, batch=True)
                r_BT_text_r = reverse.reverse(BT_text_r.argmax(dim=1), pad_mask=BT_pad_mask_r ,batch=True)
                r_BT_spec = reverse.reverse(BT_spec, pad_mask=BT_stop, batch=True)
                r_BT_spec_r = reverse.reverse(BT_spec_r, pad_mask=BT_stop_r, batch=True)

            # ===== BT LTR output =====
            # (spec to text) to spec
            BT_text_spec_out, BT_text_stop_out = forward(text_inp, speech_inp, text_enc, speech_dec, speech_out,
                                        BT_text.argmax(dim=1), unpaired_spec,
                                        BT_pad_mask.type(torch.BoolTensor).to(device),
                                        unpaired_spec_pad_mask, unpaired_spec_look_ahead_mask)


            BT_text_spec_loss_mask = 1 - unpaired_spec_pad_mask.float()
            BT_text_spec_loss = cal_loss(spec_criterion, BT_text_spec_out, unpaired_spec, unpaired_spec_pad_mask, unpaired_spec_len)
            BT_text_stop_loss = cal_loss(stop_criterion, BT_text_stop_out, unpaired_stop, unpaired_spec_pad_mask, unpaired_spec_len)

            # R(spec_r to text_r) to spec
            r_BT_text_r_spec_out, r_BT_text_r_stop_out = forward(text_inp, speech_inp, text_enc, speech_dec, speech_out,
                                        r_BT_text_r, unpaired_spec,
                                        BT_pad_mask_r.type(torch.BoolTensor).to(device),
                                        unpaired_spec_pad_mask, unpaired_spec_look_ahead_mask)
            r_BT_text_r_loss_mask = 1 - unpaired_spec_pad_mask.float()
            r_BT_text_r_spec_loss = cal_loss(spec_criterion, r_BT_text_r_spec_out, unpaired_spec, unpaired_spec_pad_mask, unpaired_spec_len)
            r_BT_text_r_stop_loss = cal_loss(stop_criterion, r_BT_text_r_stop_out, unpaired_stop, unpaired_spec_pad_mask, unpaired_spec_len)

            # (text to spec) to text
            BT_spec_text_out = forward(speech_inp, text_inp, speech_enc, text_dec, text_out,
                                        BT_spec, unpaired_text,
                                        BT_stop.type(torch.BoolTensor).to(device),
                                        unpaired_text_pad_mask, unpaired_text_look_ahead_mask)
            BT_spec_text_loss_mask = 1 - unpaired_text_pad_mask.float()
            BT_spec_text_loss = cal_loss(text_criterion, BT_spec_text_out, unpaired_text, unpaired_text_pad_mask, unpaired_text_len)

            # R(text_r to spec_r) to text
            r_BT_spec_r_text_out = forward(speech_inp, text_inp, speech_enc, text_dec, text_out,
                                        r_BT_spec_r, unpaired_text,
                                        BT_stop_r.type(torch.BoolTensor).to(device),
                                        unpaired_text_pad_mask, unpaired_text_look_ahead_mask)

            r_BT_spec_r_text_loss_mask = 1 - unpaired_text_pad_mask.float()
            r_BT_spec_r_text_loss = cal_loss(text_criterion, r_BT_spec_r_text_out, unpaired_text, unpaired_text_pad_mask, unpaired_text_len)


            # =========================
            # ===== BT RTL output =====
            # =========================
            # (spec_r to text_r) to spec_r
            BT_text_r_spec_r_out, BT_text_r_stop_r_out = forward(text_inp, speech_inp, text_enc, speech_dec, speech_out,
                                        BT_text_r.argmax(dim=1), unpaired_spec_r,
                                        BT_pad_mask_r.type(torch.BoolTensor).to(device),
                                        unpaired_spec_pad_mask, unpaired_spec_look_ahead_mask)


            BT_text_r_spec_r_loss_mask = 1 - unpaired_spec_pad_mask.float()
            BT_text_r_spec_r_loss = cal_loss(spec_criterion, BT_text_r_spec_r_out, unpaired_spec_r, unpaired_spec_pad_mask, unpaired_spec_len)
            BT_text_r_stop_r_loss = cal_loss(stop_criterion, BT_text_r_stop_r_out, unpaired_stop_r, unpaired_spec_pad_mask, unpaired_spec_len)


            # R(spec to text) to spec_r
            r_BT_text_spec_r_out, r_BT_text_stop_r_out = forward(text_inp, speech_inp, text_enc, speech_dec, speech_out,
                                        r_BT_text, unpaired_spec_r,
                                        BT_pad_mask.type(torch.BoolTensor).to(device),
                                        unpaired_spec_pad_mask, unpaired_spec_look_ahead_mask)
            r_BT_text_spec_r_loss_mask = 1 - unpaired_spec_pad_mask.float()
            r_BT_text_spec_r_loss = cal_loss(spec_criterion, r_BT_text_spec_r_out, unpaired_spec_r, unpaired_spec_pad_mask, unpaired_spec_len)
            r_BT_text_stop_r_loss = cal_loss(stop_criterion, r_BT_text_stop_r_out, unpaired_stop_r, unpaired_spec_pad_mask, unpaired_spec_len)

            # (text_r to spec_r) to text_r
            BT_spec_r_text_r_out = forward(speech_inp, text_inp, speech_enc, text_dec, text_out,
                                        BT_spec_r, unpaired_text_r,
                                        BT_stop_r.type(torch.BoolTensor).to(device),
                                        unpaired_text_pad_mask, unpaired_text_look_ahead_mask)
            BT_spec_r_text_r_loss_mask = 1 - unpaired_text_pad_mask.float()
            BT_spec_r_text_r_loss = cal_loss(text_criterion, BT_spec_r_text_r_out, unpaired_text, unpaired_text_pad_mask, unpaired_text_len)

            # R(text to spec) to text_r
            r_BT_spec_text_r_out = forward(speech_inp, text_inp, speech_enc, text_dec, text_out,
                                        r_BT_spec, unpaired_text,
                                        BT_stop.type(torch.BoolTensor).to(device),
                                        unpaired_text_pad_mask, unpaired_text_look_ahead_mask)

            r_BT_spec_text_r_loss_mask = 1 - unpaired_text_pad_mask.float()
            r_BT_spec_text_r_loss = cal_loss(text_criterion, r_BT_spec_text_r_out, unpaired_text, unpaired_text_pad_mask, unpaired_text_len)

            loss_bt = BT_text_spec_loss + BT_text_stop_loss + r_BT_text_r_spec_loss + \
                r_BT_text_r_stop_loss + BT_spec_text_loss + r_BT_spec_r_text_loss + \
                BT_text_r_spec_r_loss + BT_text_r_stop_r_loss + r_BT_text_spec_r_loss + \
                r_BT_text_stop_r_loss + BT_spec_r_text_r_loss + r_BT_spec_text_r_loss


            # ====================
            # ======= PAIR =======
            # ====================
            # pair spec to text
            sTt_text_out = forward(speech_inp, text_inp, speech_enc, text_dec, text_out,
                                                paired_spec_n, paired_text, paired_spec_pad_mask,
                                                paired_text_pad_mask, paired_text_look_ahead_mask)
            sTt_loss_mask = 1 - paired_text_pad_mask.float()
            sTt_text_loss = cal_loss(text_criterion, sTt_text_out, paired_text, paired_text_pad_mask, paired_text_len)


            # pair text to spec
            tTs_spec_out, tTs_stop_out = forward(text_inp, speech_inp, text_enc, speech_dec, speech_out,
                                                paired_text_n, paired_spec, paired_text_pad_mask,
                                                paired_spec_pad_mask, paired_spec_look_ahead_mask)

            tTs_loss_mask = 1 - paired_spec_pad_mask.float()
            tTs_spec_loss = cal_loss(spec_criterion, tTs_spec_out, paired_spec, paired_spec_pad_mask, paired_spec_len)
            tTs_stop_loss = cal_loss(stop_criterion, tTs_stop_out, paired_stop, paired_spec_pad_mask, paired_spec_len)


            # pair spec_r to text_r
            sTt_text_r_out = forward(speech_inp, text_inp, speech_enc, text_dec, text_out,
                                                paired_spec_r_n, paired_text_r, paired_spec_pad_mask,
                                                paired_text_pad_mask, paired_text_look_ahead_mask)
            sTt_r_loss_mask = 1 - paired_text_pad_mask.float()
            sTt_text_r_loss = cal_loss(text_criterion, sTt_text_r_out, paired_text_r, paired_text_pad_mask, paired_text_len)


            # pair text_r to spec_r
            tTs_spec_r_out, tTs_stop_r_out = forward(text_inp, speech_inp, text_enc, speech_dec, speech_out,
                                                paired_text_r_n, paired_spec_r, paired_text_pad_mask,
                                                paired_spec_pad_mask, paired_spec_look_ahead_mask)

            tTs_r_loss_mask = 1 - paired_spec_pad_mask.float()
            tTs_spec_r_loss = cal_loss(spec_criterion, tTs_spec_r_out, paired_spec_r, paired_spec_pad_mask, paired_spec_len)
            tTs_stop_r_loss = cal_loss(stop_criterion, tTs_stop_r_out, paired_stop_r, paired_spec_pad_mask, paired_spec_len)


            loss_pair = sTt_text_loss + tTs_spec_loss + tTs_stop_loss + sTt_text_r_loss + tTs_spec_r_loss + tTs_stop_r_loss


            # backward
            print('backward...')
            loss = loss_dae + loss_bt + loss_pair
            loss.backward()

            # optimize
            optimizer.step_and_update_lr()

            # print statistics
            running_loss += loss.item()

            if (n_iter) % log_iter == 0:
                currentDT = datetime.datetime.now()
                print('{}'.format(currentDT.strftime("%Y/%m/%d %H:%M:%S")))
                print('iter: %d loss: %.3f' % (n_iter, running_loss / 10))
                print('elapsed {} sec, {} sec/iter\n'.format(round(time.time()-start_time, 2), round((time.time()-running_time)/log_iter,2), 's/iter'))

                with open(logfile, 'a') as f:
                    f.write('{}\n'.format(currentDT.strftime("%Y/%m/%d %H:%M:%S")))
                    f.write('iter: %d loss: %.3f\n' % (n_iter, running_loss / 10))
                    f.write('elapsed {} sec, {} sec/iter\n\n'.format(round(time.time()-start_time, 2), round((time.time()-running_time)/log_iter,2), 's/iter'))
                running_time = time.time()

                writer.add_scalar('loss/total', loss, n_iter)
                writer.add_scalar('loss/dae', loss_dae, n_iter)
                writer.add_scalar('loss/bt', loss_bt, n_iter)
                writer.add_scalar('loss/pair', loss_pair, n_iter)

                writer.add_scalar('loss_dae/spec', sTs_spec_loss, n_iter)
                writer.add_scalar('loss_dae/stop', sTs_stop_loss, n_iter)
                writer.add_scalar('loss_dae/text', tTt_text_loss, n_iter)
                writer.add_scalar('loss_dae/spec_r', sTs_spec_r_loss, n_iter)
                writer.add_scalar('loss_dae/stop_r', sTs_stop_r_loss, n_iter)
                writer.add_scalar('loss_dae/text_r', tTt_text_r_loss, n_iter)

                writer.add_scalar('loss_bt/spec', BT_text_spec_loss, n_iter)
                writer.add_scalar('loss_bt/stop', BT_text_stop_loss, n_iter)
                writer.add_scalar('loss_bt/r_spec', r_BT_text_r_spec_loss, n_iter)
                writer.add_scalar('loss_bt/r_stop', r_BT_text_r_stop_loss, n_iter)
                writer.add_scalar('loss_bt/text', BT_spec_text_loss, n_iter)
                writer.add_scalar('loss_bt/r_text', r_BT_spec_r_text_loss, n_iter)

                writer.add_scalar('loss_bt/spec_r', BT_text_r_spec_r_loss, n_iter)
                writer.add_scalar('loss_bt/stop_r', BT_text_r_stop_r_loss, n_iter)
                writer.add_scalar('loss_bt/r_spec_r', r_BT_text_spec_r_loss, n_iter)
                writer.add_scalar('loss_bt/r_stop_r', r_BT_text_stop_r_loss, n_iter)
                writer.add_scalar('loss_bt/text_r', BT_spec_r_text_r_loss, n_iter)
                writer.add_scalar('loss_bt/r_text_r', r_BT_spec_text_r_loss, n_iter)

                writer.add_scalar('loss_pair/spec', tTs_spec_loss, n_iter)
                writer.add_scalar('loss_pair/stop', tTs_stop_loss, n_iter)
                writer.add_scalar('loss_pair/text', sTt_text_loss, n_iter)
                writer.add_scalar('loss_pair/spec_r', tTs_spec_r_loss, n_iter)
                writer.add_scalar('loss_pair/stop_r', tTs_stop_r_loss, n_iter)
                writer.add_scalar('loss_pair/text_r', sTt_text_r_loss, n_iter)
                running_loss = 0.0


                if ((n_iter)%save_iter == 0):
                    if not os.path.isdir(os.path.join(checkpoints_dir, str(n_iter))):
                        os.makedirs(os.path.join(checkpoints_dir, str(n_iter)))
                    torch.save(speech_inp.state_dict(), os.path.join(checkpoints_dir, str(n_iter), "speech_inp.pt"))
                    torch.save(speech_out.state_dict(), os.path.join(checkpoints_dir, str(n_iter), "speech_out.pt"))
                    torch.save(speech_enc.state_dict(), os.path.join(checkpoints_dir, str(n_iter), "speech_enc.pt"))
                    torch.save(speech_dec.state_dict(), os.path.join(checkpoints_dir, str(n_iter), "speech_dec.pt"))
                    torch.save(text_inp.state_dict(), os.path.join(checkpoints_dir, str(n_iter), "text_inp.pt"))
                    torch.save(text_out.state_dict(), os.path.join(checkpoints_dir, str(n_iter), "text_out.pt"))
                    torch.save(text_enc.state_dict(), os.path.join(checkpoints_dir, str(n_iter), "text_enc.pt"))
                    torch.save(text_dec.state_dict(), os.path.join(checkpoints_dir, str(n_iter), "text_dec.pt"))


        if( n_iter % val_iter == 0):
            infer_time = time.time()
            print('inference iter', n_iter)
            with torch.no_grad():
                running_per = 0
                running_mse = 0
                running_per_r = 0
                running_mse_r = 0

                for i, val_data in enumerate(val_loader):
                    val_spec = val_data['spec'].to(device)
                    val_spec_n = val_data['spec_noised'].to(device)
                    val_stop = val_data['spec_pad'].type(torch.FloatTensor).to(device)
                    val_text = val_data['text'].to(device)
                    val_text_n = val_data['text_noised'].to(device)

                    val_spec_r = val_data['spec_r'].to(device)
                    val_spec_r_n = val_data['spec_r_noised'].to(device)
                    val_stop_r = val_data['spec_pad'].type(torch.FloatTensor).to(device)
                    val_text_r = val_data['text_r'].to(device)
                    val_text_r_n = val_data['text_r_noised'].to(device)

                    val_spec_pad_mask = val_data['spec_pad'].type(torch.BoolTensor).to(device)
                    val_spec_look_ahead_mask = create_look_ahead_mask(val_spec.size(-1)).to(device)
                    val_text_pad_mask = val_data['text_pad'].type(torch.BoolTensor).to(device)
                    val_text_look_ahead_mask = create_look_ahead_mask(val_text.size(-1)).to(device)

                    val_spec_len = val_data['spec_len']
                    val_text_len = val_data['text_len']


                    infer_text, infer_pad_mask, infer_text_len = inference_batch(speech_inp, speech_enc, text_dec, text_out,
                                                            val_spec_n, val_spec_pad_mask, 'text')
                    infer_text_r, infer_pad_mask_r, infer_text_r_len = inference_batch(speech_inp, speech_enc, text_dec, text_out,
                                                            val_spec_r_n, val_spec_pad_mask, 'text')
                    infer_spec, infer_stop, infer_spec_len = inference_batch(text_inp, text_enc, speech_dec, speech_out,
                                                            val_text_n, val_text_pad_mask, 'spec')

                    infer_spec_r, infer_stop_r, infer_spec_r_len = inference_batch(text_inp, text_enc, speech_dec, speech_out,
                                                            val_text_r_n, val_text_pad_mask, 'spec')

                    infer_text = infer_text.argmax(dim=1)
                    infer_text_r = infer_text_r.argmax(dim=1)

                    infer_text = infer_text.cpu()
                    infer_text_r = infer_text_r.cpu()
                    val_text_pad_mask = val_text_pad_mask.cpu()
                    infer_pad_mask = infer_pad_mask.cpu()
                    infer_pad_mask_r = infer_pad_mask_r.cpu()

                    for j in range(infer_text.shape[0]):
                        lab_til = np.argwhere(val_data['text_pad'][j] == 1)[0]
                        if(len(lab_til) > 0): lab_til = lab_til[0]
                        else: lab_til = len(val_text_pad_mask[0])
                        pred_til = np.argwhere(infer_pad_mask[j] >= 0.5)[0]
                        if(len(pred_til) > 0): pred_til = pred_til[0]
                        else: pred_til = len(infer_pad_mask[0])
                        running_per += metric_text(infer_text[j][:pred_til], val_data['text'][j,:lab_til])

                        pred_til = np.argwhere(infer_pad_mask_r[j] == 1)[0]
                        if(len(pred_til) > 0): pred_til = pred_til[0]
                        else: pred_til = len(infer_pad_mask[0])
                        running_per_r += metric_text(infer_text_r[j][:pred_til], val_data['text_r'][j,:lab_til])

                    max_len = max(val_spec_r.shape[-1], infer_spec_r.shape[-1])
                    pad_tensor = torch.zeros(infer_spec_r.shape[:-1] + (max_len - infer_spec_r.shape[-1],), device=infer_spec_r.device)
                    infer_spec_r = torch.cat((infer_spec_r,  pad_tensor), axis=-1)
                    pad_tensor = torch.zeros(val_spec_r.shape[:-1] + (max_len - val_spec_r.shape[-1],), device=val_spec_r.device)
                    val_spec_r = torch.cat((val_spec_r,  pad_tensor), axis=-1)
                    if(max_len == val_spec_r.shape[-1]):
                        running_mse_r += metric_spec(infer_spec_r, val_spec, val_spec_pad_mask.float(), val_spec_len)
                    else:
                        running_mse_r += metric_spec(infer_spec_r, val_spec, infer_stop_r.float(), infer_spec_r_len)

                    max_len = max(val_spec.shape[-1], infer_spec.shape[-1])
                    pad_tensor = torch.zeros(infer_spec.shape[:-1] + (max_len - infer_spec.shape[-1],), device=infer_spec.device)
                    infer_spec = torch.cat((infer_spec,  pad_tensor), axis=-1)
                    pad_tensor = torch.zeros(val_spec.shape[:-1] + (max_len - val_spec.shape[-1],), device=val_spec.device)
                    val_spec = torch.cat((val_spec,  pad_tensor), axis=-1)
                    if(max_len == val_spec.shape[-1]):
                        running_mse += metric_spec(infer_spec, val_spec, val_spec_pad_mask.float(), val_spec_len)
                    else:
                        running_mse += metric_spec(infer_spec, val_spec, infer_stop.float(), infer_spec_len)

                running_per /= (len(val_loader) * batch_size)
                running_mse /= (len(val_loader) * batch_size)
                running_per_r /= (len(val_loader) * batch_size)
                running_mse_r /= (len(val_loader) * batch_size)

                writer.add_scalar('val/per', running_per, n_iter)
                writer.add_scalar('val/per_r', running_per_r, n_iter)
                writer.add_scalar('val/mse', running_mse, n_iter)
                writer.add_scalar('val/mse_r' , running_mse, n_iter)

                with open(logfile, 'a') as f:
                    f.write('inference iter:{}\n'.format(n_iter))
                    f.write('per: {}, per_r:{}, mse:{}, mse_r:{}\n'.format(running_per, running_per_r, running_mse, running_mse_r))
                    f.write('time usage: {}'.format(round(time.time() - infer_time,2)))

        n_iter+=1
    print('Finished Training')