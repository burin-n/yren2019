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
import pickle

import copy
import time
import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datadir = '/root/test-something/LJSpeech-1.1'
wav_dirs = os.path.join(datadir, 'wavs')

n_val = 300
n_test = 300
n_training = len(os.listdir(wav_dirs)) - n_val - n_test
n_paired = 200

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


def get_data_loader(mel_channels, batch_size=8, num_workers=8, pad_collate=pad_collate):
    dataset = LJDataset(os.path.join(datadir, 'metadata-phones.csv'),
                        os.path.join(datadir, 'wavs'),
                        transform = transforms.Compose([
                            speech_transform(mel_channels),
                            text_transform(),
                            reverse_transform(),
                            add_noise_transform()
                        ])
                    )

    # Load data (deserialize)
    with open(os.path.join(datadir, 'unpaired_sampler.pickle'), 'rb') as handle:
        unpaired_sampler = pickle.load(handle)

    with open(os.path.join(datadir,'paired_sampler.pickle'), 'rb') as handle:
        paired_sampler = pickle.load(handle)

    with open(os.path.join(datadir, 'val_sampler.pickle'), 'rb') as handle:
        val_sampler = pickle.load(handle)

    with open(os.path.join(datadir,'test_sampler.pickle'), 'rb') as handle:
        test_sampler = pickle.load(handle)


    unpaired_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                            sampler=unpaired_sampler, collate_fn=pad_collate)
    paired_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                    sampler=paired_sampler, collate_fn=pad_collate)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                    sampler=val_sampler, collate_fn=pad_collate)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                    sampler=test_sampler, collate_fn=pad_collate)
    
    return unpaired_loader, paired_loader, val_loader, test_loader

   

if __name__ == '__main__':
    # # Data loader
    # Read text data
    # transcript = pd.read_csv(os.path.join(datadir, 'metadata-phones.csv'), sep='|',
    #                         header=None, names=['ID', 'Transcription', 'Normalized Transcription']).fillna(-1)
    # transcript.iloc[-300:].to_csv(os.path.join(datadir,'test_set.csv'), sep='|', index=False , header=None)
    # transcript.iloc[-600:-300].to_csv(os.path.join(datadir,'val_set.csv'), sep='|', index=False, header=None)
    # transcript.iloc[:-600].to_csv(os.path.join(datadir,'train_set.csv'), sep='|', index=False, header=None)
    from train import mel_channels

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

    with open(os.path.join(datadir, 'unpaired_sampler.pickle'), 'wb') as handle:
        pickle.dump(unpaired_sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(datadir,'paired_sampler.pickle'), 'wb') as handle:
            pickle.dump(paired_sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(datadir, 'val_sampler.pickle'), 'wb') as handle:
            pickle.dump(val_sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(datadir,'test_sampler.pickle'), 'wb') as handle:
            pickle.dump(test_sampler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('created sampler index at', datadir)