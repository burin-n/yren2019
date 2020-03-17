#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np


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
        return cal_loss(torch.nn.MSELoss(reduction='none'), pred[:,:,1:], tgt[:,:,1:], pad[:,1:], seq_len-1)


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
        out = torch.zeros(src.shape[0], out_block.out_dim, 1, device=src.device)

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
        out = torch.zeros(src.shape[0], out_block.out_dim, 1, device=src.device)

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