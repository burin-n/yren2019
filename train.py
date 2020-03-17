#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from torchvision import transforms
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import os

from dataloader import get_data_loader, reverse_transform
from model import SpeechInput, SpeechOutput, Encoder, Decoder, TextInput, TextOutput 
from utils import *

# train parameters
exp_dir = os.path.join(os.getcwd(),'YRen', 'exp')
model_name = 'Transformer_v1.6'
load_weight_iter = 500
start_iter = 500
save_iter = 100
log_iter = 10
val_iter = 500
max_iter = 500
is_train = False


# data loader
# datadir = '../LJSpeech-1.1'
batch_size = 8

# phones parameters
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

# model parameters 
d_model=256
nhead=8
num_encoder_layers=4
num_decoder_layers=4
dim_feedforward=1024
dropout=0.1
mel_channels = 80



if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unpaired_loader, paired_loader, val_loader, test_loader = get_data_loader(mel_channels, num_workers=0)

    speech_inp = SpeechInput(in_dim=mel_channels, out_dim=d_model, h_dim=d_model)
    speech_out = SpeechOutput(in_dim=d_model, out_dim=mel_channels, h_dim=d_model)
    speech_enc = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
    speech_dec = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)

    text_inp = TextInput(in_dim=n_phones, out_dim=d_model)
    text_out = TextOutput(in_dim=d_model, out_dim=n_phones)
    text_enc = Encoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
    text_dec = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)


    spec_criterion = nn.MSELoss(reduction='none')
    stop_criterion = nn.BCELoss(reduction='none')
    # text_criterion = nn.NLLLoss(reduction='none')
    text_criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = ScheduledOptim(
        torch.optim.Adam(
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
    checkpoints_dir= os.path.join(exp_dir, 'checkpoints', model_name)
    if(is_train):
        writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'logs', model_name))
        logfile = os.path.join(exp_dir, 'logs', model_name, 'train.log')

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

    
    print('\n=== start iter', start_iter, '===')
    print('batch_size', batch_size)

    start_time = time.time()
    running_time = time.time()
    n_iter = start_iter
    running_loss = 0.0

    while (n_iter <= max_iter):
        print('start iter ', n_iter)
        print('loading data...')

        if(is_train):
            load_timing = time.time()

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

            print('load time', round(time.time() - load_timing, 2))

            print('forward...')
            forward_timing = time.time()
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

            print('forward time', round(time.time() - forward_timing, 2))

            # backward
            backward_timing = time.time()
            print('backward...')
            loss = loss_dae + loss_bt + loss_pair
            loss.backward()

            # optimize
            optimizer.step_and_update_lr()
            print('backward time', round(time.time() - backward_timing, 2))

            # print statistics
            running_loss += loss.item()

            if (n_iter) % log_iter == 0:
                currentDT = datetime.datetime.now()
                print('{}'.format(currentDT.strftime("%Y/%m/%d %H:%M:%S")))
                print('iter: %d loss: %.3f' % (n_iter, running_loss / log_iter))
                print('elapsed {} sec, {} sec/iter\n'.format(round(time.time()-start_time, 2), round((time.time()-running_time)/log_iter,2), 's/iter'))

                with open(logfile, 'a') as f:
                    f.write('{}\n'.format(currentDT.strftime("%Y/%m/%d %H:%M:%S")))
                    f.write('iter: %d loss: %.3f\n' % (n_iter, running_loss / log_iter))
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


        if(n_iter % val_iter == 0):
            infer_time = time.time()
            print('inference iter', n_iter)
            decode_dir = os.path.join(exp_dir, 'logs', model_name, 'decode')
            if not os.path.isdir(os.path.join(decode_dir)):
                os.makedirs(os.path.join(decode_dir))

            currentDT = datetime.datetime.now()
            dec_log = open(os.path.join(decode_dir, "dec_{}.log".format(n_iter)), 'w')
            dec_log.write('=== start infering at {} ===\n'.format(currentDT.strftime("%Y/%m/%d %H:%M:%S")))

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
                        dec_log.write("inference sample: {}\n".format(i * batch_size + j))

                        lab_til = np.argwhere(val_data['text_pad'][j] == 1)[0]
                        if(len(lab_til) > 0): lab_til = lab_til[0]
                        else: lab_til = len(val_text_pad_mask[0])
                        pred_til = np.argwhere(infer_pad_mask[j] >= 0.5)[0]
                        if(len(pred_til) > 0): pred_til = pred_til[0]
                        else: pred_til = len(infer_pad_mask[0])
                        cur_per = metric_text(infer_text[j][:pred_til], val_data['text'][j,:lab_til])
                        running_per += cur_per 
                        
                        dec_log.write("text:")
                        dec_log.write(" *full seq: {}\n".format(infer_text[j]))
                        dec_log.write(" *till seq: {}\n".format(infer_text[j][:pred_til]))
                        dec_log.write(" >lab seq: {}\n".format(val_data['text'][j,:lab_til]))
                        dec_log.write("per: {}\n".format(round(cur_per, 3)))

                        pred_til = np.argwhere(infer_pad_mask_r[j] == 1)[0]
                        if(len(pred_til) > 0): pred_til = pred_til[0]
                        else: pred_til = len(infer_pad_mask[0])
                        cur_per = metric_text(infer_text_r[j][:pred_til], val_data['text_r'][j,:lab_til])
                        running_per_r += cur_per 

                        dec_log.write("text reverse:")
                        dec_log.write(" *full: {}\n".format(infer_text_r[j]))
                        dec_log.write(" *till: {}\n".format(infer_text_r[j][:pred_til]))
                        dec_log.write(" > lab: {}\n".format(val_data['text_r'][j,:lab_til]))
                        dec_log.write("per: {}\n".format(round(cur_per, 3)))


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

                currentDT = datetime.datetime.now()
                dec_log.write('per: %.3f per_r: %.3f\n' % (running_per, running_per_r))
                dec_log.write('finish inference at: {}\n\n'.format(currentDT.strftime("%Y/%m/%d %H:%M:%S")))
                dec_log.close()

                if(is_train):
                    writer.add_scalar('val/per', running_per, n_iter)
                    writer.add_scalar('val/per_r', running_per_r, n_iter)
                    writer.add_scalar('val/mse', running_mse, n_iter)
                    writer.add_scalar('val/mse_r' , running_mse, n_iter)
                    with open(logfile, 'a') as f:
                        f.write('inference iter:{}\n'.format(n_iter))
                        f.write('per: {}, per_r:{}, mse:{}, mse_r:{}\n'.format(running_per, running_per_r, running_mse, running_mse_r))
                        f.write('time usage: {}'.format(round(time.time() - infer_time,2)))
                else:
                    break
            # with torch.no_grad indent
        n_iter+=1
    print('Finished Training')