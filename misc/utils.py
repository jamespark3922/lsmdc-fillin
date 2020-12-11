from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random


def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq,add_punct=False):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if len(txt) > 0 and add_punct: # for bert
            txt = txt + '.'
        txt = unicode(txt.encode('ascii', 'ignore'))
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

# split to batch_size * sent_num  x rest
def align_seq(sent_num,seq):
    # seq      = batch_size x sent_num x rest
    # aligned  = total_sent x rest
    batch_size = seq.size(0)
    total_sent = sum(sent_num)
    if len(seq.size()) >= 3:
        new_shape = [total_sent] + list(seq.size()[2:])
    else:
        new_shape = total_sent
    seq_aligned = seq.new_zeros(new_shape)

    cur = 0
    for i in range(batch_size):
        i_n = min(sent_num[i],seq[i].size(0))
        seq_aligned[cur: cur + i_n] = seq[i,:i_n]
        cur+=sent_num[i]

    return seq_aligned.contiguous()

# combine to batch_size x rest
def combine_seq(sent_num,seq):
    # seq      = batch_size x sent_num x seq_length
    # combined = batch_size x (sent_num * seq_length)
    assert len(seq.size()) == 3
    batch_size = seq.size(0)
    seq_combined = seq.new_zeros(batch_size, max(sent_num) * seq.size(2))
    cur = 0
    for i in range(batch_size):
        combined = seq[i,:sent_num[i]].view(-1)
        seq_combined[i,:combined.size(0)] = combined
    return seq_combined

def generate_paragraph_mask(sent_num,seq):
    assert len(seq.size()) == 3
    batch_size = seq.size(0)
    mask = seq.new_zeros(seq.size())
    ones = torch.ones(seq.size(2)).cuda()
    for i in range(batch_size):
        mask[i,:sent_num[i]] = ones.expand(sent_num[i],-1)
    return mask

def get_bert_masks(sent_num,seq,tokenizer,vocab,use_pair=False,eval=False,max_seq_length=40+2):
    input_ids = []
    input_masks = []
    input_tokens = []
    seq = align_seq(sent_num, seq)
    txts = decode_sequence(vocab, seq, add_punct=True)

    if not use_pair:
        for txt in txts:
            tokens_a = tokenizer.tokenize(txt)
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            input_token = [0] * len(input_id)
            padding = [0] * (max_seq_length - len(input_id))
            input_id += padding
            input_mask += padding
            input_token += padding

            assert len(input_id) == max_seq_length, \
                "input length: %d does not match maximum sequence length: %d" % (len(input_id), max_seq_length)
            assert len(input_mask) == max_seq_length, \
                "mask length: %d does not match maximum sequence length: %d" % (len(input_mask), max_seq_length)

            input_ids.append(input_id)
            input_masks.append(input_mask)
            input_tokens.append(input_token)
    else:
        prev_txt = ["[PAD]"]
        total_max_length = max_seq_length * 2 - 1
        sent_cnt = 0
        sent_idx = 0
        for txt in txts:
            tokens_a = tokenizer.tokenize(txt)
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            tokens = ["[CLS]"] + prev_txt + ["[SEP]"] + tokens_a + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_id)
            input_token = [0] * (len(prev_txt) + 2) + [1] * (len(tokens_a) + 1)
            prev_txt = tokens_a

            if eval or sent_cnt > 0: # only add if it's not the first sentence
                padding = [0] * (total_max_length - len(input_id))
                input_id += padding
                input_mask += padding
                input_token += padding

                assert len(input_id) == total_max_length, \
                "input length: %d does not match maximum sequence length: %d" % (len(input_id), total_max_length)
                assert len(input_mask) == total_max_length, \
                "mask length: %d does not match maximum sequence length: %d" % (len(input_mask), total_max_length)

                input_ids.append(input_id)
                input_masks.append(input_mask)
                input_tokens.append(input_token)

            sent_cnt+=1
            if sent_cnt == sent_num[sent_idx]:
                sent_cnt = 0
                sent_idx+=1
                prev_txt = ["[PAD]"]

    input_ids = torch.tensor(input_ids, dtype=torch.long).cuda().contiguous()
    input_masks = torch.tensor(input_masks, dtype=torch.long).cuda().contiguous()
    input_tokens = torch.tensor(input_tokens, dtype=torch.long).cuda().contiguous()
    return input_ids, input_masks, input_tokens

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
