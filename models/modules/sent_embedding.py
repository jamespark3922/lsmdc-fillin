# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Update kwargs with defaults
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import misc.utils as utils
from .attention import Attention

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.norm(X, dim=1, keepdim=True) + 1e-7
    X = torch.div(X, norm)
    return X

def pad_sentences(seqs, masks):
    len_sents = (masks > 0).long().sum(1)
    len_sents, len_ix = len_sents.sort(0, descending=True)

    # dummy 1 for captions with length 0  (or character does not appear for that batch)
    # doesn't matter, because we only update the loss for pair of characters that are relevant
    ix = (len_sents == 0).nonzero().reshape(-1)
    len_sents[ix] = 1

    inv_ix = len_ix.clone()
    inv_ix.data[len_ix.data] = torch.arange(0, len(len_ix)).type_as(inv_ix.data)

    new_seqs = seqs[len_ix].contiguous()

    return new_seqs, len_sents, len_ix, inv_ix

def before_after_embeddings(seqs, masks, blank_token):
    '''

    :param seqs: B x L
    :param masks:  B x L
    :param blank_token: Token indicating where to cut off
    :return: before and after sequence [B x L; B x L]
             before and after sequence mask [B x L; B x L]
    '''

    seq_length = seqs.size(1)
    blank_ids = (seqs == blank_token).nonzero()
    before = seqs.new_zeros(seqs.size()).cuda().long()
    before_mask = masks.new_zeros(masks.size()).cuda().long()

    for blank_id in blank_ids:
        before[blank_id[0], :blank_id[1]+1] = seqs[blank_id[0], :blank_id[1]+1]
        before_mask[blank_id[0], :blank_id[1]+1] = masks[blank_id[0], :blank_id[1]+1]

    after = seqs.new_zeros(seqs.size()).cuda().long()
    after_mask = masks.new_zeros(masks.size()).cuda().long()
    for blank_id in blank_ids:
        after_mask[blank_id[0], :seq_length - blank_id[1]] = masks[blank_id[0], blank_id[1]:]
        mask_length = after_mask[blank_id[0]].nonzero()[-1]+1
        tensor = seqs[blank_id[0], blank_id[1]:]
        inv_idx = torch.arange(tensor.size(0) - 1, -1, -1).long()
        after[blank_id[0], :mask_length] = tensor[inv_idx][-mask_length:]

    return before, before_mask, after, after_mask


class SentEmbedding(nn.Module):
    def __init__(self, opt):
        super(SentEmbedding, self).__init__()
        emb_type = opt.sent_type
        # Update kwargs here
        if emb_type == "rnn":
            self.module = BiLSTMTextEmbedding(opt)
            print('===Sentence Embedding: BLSTM===')
        elif emb_type == "transformer":
            self.module = TransformerTextEmbedding(opt)
            print('===Sentence Embedding: Transformer===')
        elif emb_type == "bert":
            self.module = BertTextEmbedding(opt)
            print('===Sentence Embedding: Bert==')
        else:
            raise NotImplementedError("Unknown question embedding '%s'" % emb_type)
        print('===Sentence Pool Type: %s ===' % opt.sent_pool_type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class TransformerTextEmbedding(nn.Module):
    def __init__(self,opt):
        super(TransformerTextEmbedding, self).__init__()

        self.vocab_size = opt.vocab_size
        self.blank_token = opt.blank_token
        self.rnn_size = opt.rnn_size

        self.glove = opt.glove_npy
        if self.glove is not None:
            self.word_encoding_size = 300
        else:
            self.word_encoding_size = opt.word_encoding_size
        self.word_embed = nn.Embedding(self.vocab_size, self.word_encoding_size)
        self.drop_prob_lm = opt.drop_prob_lm
        self.dropout = nn.Dropout(self.drop_prob_lm)

        encoder_layer = torch.nn.TransformerEncoderLayer(self.word_encoding_size, nhead=6)
        encoder_norm = torch.nn.LayerNorm(self.word_encoding_size)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, 6, encoder_norm)
        self.sent_embed = nn.Linear(self.word_encoding_size, self.rnn_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.glove is not None:
            self.word_embed.load_state_dict({'weight': torch.from_numpy(self.glove)})
        else:
            self.word_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, seqs, masks):
        masks[:,0] = 1
        transformer_masks = masks.bool() ^ 1
        words = self.word_embed(seqs)
        out = self.transformer_encoder(words.transpose(0,1), src_key_padding_mask=transformer_masks).transpose(0,1)
        batch_size = out.size(0)
        blank_tokens = (seqs == self.blank_token).nonzero()
        blank_idx = out.new_zeros(batch_size).cuda().long()
        blank_idx[blank_tokens[:,0]] = blank_tokens[:,1]
        blank_out = out[torch.arange(batch_size),blank_idx]
        label_feat = self.sent_embed(blank_out)
        return label_feat


class BiLSTMTextEmbedding(nn.Module):
    def __init__(self,opt):
        super(BiLSTMTextEmbedding, self).__init__()

        self.vocab_size = opt.vocab_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.blank_token = opt.blank_token
        self.num_layers = opt.num_layers
        self.seq_length = opt.seq_length
        if self.rnn_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif self.rnn_type.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.glove = opt.glove_npy
        if self.glove is not None:
            self.word_encoding_size = 300
        else:
            self.word_encoding_size = opt.word_encoding_size
        self.word_embed = nn.Embedding(self.vocab_size, self.word_encoding_size)
        self.before_after = opt.before_after
        self.combine_before_after = opt.combine_before_after and self.before_after
        self.bidirectional = opt.bidirectional and not self.before_after
        self.drop_prob_lm = opt.drop_prob_lm
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.sent_rnn = self.rnn_cell(self.word_encoding_size, self.rnn_size,
                                      self.num_layers, dropout=self.drop_prob_lm, batch_first=True,
                                      bidirectional=bool(self.bidirectional))
        self.pool_type = opt.sent_pool_type
        if 'att' in self.pool_type:
            self.sent_att = Attention(self.rnn_size + self.rnn_size * self.bidirectional)
        self.l2norm = opt.l2norm
        self.sent_embed = nn.Linear(2 * self.rnn_size, self.rnn_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.glove is not None:
            self.word_embed.load_state_dict({'weight': torch.from_numpy(self.glove)})
        else:
            self.word_embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(num_layers, bsz, self.rnn_size),
                    weight.new_zeros(num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(num_layers, bsz, self.rnn_size)

    def get_hidden_state(self,state):
        if self.rnn_type == "lstm":
            return_state = state[0].transpose(0,1).contiguous()
        else:
            return_state = state.transpose(0,1).contiguous()
        if self.bidirectional:
            return_state = return_state.view(return_state.size(0),1,-1)
        return return_state

    def forward(self, seqs, masks):

        if self.before_after:
            before, before_mask, after, after_mask = before_after_embeddings(seqs, masks, self.blank_token)
            before_embeddings = self._forward(before, before_mask, 'last')
            after_embeddings = self._forward(after, after_mask, 'last')
            ba = self.sent_embed(torch.cat((before_embeddings, after_embeddings),dim=1))
            if not self.combine_before_after:
                return ba

        out = self._forward(seqs, masks, self.pool_type)
        if not self.combine_before_after:
            return self.sent_embed(out)
        else:
            return torch.cat((out, ba), dim=1)

    def _forward(self, seqs, masks, pool_type):

        batch_size = seqs.size(0)
        state = self.init_hidden(batch_size)
        padded_seqs, sorted_lens, len_ix, inv_ix = pad_sentences(seqs, masks) # sort by len
        label_feat = self.dropout(self.word_embed(padded_seqs))
        label_pack = pack_padded_sequence(label_feat, list(sorted_lens.data), batch_first=True)
        out, state = self.sent_rnn(label_pack, state)
        padded = pad_packed_sequence(out, batch_first=True)
        if pool_type == 'max':
            out = padded[0]
            _masks = masks[len_ix][:, :out.size(1)].float()
            out = (out * _masks.unsqueeze(-1) + (_masks == 0).unsqueeze(-1).float() * -1e10).max(1)[0]
        elif 'blank_out' in pool_type:
            out = padded[0]
        else:
            out = self.get_hidden_state(state).squeeze(1)

        if self.l2norm:
            out = l2norm(out)
        out = out[inv_ix].contiguous() # resort by original batch index
        if 'blank_out' in pool_type:
            blank_tokens = (seqs == self.blank_token).nonzero()
            blank_idx = out.new_zeros(batch_size).cuda().long()
            blank_idx[blank_tokens[:, 0]] = blank_tokens[:, 1]
            blank_out = out[torch.arange(batch_size), blank_idx]

            if 'att' in pool_type:
                non_blank_idx = out.new_ones(batch_size,out.size(1)).cuda().long()
                non_blank_idx[torch.arange(batch_size), blank_idx] = 0
                non_blank_out = out[non_blank_idx.bool()].view(batch_size,out.size(1)-1,-1)
                a_out, out = self.sent_att(blank_out, non_blank_out)

            else:
                out = blank_out

        return out

class BertTextEmbedding(nn.Module):
    def __init__(self,opt):
        from pytorch_transformers import BertTokenizer, BertModel
        super(BertTextEmbedding, self).__init__()
        # bert_config = BertConfig.from_json_file('bert_config.json')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.blank_id = self.tokenizer._convert_token_to_id(self.tokenizer.mask_token)
        self.emb_type = opt.sent_type

        self.language_model = BertModel.from_pretrained('bert-base-uncased')
        self.rnn_size = opt.rnn_size
        self.sent_embed = nn.Sequential(
                            nn.Linear(self.language_model.config.hidden_size, self.rnn_size),
                            nn.Tanh()
        )
        self.vocab = opt.vocab
        self.pool_type = opt.sent_pool_type

    def get_bert_ids(self, seqs):
        input_ids = []
        txts = utils.decode_sequence(self.vocab, seqs[:,1:], add_punct=True)

        txts = [txt.replace('<blank>', self.tokenizer.mask_token).replace('<UNK>', self.tokenizer.unk_token) for txt in txts]
        ids = [self.tokenizer.convert_tokens_to_ids(["[CLS]"] + self.tokenizer.tokenize(txt) + ["[SEP]"]) for txt in txts]
        max_seq_length = max([len(input_id) for input_id in ids])
        pad_token = self.tokenizer.pad_token
        for input_id in ids:
            padding = [self.tokenizer._convert_token_to_id(pad_token)] * (max_seq_length - len(input_id))
            input_id += padding
            input_ids.append(input_id)

        input_ids = torch.tensor(input_ids, dtype=torch.long).cuda().contiguous()
        return input_ids

    def forward(self, seqs, masks):

        # forward bert masked lm
        input_ids = self.get_bert_ids(seqs)
        hidden, pooled_output = self.language_model(input_ids)

        # select output for blank token

        if self.pool_type == 'cls':
            label_feat = self.sent_embed(pooled_output)
        elif self.pool_type == 'max':
            label_feat = self.sent_embed(hidden.max(1)[0])
        elif self.pool_type == 'blank_out':
            batch_size = hidden.size(0)
            blank_ids = (input_ids == self.blank_id).nonzero()
            blank_idx = hidden.new_zeros(batch_size).cuda().long()
            blank_idx[blank_ids[:,0]] = blank_ids[:,1]
            blank_out = hidden[torch.arange(batch_size),blank_idx]
            label_feat = self.sent_embed(blank_out)

        return label_feat


class BertLM(nn.Module):
    def __init__(self,opt, bert_directory):
        from pytorch_transformers import BertTokenizer, BertMaskedLM, BertConfig
        super(BertLM, self, bert_directory).__init__()
        # bert_config = BertConfig.from_json_file('bert_config.json')
        # bert_config.output_hidden_states = True
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.blank_id = self.tokenizer._convert_token_to_id(self.tokenizer.mask_token)
        self.emb_type = opt.sent_type

        self.language_model = BertMaskedLM.from_pretrained('bert-base-uncased')
        self.rnn_size = opt.rnn_size
        self.sent_embed = nn.Sequential(
                            nn.Linear(self.language_model.config.hidden_size, self.rnn_size),
                            nn.Tanh()
        )
        self.vocab = opt.vocab
        self.pool_type = opt.sent_pool_type

    def get_bert_ids(self, seqs):
        input_ids = []
        txts = utils.decode_sequence(self.vocab, seqs[:,1:], add_punct=True)

        txts = [txt.replace('<blank>', self.tokenizer.mask_token).replace('<UNK>', self.tokenizer.unk_token) for txt in txts]
        ids = [self.tokenizer.convert_tokens_to_ids(["[CLS]"] + self.tokenizer.tokenize(txt) + ["[SEP]"]) for txt in txts]
        max_seq_length = max([len(input_id) for input_id in ids])
        pad_token = self.tokenizer.pad_token
        for input_id in ids:
            padding = [self.tokenizer._convert_token_to_id(pad_token)] * (max_seq_length - len(input_id))
            input_id += padding
            input_ids.append(input_id)

        input_ids = torch.tensor(input_ids, dtype=torch.long).cuda().contiguous()
        return input_ids

    def forward(self, seqs, masks):

        # forward bert masked lm
        input_ids = self.get_bert_ids(seqs)
        # if input_ids.size(1) > 50:
        #     input_ids = input_ids[:,:50]
        hidden, pooled_output = self.language_model(input_ids)

        # select output for blank token

        if self.pool_type == 'cls':
            label_feat = self.sent_embed(pooled_output)
        elif self.pool_type == 'max':
            label_feat = self.sent_embed(hidden.max(1)[0])
        elif self.pool_type == 'blank_out':
            batch_size = hidden.size(0)
            blank_ids = (input_ids == self.blank_id).nonzero()
            blank_idx = hidden.new_zeros(batch_size).cuda().long()
            blank_idx[blank_ids[:,0]] = blank_ids[:,1]
            blank_out = hidden[torch.arange(batch_size),blank_idx]
            label_feat = self.sent_embed(blank_out)

        return label_feat