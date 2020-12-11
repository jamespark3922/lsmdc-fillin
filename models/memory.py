from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .modules.face_attention import FaceAttention
from .modules.sent_embedding import SentEmbedding

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.norm(X, dim=1, keepdim=True) + 1e-7
    X = torch.div(X, norm)
    return X

class MemoryGenerator(nn.Module):
    """
        Baseline Memory with attended facial feature + visual & sent feature
    """
    def __init__(self, opt):
        super(MemoryGenerator, self).__init__()
        self.blank_token = opt.blank_token

        # sent embed
        self.rnn_size = opt.rnn_size
        self.sent_embed = SentEmbedding(opt)
        self.use_bert_embedding = 'use_bert_embedding' in vars(opt) and opt.use_bert_embedding
        if self.use_bert_embedding:
            self.bert_embedding = nn.Linear(opt.bert_size, self.rnn_size)
            print('===Use Bert Embedding===', self.use_bert_embedding)
        if opt.combine_before_after:
            self.rnn_size *=2
        self.use_both_captions = opt.use_both_captions

        # video embed
        self.use_video = opt.use_video
        self.fc_feat_size = opt.fc_feat_size
        self.video_encoding_size = opt.video_encoding_size
        if self.use_video:
            self.fc_embed = nn.Linear(self.fc_feat_size, self.video_encoding_size)

        # img embed
        self.use_img = opt.use_img
        self.img_feat_size = opt.img_feat_size
        self.img_encoding_size = opt.img_encoding_size
        if self.use_img:
            self.img_embed = nn.Linear(self.img_feat_size, self.img_encoding_size)

        # memory embed
        self.memory_encoding_size = opt.encoding_size

        # face embed
        self.face_encoding_size = opt.face_encoding_size
        self.face_feat_size = opt.face_feat_size - 6
        self.face_embed = nn.Linear(self.face_feat_size, self.face_encoding_size)
        self.face_attention = FaceAttention(self.video_encoding_size + self.img_encoding_size + self.rnn_size +
                                             self.rnn_size * self.use_both_captions,
                                             self.face_encoding_size)

        # final encoder
        self.encoder = nn.Linear(self.rnn_size + self.rnn_size * self.use_both_captions + self.face_encoding_size + 2,
                                 self.memory_encoding_size)

        # dropout
        self.drop_prob_lm = opt.drop_prob_lm
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.classify_gender = opt.classify_gender
        if self.classify_gender:
            self.gender_face_embed = nn.Linear(self.face_encoding_size, self.face_encoding_size)
            self.gender_logit = nn.Sequential(
                nn.Linear(self.memory_encoding_size, self.memory_encoding_size),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(self.memory_encoding_size, 3),
            )

        self.init_weights()

        self.alpha = None

    def init_weights(self):
        initrange = 0.1

        if self.use_video:
            self.fc_embed.weight.data.uniform_(-initrange, initrange)
        if self.use_img:
            self.img_embed.weight.data.uniform_(-initrange, initrange)
        self.face_embed.weight.data.uniform_(-initrange, initrange)
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def fill_slots(self, slots, slot_size):
        batch_size = len(slots)
        filled_slots = np.zeros((batch_size, slot_size))
        for i in range(batch_size):
            filled_slots[i, :len(slots[i])] = slots[i]
        return torch.from_numpy(filled_slots).cuda().float()

    def get_alpha(self):
        return self.alpha

    def forward(self, fc_feats, img_feats, face_feats, face_masks, seqs, masks, bert_emb, slots, slot_size):
        # fc_feats = batch_size x sent_num x frame_num x feat_dim
        # seqs = batch_size x sent_num x seq_length

        memory = None
        batch_size = fc_feats.size(0)

        self.alpha = np.zeros((face_feats.size()[:3]))

        gender_sequence = []
        gender_logprobs = None
        for s in range(slot_size):
            seg_size = fc_feats[:,s].size(1)
            # get visual feature for corresponding slot
            if self.use_video:
                fc_feat = self.fc_embed(fc_feats[:,s])
            else:
                fc_feat = fc_feats.new_zeros((batch_size, seg_size, self.video_encoding_size))
            if self.use_img:
                img_feat = self.img_embed(img_feats[:,s])
            else:
                img_feat = fc_feats.new_zeros((batch_size, seg_size, self.img_encoding_size))

            face_feat = self.face_embed(face_feats[:,s,:,6:])
            face_mask = face_masks[:,s]

            if self.use_bert_embedding:
                bert_feat = self.bert_embedding(bert_emb[:,s])
                if self.use_both_captions:
                    sent_feat = self.sent_embed(seqs[:,s],masks[:,s])
                    label_feat = torch.cat((sent_feat, bert_feat), dim=1)
                else:
                    label_feat = bert_feat
            else:
                label_feat = self.sent_embed(seqs[:,s],masks[:,s])

            # attention for facial features with visual and sentence feature as query
            face_alpha, face_att_feat = self.face_attention(fc_feat, img_feat, label_feat, face_feat, face_mask)

            if self.classify_gender:
                gender_logprobs = F.log_softmax(self.gender_logit(self.gender_face_embed(face_att_feat)), dim=1)
                gender_sequence.append(gender_logprobs)

            memory_idx = fc_feats.new_full((batch_size,1),fill_value=s).cuda()
            slot_idx = slots[:,s].float().unsqueeze(1)

            encoded = self.encoder(torch.cat((memory_idx, slot_idx, label_feat, face_att_feat),dim=1)).unsqueeze(1)
            # add final feature to memory
            if s > 0:
                memory = torch.cat((memory, encoded),dim=1)
            else:
                memory = encoded

            self.alpha[:,s] = face_alpha.data.cpu().numpy()

        if self.classify_gender:
            gender_logprobs = torch.cat([_.unsqueeze(1) for _ in gender_sequence], 1).contiguous()

        return memory, gender_logprobs