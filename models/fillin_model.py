from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import MemoryGenerator
from .modules.attention import Attention

def print_norm(output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('output: ', output)
    print('grad size:', output.data.size())
    print('grad norm:', output.data.norm())

class FillInCharacter(nn.Module):
    def __init__(self, opt):
        super(FillInCharacter, self).__init__()
        self.memory_generator = MemoryGenerator(opt)
        self.memory_encoding_size = opt.encoding_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.classifier_type = opt.classifier_type
        if self.classifier_type == 'transformer':
            self.transformer = torch.nn.Transformer(self.memory_encoding_size)
            print('===fillin classifier: transformer===')
        else :
            self.rnn_size = opt.rnn_size
            self.num_layers = opt.num_layers
            self.seq_length = opt.seq_length
            self.rnn_type = opt.rnn_type
            if self.rnn_type.lower() == 'lstm':
                self.rnn_cell = nn.LSTM
            elif self.rnn_type.lower() == 'gru':
                self.rnn_cell = nn.GRU
            self.decoder = self.rnn_cell(self.rnn_size + self.memory_encoding_size, self.rnn_size,
                                       self.num_layers, dropout=self.drop_prob_lm, batch_first=True)
            self.attention = Attention(self.rnn_size)
            print('===fillin classifier: rnn===')
        num_classes = opt.unique_characters + 1
        self.logit = nn.Linear(self.memory_encoding_size, num_classes)
        self.character_embed = nn.Embedding(num_classes, self.memory_encoding_size)
        self.loss = IdentificationLoss()

        self.classify_gender = opt.classify_gender
        if opt.classify_gender:
            self.gender_logit = nn.Sequential(
                                        nn.Linear(self.memory_encoding_size,self.memory_encoding_size),
                                        nn.Dropout(),
                                        nn.ReLU(),
                                        nn.Linear(self.memory_encoding_size, 3),
                                )
            self.gender_loss = IdentificationLoss()
            self.gender_loss_weight = opt.gender_loss
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.character_embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def get_hidden_state(self,state):
        if self.rnn_type == "lstm":
            return state[0].transpose(0,1).cuda()
        else:
            return state.transpose(0,1).cuda()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _forward(self, fc_feats, img_feats, face_feats, face_masks, captions, caption_masks, bert_emb, slots, slot_masks, slot_size,
                characters, genders=None):
        # get memory of features for each slot
        memory, gender_logprobs = self.memory_generator(fc_feats, img_feats, face_feats, face_masks, captions, caption_masks, bert_emb, slots, slot_size)
        characters = characters[:,:slot_size+1]
        character_embed = self.character_embed(characters)
        masks = slot_masks[:,:slot_size+1].bool()

        if self.classifier_type == 'transformer':
            transformer_masks = ~masks
            src_mask = self.transformer.generate_square_subsequent_mask(masks.size(1)-1).cuda()
            tgt_mask = self.transformer.generate_square_subsequent_mask(masks.size(1)).cuda()
            logits = self.transformer(memory.transpose(0, 1), character_embed.transpose(0, 1),
                                      src_mask = src_mask,
                                      tgt_mask = tgt_mask,
                                      src_key_padding_mask=transformer_masks[:,1:],
                                      tgt_key_padding_mask=transformer_masks,
                                      memory_key_padding_mask=transformer_masks[:,1:])
            logprobs = F.log_softmax(self.logit(self.dropout(logits.transpose(0, 1)[:,:-1])),dim=2)

        else:
            sequence = []
            batch_size = memory.size(0)

            # decoder
            state = self.init_hidden(batch_size)
            for i in range(memory.size(1)):
                att_encode_output = self.attention(self.get_hidden_state(state).squeeze(1), memory)
                dec_input = torch.cat((att_encode_output, character_embed[:,i]),dim=1).unsqueeze(1)
                output, state = self.decoder(dec_input, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(1))), dim=1)
                sequence.append(logprobs)
            logprobs = torch.cat([_.unsqueeze(1) for _ in sequence], 1).contiguous()

        if self.classify_gender:
            genders = genders[:, :memory.size(1)]
            loss = self.loss(logprobs, characters[:,1:], masks[:,1:].float()) + \
                   self.gender_loss_weight * self.gender_loss(gender_logprobs, genders, masks[:,1:].float())
        else:
            loss = self.loss(logprobs, characters[:,1:], masks[:,1:].float())
        return loss

    def _predict(self, fc_feats, img_feats, face_feats, face_masks, captions, caption_masks, bert_emb, slots, slot_masks, slot_size):
        # get memory of features for each slot
        memory, gender_logprobs = self.memory_generator(fc_feats, img_feats, face_feats, face_masks, captions, caption_masks, bert_emb, slots, slot_size)
        batch_size = fc_feats.size(0)
        masks = slot_masks[:, :slot_size + 1].bool()

        if self.classifier_type == 'transformer':
            transformer_masks = ~masks
            src_mask = self.transformer.generate_square_subsequent_mask(masks.size(1) - 1).cuda()
            enc_output = self.transformer.encoder(memory.transpose(0, 1),
                                                  mask=src_mask,
                                                  src_key_padding_mask=transformer_masks[:,1:])

            predictions = fc_feats.new_zeros(batch_size, slot_size+1, dtype=torch.long)
            for i in range(slot_size):
                character_embed = self.character_embed(predictions[:,:i+1]).transpose(0,1)
                tgt_transformer_masks = transformer_masks[:,:i+1]
                dec_output = self.transformer.decoder(character_embed, enc_output,
                                                tgt_mask=self.transformer.generate_square_subsequent_mask(i+1).cuda(),
                                                tgt_key_padding_mask=tgt_transformer_masks,
                                                memory_key_padding_mask=transformer_masks[:,1:]).transpose(0,1)
                dec_logit = self.logit(dec_output)[:,-1,:]
                _, tgt = torch.max(dec_logit.data, 1)
                predictions[:,i+1] = tgt
            predictions = predictions[:,1:]
        else:
            batch_size = memory.size(0)
            predictions = fc_feats.new_zeros(batch_size, slot_size, dtype=torch.long)

            # decoder
            state = self.init_hidden(batch_size)
            it = fc_feats.new_zeros(batch_size, dtype=torch.long).cuda()
            for i in range(memory.size(1)):
                att_encode_output = self.attention(self.get_hidden_state(state).squeeze(1), memory)
                character_embed = self.character_embed(it)
                dec_input = torch.cat((att_encode_output, character_embed),dim=1).unsqueeze(1)
                output, state = self.decoder(dec_input, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(1))), dim=1)
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
                predictions[:,i] = it
        predicted_genders = predictions.new_zeros(predictions.size(0), predictions.size(1),dtype=torch.long)
        if self.classify_gender:
            _, predicted_genders = torch.max(gender_logprobs,2)
        return predictions, predicted_genders

    def get_alpha(self):
        return self.memory_generator.alpha

class IdentificationLoss(nn.Module):
    def __init__(self):
        super(IdentificationLoss, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

