import torch

import numpy as np

def get_characters(f_fc_feats, f_img_feats, f_face_feats, f_face_masks, sents, sent_num,
                   fillin_model, gender_model, tokenizer):
    # get the slots
    slots, slot_masks, slot_size = get_fillin_slots(sents, sent_num)
    f_fc_feats = expand_features(f_fc_feats, slots, slot_size)
    f_img_feats = expand_features(f_img_feats, slots, slot_size)
    f_face_feats = expand_features(f_face_feats, slots, slot_size)
    f_face_masks = expand_features(f_face_masks, slots, slot_size)

    # mask out captions with blanks and get their BERT embedding
    pars = sent2par(sents, sent_num)
    bert_emb = get_bert_emb(gender_model, tokenizer, pars, slot_size)
    slots = torch.from_numpy(slots).cuda()
    slot_masks = torch.from_numpy(slot_masks).cuda()
    predicted_characters, predicted_genders, character_logits = fillin_model(f_fc_feats, f_img_feats, f_face_feats,
                                                                             f_face_masks,
                                                                             None, None, bert_emb, slots, slot_masks,
                                                                             slot_size,
                                                                             mode='predict')
    predicted_characters, predicted_genders = parse_predictions(predicted_characters, predicted_genders, slots)
    torch.cuda.synchronize()

    return slots, predicted_characters, predicted_genders, character_logits

def sent2par(sents, sent_num):
    batch_size = len(sent_num)
    pars = []
    sent_c = 0
    for i in range(batch_size):
        txt = ""
        for j in range(sent_num[i]):
            txt += sents[sent_c].capitalize() + ' '
            sent_c += 1
        pars.append(txt[:-1])
    return pars

# get slot information for general caption with someone labels
def get_fillin_slots(sents, sent_num):
    batch_size = len(sent_num)
    slots = []
    sent_c = 0
    for i in range(batch_size):
        slot = []
        for j in range(sent_num[i]):
            slot = slot + [j] * sents[sent_c].count('<blank>')
            # slot = slot + [j] * sents[sent_c].count('someone')
            sent_c +=1
        slots.append(np.array(slot))

    slot_size = max([len(slot) for slot in slots])
    slot_batch = np.ones((batch_size, slot_size), dtype='int') * -1
    slot_masks = np.zeros((batch_size, slot_size + 1), dtype='int')
    for i in range(batch_size):
        slot_num = len(slots[i])
        if slot_num > 0:
            slot_batch[i, :slot_num] = slots[i]
            slot_masks[i, :slot_num+1] = 1

    return slot_batch, slot_masks, slot_size

def get_bert_emb(model, tokenizer, texts, slot_size):
    batch = get_bert_batch(tokenizer, texts)
    batch, masks = torch.tensor(batch[0]).cuda(), torch.tensor(batch[1]).cuda()
    inputs, labels = mask_tokens(batch, tokenizer)

    with torch.no_grad():
        outputs = model(inputs, attention_mask=masks, masked_lm_labels=labels)
        last_hidden_states = outputs[2][-1]
        cls = last_hidden_states[:,0]

        character_embedding = cls.new_zeros(batch.size(0), slot_size, 2*cls.size(1)).cuda()
        character_indices = (inputs == tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
        blank_indices = character_indices.nonzero()
        cur_b = -1
        for bi in blank_indices:
            b, i = bi
            if cur_b != b:
                j = 0
                cur_b+=1
            character_embedding[b,j] = torch.cat((cls[b], last_hidden_states[b,i]))
            j+=1
    return character_embedding

def get_bert_batch(tokenizer, texts):
    text_ids, attention_masks = [],[]
    pad_id = tokenizer._convert_token_to_id(tokenizer.pad_token)
    block_size = tokenizer.max_len_single_sentence
    for i in range(len(texts)):
        text = texts[i].replace('Someone', 'someone').replace('<blank>', '[person 1]')
        # text = texts[i].replace('Someone', 'someone').replace('someone', '[person 1]')
        tt = tokenizer.tokenize(text)
        tokenized_text = []

        j = 0
        while j < len(tt):
            if tt[j] == '[' and j <= len(tt) - 4 and tt[j + 1] == 'person' and tt[j + 3] == ']':
                tokenized_text.append('[unused%s]' % tt[j + 2])
                j += 4
            else:
                tokenized_text.append(tt[j])
                j += 1
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokenized_text = tokenizer.add_special_tokens_single_sequence(tokenized_text)
        attention_mask = [1] * len(tokenized_text)

        pad_length = (block_size + 2 - len(tokenized_text))
        tokenized_text += [pad_id] * pad_length
        attention_mask += [0] * pad_length
        text_ids.append(tokenized_text)
        attention_masks.append(attention_mask)
    return text_ids, attention_masks

def mask_tokens(inputs, tokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = (labels > 0) & (labels < 19)
    labels[~masked_indices] = -1
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels

def expand_features(feats, slots, slot_size):
    """ Copy features up to number of slots for each video """
    batch_size, sent_num = feats.size(0), feats.size(1)
    new_feats_shape = [batch_size, slot_size] + list(feats.size()[2:])
    new_feats = feats.new_zeros(new_feats_shape).cuda()
    for i in range(batch_size):
        if np.sum(slots[i]+1) != 0:
            slot = np.delete(slots[i], np.where(slots[i]==-1))
            el, freq = np.unique(slot, return_counts=True)
            c = 0
            for j in range(len(freq)):
                new_feats[i,c:c+freq[j],:] = feats[i,el[j]]
                c+=freq[j]
    return new_feats

# fill caption with character labels
def fillin_captions(caption, characters):
    for c in range(len(characters)):
        caption.replace('someone')
    return characters

def parse_predictions(predicted_characters, predicted_genders, slots):
    def count_slot(slot):
        return np.count_nonzero(slot + 1)
    slots = slots.data.cpu().numpy()
    predicted_characters = predicted_characters.data.cpu().numpy()
    predicted_genders = predicted_genders.data.cpu().numpy()

    total_pair = sum([count_slot(s) for s in slots])
    characters = np.zeros(total_pair)
    genders = np.zeros(total_pair)
    n = 0

    for i in range(len(slots)):
        cs = count_slot(slots[i])
        characters[n:n + cs] = predicted_characters[i, :cs]
        genders[n:n + cs] = predicted_genders[i, :cs]
        n += cs
    return characters.astype(int), genders.astype(int)
