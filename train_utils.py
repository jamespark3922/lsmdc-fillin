import torch
import misc.utils as utils

def train_generator(gen_model, gen_optimizer, loader, grad_clip=0.1):
    gen_model.train()
    data = loader.get_batch('train')
    torch.cuda.synchronize()
    tmp = [data['fc_feats'], data['img_feats'], data['face_feats'], data['face_masks'],
           data['captions'], data['masks'], data['bert_emb'], data['slots'], data['slot_masks'], data['characters'], data['genders']]

    tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
    fc_feats, img_feats, face_feats, face_masks, captions, masks, bert_emb, slots, slot_masks, characters, genders = tmp

    sent_num = data['sent_num']
    slot_size = data['slot_size']

    wrapped = data['bounds']['wrapped']
    gen_optimizer.zero_grad()

    loss = gen_model(fc_feats, img_feats, face_feats, face_masks, captions, masks, bert_emb, slots, slot_masks, slot_size,
                     characters, genders)
    loss = loss.mean()
    loss.backward()
    gen_loss = loss.item()

    utils.clip_gradient(gen_optimizer, grad_clip)
    gen_optimizer.step()
    torch.cuda.synchronize()

    return gen_loss, wrapped, sent_num