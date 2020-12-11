from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import numpy as np
import json
import string
import random
import os
import sys
import subprocess
import pickle
import csv

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def eval_fillin(preds, model_id, split, remove=False):
    import sys
    sys.path.append("codalab-2019-fill_in")
    results = []
    for pred in preds:
        info = {'video_id': pred['video_id']}
        info['characters'] = pred['characters']
        results.append(info)
    if remove:
        model_id += id_generator() # to avoid processing and removing same ids
    with open(os.path.join('character_eval', 'characters', 'character_' + model_id + '.csv'), 'w') as f:
        keys = ['video_id', 'characters']
        dict_writer = csv.DictWriter(f, keys, delimiter='\t')
        dict_writer.writerows(results)
        f.close()

    eval_command = ["python","eval_characters.py", "-s",'characters/character_' + model_id + '.csv', "--split", split ,
                    "-o", 'results/result_' + model_id + '.json']
    subprocess.call(eval_command,cwd='character_eval')

    with open(os.path.join('character_eval', 'results','result_' + model_id + '.json'),'r') as f:
        output = json.load(f)
        f.close()
    if remove: # remove for validation
        os.remove(os.path.join('character_eval','characters','character_' + model_id + '.csv'))
        os.remove(os.path.join('character_eval','results','result_' + model_id + '.json'))
    return output

def calculate_gender_accuracy(preds):
    correct = []
    f_recall = []
    f_precision = []
    for pred in preds:
        if type(pred['genders']) is not str:
            for i in range(len(pred['genders'])):
                correct.append(pred['genders'][i] == pred['gt_genders'][i])
                if pred['gt_genders'][i] == 1:
                    f_recall.append(pred['genders'][i] == pred['gt_genders'][i])
                if pred['genders'][i] == 1:
                    f_precision.append(pred['genders'][i] == pred['gt_genders'][i])

    return np.mean(correct), np.mean(f_recall), np.mean(f_precision)

def decode_sequence(ix_to_word, seq):
    D = seq.shape[0]
    txt = ''
    for j in range(D):
        ix = seq[j]
        if ix > 0 :
            if j >= 1:
                txt = txt + ' '
            txt = txt + ix_to_word[str(ix)]
        else:
            break
    return txt

def lst2string(lst):
    txt = ''
    for l in lst:
        txt+= '[' + str(l) + ']' + ','
    return txt[:-1]

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

def eval_split(gen_model, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_videos = eval_kwargs.get('num_videos', eval_kwargs.get('val_videos_use', -1))
    split = eval_kwargs.get('split', 'val')
    eval_accuracy = eval_kwargs.get('eval_accuracy', 0)
    ensemble = eval_kwargs.get('ensemble', 0)


    sample_max = eval_kwargs.get('sample_max', 1)
    beam_size = eval_kwargs.get('beam_size', 1)
    num_samples = eval_kwargs.get('num_samples', 1)
    num_captions = eval_kwargs.get('num_captions', 1)
    remove_result = eval_kwargs.get('remove', 0) # usually remove captions in validation stage but not in test.
    seed = eval_kwargs.get('seed', 1234)


    model_id = eval_kwargs.get('id', eval_kwargs.get('val_id', ''))

    if split == 'val':
        model_id = 'val_' + model_id

    if sample_max:
        assert num_captions <= beam_size
    else:
        assert num_captions <= num_samples

    # if use_context:
    #     gen_model.use_context()
    # Make sure in the evaluation mode
    gen_model.eval()

    loader.reset_iterator(split)

    n = 0
    losses = []
    loss = 0
    predictions = []
    first_id = None
    saw_first = False
    classify_gender = False

    visualize = []

    max_alphas =[]

    while True:
        data = loader.get_batch(split)

        tmp = [data['fc_feats'], data['img_feats'], data['face_feats'], data['face_masks'],
               data['captions'], data['masks'], data['bert_emb'], data['slots'], data['slot_masks'], data['characters'],
               data['genders']]

        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, img_feats, face_feats, face_masks, captions, masks, bert_emb, slots, slot_masks, characters, genders = tmp

        slot_size = data['slot_size']
        torch.manual_seed(seed)

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            # calculate loss
            if split != 'test' and not ensemble:
                loss = gen_model(fc_feats, img_feats, face_feats, face_masks, captions, masks, bert_emb,
                                 slots, slot_masks,slot_size, characters, genders)
                loss = loss.mean()
                losses.append(loss.item())
            if split == 'test' or eval_accuracy:
                predicted_characters, predicted_genders = gen_model(fc_feats, img_feats, face_feats, face_masks,
                                                                    captions, masks, bert_emb, slots, slot_masks, slot_size,
                                                                    mode='predict')
                torch.cuda.synchronize()
                predicted_characters, predicted_genders = parse_predictions(predicted_characters, predicted_genders, slots)

                # if split == 'test':
                alphas = gen_model.get_alpha()

        g_id = -1
        b = -1
        s = 0
        b_s = 0
        slots = slots.data.cpu().numpy()
        # print and store actual decoded sentence
        for info in data['infos']:
            entry = {'video_id': info['id'],'group_id' : info['g_index'],
                     'caption' : decode_sequence(loader.get_vocab(), info['caption'][1:-1]),
                     'characters' : '_', 'genders' : '_',
                     'gt_characters' : '_', 'gt_genders' : '_'}
            if g_id != entry['group_id']:
                if not info['skipped']:
                    b+=1
                s_k = 0
                b_s = 0
                g_id = entry['group_id']

            if not info['skipped'] and s_k in slots[b] and eval_accuracy:
                num_clips = np.count_nonzero(slots[b]==s_k)
                entry['characters'] = lst2string(predicted_characters[s:s+num_clips])
                entry['genders'] = predicted_genders[s:s+num_clips]

                if alphas is not None:
                    entry['alphas'] = alphas[b] # , s_k]
                    # entry['detections'] = data['face_feats'][b,:,:,:6]# s_k, :, :6]

                if split != 'test':
                    entry['gt_characters'] = lst2string(characters[b][b_s:b_s+num_clips].data.cpu().numpy())
                    if genders is not None:
                        classify_gender = True
                        entry['gt_genders'] = genders[b][b_s:b_s+num_clips].data.cpu().numpy()

                s += num_clips
                b_s+=num_clips

            s_k+=1
            predictions.append(entry)


            if verbose:
                print('video %s: caption: %s; predicted characters: %s ; gt_characters: %s; predicted_genders: %s; gt_genders: %s' %
                      (entry['video_id'], entry['caption'].encode('ascii','ignore'), entry['characters'], entry['gt_characters'],
                       entry['genders'], entry['gt_genders']))

        # if we wrapped around the split or used up val imgs budget then bail
        if n == 0:
            first_id = predictions[0]['group_id']
        n = n + loader.batch_size
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        sys.stdout.write("\revaluating validation preformance... %d/%d (%f)" %(ix0 - 1, ix1, loss))
        sys.stdout.flush()

        if data['bounds']['wrapped']:
            img_id = predictions[-1]['group_id']
            while True:
                predictions.pop()
                cur_id = predictions[-1]['group_id']
                if cur_id != img_id:
                    img_id = cur_id
                    if saw_first:
                        break
                if cur_id == first_id:
                    saw_first = True
            break
        if num_videos >= 0 and n >= num_videos:
            break

    # Switch back to training mode
    gen_model.train()

    # calculate accuracy scores
    gen_loss = np.mean(losses)
    accuracy = None
    if eval_accuracy:
        accuracy = eval_fillin(predictions, model_id, split, remove=remove_result)

        if split != 'test' and classify_gender:
            gender_accuracy, female_recall, female_precision = calculate_gender_accuracy(predictions)
            print('gender_accuracy: ', gender_accuracy)
            print('female recall: ', female_recall)
            print('female precision: ', female_precision)
            print('female F1: ', 2 * female_recall * female_precision / (female_recall + female_precision))

    return gen_loss, predictions, accuracy
