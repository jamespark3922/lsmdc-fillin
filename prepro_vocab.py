import re
import json
import argparse
import os
import csv

import numpy as np
import h5py

def build_vocab(params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    max_len = []
    csvs = ['LSMDC16_annos_training', 'LSMDC16_annos_val', 'LSMDC16_annos_test']
    for c in csvs:
        with open(os.path.join(params['input_path'], '%s_blank.csv' % c)) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter='\t')
            for row in csv_reader:
                # remove punctuation but keep possessive because we want to separate out character names
                row[5] = row[5].replace("[...]'s", "SOMEONE's")
                # row[5] = row[5].replace("[...]", " [...] ")
                row[5] = row[5].replace("[...]", "<blank>")
                ws = re.sub(r'[.!,;?]', ' ', str(row[5]).lower()).replace("'s", " 's").split()
                if "<blank>" in row[5]:
                    max_len.append(len(ws))
                for w in ws:
                    counts[w] = counts.get(w, 0) + 1
    max_len = np.array(max_len)
    print('avg seq length', np.mean(max_len))
    print('max seq length', max_len[max_len.argsort()[-20:][::-1]])

    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
    vocab.append('SOMEONE')

    splits = ['train', 'val', 'test']
    videos = []
    movie_ids = {}
    movie_set = set()
    vid = 0
    groups = []
    gid = -1

    max_count = -1
    max_unique_count = -1
    c_ids = []

    for i,c in enumerate(csvs):
        split = splits[i]
        for j in range(5):
            with open(os.path.join(params['input_path'], '%s_blank.csv' % c)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                if split != 'test':
                    id_list = list(csv.reader(open(os.path.join(params['input_path'], '%s_onlyIDs_NEW.csv' % c)),
                                              delimiter='\t'))
                skip_first = 0
                for r, row in enumerate(csv_reader):
                    clip = row[0]
                    movie = clip[:clip.rfind('_')]
                    info = {'id': vid, 'split': split, 'movie': movie, 'clip': clip}
                    if split != 'test':
                        c_id = id_list[r][-1]
                        if c_id == '_':
                            c_id = []
                        else:
                            c_id = c_id.split(',') if ',' in c_id else [c_id]
                        info['character_id'] = c_id
                    else:
                        c_id = []

                    if movie not in movie_set:
                        if skip_first < j:
                            skip_first += 1
                            vid += 1
                            continue
                        movie_set.add(movie)
                        gcount = 0
                        skip_first = 0

                        gid+=1
                        ginfo = {'id': gid, 'split': split, 'movie': movie, 'videos': [vid]}
                        groups.append(ginfo)
                        if movie not in movie_ids:
                            movie_ids[movie] = [gid]
                        c_ids = c_id

                    else:
                        if gcount >= params['group_by']:
                            gcount = 0
                            gid += 1
                            ginfo = {'id': gid, 'split': split, 'movie': movie, 'videos': [vid]}
                            groups.append(ginfo)
                            movie_ids[movie].append(gid)
                            c_ids = c_id
                        else:
                            groups[gid]['videos'].append(vid)
                            c_ids = c_ids + c_id

                    max_count = max(max_count, len(c_ids))
                    max_unique_count = max(max_unique_count, len(set(c_ids)))

                    row[5] = row[5].replace("[...]", " [...] ")
                    row[5] = row[5].replace("[...]", "<blank>")
                    ws = re.sub(r'[.!,;?]', ' ', str(row[5]).lower()).split()

                    caption = ['<eos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
                    info['final_caption'] = caption
                    info['num_blanks'] = caption.count('<blank>')
                    if j == 0:
                        videos.append(info)
                    vid+=1
                    gcount+=1

                if split != 'train':
                    break
                else:
                    if j < 4:
                        vid = 0
                    gcount = 0
                    movie_set = set()

    print('max number of characters per video', max_count)
    print('max number of unique characters per video', max_unique_count)

    return videos, groups, movie_ids, vocab, max_count, max_unique_count

def build_label(params, videos, wtoi):
    max_length = params['max_length']
    N = len(videos)

    label_arrays = []
    label_lengths = np.zeros(N, dtype='uint32')
    bt = 0
    for i, video in enumerate(videos):
        if 'final_caption' not in video:
            bt+=1
            continue
        s = video['final_caption']
        assert len(s) > 0, 'error: some video has no captions'

        Li = np.zeros((max_length), dtype='uint32')
        label_lengths[i] = min(max_length, len(s))  # record the length of this sequence
        for k, w in enumerate(s):
            if k < max_length - 1:
                Li[k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
    total = N - bt
    labels = np.array(label_arrays)[:total]  # put all the labels together
    label_lengths = label_lengths[:total]
    assert labels.shape[0] == total, 'lengths don\'t match? that\'s weird'
    assert labels[:,-1].sum() == 0 , 'sequences do not end on <eos>'
    assert np.all(label_lengths > 2), 'error: some caption had no words?'

    print('encoded captions to array of size ', labels.shape)
    return labels, label_lengths

def main(params):
    # create the vocab
    videos, groups, movie_ids, vocab, max_character_count, unique_character_count = build_vocab(params)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    output_path = params['output_path']
    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    out['blank'] = wtoi['<blank>']
    out['someone'] = wtoi['SOMEONE']
    out['max_character_count'] = max_character_count
    out['unique_character_count'] = unique_character_count
    out['videos'] = videos
    out['groups'] = groups
    out['movie_ids'] = movie_ids
    out['max_seq_length'] = params['max_length']
    json.dump(out, open(os.path.join(output_path,'LSMDC16_info_fillin_new_augmented.json'), 'w'))

    labels, ll = build_label(params,videos,wtoi)
    f_lb = h5py.File(os.path.join(output_path,"LSMDC16_labels_fillin_new_augmented.h5"),"w")
    f_lb.create_dataset("labels", dtype='uint32', data=labels)
    f_lb.create_dataset("label_length", dtype='uint32', data=ll)

    print(f'saved output to `{output_path}`')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_path', type=str, 
                        help='directory containing csv files')
    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--max_length', default=100, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--group_by', default=5, type=int,
                        help='group # of clips as one video')
    parser.add_argument('--output_path', type=str, required=True,
                        help='path to save your output')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
