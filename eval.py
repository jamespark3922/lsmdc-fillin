from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

from dataloader import *
import eval_utils
import misc.utils as utils
from models.fillin_model import FillInCharacter

import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--g_model_path', type=str, required=True,
                help='path to generator to evaluate')
parser.add_argument('--infos_path', type=str, required=True,
                help='path to infos to evaluate')
parser.add_argument('--caption_path', type=str,
                help='caption you wish to fill in')

# Basic options
parser.add_argument('--split', type=str, default='val',
                help='which split to use: val|test')
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_videos', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--use_context', type=int, default=0,
                help='use context during evaluation for dense model (1=yes,0=no)')

# weights for hybrid discriminator
parser.add_argument('--vis_weight', type=float, default=0.8,
                help='weight for visual discriminator in adversarial inference')
parser.add_argument('--lang_weight', type=float, default=0.2,
                help='weight for lang discriminator in adversarial inference')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--top_num', type=float, default=1.0,
                help='top k samples to maintain')
parser.add_argument('--num_samples', type=int, default=100,
                help='number to sample for each image/video. Used when sample_max is 0')
parser.add_argument('--num_captions', type=int, default=1,
                help='number of captions to consider for each image/video.')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=1,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')

# Input options to overwrite if needed
parser.add_argument('--input_fc_dir', type=str, default='',
                help='path to video features')
parser.add_argument('--input_img_dir', type=str, default='',
                help='path to image features')
parser.add_argument('--input_face_dir', type=str, default='',
                help='path to the face clusters')
parser.add_argument('--input_label_h5', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--bert_embedding_dir', type=str, default='',
                help='path to bert embedding')
parser.add_argument('--glove_npy', type=str, default='',
                help='path to glove numpy')
parser.add_argument('--input_json', type=str, default='',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')

parser.add_argument('--classify_gender', action='store_true'),
parser.add_argument('--clip_gender_json', type=str, help='clip gender json created by notebook')

# misc
parser.add_argument('--id', type=str, default='',
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--seed', type=int, default=1234,
                help='random seed to use')
parser.add_argument('--verbose', type=int, default=1,
                help='verbse print.')
parser.add_argument('--verbose_beam', type=int, default=0,
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0,
                help='if we need to calculate loss.')
parser.add_argument('--verbose_video', type=int, default=1,
                help='if we need to get all the language metrics for video evaluation.')
opt = parser.parse_args()

# Load infos
with open(opt.infos_path,'rb') as f:
    if sys.version_info[0] == 2:
        from six.moves import cPickle
        infos = cPickle.load(f)
    else:
        import pickle
        infos = pickle.load(f)
# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_img_dir = infos['opt'].input_img_dir
    opt.input_face_dir = infos['opt'].input_face_dir
if len(opt.input_label_h5) == 0:
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.glove_npy) == 0:
    opt.glove_npy = infos['opt'].glove_npy
if len(opt.id) == 0:
    opt.id = infos['opt'].id

ignore1 =["input_fc_dir", "input_img_dir", "input_box_dir", "input_face_dir", "input_label_h5", "glove_npy", "clip_gender_json", "input_json", "bert_embedding_dir"]
ignore2 = ["id", "batch_size", "beam_size", "start_from", "language_eval", "g_start_from", "d_start_from", "temperature",
          "vis_weight", "lang_weight", "pair_weight"]
for k in vars(infos['opt']).keys():
    if k not in ignore1 and k not in ignore2:
        if k in vars(opt) and 'gender' not in k:
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
opt.vocab = infos['vocab'] # ix -> word mapping

# Setup the model
gen_model = FillInCharacter(opt)
gen_model.load_state_dict(torch.load(opt.g_model_path))
gen_model.cuda()
gen_model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
loader = DataLoader(opt)

# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']
loss, split_predictions, lang_stats = eval_utils.eval_split(gen_model, loader, eval_kwargs=vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)
