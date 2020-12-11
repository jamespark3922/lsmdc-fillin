import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    # ============================
    # General Options
    # ============================

    # Data input settings
    parser.add_argument('--input_json', type=str,
                    help='path to the json file containing additional info and vocab (img/video)')
    parser.add_argument('--input_fc_dir', type=str,
                        help='path to the directory containing the preprocessed fc video features')
    parser.add_argument('--input_img_dir', type=str,
                        help='path to the directory containing the image features (img)')
    parser.add_argument('--input_face_dir', type=str,
                        help='path to the directory containing the face features')

    parser.add_argument('--input_label_h5', type=str,
                    help='path to the h5file containing the preprocessed dataset (img/video)')
    parser.add_argument('--clip_gender_json', type=str, help='clip gender json provided in data')


    # Checkpoint Options
    parser.add_argument('--start_from', type=str, default=None,
                     help="""skip pre training step and continue training from saved generator model at this path.
                          'infos_{id}.pkl'         : configuration;
                          'gen_optimizer_{epoch}.pth'     : optimizer;
                          'gen_{epoch}.pth'         : model
                     """)
    parser.add_argument('--start_epoch', type=str, default="latest",
                     help="""start training generator at epoch (int, latest, latest_ce, latest_scst)
                     """)
    parser.add_argument('--pre_nepoch', type=int, default=80,
                        help='number of epochs to pre-train generator with cross entropy')

    # Feature options
    parser.add_argument('--fc_feat_size', type=int, default=1024,
                        help='1024 for i3d, 2048 for resnet, 4096 for vgg (img) \
                              500  for c3d,    8192 for r3d (video')
    parser.add_argument('--img_feat_size', type=int, default=2048,
                        help='img feat size')
    parser.add_argument('--face_feat_size', type=int, default=512 + 6,
                        help='face feat size')

    # Visual Input Options
    parser.add_argument('--use_video', type=int, default=1,
                        help='use video features (c3d/resnext101-64f) specified in input_fc_dir')
    parser.add_argument('--use_img', type=int, default=0,
                        help='use resnet features specified in input_img_dir')
    parser.add_argument('--use_face', type=int, default=1,
                        help='use face features')
    parser.add_argument('--max_face', type=int, default=10,
                        help='number of face features per clip')
    parser.add_argument('--max_sent_num', type=int, default=5,
                        help='max number of sentences per group (LSMDC has a group of 5 clips)')
    parser.add_argument('--max_seg', type=int, default=5,
                        help='max number of segments to divide the clip features')

    # ============================
    # Model Options
    # ============================

    # model type
    parser.add_argument('--classifier_type', type=str, default='transformer',
                 help='fillin_model classifier used given memory (rnn/transformer)')

    # gender options
    parser.add_argument('--classify_gender', action='store_true')
    parser.add_argument('--gender_loss', type=float, default=0.2)

    # bert embeddings
    parser.add_argument('--use_bert_embedding', action='store_true', help='use pretrained bert embedding to encode captions instead of from scratch')
    parser.add_argument('--bert_embedding_dir', type=str)
    parser.add_argument('--bert_size', type=int, default=1536)
    parser.add_argument('--use_both_captions', action='store_true')


    # Memory: Sentence Embedding Options
    parser.add_argument('--sent_type', type=str, default='rnn',
                        help='rnn or transformer for encoding sentence')
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--bidirectional', type=int, default=1)
    parser.add_argument('--before_after', action='store_true',
                        help='encode sentences before and after blank with rnn as two features')
    parser.add_argument('--combine_before_after', action='store_true',
                        help='combine before after')
    parser.add_argument('--sent_pool_type', type=str, default='last',
                        help='rnn pooling operation to use to get final sentence features (last/max)')

    # Memory: Encoding Options
    parser.add_argument('--video_encoding_size', type=int, default=256,
                        help='the encoding size of video fc features.')
    parser.add_argument('--img_encoding_size', type=int, default=256,
                        help='the encoding size of image features.')
    parser.add_argument('--face_encoding_size', type=int, default=512,
                        help='the encoding size of each frame of facial features.')
    parser.add_argument('--word_encoding_size', type=int, default=512,
                        help='the encoding size of each token in the vocabulary')
    parser.add_argument('--encoding_size', type=int, default=512,
                        help='encoding size for the final feature')
    parser.add_argument('--memory_attention_size', type=int, default=32,
                        help='memory attention size for face attention prediction')
    parser.add_argument('--l2norm', type=int, default=0,
                        help='If 1, then l2 normalize visual and language encoding space')

    # ============================
    # Optimization Options
    # ============================

    # Optimization: General
    parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                    help='Maximum scheduled sampling prob.')
    parser.add_argument('--glove', type=str, default=None,
                        help='text or npy containing glove vector associated with word_idx labels. \
                                 builds a npy file in the same directory if text file is given')

    # ============================
    # Evaluation
    # ============================

    # Evaluation/Checkpointing
    parser.add_argument('--val_id', type=str, default='',
                        help='id to use to save captions for validation')
    parser.add_argument('--val_videos_use', type=int, default=-1,
                    help='how many videos to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--losses_print_every', type=int, default=50,
                    help='How often do we want to print losses? (0 = disable)')
    parser.add_argument('--save_checkpoint_every', type=int, default=5,
                    help='how often to save a model checkpoint in iterations? the code already saves checkpoint every epoch (0 = dont save; 1 = every epoch)')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--losses_log_every', type=int, default=50,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--eval_accuracy', type=int, default=1,
                    help='Evaluate accuracy during validation')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')
    parser.add_argument('--reset_tensorboard', action='store_true')


    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.eval_accuracy == 0 or args.eval_accuracy == 1, "eval_accuracy should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.save_checkpoint_every >= 0, "saving checkpoint at every $epoch should be non-negative"

    return args
