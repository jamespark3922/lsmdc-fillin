# Identity-Aware Multi-Sentence Video Description (ECCV 2020)

Repo for [Identity-Aware Multi-Sentence Video Description](https://arxiv.org/abs/2008.09791). (Baseline for [LSMDC 2021](https://sites.google.com/site/describingmovies/lsmdc-2021?authuser=0)) <br>

## Fill in Identities within a set of Clips

Standard video and movie description tasks **abstract away from person identitie**s, thus failing to link identities across sentences. We propose a multi-sentence Identity-Aware Video Description task, which overcomes this limitation and requires to **re-identify persons locally within a set of consecutive clips**. We introduce an auxiliary task of **Fill-in the Identity**, that aims to predict persons' IDs consistently within a set of clips, when the video descriptions are given.

<img src="https://storage.googleapis.com/jamesp/lsmdc/3739_teaser.png" width="70%">

# Code

## Getting Started

Clone this repository.
```
git clone https://github.com/jamespark3922/lsmdc-fillin.git
cd lsmdc-fillin
```

Then install the requirements. The following code was tested on Python3.6 and pytorch >= 1.2
```
pip install -r requirements.txt
```

## Preprocessed Data and Features

Download Task2 annotations from the [LSMDC download page](https://sites.google.com/site/describingmovies/download?authuser=0). You will need to consult the organizer to have access to the dataset.
Save the file in your `$TASK2_ANNOTATION` directory.

Videos, caption annotations with character information, and features are available in the [LSMDC project page](https://sites.google.com/site/describingmovies/download?authuser=0).
For simplicity, we have also included features and preprocessed data to reproduce our code.
All you need to do is run:
```
bash get_data_and_features.sh
```
This will create `data` directory that contains the i3d features `i3d`, preprocessed json files, face clusters, and bert gender embeddings (described in paper) in `fillin_data`. We used the [Facenet](https://github.com/davidsandberg/facenet) repository to detect and cluster the faces. More details on extracting the face features will be added later.

Then, run the following to do further preprocessing:
```
python prepro_vocab.py --input_path $TASK2_ANNOTATION --output_path data/fillin_data
```

After running the above code, the `data` folder has the following files:
```
data/
data/i3d                                                (i3d features for each clip)
data/fillin_data
data/fillin_data/LSMDC16_info_fillin_augmented.json     (preprocessed annotation with clip and character information with training time data augmentation.)
data/fillin_data/LSMDC16_labels_fillin.h5               (caption label information)
data/fillin_data/LSMDC16_annos_gender.json              (gender information for each clip)
data/fillin_data/bert_text_gender_embedding             (bert trained to do gender classification that will be used to encode the sentence)
data/fillin_data/face_features_rgb_mtcnn_cluster        (face clusters)
```



## Training
Before training, you might want to create a separate directory to save your experiments and model checkpoints.
```
mkdir experiments
```

Then, you run the following code to use **data augmentation** trick, **gender loss**, and **bert embedding** (Last row of Table 2 in the paper).
```
python train.py --input_json data/fillin_data/LSMDC16_info_fillin_new_augmented.json \
                --input_fc_dir data/i3d/ \
                --input_face_dir data/fillin_data/face_features_rgb_mtcnn_cluster/ \
                --input_label_h5 data/fillin_data/LSMDC16_labels_fillin.h5 \
                --clip_gender_json data/fillin_data/LSMDC16_annos_gender.json \
                --use_bert_embedding --bert_embedding_dir data/fillin_data/bert_text_gender_embedding/ \
                --learning_rate 5e-5  --gender_loss 0.2 --batch_size 64 \
                --pre_nepoch 30 --save_checkpoint_every 5\
                --checkpoint_path experiments/exp1
```
This will train for 30 epochs and evaluate and save the model every 5 epochs.
Here, the best model will be saved in `experiments/exp1/gen_best.pth`.
Usually, it took me within a day to finish training.

## Character Label Generation and Evaluation

### Validation Set
To run evaluation on your saved model:
```
python eval.py --batch_size 64 --g_model_path {$exp_path}/gen_best.pth  --infos_path {$exp_path}/infos.pkl --id $eval_id --split val
```
This will save your character predictions in the validation set in `character_eval/characters/character_val_{$eval_id}.csv` and print out the accuracy scores.
You can run the following code to get the accuracy scores as in the paper. Note that we have included `character_eval/characters/character_val_different_ids.csv` that predicts different ids all the time (Second row of Table 2 in the paper).
```
cd character_eval
python eval_characters.py --subission character_eval/characters/character_val_{$eval_id}.csv --output character_eval/results/result_val_{$eval_id}.csv
```

### Test Set
If you want to run on test set, set `--split test` in `eval.py` which will not run the evaluation code but give the prediction in the end in `character_eval/characters/character_{$eval_id}.csv` (note that the *val* prefix is removed).

#### Leaderboard
We use codalab for our leaderboard. Here is the [codalab competition] for Task 2 (Fill-in the Characters) challenge(https://competitions.codalab.org/competitions/32769).

Other relevant challenges are:
- [Task 1](https://competitions.codalab.org/competitions/32767) (Multi-Sentence Video Description)
- [Task 3](https://competitions.codalab.org/competitions/32914) (Multi-Sentence Video Description with Characters)

You can find more info on the [LSMDC proejct website](https://sites.google.com/site/describingmovies/lsmdc-2021?authuser=0).

## Pretrained Checkpoints
We share the pretrained checkpoints used in the paper. This is the Last row of Table 2 in the paper.
```
wget https://storage.googleapis.com/jamesp/lsmdc/fillin2_transformer_memory9_mtcnn_cluster_gender0.2_bert_gender_sent_emb_bs64_augmented_new_no_img.zip
unzip fillin2_transformer_memory9_mtcnn_cluster_gender0.2_bert_gender_sent_emb_bs64_augmented_new_no_img.zip
```

### Evaluation
You will have to overwrite some input directories to match the one you have.
If you have followed the above data preprocessing step, you should be abel to run the following command:
```
python eval.py --infos_path fillin2_transformer_memory9_mtcnn_cluster_gender0.2_bert_gender_sent_emb_bs64_augmented_new_no_img/infos.pkl \
               --g_model_path fillin2_transformer_memory9_mtcnn_cluster_gender0.2_bert_gender_sent_emb_bs64_augmented_new_no_img/gen_best.pth \
               --input_json data/fillin_data/LSMDC16_info_fillin_new_augmented.json \
               --input_fc_dir data/i3d/ \
               --input_face_dir data/fillin_data/face_features_rgb_mtcnn_cluster/ \
               --input_label_h5 data/fillin_data/LSMDC16_labels_fillin.h5 \
               --clip_gender_json data/fillin_data/LSMDC16_annos_gender.json \
               --bert_embedding_dir data/fillin_data/bert_text_gender_embedding \
               --id fillin2_transformer_memory9_mtcnn_cluster_gender0.2_bert_gender_sent_emb_bs64_augmented_new_no_img
```
This will save the character predictions in `character_eval/characters/character_val_fillin2_transformer_memory9_mtcnn_cluster_gender0.2_bert_gender_sent_emb_bs64_augmented_new_no_img.csv`
and results in 
`character_eval/results/result_val_fillin2_transformer_memory9_mtcnn_cluster_gender0.2_bert_gender_sent_emb_bs64_augmented_new_no_img.json`

The numbers should be 
| | Same Acc  | Diff Acc | Class Acc | Instance Acc|
| ------------- |------------- | ------------- | ------------- | ------------- |
| Best Model | 63.5  | 68.4  | 65.9|  69.8|

See the paper for more details about the evaluation. Overall, we use a combination of *Class Acc* and *Instance Acc* to measure the performance.

## Filling in Non-GT captions
If you wish to fill in characters for generated captions, first download this [caption](https://storage.googleapis.com/jamesp/lsmdc/caption_val_i3d_resnet_someone_context_greedy.json) and parse the generated captions into the same format.
Then, use `--caption_path` option in `eval.py` to fill in the captions.

#### Bibtex
```
@InProceedings{park2020,
  author = {Park, Jae Sung and Darrell, Trevor and Rohrbach, Anna},
  title = {Identity-Aware Multi-Sentence Video Description},
  booktitle = {In Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```
