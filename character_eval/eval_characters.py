import sys, os.path
import json

import argparse
import numpy as np

class FillInCharacters():

    def __init__(self, ground_truth_filename, set_filename, prediction_filename):
        self.ref_data = []
        self.res_data = []
        self.sets_data = []

        with open(ground_truth_filename) as f:
            for l in f.readlines():
                line = l.strip().split("\t")
                self.ref_data.append(line)
            f.close()

        with open(set_filename) as f:
            for l in f.readlines():
                line = l.strip().split("\t")
                self.sets_data.append(line)
            f.close()

        with open(prediction_filename) as f:
            for l in f.readlines():
                line = l.strip().split("\t")
                self.res_data.append(line)
            f.close()

    def evaluate(self):
        # =================================================
        # evaluation
        # =================================================


        ref_len = len(self.ref_data)
        res_len = len(self.res_data)

        if ref_len != res_len:
            print('the length of the results file does not match the references')
            exit()

        accuracy = []  # [None] * (ref_len/5 + 1*(ref_len-ref_len/5*5 <> 0))
        accuracy_same = []
        accuracy_diff = []

        j = 0
        i = 0
        while j < res_len:
            max_k = len(self.sets_data[i])
            gt = []
            gt_id = {}
            res = []
            res_id = {}
            countBlanks = 0
            for k in range(max_k):
                gt_id[k] = self.ref_data[j + k][0]
                gt_items = self.ref_data[j + k][1].strip().split(",")
                res_id[k] = self.res_data[j + k][0]
                res_items = self.res_data[j + k][1].strip().split(",")
                if gt_id[k] != res_id[k]:
                    print('the clip IDs in the results file do not match the references')
                    exit()
                if self.ref_data[j + k][1] != '_':
                    countBlanks += len(gt_items)
                    for it in gt_items:
                        gt.append(it)
                    for it in res_items:
                        res.append(it)
            # compare gt and res
            if countBlanks < 2:
                # 0 or 1 blank, nothing to do
                j += max_k
                i += 1
                continue  # accuracy.append(1)
            else:
                # build a pairwise matrix (upper diag)
                gt_pair = np.zeros((countBlanks, countBlanks)) - 1
                res_pair = np.zeros((countBlanks, countBlanks)) - 1
                acc = 0
                acc_count = 0
                acc_same = 0
                same_count = 0
                acc_diff = 0
                diff_count = 0
                for x in range(countBlanks - 1):
                    for y in range(x + 1, countBlanks):
                        gt_pair[x, y] = gt[x] == gt[y]
                        res_pair[x, y] = res[x] == res[y]
                        if gt_pair[x, y] == res_pair[x, y]:
                            acc += 1
                        acc_count += 1
                        if gt_pair[x,y]: #same
                            if gt_pair[x, y] == res_pair[x, y]:
                                acc_same += 1
                            same_count += 1
                        else: #diff
                            if gt_pair[x, y] == res_pair[x, y]:
                                acc_diff += 1
                            diff_count += 1

                accuracy.append(float(acc) / float(acc_count))
                if same_count>0:
                    acc_f = float(acc_same) / float(same_count)
                    accuracy_same.append(acc_f)
                if diff_count>0:
                    acc_f = float(acc_diff) / float(diff_count)
                    accuracy_diff.append(acc_f)
            j += max_k
            i += 1

        acc = sum(accuracy) / len(accuracy)
        print('Instance Accuracy: %03f' % (acc))
        
        acc_same = sum(accuracy_same)/len(accuracy_same)
        print('Same Accuracy:\t%03f'%(acc_same))

        acc_diff = sum(accuracy_diff)/len(accuracy_diff)
        print('Diff Accuracy:\t%03f'%(acc_diff))
       
        acc_mean = 2 * acc_same * acc_diff / (acc_same + acc_diff)
        print('Class Accuracy:\t%03f'%(acc_mean))
        return {'Instance Accuracy': acc, 'Same Accuracy': acc_same, 'Diff Accuracy': acc_diff,
                'Class Accuracy': acc_mean}

def main(args):
    # Call coco eval
    reference = os.path.join(args.reference, 'references_%s_new.csv' % args.split)
    set = os.path.join(args.reference, '%sSets.csv' % args.split)
    evaluator = FillInCharacters(ground_truth_filename=reference,
                                 set_filename=set,
                             prediction_filename=args.submission)
    output = evaluator.evaluate()
    json.dump(output,open(args.output,'w'))
    print(output)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='sample_submission.csv',
                        help='sample submission file for LSMDC.')
    parser.add_argument('-r', '--reference', type=str, default='input',
                        help='directory containing reference csv and set csv files')
    parser.add_argument('-o', '--output', type=str,  default='result.json',
                        help='output file with final language metrics.')
    parser.add_argument('--split', type=str, default='val')
    args = parser.parse_args()

    main(args)
