import os
import re
import json
from statistics import mean, stdev
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def detect_file(filename):
    # pattern = r'use_annotator_embed-(?P<use_annotator_embed>\w+)-use_annotation_embed-(?P<use_annotation_embed>\w+)-annotator_embed_weight-(?P<annotator_embed_weight>[\d.]+)-annotation_embed_weight-(?P<annotation_embed_weight>[\d.]+)-pad-(?P<include_pad_annotation>\w+)-method-(?P<method>\w+)-test_mode-(?P<test_mode>\w+)-seed-(?P<seed>\d+)-(?P<i>\d+)-(?P<test_split>\w+)-(?P<tt_idx>\d+)'
    pattern = r'use_annotator_embed-(?P<use_annotator_embed>\w+)-use_annotation_embed-(?P<use_annotation_embed>\w+)-pad-(?P<include_pad_annotation>\w+)-method-(?P<method>\w+)-test_mode-(?P<test_mode>\w+)-seed-(?P<seed>\d+)-(?P<i>\d+)-(?P<test_split>\w+)-(?P<tt_idx>\d+)'
    match = re.match(pattern, filename)
    if match and filename.endswith(".jsonl"):
        key_value_pairs = match.groupdict()
        return key_value_pairs
    else:
        None

# PARTS = ["all", "0.8", "0.7", "0.6", "0.8_smaller", "0.7_smaller", "0.6_smaller"]
PARTS = ["all"]
TEST_SPLITS = ["annotation"]
# TEST_SPLITS = ["annotation"]
USE_ANNOTATOR_EMBED = [True, False]
USE_ANNOTATION_EMBED = [True, False]
TEST_CASE = "normal"
# DATA_LIST = ["friends_qia", "pejorative", "md-agreement", "goemotions", "hs_brexit",  "humor", "commitmentbank", "sentiment"]
DATA_LIST = ["toxic-ratings"]
EXTRA_PATH = "overall_majority_vote"

def print_result(dir, fns, tt_idx, part, test_split):
    tt_idx = 0
    setting_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    for fn in fns:
        if p := detect_file(fn):
            if not (int(p["tt_idx"]) == tt_idx and p["test_split"] == test_split):
                continue
            with open(f"{dir}/{EXTRA_PATH}/{fn}", 'r') as f:
                data = f.readlines()
            data = [json.loads(d) for d in data]
            p.pop("i")
            p.pop("seed")
            name = ""
            # # Do majority vote baselines
            # label_nums = defaultdict(int)
            # all_golds = [d["gold"] for d in data]
            # for l in all_golds:
            #     label_nums[l] += 1
            # sorted_l = sorted(label_nums.items(), key=lambda x: x[1])
            # ll, nn = sorted_l[-1]
            # print(round(nn/len(data) * 100, 2))
            # return
        
            score_list = []
            # # accuracy
            # per_ann_accs = defaultdict(list)
            # for d in data:
            #     if d["pred"] == d["gold"]:
            #         per_ann_accs[d["respondent_id"]].append(1)
            #     else:
            #         per_ann_accs[d["respondent_id"]].append(0)
            # per_ann_acc_scores = dict()
            # for atr, l in per_ann_accs.items():
            #     if len(l) < 5:
            #         continue
            #     per_ann_acc_scores[atr] = sum(l) / len(l)
            
            # score = sum(per_ann_acc_scores.values()) / len(per_ann_acc_scores)
            # accuracy
            score = len([d for d in data if d["pred"] == d["gold"]]) / len(data)
            # precision
            # score = precision_score([d["gold"] for d in data], [d["pred"] for d in data], average="micro")
            # recall
            # score = recall_score([d["gold"] for d in data], [d["pred"] for d in data], average="micro")
            # f1 score
            # score = f1_score([d["gold"] for d in data], [d["pred"] for d in data], average="macro")
            score_list.append((len(labels), score))

            for label in labels:
                score = len([d for d in data if d["pred"] == d["gold"] and d["gold"] == label]) / len([d for d in data if d["gold"] == label])
                score_list.append((len([d for d in data if d["gold"] == label]), score))
            setting_scores[p["test_mode"]][p["method"]][p["use_annotator_embed"]][p["use_annotation_embed"]][p["include_pad_annotation"]].append(score_list)

    x=' '
    current_annotation_weight = None
    current_annotator_weight = None
    print(f"Setting: {part} {test_split}")
    d_scores = dict()
    for n_1, v_1 in setting_scores.items():
        if n_1 != TEST_CASE:
            continue
        for n_2, v_2 in v_1.items():
            # print(f"{15*x}method: {n_2}")
            for n_3, v_3 in v_2.items():
                # print(f"{30*x}use_annotator_embed: {n_3}")
                for n_4, v_4 in v_3.items():
                    if n_4 != current_annotator_weight:
                        current_annotator_weight = n_4
                    # print(f"{45*x}annotator_embed_weight: {n_4}")
                    for n_5, s_list in v_4.items():
                        # print(f"{90*x}include_pad_annotation: {n_7}")
                        # print(name)
                        # print(f"{x*105}{mean(s_list)} +/- {stdev(s_list)}")
                        print(n_4, n_3)
                        # if len(s_list) != 10:
                        #     print(f"{10 - len(s_list)} Away.")
                        #     continue
                        tt_scores = [s[0][1] for s in s_list]
                        # print(f"Amount: {[s[0][0] for s in s_list][0]}; Mean: {round(100 * mean(tt_scores), 2)}, Stdev: {round(100 * stdev(tt_scores), 2)}")
                        print(f"{round(100 * mean(tt_scores), 2)}, {round(100 * stdev(tt_scores), 2)}")
                        # d_scores[n_4 + n_3] = f"{round(100 * mean(tt_scores), 2)} \\tiny{{{round(100 * stdev(tt_scores), 2)}}}"
                        d_scores[n_4 + n_3] = f"{round(100 * mean(tt_scores), 2)}"
                        # print(round(100 * mean(tt_scores), 2))
                        # print(round(100 * stdev(tt_scores), 2))
                        for i, label in enumerate(labels):
                            scores = [s[i + 1][1] for s in s_list]
                            amount = [s[i + 1][0] for s in s_list][0]
                            print(f"For {label}: Amount: {amount}; Mean: {round(100 * mean(scores), 2)}, Stdev: {round(100 * stdev(scores), 2)}")
                            # print(f"{round(100 * mean(scores), 2)}, {round(100 * stdev(scores), 2)}")
                        # print()
    # print(d_scores["FalseFalse"] + "&  &" + d_scores["TrueFalse"] + "& " + d_scores["FalseTrue"] + "& " + d_scores["TrueTrue"]+ "\\\\")
    print("\n\n\n")

for dir in DATA_LIST:
    # load possible labels:
    print(dir)
    with open(f"../src/example-data/{dir}-processed/annotation_labels.json", 'r') as f:
        labels = json.load(f)
    labels.sort(reverse=True)
    # labels = ["-3", "-2", "-1", "0", "1", "2", "3"]
    # labels = ["Very positive", "Somewhat positive", "Neutral", "Somewhat negative", "Very negative"]
    fns = os.listdir(f"{dir}/{EXTRA_PATH}")
    for tt_idx, part in enumerate(PARTS):
        for test_split in TEST_SPLITS:
            print_result(dir, fns, tt_idx, part, test_split)
    
    