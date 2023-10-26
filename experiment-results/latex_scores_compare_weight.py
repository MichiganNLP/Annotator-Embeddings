""" 
Read baseline models performances, output latex-formatted scores
with the p value in the t-test
This file calculates the annotator split for bert base model only
"""

import scipy.stats as stats
from sklearn.metrics import precision_score, recall_score, f1_score
from statistics import mean, stdev
import json 
import re
import os
from collections import defaultdict

def detect_file(filename):
    # pattern = r'use_annotator_embed-(?P<use_annotator_embed>\w+)-use_annotation_embed-(?P<use_annotation_embed>\w+)-annotator_embed_weight-(?P<annotator_embed_weight>[\d.]+)-annotation_embed_weight-(?P<annotation_embed_weight>[\d.]+)-pad-(?P<include_pad_annotation>\w+)-method-(?P<method>\w+)-test_mode-(?P<test_mode>\w+)-seed-(?P<seed>\d+)-(?P<i>\d+)-(?P<test_split>\w+)-(?P<tt_idx>\d+)'
    pattern = r'use_annotator_embed-(?P<use_annotator_embed>\w+)-use_annotation_embed-(?P<use_annotation_embed>\w+)-pad-(?P<include_pad_annotation>\w+)-method-(?P<method>\w+)-test_mode-(?P<test_mode>\w+)-seed-(?P<seed>\d+)-(?P<i>\d+)-(?P<test_split>\w+)-(?P<tt_idx>\d+)'
    match = re.match(pattern, filename)
    if match and filename.endswith(".jsonl"):
        key_value_pairs = match.groupdict()
        return key_value_pairs
    else:
        None

def stats_diff(g1, g2):
    output = stats.ttest_ind(a=g1, b=g2, equal_var=True)
    p_value = output[-1]
    return p_value < 0.05, p_value

def calculate_score(p, score_type):
    # calculate score for a single file
    with open(p, 'r') as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    if score_type == "acc":
        return len([d for d in data if d["pred"] == d["gold"]]) / len(data)
    elif score_type == "f1":
        return f1_score([d["gold"] for d in data], [d["pred"] for d in data], average="macro")

def process(dataset, setting, score_type):
    fns = os.listdir(f"{dataset}/{setting}")
    score_list = defaultdict(list)
    for fn in fns:
        if setting in {"random", "individual_majority_vote", "overall_majority_vote"}:
            score = calculate_score(f"{dataset}/{setting}/{fn}", score_type=score_type)
            kv = detect_file(fn)
            if kv and "test_split" in kv and kv["test_split"] == "annotation":
                score_list[setting].append(score) 
            continue
        if kv := detect_file(fn):
            if not (kv["method"] == "add" and kv["include_pad_annotation"] == "True" \
                   and kv["test_mode"] == "normal" and kv["test_split"] == "annotation"):
                continue
            score = calculate_score(f"{dataset}/{setting}/{fn}", score_type=score_type)
            if "naiive-concat" in setting:
                assert kv["use_annotator_embed"] == "False" and kv["use_annotation_embed"] == "False"
                score_list["NC"].append(score)
            else:
                use_annotator_embed = kv["use_annotator_embed"]
                use_annotation_embed = kv["use_annotation_embed"]
                if use_annotator_embed == "False" and use_annotation_embed == "False":
                    score_list["B"].append(score)
                if use_annotation_embed == "True" and use_annotator_embed == "False":
                    score_list["B + En"].append(score)
                if use_annotator_embed == "True" and use_annotation_embed == "False":
                    score_list["B + Ea"].append(score)
                if use_annotator_embed == "True" and use_annotation_embed == "True":
                    score_list["B + En +Ea"].append(score)
    return score_list

def find_largest(method_ss):
    l = [mean(method_ss["B + En"]), mean(method_ss["B + Ea"]), mean(method_ss["B + En +Ea"])]
    i = l.index(max(l))
    if i == 0:
        return "B + En"
    elif i == 1:
        return "B + Ea"
    else:
        return "B + En +Ea"
    
def process_dataset(dataset, settings, score_type, line_colors, last_row):
    scores = defaultdict(dict)
    for setting in settings:
        score_list = process(dataset, setting, score_type)
        for method_type, s_list in score_list.items():
            assert len(s_list) == 10
            scores[setting][method_type] = s_list
    # for each model, find the statistically larger score
    latex_ss = []
    for mtype in {"B + En", "B + Ea", "B + En +Ea"}:
        stats_sig1, p_value1 = stats_diff(scores["bert-base-embed-wo-weight"][mtype], \
            scores["bert-base"][mtype])
        if stats_sig1:
            if mean(scores["bert-base-embed-wo-weight"][mtype]) > mean(scores["bert-base"][mtype]):
                latex_ss.append(f' \\textbf{{{round(100 * mean(scores["bert-base-embed-wo-weight"][mtype]), 2):.2f}}} & {round(100 * mean(scores["bert-base"][mtype]), 2):.2f}')
            else:
                latex_ss.append(f' {round(100 * mean(scores["bert-base-embed-wo-weight"][mtype]), 2):.2f} & \\textbf{{{round(100 * mean(scores["bert-base"][mtype]), 2):.2f}}}')
        else:
            latex_ss.append(f' {round(100 * mean(scores["bert-base-embed-wo-weight"][mtype]), 2):.2f} & {round(100 * mean(scores["bert-base"][mtype]), 2):.2f}')
    
    latex_ss = " & ".join(latex_ss)
    latex_ss += " \\\\"
    print(latex_ss)
    print("\n")
            
def main():
    # hyperparameters
    # datasets = ["md-agreement", "goemotions", "humor", "commitmentbank", "sentiment"]
    datasets = ["pejorative"]
    settings = ["bert-base", "bert-base-embed-wo-weight"]  # wo_embed corresponds to the bert-base ones
    line_colors = ["\\rlb\\cw", "\\rdb\\cw", "\\rlr\\cw", "\\rdr\\cw", "\\rld\\cw", "\\rdd\\cw"]
    # last_rows = {
    #     "md-agreement": "\\multirow{-6}{*}{MDA}",
    #     "goemotions": "\\multirow{-6}{*}{GOE}", 
    #     "humor": "\\multirow{-6}{*}{HUM}", 
    #     "commitmentbank": "\\multirow{-6}{*}{COM}", 
    #     "sentiment": "\\multirow{-6}{*}{SNT}"}
    last_rows = {
        "md-agreement": "\\multirow{-6}{*}{MDA}",
        "goemotions": "\\multirow{-6}{*}{GOE}", 
        "humor": "\\multirow{-6}{*}{HUM}", 
        "commitmentbank": "\\multirow{-6}{*}{COM}", 
        "sentiment": "\\multirow{-6}{*}{SNT}",
        "friends_qia": "\\multirow{-6}{*}{FIA}",
        "pejorative": "\\multirow{-6}{*}{PEJ}",
        "hs_brexit": "\\multirow{-6}{*}{HSB}"
    }
    score_type = "f1"
    # read in different settings 
    for dataset in datasets:
        process_dataset(dataset, settings, score_type, line_colors, last_rows[dataset])


if __name__ == "__main__":
    main()
