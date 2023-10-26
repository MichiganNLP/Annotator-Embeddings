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
            if kv and "test_split" in kv and kv["test_split"] == "annotator":
                score_list[setting].append(score) 
            continue
        if kv := detect_file(fn):
            if not (kv["method"] == "add" and kv["include_pad_annotation"] == "True" \
                   and kv["test_mode"] == "normal" and kv["test_split"] == "annotator"):
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
        if setting in {"wo_embed", "bert-base-naiive-concat"}:
            model_type = "bert-base"
        elif setting in {"random", "individual_majority_vote", "overall_majority_vote"}:
            model_type = setting
        else:
            model_type = setting.split("-naiive-concat")[0]
        # baseline methods
        for method_type, s_list in score_list.items():
            assert len(s_list) == 10
            scores[model_type][method_type] = s_list

    # for each model, find the statistically larger score
    latex_ss = []
    for model_type, method_ss in scores.items():
        # our methods
        if model_type in {"random", "individual_majority_vote", "overall_majority_vote"}:
            continue
        best_method_type = find_largest(method_ss)
        stats_sig1, p_value1 = stats_diff(method_ss["B"], method_ss[best_method_type])
        stats_sig2, p_value2 = stats_diff(method_ss["NC"], method_ss[best_method_type])
        latex_s = None
        if stats_sig1 and stats_sig2:
            if mean(method_ss[best_method_type]) > mean(method_ss["B"]) and mean(method_ss[best_method_type]) > mean(method_ss["NC"]):
                # it is significantly better
                if best_method_type == "B + En":
                    latex_s = f'{round(100 *mean(method_ss["B"]), 2):.2f} & {round(100 *mean(method_ss["NC"]), 2):.2f} & \\textbf{{{round(100 *mean(method_ss["B + En"]), 2):.2f}}} & {round(100 *mean(method_ss["B + Ea"]), 2):.2f} & {round(100 *mean(method_ss["B + En +Ea"]), 2):.2f} \\\\% {p_value1} {p_value2}'
                elif best_method_type == "B + Ea":
                    latex_s = f'{round(100 *mean(method_ss["B"]), 2):.2f} & {round(100 *mean(method_ss["NC"]), 2):.2f} & {round(100 *mean(method_ss["B + En"]), 2):.2f} & \\textbf{{{round(100 *mean(method_ss["B + Ea"]), 2):.2f}}} & {round(100 *mean(method_ss["B + En +Ea"]), 2):.2f} \\\\% {p_value1} {p_value2}'
                elif best_method_type == "B + En +Ea":
                    latex_s = f'{round(100 *mean(method_ss["B"]), 2):.2f} & {round(100 *mean(method_ss["NC"]), 2):.2f} & {round(100 *mean(method_ss["B + En"]), 2):.2f} & {round(100 *mean(method_ss["B + Ea"]), 2):.2f} &  \\textbf{{{round(100 *mean(method_ss["B + En +Ea"]), 2):.2f}}} \\\\% {p_value1} {p_value2}'
                else:
                    raise RuntimeError("No method selected")
            else:
                if mean(method_ss["B"]) >  mean(method_ss[best_method_type]):
                    latex_s = f'\\textbf{{{round(100 *mean(method_ss["B"]), 2):.2f}}} & {round(100 *mean(method_ss["NC"]), 2):.2f} & {round(100 *mean(method_ss["B + En"]), 2):.2f} & {round(100 *mean(method_ss["B + Ea"]), 2):.2f} & {round(100 *mean(method_ss["B + En +Ea"]), 2):.2f} \\\\% {p_value1} {p_value2}'
                else:
                    latex_s = f'{round(100 *mean(method_ss["B"]), 2):.2f} & \\textbf{{{round(100 *mean(method_ss["NC"]), 2):.2f}}} & {round(100 *mean(method_ss["B + En"]), 2):.2f} & {round(100 *mean(method_ss["B + Ea"]), 2):.2f} & {round(100 *mean(method_ss["B + En +Ea"]), 2):.2f} \\\\% {p_value1} {p_value2}'
        else:
            latex_s = f'{round(100 *mean(method_ss["B"]), 2):.2f} & {round(100 *mean(method_ss["NC"]), 2):.2f} & {round(100 *mean(method_ss["B + En"]), 2):.2f} & {round(100 *mean(method_ss["B + Ea"]), 2):.2f} & {round(100 *mean(method_ss["B + En +Ea"]), 2):.2f} \\\\% {p_value1} {p_value2}'
        latex_ss.append(latex_s)
    c_latex_ss = []
    for i, line in enumerate(latex_ss):
        # prefix = line_colors[i]
        # if i == len(latex_ss) - 1:
        #     prefix += last_row
        #     c_latex_ss.append(f' & {round(100 *mean(scores["random"]["random"]), 2):.2f} & {round(100 *mean(scores["individual_majority_vote"]["individual_majority_vote"]), 2):.2f} & {round(100 *mean(scores["overall_majority_vote"]["overall_majority_vote"]), 2):.2f} & ' + line)
        # else:
            # c_latex_ss.append(" & \\cw & \\cw & \\cw & " + line)
        # c_latex_ss.append(line)
        c_latex_ss.append(f' & {round(100 *mean(scores["random"]["random"]), 2):.2f} & {round(100 *mean(scores["overall_majority_vote"]["overall_majority_vote"]), 2):.2f} & ' + line)
    print("\n".join(c_latex_ss))
    print("\n")
            
def main():
    # hyperparameters
    # datasets = ["md-agreement", "goemotions", "humor", "commitmentbank", "sentiment"]
    datasets = ["md-agreement", "goemotions", "humor", "commitmentbank", "sentiment", "friends_qia", "pejorative", "hs_brexit"]
    settings = ["wo_embed", "bert-base-naiive-concat", "random", "overall_majority_vote"]  # wo_embed corresponds to the bert-base ones
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
    score_type = "acc"
    # read in different settings 
    for dataset in datasets:
        process_dataset(dataset, settings, score_type, line_colors, last_rows[dataset])


if __name__ == "__main__":
    main()
