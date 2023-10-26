import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean, stdev
from scipy.signal import savgol_filter


width = 0.2
RANGE = 100

# plot setting
FONT_SIZE = 12
TICK_SIZE = 12
LEGEND_FONT_SIZE = 8


plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=42)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"


def detect_file(filename):
    # pattern = r'use_annotator_embed-(?P<use_annotator_embed>\w+)-use_annotation_embed-(?P<use_annotation_embed>\w+)-annotator_embed_weight-(?P<annotator_embed_weight>[\d.]+)-annotation_embed_weight-(?P<annotation_embed_weight>[\d.]+)-pad-(?P<include_pad_annotation>\w+)-method-(?P<method>\w+)-test_mode-(?P<test_mode>\w+)-seed-(?P<seed>\d+)-(?P<i>\d+)-(?P<test_split>\w+)-(?P<tt_idx>\d+)'
    pattern = r'use_annotator_embed-(?P<use_annotator_embed>\w+)-use_annotation_embed-(?P<use_annotation_embed>\w+)-pad-(?P<include_pad_annotation>\w+)-method-(?P<method>\w+)-test_mode-(?P<test_mode>\w+)-seed-(?P<seed>\d+)-(?P<i>\d+)-(?P<test_split>\w+)-(?P<tt_idx>\d+)'
    match = re.match(pattern, filename)
    if match:
        key_value_pairs = match.groupdict()
        return key_value_pairs
    else:
        None

def calculate_acc(x):
    # x is a dict of dict of list
    respondent_accs = dict()
    for respondent_id, turns in x.items():
        accs = []
        for turn_idx, turn_values in turns.items():
            acc = sum(turn_values) / len(turn_values) * 100
            accs.append(acc)
        respondent_accs[respondent_id] = [mean(accs), stdev(accs), len(turns["0"])]
    sorted_dict = dict(sorted(respondent_accs.items(), key=lambda x: x[1][2], reverse=True))
    return sorted_dict

def plot(ax, x, label, wd):
    # x is a dict of mean and stdev
    means = [v[0] for v in x.values()]
    stds = [v[1] for v in x.values()]
    counts = [v[2] for v in x.values()]
    # Plotting the bar chart
    # ax.bar(range(len(list(x.keys()))), means, yerr=stds, align='center', alpha=0.2, ecolor='black', capsize=1, label=label)
    ax.bar(np.arange(len(list(x.keys()))) + wd, means, width, align='center', alpha=0.5, ecolor='black', label=label)
    # ax.set_ylabel('Accuracy')
    # ax.set_xticks(x.keys())
    # ax.set_xticklabels(x.keys())
    ax.set_title('Accuracy by person')
    ax.yaxis.grid(True)
    return range(len(list(x.keys()))), counts

def add_value(data, d, kvs):
    for datapoint in data:
        if datapoint["pred"] == datapoint["gold"]:
            d[datapoint["respondent_id"]][kvs["i"]].append(1)
        else:
            d[datapoint["respondent_id"]][kvs["i"]].append(0)

def aaverage(x):
    means = [v[0] for v in x.values()]
    print(f"{round(100 * mean(means), 2)}")

DATASET = "commitmentbank"
PATH = f"../experiment-results/{DATASET}/wo_embed"
TEST_SPLIT = "annotation"
TEST_MODE = "normal"
files = os.listdir(PATH)

question_d = defaultdict(lambda : defaultdict(list))
annotator_d = defaultdict(lambda : defaultdict(list))
annotation_d = defaultdict(lambda : defaultdict(list))
both_d = defaultdict(lambda : defaultdict(list))

for file in files:
    # assert file.endswith(".jsonl")
    kvs = detect_file(file)
    if not kvs:
        continue
    if not (kvs["test_split"] == TEST_SPLIT and kvs["test_mode"] == TEST_MODE):
        continue
    
    with open(f"{PATH}/{file}", 'r') as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    if kvs["use_annotator_embed"] == "False" and kvs["use_annotation_embed"] == "False":
        add_value(data, question_d, kvs)
    elif kvs["use_annotator_embed"] == "True" and kvs["use_annotation_embed"] == "False":
        add_value(data, annotator_d, kvs)
    elif kvs["use_annotator_embed"] == "False" and kvs["use_annotation_embed"] == "True":
        add_value(data, annotation_d, kvs)
    elif kvs["use_annotator_embed"] == "True" and kvs["use_annotation_embed"] == "True":
        add_value(data, both_d, kvs)

print(calculate_acc(both_d))
exit(0)
fig, ax = plt.subplots(figsize=(12, 6))
plot(ax, calculate_acc(question_d), "Question", 0)
plot(ax, calculate_acc(annotation_d), "Annotation", width * 1)
plot(ax, calculate_acc(annotator_d), "Annotator",  width * 2)
x_values, counts = plot(ax, calculate_acc(both_d), "Both", width * 3)

plt.legend()
ax.set_xlim(59.9, 80)
# ax.set_xlim(59.9, 80.9)
ax.set_ylabel('Accuracy')
# ax.set_xlim(480.9, 496.9)
# smoothed_counts = savgol_filter(counts, window_length=11, polyorder=8)
ax2 = ax.twinx()
ax2.plot(x_values, counts, color='red', label='Item Count')
# ax2.set_ylim(0, 200)
ax2.set_ylabel('Number of examples')
plt.xticks(range(0, 21))
# plt.xticks(range(60, 80))
ax.set_facecolor('#EBEBEB')
[ax.spines[side].set_visible(False) for side in ax.spines]
[ax2.spines[side].set_visible(False) for side in ax2.spines]
plt.savefig(f"annotator_accs/{TEST_SPLIT}-{DATASET}-annotator-acc.pdf", bbox_inches='tight')

# # calculate the average average for each person
# aaverage(calculate_acc(question_d))
# aaverage(calculate_acc(annotator_d))
# aaverage(calculate_acc(annotation_d))
# aaverage(calculate_acc(both_d))
