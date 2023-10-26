import json
import os
import matplotlib.pyplot as plt
import tqdm
from collections import defaultdict


# plot setting
FONT_SIZE = 15
TICK_SIZE = 15
LEGEND_FONT_SIZE = 13


plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=42)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"

DATA_LIST = ["friends_qia", "pejorative", "hs_brexit", "md-agreement", "goemotions", "humor", "commitmentbank", "sentiment",]
names = ["Friends QIA", "Pejorative", "HS Brexit", "MD Agreement", "Go Emotions", "Humor", "Commitmentbank", "Sentiment"]
sh_datasets = ["FIA", "PEJ", "HSB", "MDA", "GOE", "HUM", "COM", "SNT"]
TASKS = ["indirect_ans", "pejorative", "hs_brexit", "offensive", "emotion", "humor", "certainty", "sentiment"]

numbers = [
    [(1, 5595), (2, 4)],
    [(1, 856), (2, 76)],
    [(1, 774), (2, 345)],
    [(1, 4941), (2, 5812)],
    # [(1, 39565)],
    [(1, 24331), (2, 25198), (3, 6954), (4, 941)],
    [(1, 2402), (2, 14447), (3, 11361)],
    [(1, 25), (2, 154), (3, 310), (4, 367), (5, 231), (6, 94), (7, 19)],
    [(1, 892), (2, 6660), (3, 5699), (4, 809), (5, 10)],
]

# for i, l in enumerate(numbers):
#     print(sh_datasets[i] + " & " + "& ".join([str(v[1]) for v in l]))
# exit(0)

# # markers = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^"]
marker_size = 160
fig, ax1 = plt.subplots()
alpha = 0.9

for i, (dataset, task) in enumerate(zip(DATA_LIST, TASKS)):
    # if dataset != "pejorative":
    #     continue
    # PATH = f"../src/example-data/{dataset}-processed/"
    # example_annotations = defaultdict(set)
    # for split in ["train", "dev", "test"]:
    #     if os.path.exists(f"{PATH}/annotation_split_{split}.json"):
    #         with open(f"{PATH}/annotation_split_{split}.json", 'r') as f:
    #             data_paths = json.load(f)
    #         for data_path in tqdm.tqdm(iter(data_paths)):
    #             with open(f"../src/{data_path['path']}", 'r') as f:
    #                 data = json.load(f)
    #             sentence = data["sentence"]
    #             annotation = data[task]
    #             example_annotations[sentence].add(annotation)
    
    # annotation_nums = defaultdict(int)
    # for k, v in example_annotations.items():
    #     num_annotations = len(v)
    #     annotation_nums[num_annotations] += 1
    # sorted_d = sorted(annotation_nums.items(), key=lambda item: item[0], reverse=False)
    # print(sorted_d)
    sorted_d = numbers[i]
    xs = [v[0] for v in sorted_d]
    tt_num = sum(v[1] for v in sorted_d)
    ys = [sum([v[1] for v in sorted_d[:i + 1]])/tt_num * 100 for i, _ in enumerate(sorted_d)]
    # ax1.set_ylim(1000, 40000)
    # ax1.scatter(xs, ys, label=f"{names[i]} ({sh_datasets[i]})", marker='*',  s=marker_size, alpha=alpha)
    ax1.plot(xs, ys, label=f"{names[i]} ({sh_datasets[i]})", marker="o")

# for i, (dataset, task) in enumerate(zip(DATA_LIST, TASKS)):
#     # PATH = f"../src/example-data/{dataset}-processed/"
#     # example_annotations = defaultdict(set)
#     # for split in ["train", "dev", "test"]:
#     #     if os.path.exists(f"{PATH}/annotation_split_{split}.json"):
#     #         with open(f"{PATH}/annotation_split_{split}.json", 'r') as f:
#     #             data_paths = json.load(f)
#     #         for data_path in tqdm.tqdm(iter(data_paths)):
#     #             with open(f"../src/{data_path['path']}", 'r') as f:
#     #                 data = json.load(f)
#     #             sentence = data["sentence"]
#     #             annotation = data[task]
#     #             example_annotations[sentence].add(annotation)
    
#     # annotation_nums = defaultdict(int)
#     # for k, v in example_annotations.items():
#     #     num_annotations = len(v)
#     #     annotation_nums[num_annotations] += 1
    
#     # sorted_d = sorted(annotation_nums.items(), key=lambda item: item[0], reverse=False)
#     # print(sorted_d)
#     sorted_d = numbers[i]
#     xs = [v[0] for v in sorted_d]
#     # ys = [v[1] for v in sorted_d]
#     tt_num = sum(v[1] for v in sorted_d)
#     ys = [sum([v[1] for v in sorted_d[:i]])/tt_num for i, _ in enumerate(sorted_d)]
#     ax2.set_ylim(0, 500)
#     # ax2.scatter(xs, ys, marker='*', s=marker_size, alpha=alpha)
#     ax2.plot(xs, ys, marker="*")
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax1.set_facecolor('#EBEBEB')
[ax1.spines[side].set_visible(False) for side in ax1.spines]
ax1.grid(color='white', linewidth=1)
# ax2.set_facecolor('#EBEBEB')
# [ax2.spines[side].set_visible(False) for side in ax2.spines]
# ax2.grid(color='white', linewidth=1)
ax1.legend()
# plt.ylabel("Example Coverage (%)")
plt.xlabel("Number of Answers")
plt.savefig(f"disagreement_examples/all.pdf", bbox_inches='tight')
