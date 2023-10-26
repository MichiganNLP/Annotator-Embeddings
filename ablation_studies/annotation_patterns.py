import json
import os
import matplotlib.pyplot as plt
from math import log10
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# plot setting
FONT_SIZE = 20
TICK_SIZE = 20
LEGEND_FONT_SIZE = 20
LINEWIDTH = 2.0
ALPHA = 0.7
TYPE = "few"

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=42)  # fontsize of the figure title
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams["font.family"] = "Times New Roman"

# DATA_LIST = ["friends_qia", "pejorative", "hs_brexit", "md-agreement", "goemotions", "humor", "commitmentbank", "sentiment",]
# names = ["Friends QIA", "Pejorative", "HS Brexit", "MD Agreement", "Go Emotions", "Humor", "Commitmentbank", "Sentiment"]
# sh_datasets = ["FIA", "PEJ", "HSB", "MDA", "GOE", "HUM", "COM", "SNT"]
if TYPE == "many":
    DATA_LIST = ["md-agreement", "goemotions", "humor", "commitmentbank", "sentiment",]
    names = ["MD Agreement", "Go Emotions", "Humor", "Commitmentbank", "Sentiment"]
    sh_datasets = ["MDA", "GOE", "HUM", "COM", "SNT"]
else:
    DATA_LIST = ["friends_qia", "pejorative", "hs_brexit"]
    names = ["Friends QIA", "Pejorative", "HS Brexit"]
    sh_datasets = ["FIA", "PEJ", "HSB"]
colors = []


# for dataset in DATA_LIST:
#     PATH = f"../src/example-data/{dataset}-processed/stats.json"

#     with open(PATH, 'r') as f:
#         stats = json.load(f)
#     fig, ax = plt.subplots()
#     # ax.set_facecolor('#EBEBEB')
#     # [ax.spines[side].set_visible(False) for side in ax.spines]
#     sorted_d = sorted(stats.items(), key=lambda item: item[1], reverse=True)
#     xs = range(1, len(sorted_d) + 1)
#     # xs = [log10(x) for x in xs]
#     tt_num = sum([num for _, num in sorted_d])
#     # import ipdb;ipdb.set_trace()
#     # ys = [100 * sum([num for _, num in sorted_d[:i]])/tt_num for i, _ in enumerate(sorted_d)]
#     if dataset == "pejorative" or dataset == "friends_qia":
#         ax.set_xticks([0, 1, 2])
#     plt.bar(range(len(sorted_d)), sorted_d.values())
#     plt.savefig(f"annotation_patterns/{dataset}.pdf", bbox_inches='tight')

# exit(0)
if TYPE == "many":
    fig, ax2 = plt.subplots(figsize=(8, 6))
else:
    fig, ax2 = plt.subplots(figsize=(4, 6.7))
# 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 3]})

# for dataset in DATA_LIST:
#     PATH = f"../src/example-data/{dataset}-processed/stats.json"

#     with open(PATH, 'r') as f:
#         stats = json.load(f)

#     # ax.set_facecolor('#EBEBEB')
#     # [ax.spines[side].set_visible(False) for side in ax.spines]
#     sorted_d = sorted(stats.items(), key=lambda item: item[1], reverse=True)
#     xs = range(1, len(sorted_d) + 1)
#     # xs = [log10(x) for x in xs]
#     tt_num = sum([num for _, num in sorted_d])
#     # import ipdb;ipdb.set_trace()
#     ys = [100 * sum([num for _, num in sorted_d[:i]])/tt_num for i, _ in enumerate(sorted_d)]
#     # if dataset == "pejorative" or dataset == "friends_qia":
#     #     ax.set_xticks([0, 1, 2])
#     # plt.bar(range(len(sorted_d)), sorted_d.values())
#     # set the limits of the x-axis for the first region (0-100)
#     ax1.set_xlim(0, 100)
#     # ax1.set_aspect(50) 
#     # set the limits of the y-axis for the first region
#     ax1.plot(xs, ys, linewidth=LINEWIDTH, alpha=ALPHA)

if TYPE == "many":
    colors = [u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
# ax2 = ax1.inset_axes([0.5, 0.1, 0.4, 0.4])
for i, dataset in enumerate(DATA_LIST):
    PATH = f"../src/example-data/{dataset}-processed/stats.json"

    with open(PATH, 'r') as f:
        stats = json.load(f)

    # ax.set_facecolor('#EBEBEB')
    # [ax.spines[side].set_visible(False) for side in ax.spines]
    sorted_d = sorted(stats.items(), key=lambda item: item[1], reverse=True)
    xs = range(1, len(sorted_d) + 1)
    # xs = [log10(x) for x in xs]
    tt_num = sum([num for _, num in sorted_d])
    # import ipdb;ipdb.set_trace()
    ys = [100 * sum([num for _, num in sorted_d[:i]])/tt_num for i, _ in enumerate(sorted_d)]
    # if dataset == "pejorative" or dataset == "friends_qia":
    #     ax.set_xticks([0, 1, 2])
    # plt.bar(range(len(sorted_d)), sorted_d.values())
    # set the limits of the x-axis for the second region (100-10000)
    # ax2.set_xlim(0, 10000)

    # set the limits of the y-axis for the second region
    if TYPE == "many":
        ax2.plot(xs, ys, label=f"{sh_datasets[i]}", linewidth=LINEWIDTH, alpha=ALPHA, color=colors[i])
    else:
        # {names[i]} 
        ax2.plot(xs, ys, label=f"{sh_datasets[i]}", linewidth=LINEWIDTH, alpha=ALPHA)
    # mark_inset(ax1, ax2, loc1=4, loc2=2, fc="none", ec="0.6")

# ax1.set_facecolor('#EBEBEB')
# [ax1.spines[side].set_visible(False) for side in ax1.spines]
# ax1.grid(color='white', linewidth=1)
ax2.set_facecolor('#EBEBEB')
[ax2.spines[side].set_visible(False) for side in ax2.spines]
ax2.grid(color='white', linewidth=1)
if TYPE == "few":
    plt.ylabel("Example Coverage (%)")
plt.xlabel("Number of Annotators")
ax2.legend()
if TYPE == "many":
    plt.savefig(f"annotation_patterns/all-many.pdf", bbox_inches='tight')
else:
    plt.savefig(f"annotation_patterns/all-few.pdf", bbox_inches='tight')
