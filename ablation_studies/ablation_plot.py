# annotation split
# results

# dataset_order = ["friends_qia", "pejorative", "md-agreement", "goemotions", "hs_brexit",  "humor", "commitmentbank", "sentiment",]
dataset_order = ["md-agreement", "goemotions",  "humor", "commitmentbank", "sentiment"]
# sh_datasets_order = ["FIA", "PEJ", "MDA", "GOE", "HSB", "HUM", "COM", "SNT"]
sh_datasets_order = ["MDA", "GOE", "HUM", "COM", "SNT"]

datasets = ["commitmentbank", "sentiment", "pejorative", "friends_qia", "goemotions", "hs_brexit", "md-agreement", "humor"]
sh_datasets = ["COM", "SNT", "PEJ", "FIA", "GOE", "HSB", "MDA", "HUM"]
cls_only_q_avg = [13.42, 20.85, 48.47, 39.04, 21.29, 86.05, 58.36, 24.88]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
cls_only_q_std = [1.76, 9.04, 3.44, 13.19, 5.12, 2.7, 5.79, 8.42]                                                                                                                                                                                                                                                                                                                                        
cls_only_annotation_avg = [15.12, 27.77, 49.02, 33.53, 26.68, 86.9, 63.58, 31.98]                                                                                                                                                                                                                         
cls_only_annotation_std = [3.0, 11.08, 2.85, 12.15, 9.06, 0.0, 0.0, 10.57]                                                                                                                                                                                                                           
cls_only_both_avg = [14.68, 19.62, 48.9, 35.72, 24.76, 86.9, 63.32, 34.02]                                                                                                                                                                                                                         
cls_only_both_std = [0.94, 3.78, 3.15, 12.03, 2.91, 0.0, 0.53, 5.13]                                                                                                              
cls_only_annotator_avg = [14.11, 19.25, 47.85, 38.09, 26.38, 85.46, 62.35, 30.94] 
cls_only_annotator_std = [1.38, 9.33, 3.18, 11.18, 3.11, 4.58, 1.79, 5.76] 

question_only_q_avg = [40.83, 47.09, 70.29, 56.36, 63.04, 86.87, 75.06, 54.26]                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
question_only_q_std = [0.72, 0.5, 1.69, 1.32, 0.24, 0.53, 0.56, 0.14]    
question_only_annotation_avg = [40.4, 37.66, 49.4, 48.86, 62.05, 80.6, 74.08, 53.49]                                                                                                                                                                                                                                                                                                                                                       
question_only_annotation_std = [0.64, 8.85, 4.31, 8.48, 0.64, 19.95, 0.86, 1.52]                                                                                                                                                          
question_only_both_avg = [39.11, 32.47, 58.48, 50.88, 61.33, 83.02, 72.49, 50.18]                                                                                                                                                         
question_only_both_std = [1.7, 8.11, 8.99, 4.89, 1.08, 10.37, 2.75, 2.8]                                                                                                                                                           
question_only_annotator_avg = [38.27, 32.68, 61.37, 48.69, 60.62, 85.24, 70.93, 49.97]                                                                                                                                                         
question_only_annotator_std = [2.21, 8.79, 6.33, 4.92, 1.24, 4.69, 4.98, 3.23]

full_annotation_avg = [44.0, 62.88, 70.55, 55.3, 68.49, 86.9, 76.7, 56.72]                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
full_annotation_std = [0.42, 1.11, 2.35, 1.91, 0.35, 0.69, 0.62, 0.36]                                                                                                                                                                                                                                                                                                                            
full_q_avg = [40.83, 47.09, 70.29, 56.36, 63.04, 86.87, 75.06, 54.26]                                                                                                                                                                                                                                                                                                                          
full_q_std = [0.72, 0.5, 1.69, 1.32, 0.24, 0.53, 0.56, 0.14]                                                                                                                                                                                                                                                                                                                            
full_both_avg = [44.41, 64.61, 69.72, 55.22, 69.9, 87.68, 75.76, 53.89]                                                                                                                                                                                                                                                                                                                          
full_both_std = [0.79, 0.78, 2.71, 3.35, 0.26, 0.97, 1.29, 2.68]                                                                                                                                                                                                                                                                                                                            
full_annotator_avg = [44.22, 60.23, 69.71, 55.62, 69.98, 87.8, 75.72, 58.15]                                                                                                                                                                                                                                                                                                                          
full_annotator_std = [0.58, 0.86, 3.55, 1.9, 0.22, 0.57, 1.2, 0.19]          


import matplotlib.pyplot as plt
import numpy as np

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

# Bar plot
labels = sh_datasets_order

x = np.arange(len(labels))
width = 0.2
error_kw = dict(lw=1, capsize=1.5, capthick=1)

# map the old order to the new order
def reorder(l):
    # mapping of the old order to the new order
    order_maps = []
    for dname in dataset_order:
        order_maps.append(datasets.index(dname))
    ordered_l = []
    for ord in order_maps:
        ordered_l.append(l[ord])
    return ordered_l


fig, ax = plt.subplots()
rects1 = ax.bar(x - width, reorder(cls_only_both_avg), width, yerr=reorder(cls_only_both_std),\
                 color='#bae1e8', ecolor='#8B8F9E', error_kw=error_kw, label='Embedding Only')
rects2 = ax.bar(x, reorder(question_only_both_avg), width, yerr=reorder(question_only_both_std),\
                 color='#fce3bd', ecolor='#8B8F9E', error_kw=error_kw, label='Text Only')
rects3 = ax.bar(x + width, reorder(full_both_avg), width, yerr=reorder(full_both_std),\
                 color='#78bf9d', ecolor='#8B8F9E', error_kw=error_kw, label='Combination')

ax.set_facecolor('#EBEBEB')
# ax.grid(which='major', color='white', linewidth=0.001)
# ax.grid(which='minor', color='white', linewidth=0.0005)
# ax.minorticks_on()
# ax.tick_params(which='minor', bottom=False, left=False)
[ax.spines[side].set_visible(False) for side in ax.spines]
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('performance_ablation/annotation-bar.pdf', bbox_inches='tight')
