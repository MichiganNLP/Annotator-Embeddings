import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plot setting
FONT_SIZE = 12
TICK_SIZE = 12
LEGEND_FONT_SIZE = 12

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=42)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"


# Example data
data = [[1.0, 0.40750853242320817, 0.450552873384208, 0.2225982457352027, 0.20543698581137493, 0.2814058030241111], [0.40750853242320817, 1.0, 0.4408104196816208, 0.21728561782579336, 0.19906213364595537, 0.23172085646312457], [0.450552873384208, 0.4408104196816208, 1.0, 0.28404320519833137, 0.24901868244448788, 0.2689575734306008], [0.2225982457352027, 0.21728561782579336, 0.28404320519833137, 1.0, 0.6649218485936401, 0.5566697453489906], [0.20543698581137493, 0.19906213364595537, 0.24901868244448788, 0.6649218485936401, 1.0, 0.5196510782650836], [0.2814058030241111, 0.23172085646312457, 0.2689575734306008, 0.5566697453489906, 0.5196510782650836, 1.0]]
# data = [[ 1.0000001 ,  0.0389656,  -0.03330534, -0.01412641,  0.0274183,  -0.02302336],
#  [ 0.0389656 ,  1.0000001,  -0.01841876,  0.02257179, -0.02935919,  0.02033792],
#  [-0.03330534, -0.01841876,  1.0000004,  -0.00151378, -0.01719022, -0.01987514],
#  [-0.01412641,  0.02257179, -0.00151378,  1.0000002,   0.03421939,  0.00708286],
#  [ 0.0274183 , -0.02935919, -0.01719022,  0.03421939,  1.,          0.09329832],
#  [-0.02302336,  0.02033792, -0.01987514,  0.00708286,  0.09329832,  1.0000001 ]]
# data = [[1.        , 0.9999412,  0.9998168,  0.98315823, 0.96477413, 0.99015665],
#  [0.9999412 , 1.0000002,  0.99996567, 0.9850851,  0.9675741,  0.99161845],
#  [0.9998168 , 0.99996567, 0.99999976, 0.98647743, 0.9696348,  0.99265504],
#  [0.98315823, 0.9850851,  0.98647743, 0.99999976, 0.9966048,  0.9990597 ],
#  [0.96477413, 0.9675741,  0.9696348,  0.9966048,  1.0000002,  0.9920989 ],
#  [0.99015665, 0.99161845, 0.99265504, 0.9990597,  0.9920989,  0.99999994]]
# data = [[1.0, 0.35275704310137257, 0.37205651491365777, 0.2428556796066783, 0.20712917464812186, 0.29406220546654105], [0.35275704310137257, 1.0, 0.42279942279942284, 0.24114272700098205, 0.22157717062458382, 0.22909894682095966], [0.37205651491365777, 0.42279942279942284, 1.0, 0.3000683526999316, 0.23207227555053644, 0.24844720496894412], [0.2428556796066783, 0.24114272700098205, 0.3000683526999316, 1.0, 0.6411698222475457, 0.5628478049642258], [0.20712917464812186, 0.22157717062458382, 0.23207227555053644, 0.6411698222475457, 1.0, 0.528145306505911], [0.29406220546654105, 0.22909894682095966, 0.24844720496894412, 0.5628478049642258, 0.528145306505911, 1.0]]
# data = [[1.0, 0.5303983228511531, 0.5749525616698292, 0.19791666666666663, 0.2106499608457323, 0.34811529933481156], [0.5303983228511531, 1.0, 0.4686907020872866, 0.10069444444444431, 0.12294440093970249, 0.22394678492239473], [0.5749525616698292, 0.4686907020872866, 1.0, 0.21959459459459474, 0.2708705642073015, 0.3749114103472714], [0.19791666666666663, 0.10069444444444431, 0.21959459459459474, 1.0, 0.6802884615384616, 0.6067415730337078], [0.2106499608457323, 0.12294440093970249, 0.2708705642073015, 0.6802884615384616, 1.0, 0.5028496273564227], [0.34811529933481156, 0.22394678492239473, 0.3749114103472714, 0.6067415730337078, 0.5028496273564227, 1.0]]

data = np.array(data)

mask = np.triu(data)
mask[np.diag_indices_from(mask)] = False 

# Create heatmap
ax = sns.heatmap(data, annot=True, cmap="PiYG", center=0, fmt=".2f", cbar=True, mask=mask)

ax.set_xticks(np.arange(6) + 0.5)
ax.set_xticklabels([1, 2, 3, 4, 5, 6])

ax.set_yticks(np.arange(6) + 0.5)
ax.set_yticklabels([1, 2, 3, 4, 5, 6])

plt.savefig(f"heatmaps/hs_brexit-cohens.pdf", bbox_inches='tight')