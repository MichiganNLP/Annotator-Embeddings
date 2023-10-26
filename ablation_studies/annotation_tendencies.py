import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm

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

# Example dictionary of lists and correlation values
correlation_dict = {('Somewhat Negative', 'Somewhat Positive'): 0.21286779455424362, ('Somewhat Negative', 'Very Positive'): -0.22492140409261166, ('Somewhat Negative', 'Very Negative'): 0.030670168538675387, ('Somewhat Negative', 'Neutral'): -0.5270749887760589, ('Somewhat Positive', 'Somewhat Negative'): 0.21286779455424362, ('Somewhat Positive', 'Very Positive'): -0.12696462837699637, ('Somewhat Positive', 'Very Negative'): -0.18545347097891693, ('Somewhat Positive', 'Neutral'): -0.6013315146614131, ('Very Positive', 'Somewhat Negative'): -0.22492140409261166, ('Very Positive', 'Somewhat Positive'): -0.12696462837699637, ('Very Positive', 'Very Negative'): 0.20846684234254956, ('Very Positive', 'Neutral'): -0.44431802503462003, ('Very Negative', 'Somewhat Negative'): 0.030670168538675387, ('Very Negative', 'Somewhat Positive'): -0.18545347097891693, ('Very Negative', 'Very Positive'): 0.20846684234254956, ('Very Negative', 'Neutral'): -0.32191188125686915, ('Neutral', 'Somewhat Negative'): -0.5270749887760589, ('Neutral', 'Somewhat Positive'): -0.6013315146614131, ('Neutral', 'Very Positive'): -0.44431802503462003, ('Neutral', 'Very Negative'): -0.32191188125686915}

# correlation_dict = {(0, -3): 0.672087330304234, (0, 3): 0.7062896104481817, (0, -1): 0.6697057767909749, (0, -2): 0.6485861474079335, (0, 1): 0.8385403016906361, (0, 2): 0.8610908574515229, (-3, 0): 0.672087330304234, (-3, 3): 0.9054555374360802, (-3, -1): 0.4426387742557081, (-3, -2): 0.3991405512466931, (-3, 1): 0.6336372203876766, (-3, 2): 0.6484766904229875, (3, 0): 0.7062896104481817, (3, -3): 0.9054555374360802, (3, -1): 0.48352281367706434, (3, -2): 0.3991370635089841, (3, 1): 0.7132771991899474, (3, 2): 0.7684143371412747, (-1, 0): 0.6697057767909749, (-1, -3): 0.4426387742557081, (-1, 3): 0.48352281367706434, (-1, -2): 0.7156867110567123, (-1, 1): 0.7953388721353951, (-1, 2): 0.6824244711196443, (-2, 0): 0.6485861474079335, (-2, -3): 0.3991405512466931, (-2, 3): 0.3991370635089841, (-2, -1): 0.7156867110567123, (-2, 1): 0.6318869052225791, (-2, 2): 0.6453137886897988, (1, 0): 0.8385403016906361, (1, -3): 0.6336372203876766, (1, 3): 0.7132771991899474, (1, -1): 0.7953388721353951, (1, -2): 0.6318869052225791, (1, 2): 0.8804904507137649, (2, 0): 0.8610908574515229, (2, -3): 0.6484766904229875, (2, 3): 0.7684143371412747, (2, -1): 0.6824244711196443, (2, -2): 0.6453137886897988, (2, 1): 0.8804904507137649}

# Extract unique list names
# list_names = [-3, -2, -1, 0, 1, 2, 3]
list_names = ["Very Negative", "Somewhat Negative", "Neutral", "Somewhat Positive", "Very Positive"]


# Create a matrix of correlation values
correlation_matrix = []
for list1 in list_names:
    row = []
    for list2 in list_names:
        if (list1, list2) in correlation_dict:
            correlation = correlation_dict[(list1, list2)]
        elif (list2, list1) in correlation_dict:
            correlation = correlation_dict[(list2, list1)]
        else:
            correlation = 1.0
        row.append(correlation)
    correlation_matrix.append(row)

correlation_matrix = np.array(correlation_matrix)
mask = np.triu(correlation_matrix)
mask[np.diag_indices_from(mask)] = False 


# Create the heatmap using seaborn
ax = sns.heatmap(correlation_matrix, annot=True, center=0, cmap="PiYG", fmt=".2f", mask=mask)
ax.set_xticklabels(list_names, rotation=25, ha='right')
ax.set_yticklabels(list_names, rotation=0)
# plt.xlabel('Lists')
# plt.ylabel('Lists')
# plot setting

plt.savefig('annotation_tendencies/sentiment.pdf', bbox_inches='tight')

