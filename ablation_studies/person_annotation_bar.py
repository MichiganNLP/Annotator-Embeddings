import matplotlib.pyplot as plt

# plot setting
FONT_SIZE = 11
TICK_SIZE = 11
LEGEND_FONT_SIZE = 8


plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=42)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"

data = {'Very negative': 6, 'Somewhat Negative': 14, 'Neutral': 24,  'Somewhat Positive': 16, 'Very Positive': 20}

# Extract labels and values
labels = list(data.keys())
values = list(data.values())


fig, ax = plt.subplots(figsize=(5, 4))

# Create bar plot
plt.bar([-2, -1, 0, 1, 2], values, width=0.5, color="#78bf9d")

# Add labels and title
plt.ylabel('Counts')
# plt.xticks(rotation = 18) # Rotates X-Axis Ticks by 45-degrees
ax.set_facecolor('#EBEBEB')
[ax.spines[side].set_visible(False) for side in ax.spines]

# Display the plot
plt.savefig("persona_annotation_bars/sentiment-person.pdf", bbox_inches='tight')

