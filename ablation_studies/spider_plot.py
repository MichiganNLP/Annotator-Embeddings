import matplotlib.pyplot as plt
import numpy as np
import os

# plot setting
FONT_SIZE = 20
TICK_SIZE = 20
LEGEND_FONT_SIZE = 20


plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=42)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"

cluster_num = 5
# Example data
# data = [{'age': 1.0, 'race': 0.8662379629263522, 'hispanic_latino': 0.0, 'grew_up_area': 0.2002427184466018, 'current_live_area': 0.0, 'current_live_region': 0.014304800926338002, 'annual_householdincome': 0.5075895079263735, 'education': 0.7405480444362736, 'employment_status': 0.5274779602611382, 'living_situation': 0.3019451456057094, 'political_identification': 0.4418727776148129, 'gender': 0.5580007703155377}, {'age': 0.6917685993316246, 'race': 0.0, 'hispanic_latino': 1.0, 'grew_up_area': 0.1155462184873945, 'current_live_area': 0.5633139331682044, 'current_live_region': 0.15756687996299368, 'annual_householdincome': 0.0, 'education': 0.8390030691945669, 'employment_status': 0.2524720699604179, 'living_situation': 0.2677908022635475, 'political_identification': 1.0, 'gender': 0.07655333351927768}, {'age': 0.0, 'race': 0.28306014607384467, 'hispanic_latino': 0.015792966157929602, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'current_live_region': 1.0, 'annual_householdincome': 0.3927027917706341, 'education': 0.0, 'employment_status': 0.9227022265213413, 'living_situation': 0.0, 'political_identification': 0.26921273273520885, 'gender': 0.6622771096455308}, {'age': 0.9486637475464287, 'race': 0.9120251344939488, 'hispanic_latino': 0.6916213994595835, 'grew_up_area': 0.0, 'current_live_area': 0.11918231686989215, 'current_live_region': 0.8427451181384866, 'annual_householdincome': 1.0, 'education': 1.0, 'employment_status': 0.0, 'living_situation': 1.0, 'political_identification': 0.5896225115501944, 'gender': 1.0}, {'age': 0.4799627213420321, 'race': 1.0, 'hispanic_latino': 0.7213745413117489, 'grew_up_area': 0.5689655172413788, 'current_live_area': 0.4540749516742536, 'current_live_region': 0.0, 'annual_householdincome': 0.5171913784174691, 'education': 0.6264100629127548, 'employment_status': 1.0, 'living_situation': 0.2292527925036072, 'political_identification': 0.0, 'gender': 0.0}]

# normalized considering group distribution
# data = [{'age': 0.06631768599177228, 'race': 0.0, 'hispanic_latino': 0.0, 'grew_up_area': 0.0729306776125505, 'current_live_area': 0.0, 'current_live_region': 0.0, 'annual_householdincome': 0.34324859340051833, 'education': 0.37514077719750116, 'employment_status': 0.5489285926091695, 'living_situation': 0.7207554871557765, 'political_identification': 0.3751048068126521, 'gender': 0.016043438171866573}, {'age': 0.5792946872492846, 'race': 0.3247165448988325, 'hispanic_latino': 1.0, 'grew_up_area': 0.19248242436975857, 'current_live_area': 0.8264686879423405, 'current_live_region': 0.15371173924098078, 'annual_householdincome': 0.5387680131631734, 'education': 0.31882595350261805, 'employment_status': 0.1682289808618143, 'living_situation': 0.2514377646832391, 'political_identification': 1.0, 'gender': 0.0022010340101723215}, {'age': 1.0, 'race': 0.4757554755227374, 'hispanic_latino': 0.01579296615792966, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'current_live_region': 1.0, 'annual_householdincome': 1.0, 'education': 0.2865296687353312, 'employment_status': 0.8739858050928804, 'living_situation': 0.0, 'political_identification': 0.2776800407737745, 'gender': 0.019041554109741}, {'age': 0.12359342297499963, 'race': 0.2561848993644985, 'hispanic_latino': 0.6916213994595836, 'grew_up_area': 0.0, 'current_live_area': 0.27169061648443604, 'current_live_region': 0.6924381744548591, 'annual_householdincome': 0.0, 'education': 1.0, 'employment_status': 0.0, 'living_situation': 1.0, 'political_identification': 0.5191344147481848, 'gender': 1.0}, {'age': 0.0, 'race': 1.0, 'hispanic_latino': 0.721374541311749, 'grew_up_area': 0.4691998204905285, 'current_live_area': 0.518829848404944, 'current_live_region': 0.05905346896821951, 'annual_householdincome': 0.3455806264781424, 'education': 0.0, 'employment_status': 1.0, 'living_situation': 0.36665069568363834, 'political_identification': 0.0, 'gender': 0.0}]

# normalized group distribution
# data = [{'age': 3.0, 'race': 8.0, 'hispanic_latino': 0.0, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'current_live_region': 1.0, 'annual_householdincome': 5.0, 'education': 2.0, 'employment_status': 2.0, 'living_situation': 5.0, 'political_identification': 2.0, 'gender': 1.0}, {'age': 4.0, 'race': 10.0, 'hispanic_latino': 1.0, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'current_live_region': 1.0, 'annual_householdincome': 5.0, 'education': 2.0, 'employment_status': 2.0, 'living_situation': 6.0, 'political_identification': 2.0, 'gender': 0.0}, {'age': 4.0, 'race': 13.0, 'hispanic_latino': 0.0, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'current_live_region': 2.0, 'annual_householdincome': 5.0, 'education': 2.0, 'employment_status': 2.0, 'living_situation': 3.0, 'political_identification': 2.0, 'gender': 1.0}, {'age': 1.0, 'race': 10.0, 'hispanic_latino': 1.0, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'current_live_region': 2.0, 'annual_householdincome': 4.0, 'education': 2.0, 'employment_status': 2.0, 'living_situation': 8.0, 'political_identification': 2.0, 'gender': 2.0}, {'age': 2.0, 'race': 11.0, 'hispanic_latino': 1.0, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'current_live_region': 1.0, 'annual_householdincome': 4.0, 'education': 2.0, 'employment_status': 2.0, 'living_situation': 5.0, 'political_identification': 2.0, 'gender': 0.0}]
# data = [{'age': 0.5251263772418987, 'race': 0.0, 'hispanic_latino': 0.0, 'grew_up_area': 0.23080264323115304, 'current_live_area': 0.0, 'current_live_region': 0.011073697190017722, 'annual_householdincome': 0.8397345599776764, 'education': 0.5372032547318476, 'employment_status': 0.8388780695283291, 'living_situation': 0.45769722440116023, 'political_identification': 0.3404343775771376, 'gender': 0.10160251492374088}, {'age': 0.8603120896619711, 'race': 0.38333086400613836, 'hispanic_latino': 1.0, 'grew_up_area': 0.09461087856012577, 'current_live_area': 0.5588804791733013, 'current_live_region': 0.13543097203867796, 'annual_householdincome': 0.7543055761870403, 'education': 0.5899829119993322, 'employment_status': 0.0, 'living_situation': 0.5301693165974007, 'political_identification': 1.0, 'gender': 0.013952324656226698}, {'age': 1.0, 'race': 1.0, 'hispanic_latino': 0.020315956813468466, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'current_live_region': 1.0, 'annual_householdincome': 1.0, 'education': 0.0, 'employment_status': 0.28783394140952984, 'living_situation': 0.0, 'political_identification': 0.3433578295448513, 'gender': 0.12056466572598275}, {'age': 0.0, 'race': 0.40601279923653344, 'hispanic_latino': 0.7434844512752281, 'grew_up_area': 0.0, 'current_live_area': 0.13442695033656113, 'current_live_region': 0.8248630560351092, 'annual_householdincome': 0.30397733737333066, 'education': 1.0, 'employment_status': 0.4800182496676102, 'living_situation': 1.0, 'political_identification': 0.482345762951701, 'gender': 1.0}, {'age': 0.45142183360107735, 'race': 0.5672549164248211, 'hispanic_latino': 0.7698984988885409, 'grew_up_area': 0.5974613082941508, 'current_live_area': 0.4634104503501635, 'current_live_region': 0.0, 'annual_householdincome': 0.0, 'education': 0.1943110306833776, 'employment_status': 1.0, 'living_situation': 0.40997526671398665, 'political_identification': 0.0, 'gender': 0.0}]

# majority
data = [{'age': 0.25, 'grew_up_area': 0.5, 'current_live_area': 0.5, 'annual_householdincome': 1.0, 'education': 0.6666666666666666, 'political_identification': 1.0, 'gender': 1.0}, {'age': 0.75, 'grew_up_area': 0.0, 'current_live_area': 1.0, 'annual_householdincome': 0.0, 'education': 0.6666666666666666, 'political_identification': 0.0, 'gender': 0.0}, {'age': 0.75, 'grew_up_area': 1.0, 'current_live_area': 1.0, 'annual_householdincome': 0.125, 'education': 1.0, 'political_identification': 0.6666666666666666, 'gender': 1.0}, {'age': 1.0, 'grew_up_area': 0.0, 'current_live_area': 0.0, 'annual_householdincome': 0.5, 'education': 0.0, 'political_identification': 1.0, 'gender': 0.5}, {'age': 0.0, 'grew_up_area': 0.5, 'current_live_area': 0.5, 'annual_householdincome': 1.0, 'education': 0.3333333333333333, 'political_identification': 0.6666666666666666, 'gender': 0.0}]

# mappings = {'age': 'AG', 'race': 'RA', 'hispanic_latino': 'HL', 'grew_up_area': 'GU', 'current_live_area': 'CA', 'current_live_region': 'CR', 'annual_householdincome': 'AH', 'education': 'ED', 'employment_status': 'ES', 'living_situation': 'LS', 'political_identification': 'PI', 'gender': 'GE'}
mappings = {
    'age': 'AG', 
    'annual_householdincome': 'AH', 
    'education': 'ED', 
    'political_identification': 'PI', 
    'current_live_area': 'CA',
    'grew_up_area': 'GU',
    'gender': 'GE'}

def filter_data(data):
    pdata = []
    for d in data:
        pd = {k: v for k, v in d.items() if k in {"age", "annual_householdincome", "political_identification", "education", "current_live_area", "grew_up_area", "gender"}}
        pdata.append(pd)
    return pdata

data = filter_data(data)


# Example data
# data = [
#     {'A': 0.6, 'B': 0.8, 'C': 0.4, 'D': 0.7, 'E': 0.5},
#     {'A': 0.3, 'B': 0.5, 'C': 0.2, 'D': 0.6, 'E': 0.4}
# ]

# Extract categories from the first dictionary
categories = list(data[0].keys())

# Calculate angles for each category
angles = [i / float(len(categories)) * 2 * 3.14159 for i in range(len(categories))]
angles += angles[:1]  # Close the plot

# Plot the spider charts for each dictionary
for di, d in enumerate(data):
    values = list(d.values())
    values += values[:1]  # Repeat the first value to complete the loop
    plt.polar(angles, values, marker='o', clip_on=False, label=f"Group {di}")

# Fill the areas inside the plots
for d in data:
    values = list(d.values())
    values += values[:1]  # Repeat the first value to complete the loop
    plt.fill(angles, values, alpha=0.25)


# Set the labels for each category
plt.xticks(angles[:-1], [mappings[ca] for ca in categories])

# Set the range for the radial axis
plt.ylim(0, 1)

# Add a grid
plt.grid(True)
plt.legend(loc=(-0.6, 0.25))
# Show the plot
plt.savefig(f"spider_plots/sentiment-spider-{cluster_num}-1.pdf", bbox_inches='tight')
plt.close()

if not os.path.exists(f"spider_plots/{cluster_num}-clusters-1"):
    os.makedirs(f"spider_plots/{cluster_num}-clusters-1")
for di, d in enumerate(data):
    plt.figure(figsize=(6, 6))
    values = list(d.values())
    values += values[:1]  # Repeat the first value to complete the loop
    plt.polar(angles, values, marker='o', clip_on=False, )
    plt.fill(angles, values, alpha=0.25)
    # Set the labels for each category
    plt.xticks(angles[:-1], [mappings[ca] for ca in categories])

    # Set the range for the radial axis
    plt.ylim(0, 1)

    # Add a grid
    plt.grid(True)

    # Show the plot
    plt.savefig(f"spider_plots/{cluster_num}-clusters-1/spider-{di}.pdf", bbox_inches='tight')
    plt.close()

