import json
import os
import json
import math
import tqdm

from scipy.stats import pearsonr
from collections import defaultdict

dataset = "commitmentbank"
split = "annotation"
task = "certainty"


annotations = defaultdict(lambda: defaultdict(int))
PATH = f"../src/example-data/{dataset}-processed/"
example_annotations = list()
# mappings = {
#     "Somewhat negative": 0, 
#     "Somewhat positive": 1,
#     "Very positive": 2,
#     "Very negative": 3,
#     "Neutral": 4
# }
mappings = {
    "-3", "-2", "-1", "0", "1", "2", "3"
}
for split in ["train", "dev", "test"]:
    if os.path.exists(f"{PATH}/annotation_split_{split}.json"):
        with open(f"{PATH}/annotation_split_{split}.json", 'r') as f:
            data_paths = json.load(f)
    for data_path in tqdm.tqdm(iter(data_paths)):
        with open(f"../src/{data_path['path']}", 'r') as f:
            data = json.load(f)
        annotations[data["respondent_id"]][data[task]] += 1

annotation_types = defaultdict(list)
counts = 0

for annotator, anns in annotations.items():
    if sum(anns.values()) > 50:
        counts += 1
        for k in mappings:
            if k not in anns:
                annotation_types[k].append(0)
            else:
                annotation_types[k].append(anns[k])

print(counts)
def calculate_correlation(list1, list2):
    correlation, _ = pearsonr(list1, list2)
    return correlation

# Calculate correlations between all pairs of lists
correlation_dict = {}
for key1, list1 in annotation_types.items():
    for key2, list2 in annotation_types.items():
        if key1 != key2:
            correlation = calculate_correlation(list1, list2)
            correlation_dict[(int(key1), int(key2))] = correlation

sorted_dict = dict(sorted(correlation_dict.items(), key=lambda x: (x[0][0], x[0][1])))

# Print the correlation results
# for (key1, key2), correlation in correlation_dict.items():
#     print(f"Pearson correlation between {key1} and {key2}: {correlation}")
print(correlation_dict)
