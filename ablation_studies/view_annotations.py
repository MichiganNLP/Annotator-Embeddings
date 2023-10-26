import json
import os
import json
import tqdm
from collections import defaultdict

annotator_id = "R_eFB3dtl1ojJmk7L"
dataset = "sentiment"
split = "annotation"
task = "sentiment"

PATH = f"../src/example-data/{dataset}-processed/"
example_annotations = list()
for split in ["train", "dev"]:
    if os.path.exists(f"{PATH}/annotation_split_{split}.json"):
        with open(f"{PATH}/annotation_split_{split}.json", 'r') as f:
            data_paths = json.load(f)
    for data_path in tqdm.tqdm(iter(data_paths)):
        with open(f"../src/{data_path['path']}", 'r') as f:
            data = json.load(f)
        if data["respondent_id"] == annotator_id:
            example_annotations.append(data[task])
ea = defaultdict(int)
for k in example_annotations:
    ea[k] += 1
print(ea)
