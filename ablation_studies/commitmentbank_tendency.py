import json
import os
import json
import tqdm
import random
from collections import defaultdict

random.seed(42)
dataset = "commitmentbank"
split = "annotation"
task = "certainty"

PATH = f"../src/example-data/{dataset}-processed/"
example_annotations = defaultdict(dict)
sentences = set()
for split in ["train", "dev", "test"]:
    if os.path.exists(f"{PATH}/annotation_split_{split}.json"):
        with open(f"{PATH}/annotation_split_{split}.json", 'r') as f:
            data_paths = json.load(f)
    for data_path in tqdm.tqdm(iter(data_paths)):
        with open(f"../src/{data_path['path']}", 'r') as f:
            data = json.load(f)
        sentences.add(data["sentence"])

sel_sents = random.sample(sentences, k=3)
for split in ["train", "dev", "test"]:
    if os.path.exists(f"{PATH}/annotation_split_{split}.json"):
        with open(f"{PATH}/annotation_split_{split}.json", 'r') as f:
            data_paths = json.load(f)
    for data_path in tqdm.tqdm(iter(data_paths)):
        with open(f"../src/{data_path['path']}", 'r') as f:
            data = json.load(f)
        if data["sentence"] in sel_sents:
            example_annotations[data["respondent_id"]][data["sentence"]] = data[task]

filtered_anns = dict()
for person, v in example_annotations.items():
    if len(v) < 3:
        continue
    filtered_anns[person] = v
