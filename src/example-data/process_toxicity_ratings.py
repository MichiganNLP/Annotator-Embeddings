import os
import pandas as pd
import numpy as np
import json
import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy

from utils import create_annotation_split, create_annotator_split


DATASET_NAME = "toxic-ratings"
TASK = "toxic_score"

annotator_data = defaultdict(list)
annotation_labels = set()
stats = defaultdict(int)
idx = 0
with open(f'raw-{DATASET_NAME}/toxicity_ratings.json', 'r') as f:
    lines = f.readlines()
for uid, line in enumerate(lines):
    data = json.loads(line)
    # process each row differently this time
    item_dict = {}
    item_dict["uid"] = uid
    ratings = data["ratings"]
    for rating in ratings:
        item_dict["id"] = idx
        item_dict["respondent_id"] = rating["worker_id"]
        item_dict[TASK] = str(rating["toxic_score"])
        annotation_labels.add(str(rating["toxic_score"]))
        item_dict["sentence"] = data["comment"]
        annotator_data[rating["worker_id"]].append(deepcopy(item_dict))
        stats[rating["worker_id"]] += 1
        idx += 1

create_annotation_split(DATASET_NAME, annotator_data, TASK)
create_annotator_split(DATASET_NAME, annotator_data, TASK)


with open(f"{DATASET_NAME}-processed/annotation_labels.json", 'w') as f:
    json.dump(list(annotation_labels), f, indent=4)

with open(f"{DATASET_NAME}-processed/stats.json", 'w') as f:
    json.dump(stats, f, indent=4)

