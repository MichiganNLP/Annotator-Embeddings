import os
import pandas as pd
import numpy as np
import json
import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy

from utils import create_annotation_split, create_annotator_split,\
     create_th_annotation_split, create_th_annotator_split

DATASET_NAME = "humor"
TASK = "humor"

annotator_data = defaultdict(list)
annotation_labels = set()
stats = defaultdict(int)
idx = 0
prev_sent = None
uid = -1
with open(f'raw-{DATASET_NAME}/data/pl-humor-full/results.tsv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
    next(reader)    # skip the row of column names

    for row in reader:
        # process each row differently this time
        item_dict = {}
        item_dict["id"] = idx
        worker_id, answer, text_a, text_b = row[0], row[1], row[2], row[3]
        item_dict["respondent_id"] = worker_id
        # need to convert to str to run the model
        item_dict[TASK] = answer
        annotation_labels.add(answer)
        item_dict["text_a"] = text_a
        item_dict["text_b"] = text_b
        item_dict["sentence"] = f"{text_a}</s>{text_b}"
        if f"{text_a}</s>{text_b}" != prev_sent:
            prev_sent = f"{text_a}</s>{text_b}"
            uid += 1
        item_dict["uid"] = uid
        assert answer in {"X", "A", "B"}
        annotator_data[worker_id].append(deepcopy(item_dict))
        stats[worker_id] += 1
        idx += 1

create_annotation_split(DATASET_NAME, annotator_data, TASK)
create_annotator_split(DATASET_NAME, annotator_data, TASK)


with open(f"{DATASET_NAME}-processed/annotation_labels.json", 'w') as f:
    json.dump(list(annotation_labels), f, indent=4)

with open(f"{DATASET_NAME}-processed/stats.json", 'w') as f:
    json.dump(stats, f, indent=4)

