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

DATASET_NAME = "commitmentbank"
TASK = "certainty"

annotator_history = defaultdict(list)
annotator_data = defaultdict(list)
annotation_labels = set()
stats = defaultdict(int)
text_data = defaultdict(list)
idx = 0
with open('raw-commitmentbank/CommitmentBank-All.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)    # skip the row of column names
    for row in reader:
        # process each row
        annotator_history[row[2]].append(row[4])
    
    # Reset the file pointer to the beginning of the file
    csvfile.seek(0)

    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    weak_labels = defaultdict(list)
    for row in reader:
        # process each row differently this time
        item_dict = {}
        weak_labels[row[0]].append(str(row[4]))
        item_dict["uid"] = row[0]
        item_dict["id"] = idx
        item_dict["HitID"] = row[1]
        worker_id = row[2]
        item_dict["Verb"] = row[3]
        item_dict["respondent_id"] = worker_id
        # need to convert to str to run the model
        item_dict["certainty"] = str(row[4])
        annotation_labels.add(str(row[4]))
        item_dict["Context"] = row[5]
        item_dict["Prompt"] = row[7]
        item_dict["Target"] = row[6]
        item_dict["ModalType"] = row[9]
        item_dict["Embedding"] = row[8]
        item_dict["MatTense"] = row[10]
        # Following Frederick's setting
        item_dict["sentence"] = f"{row[5]}</s>{row[6]}</s>{row[7]}"
        assert row[9] in {"", "AB", "CI", "DE", "EP"}
        annotator_data[worker_id].append(deepcopy(item_dict))
        stats[worker_id] += 1
        uid = row[0]
        text_data[uid].append(item_dict)    # uid is the id for the same text
        idx += 1

for aid, item_l in annotator_data.items():
    for item in item_l:
        item["weak_labels"] = weak_labels[item["uid"]]

create_annotation_split(DATASET_NAME, annotator_data, TASK)
create_annotator_split(DATASET_NAME, annotator_data, TASK)

# for th in [0.6, 0.7, 0.8]:
#     create_th_annotation_split(DATASET_NAME, text_data, TASK, th=th)
#     create_th_annotator_split(DATASET_NAME, text_data, TASK, th=th)
#     create_th_annotation_split(DATASET_NAME, text_data, TASK, th=th, part="smaller")
#     create_th_annotator_split(DATASET_NAME, text_data, TASK, th=th, part="smaller")

with open("commitmentbank-processed/annotation_labels.json", 'w') as f:
    json.dump(list(annotation_labels), f, indent=4)

with open("commitmentbank-processed/stats.json", 'w') as f:
    json.dump(stats, f, indent=4)

