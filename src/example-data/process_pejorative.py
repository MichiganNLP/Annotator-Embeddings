import os
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy

from utils import create_annotation_split, create_annotator_split

TASK="pejorative"
DATASET_NAME="pejorative"

df = pd.read_csv(os.path.join("raw-pejorative_dataset", \
    "tweet_datasets", "English", "PEJOR1_annotated.csv"), skiprows=[0])

if not os.path.exists("pejorative-processed"):
    os.makedirs("pejorative-processed")

data = []
stats = defaultdict(int)

# two splits of one of same set of annotators in the training and testing datasets
# another of different annotators in the training and testing datasets
annotator_data = defaultdict(list)

def process_v(v):
    if v == "1" or v == 1 or v == "1.0":
        return "pejorative"
    elif v == "0" or v == 0 or v == "0.0":
        return "non-pejorative"
    elif v in {"X", "1 or 0", "1 or0", "0 or 1", "1/0"}:
        return "undecided"
    else:
        raise RuntimeError("unexpected value")

tid = 0
for i in range(len(df)):
    example_dict = {}
    for k, v in df.iloc[i].items():
        if k == "tweet":
            example_dict["sentence"] = v
        else:
            example_dict[k] = v
    example_dict["uid"] = i

    example_dict["annotator-1"] = example_dict["Label (pejorative=1, non-pejorative=0)"]
    example_dict["annotator-2"] = example_dict["P1:Label (pejorative=1, non-pejorative=0)"]
    example_dict["annotator-3"] = example_dict["P3: Label (pejorative=1, non-pejorative=0)"]
    example_dict.pop("Label (pejorative=1, non-pejorative=0)")
    example_dict.pop("P1:Label (pejorative=1, non-pejorative=0)")
    example_dict.pop("P3: Label (pejorative=1, non-pejorative=0)")
    if (not isinstance(example_dict["annotator-1"], str)) and np.isnan(example_dict["annotator-1"]):
        example_dict.pop("annotator-1")
    if (not isinstance(example_dict["annotator-2"], str)) and np.isnan(example_dict["annotator-2"]):
        example_dict.pop("annotator-2")
    if (not isinstance(example_dict["annotator-3"], str)) and np.isnan(example_dict["annotator-3"]):
        example_dict.pop("annotator-3")

    if "annotator-1" in example_dict:
        example_dict["annotator-1"] = str(example_dict["annotator-1"])
    if "annotator-2" in example_dict:
        example_dict["annotator-2"] = str(example_dict["annotator-2"])
    if "annotator-3" in example_dict:
        example_dict["annotator-3"] = str(example_dict["annotator-3"])

    if example_dict.get("annotator-1") and example_dict["annotator-1"] != "None":
        example_dict["id"] = tid
        example_dict["pejorative"] = process_v(example_dict["annotator-1"])
        example_dict["respondent_id"] = "annotator-1"
        if "annotator-2" not in example_dict:
            example_dict["annotator-2"] = "None"
        if "annotator-3" not in example_dict:
            example_dict["annotator-3"] = "None"
        annotator_data["annotator-1"].append(deepcopy(example_dict))
        stats["annotator-1"] += 1
        tid += 1
    if example_dict.get("annotator-2") and example_dict["annotator-2"] != "None":
        example_dict["id"] = tid
        example_dict["pejorative"] = process_v(example_dict["annotator-2"])
        example_dict["respondent_id"] = "annotator-2"
        if "annotator-1" not in example_dict:
            example_dict["annotator-1"] = "None"
        if "annotator-3" not in example_dict:
            example_dict["annotator-3"] = "None"
        annotator_data["annotator-2"].append(deepcopy(example_dict))
        stats["annotator-2"] += 1
        tid += 1
    if example_dict.get("annotator-3") and example_dict["annotator-3"] != "None":
        example_dict["id"] = tid
        example_dict["pejorative"] = process_v(example_dict["annotator-3"])
        example_dict["respondent_id"] = "annotator-3"
        if "annotator-1" not in example_dict:
            example_dict["annotator-1"] = "None"
        if "annotator-2" not in example_dict:
            example_dict["annotator-2"] = "None"
        annotator_data["annotator-3"].append(deepcopy(example_dict))
        stats["annotator-3"] += 1
        tid += 1

create_annotation_split(DATASET_NAME, annotator_data, TASK)

create_annotator_split(DATASET_NAME, annotator_data, TASK)
    
with open(os.path.join("pejorative-processed", "stats.json"), "w") as f:
    json.dump(stats, f, indent=4)

print(f"Number of annotators: {len(stats)}")
