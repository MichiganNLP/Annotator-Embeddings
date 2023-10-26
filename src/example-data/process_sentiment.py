import os
import pandas as pd
import math
import json
from collections import defaultdict
from copy import deepcopy

from utils import create_annotator_split

DATASET_NAME = "sentiment"
TASK = "sentiment"

demographics = pd.read_csv(f"raw-{DATASET_NAME}/demographics_responses.csv", encoding = "ISO-8859-1")
test = pd.read_csv(f"raw-{DATASET_NAME}/test_annotations.csv", encoding = "ISO-8859-1")
train = pd.read_csv(f"raw-{DATASET_NAME}/train_older_adult_annotations.csv", encoding = "ISO-8859-1")

raw_demographics = demographics.to_dict()
demographics = {}

KEYS = {
    'respondent_id', 
    'Please indicate your age', 
    'Please indicate your race - Selected Choice', 
    'Please indicate your race - Other - Text', 
    'Are you Hispanic or Latino?', 
    'How would you describe the area where you grew up?', 
    'How would you describe the area in which you currently live?', 
    'In which region of the United States do you currently live?', 
    'Please indicate your annual household income', 
    'Education', 
    'Which employment status best describes you?', 
    'How would you describe your current living situation (check all that apply) - Selected Choice', 
    'How would you describe your current living situation (check all that apply) - Other - Text', 
    'Please indicate your political identification', 
    'Please indicate your gender'
}

def check(val):
    if isinstance(val, str):
        return True
    return False

def fix_race(val):
    if val in {"mixed", "Mixed ", "Mixed", "Mixedd"}:
        return "Mixed"
    if val in {"Mexican", "mexican"}:
        return "Mexican"
    if val in {"human", "human race", "prefer not to answer"}:
        return "human"
    if val in {"Hispanic", "hispanic"}:
        return "hispanic"
    return val

def fix_live(val):
    if val in {"Roomate", "I have a roommate", "live with room-mate", "roommates", "Roomate 9", "Roommates"}:
        return "Roomate"
    if val in {"And kids", "kids", "Children", "Children under 18"}:
        return "kids"
    return val

for k, val in raw_demographics["respondent_id"].items():
    demographics[val] = {
        "age": raw_demographics['Please indicate your age'][k],
        "race": fix_race(raw_demographics['Please indicate your race - Other - Text'][k] if check(raw_demographics['Please indicate your race - Other - Text'][k]) \
            else raw_demographics['Please indicate your race - Selected Choice'][k]),
        "hispanic_latino": raw_demographics['Are you Hispanic or Latino?'][k],
        "grew_up_area": raw_demographics['How would you describe the area where you grew up?'][k],
        "current_live_area": raw_demographics['How would you describe the area in which you currently live?'][k],
        "current_live_region": raw_demographics['In which region of the United States do you currently live?'][k],
        "annual_householdincome": raw_demographics['Please indicate your annual household income'][k],
        "education": raw_demographics['Education'][k],
        "employment_status": raw_demographics['Which employment status best describes you?'][k],
        "living_situation": fix_live(raw_demographics['How would you describe your current living situation (check all that apply) - Other - Text'][k] if check(raw_demographics['How would you describe your current living situation (check all that apply) - Other - Text'][k]) \
            else raw_demographics['How would you describe your current living situation (check all that apply) - Selected Choice'][k]),
        "political_identification": raw_demographics['Please indicate your political identification'][k],
        "gender": raw_demographics['Please indicate your gender'][k],
        "respondent_id": val,
    }

choice_types = defaultdict(set)
for k, vs in demographics.items():
    for kk, v in vs.items():
        choice_types[kk].add(v)

choice_types = {k: list(v) for k, v in choice_types.items()}

raw_train = train.to_dict()

# answer for the annotator:
annotator_ans = defaultdict(list)
for rid, ann in zip(raw_train["respondent_id"].values(), raw_train["annotation"].values()):
    annotator_ans[rid].append(ann)

annotation_labels = set()
stats = defaultdict(int)
train_paths = []
annotator_data = defaultdict(list)
id = 0
train_fn_path = f"{DATASET_NAME}-processed/annotation_split_train"
if not os.path.exists(train_fn_path):
    os.makedirs(train_fn_path)

test_fn_path = f"{DATASET_NAME}-processed/annotation_split_test"
if not os.path.exists(test_fn_path):
    os.makedirs(test_fn_path)

train_fn = open(f"huggingface-data/{DATASET_NAME}/ann_train.jsonl", "a+")
for _, (text, annotation, respondent_id, uid) in enumerate(zip(raw_train["unit_text"].values(), \
    raw_train["annotation"].values(), raw_train["respondent_id"].values(),\
        raw_train["unit_id"].values())):
    anns = annotator_ans[respondent_id]
    anns_except_current_one = deepcopy(anns)
    anns_except_current_one.remove(annotation)
    ann = {
        "sentence": text,
        "sentiment": annotation,
        "respondent_id": respondent_id,
        "id": id,
        "uid": uid,
        "annotations": anns,
        "anns_except_current_one": anns_except_current_one
    }

    train_fn.write(json.dumps(ann) + "\n")

    with open(f"{train_fn_path}/train_{ann['id']}.json", 'w') as f:
        json.dump(ann, f, indent=4)
    train_paths.append({
        "id": ann["id"],
        "path": f"example-data/{train_fn_path}/train_{ann['id']}.json"
    })
    annotator_data[respondent_id].append({
        "sentence": text,
        "sentiment": annotation,
        "respondent_id": respondent_id,
        "id": id,
        "uid": uid
    })
    stats[respondent_id] += 1
    annotation_labels.add(annotation)
    id += 1

train_fn.close()

raw_test = test.to_dict()

test_fn = open(f"huggingface-data/{DATASET_NAME}/ann_test.jsonl", "a+")

test_paths = []
for _, (text, annotation, respondent_id, uid) in enumerate(zip(raw_test["unit_text"].values(), \
    raw_test["annotation"].values(), raw_test["respondent_id"].values(),\
         raw_test["unit_id"].values())):
    anns = annotator_ans[respondent_id]
    ann = {
        "sentence": text,
        "sentiment": annotation,
        "respondent_id": respondent_id,
        "id": id,
        "uid": uid,
        "annotations": anns,    # for experiment-purpose
        "anns_except_current_one": anns # for experiment-purpose
    }
    
    test_fn.write(json.dumps(ann) + "\n")

    with open(f"{test_fn_path}/test_{ann['id']}.json", 'w') as f:
        json.dump(ann, f, indent=4)
    test_paths.append({
        "id": ann["id"],
        "path": f"example-data/{test_fn_path}/test_{ann['id']}.json"
    })
    annotator_data[respondent_id].append({
        "sentence": text,
        "sentiment": annotation,
        "respondent_id": respondent_id,
        "id": id,
        "uid": uid
    })
    stats[respondent_id] += 1
    annotation_labels.add(annotation)
    id += 1

test_fn.close()

with open(os.path.join(f"{DATASET_NAME}-processed", "annotation_split_train.json"), 'w') as f:
    json.dump(train_paths, f, indent=4)

with open(os.path.join(f"{DATASET_NAME}-processed", "annotation_split_test.json"), 'w') as f:
    json.dump(test_paths, f, indent=4)

print(f"For annotation split: Train: {len(train_paths)}; Test: {len(test_paths)}")

create_annotator_split(DATASET_NAME, annotator_data, TASK)

with open(f"{DATASET_NAME}-processed/demographics.json", 'w') as f:
    json.dump(demographics, f, indent=4)

with open(f"{DATASET_NAME}-processed/demographics_choices.json", 'w') as f:
    json.dump(choice_types, f, indent=4)

with open(f"{DATASET_NAME}-processed/stats.json", 'w') as f:
    json.dump(stats, f, indent=4)

with open(f"{DATASET_NAME}-processed/annotation_labels.json", 'w') as f:
    json.dump(list(annotation_labels), f, indent=4)

print(f"Number of annotators: {len(stats)}")
print(f"Task: {TASK}")
print(f"Number of labels: {len(annotation_labels)}")
