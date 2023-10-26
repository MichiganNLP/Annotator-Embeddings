import os
import pandas as pd
import numpy as np
import json
import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy

from utils import create_annotator_split

DATASET_NAME = "friends_qia"
TASK = "indirect_ans"

if not os.path.exists("friends_qia-processed"):
    os.makedirs("friends_qia-processed")

annotator_data = defaultdict(list)
stats = defaultdict(int)
annotator_annotations = defaultdict(list)
train, dev, test = [], [], []
tidx = 0
for split in ["train", "dev", "test"]:
    df = pd.read_csv(f"raw-friends_qia/Data/Friends_data/Final_QA_datasets/qa_data_{split}.csv",
                     sep='\t')

    # two splits of one of same set of annotators in the training and testing datasets
    # another of different annotators in the training and testing datasets
    data_dict = df.to_dict(orient="list")
    list_of_dicts = [{key: value[i] for key, value in data_dict.items()} for i in range(len(df))]

    for i, itm in tqdm.tqdm(iter(enumerate(list_of_dicts))):
        example_dict = {}
        for k, v in itm.items():
            if k == "Q_modified":
                example_dict["sentence"] = v
            elif k == "A_modified":
                example_dict["sentence"] += "</s>" + v
            elif k == "Unnamed: 0":
                continue
            example_dict[k] = str(v)

        example_dict["id"] = tidx
        example_dict["uid"] = i
        if example_dict.get("Annotation_1"):
            example_dict["indirect_ans"] = str(example_dict["Annotation_1"])
            example_dict["respondent_id"] = "Annotation_1"
            annotator_data["Annotation_1"].append(deepcopy(example_dict))
            stats["Annotation_1"] += 1
            if split == "train":
                annotator_annotations["Annotation_1"].append(str(example_dict["Annotation_1"]))

        if split == "train":
            train.append(deepcopy(example_dict))
        elif split == "dev":
            dev.append(deepcopy(example_dict))
        elif split == "test":
            test.append(deepcopy(example_dict))
        tidx += 1
        example_dict["id"] = tidx

        if example_dict.get("Annotation_2"):
            example_dict["indirect_ans"] = str(example_dict["Annotation_2"])
            example_dict["respondent_id"] = "Annotation_2"
            annotator_data["Annotation_2"].append(deepcopy(example_dict))
            stats["Annotation_2"] += 1
            if split == "train":
                annotator_annotations["Annotation_2"].append(str(example_dict["Annotation_2"]))

        if split == "train":
            train.append(deepcopy(example_dict))
        elif split == "dev":
            dev.append(deepcopy(example_dict))
        elif split == "test":
            test.append(deepcopy(example_dict))
        tidx += 1
        example_dict["id"] = tidx

        if example_dict.get("Annotation_3"):
            example_dict["indirect_ans"] = str(example_dict["Annotation_3"])
            example_dict["respondent_id"] = "Annotation_3"
            annotator_data["Annotation_3"].append(deepcopy(example_dict))
            stats["Annotation_3"] += 1
            if split == "train":
                annotator_annotations["Annotation_3"].append(str(example_dict["Annotation_3"]))
        
        if split == "train":
            train.append(deepcopy(example_dict))
        elif split == "dev":
            dev.append(deepcopy(example_dict))
        elif split == "test":
            test.append(deepcopy(example_dict))
        tidx += 1

train_fn_path = f"{DATASET_NAME}-processed/annotation_split_train"
if not os.path.exists(train_fn_path):
    os.makedirs(train_fn_path)

dev_fn_path = f"{DATASET_NAME}-processed/annotation_split_dev"
if not os.path.exists(dev_fn_path):
    os.makedirs(dev_fn_path)

test_fn_path = f"{DATASET_NAME}-processed/annotation_split_test"
if not os.path.exists(test_fn_path):
    os.makedirs(test_fn_path)

train_paths, dev_paths, test_paths = [], [], []
train_fn = open(f"huggingface-data/{DATASET_NAME}-ann/train.jsonl", "a+")
for itm in train:
    respondent_id = itm["respondent_id"]
    label_annotations_except_current_one = deepcopy(annotator_annotations[respondent_id])
    label_annotations_except_current_one.remove(itm["indirect_ans"])
    itm["anns_except_current_one"] = label_annotations_except_current_one
    train_fn.write(json.dumps(itm) + "\n")
    with open(f"{train_fn_path}/train_{itm['id']}.json", 'w') as f:
        json.dump(itm, f, indent=4)
    train_paths.append({
        "id": itm["id"],
        "path": f"example-data/{train_fn_path}/train_{itm['id']}.json"
    })
train_fn.close()

dev_fn = open(f"huggingface-data/{DATASET_NAME}-ann/dev.jsonl", "a+")
# for dev and test, we do not
for itm in dev:
    respondent_id = itm["respondent_id"]
    label_annotations_except_current_one = deepcopy(annotator_annotations[respondent_id])
    itm["anns_except_current_one"] = label_annotations_except_current_one
    dev_fn.write(json.dumps(itm) + "\n")
    with open(f"{dev_fn_path}/dev_{itm['id']}.json", 'w') as f:
        json.dump(itm, f, indent=4)
    dev_paths.append({
        "id": itm["id"],
        "path": f"example-data/{dev_fn_path}/dev_{itm['id']}.json"
    })
dev_fn.close()

test_fn = open(f"huggingface-data/{DATASET_NAME}-ann/test.jsonl", "a+")
for itm in test:
    respondent_id = itm["respondent_id"]
    label_annotations_except_current_one = deepcopy(annotator_annotations[respondent_id])
    itm["anns_except_current_one"] = label_annotations_except_current_one
    test_fn.write(json.dumps(itm) + "\n")
    with open(f"{test_fn_path}/test_{itm['id']}.json", 'w') as f:
        json.dump(itm, f, indent=4)
    test_paths.append({
        "id": itm["id"],
        "path": f"example-data/{test_fn_path}/test_{itm['id']}.json"
    })
test_fn.close()
print(f"For annotation split: Train: {len(train_paths)}, Dev: {len(dev_paths)}, Test: {len(test_paths)}")

with open(os.path.join("friends_qia-processed", "annotation_split_train.json"), 'w') as f:
    json.dump(train_paths, f, indent=4)

with open(os.path.join("friends_qia-processed", "annotation_split_dev.json"), 'w') as f:
    json.dump(dev_paths, f, indent=4)

with open(os.path.join("friends_qia-processed", "annotation_split_test.json"), 'w') as f:
    json.dump(test_paths, f, indent=4)

annotation_labels = []
for k, ann_l in annotator_annotations.items():
    annotation_labels.extend(ann_l)
annotation_labels = list(set(annotation_labels))

with open(os.path.join("friends_qia-processed", "stats.json"), 'w') as f:
    json.dump(stats, f, indent=4)

with open(os.path.join("friends_qia-processed", "annotation_labels.json"), 'w') as f:
    json.dump(annotation_labels, f, indent=4)


# create annotator split

create_annotator_split(DATASET_NAME, annotator_data, TASK)
