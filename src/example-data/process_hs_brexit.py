import os
import pandas as pd
import numpy as np
import json
import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy

from utils import create_annotation_split, create_annotator_split


DATASET_NAME = "hs_brexit"
TASK = "hs_brexit"

annotator_data = defaultdict(list)
all_annotator_data = defaultdict(list)
stats = defaultdict(int)

train_fn_path = f"{DATASET_NAME}-processed/annotation_split_train"
if not os.path.exists(train_fn_path):
    os.makedirs(train_fn_path)

dev_fn_path = f"{DATASET_NAME}-processed/annotation_split_dev"
if not os.path.exists(dev_fn_path):
    os.makedirs(dev_fn_path)

test_fn_path = f"{DATASET_NAME}-processed/annotation_split_test"
if not os.path.exists(test_fn_path):
    os.makedirs(test_fn_path)

tidx = 0
train_fn = open(f"huggingface-data/{DATASET_NAME}/ann_train.jsonl", "a+")
dev_fn = open(f"huggingface-data/{DATASET_NAME}/ann_dev.jsonl", "a+")
test_fn = open(f"huggingface-data/{DATASET_NAME}/ann_test.jsonl", "a+")

for split in ["train", "dev", "test"]:
    with open(f"raw-HS-Brexit_dataset/HS-Brexit_{split}.json", 'r') as f:
        data = json.load(f)

    if split == "train":
        # we only collect data in the training phase once
        for key, value in tqdm.tqdm(iter(data.items())):
            ann_list=value['annotations'].split(',')
            for i in range(6):
                annotator_data[f"annotator_{i+1}"].append('hate_speech' if ann_list[i]=='1' else 'not_hate_speech')

    data_paths = []
    for j, (key, value) in tqdm.tqdm(enumerate(iter(data.items()))):
        ann_list=value['annotations'].split(',')
        aggressive_detection=value['other_info']['other annotations']['aggressive language detection'].split(',')
        offensive_detection=value['other_info']['other annotations']['offensive language detection'].split(',')
        assert value["number of annotations"] == 6
        for i in range(6):
            item_dict = {}
            item_dict["sentence"]=value['text']
            item_dict["task"]=value['annotation task']
            item_dict["respondent_id"] = f"annotator_{i+1}"
            stats[f"annotator_{i+1}"] += 1
            item_dict["id"] = tidx
            item_dict["uid"] = j
            item_dict[TASK]='hate_speech' if  ann_list[i]=='1' else 'not_hate_speech'
            if split == "train":
                label_annotations_except_current_one = deepcopy(annotator_data[f"annotator_{i+1}"])
                label_annotations_except_current_one.remove('hate_speech' if  ann_list[i]=='1' else 'not_hate_speech')
            else:
                label_annotations_except_current_one = annotator_data[f"annotator_{i+1}"]
            item_dict["anns_except_current_one"]=label_annotations_except_current_one
            item_dict["other annotations"]=['aggressive speech' if  aggressive_detection[i]=='1' else 'not aggressive speech', 'offensive speech' if  offensive_detection[i]=='1' else 'not offensive speech']
            
            all_annotator_data[f"annotator_{i+1}"].append(item_dict)
            if split == "train":
                train_fn.write(json.dumps(item_dict) + "\n")
                with open(f"{train_fn_path}/train_{item_dict['id']}.json", 'w') as f:
                    json.dump(item_dict, f, indent=4)
                data_paths.append({
                    "id": item_dict["id"],
                    "path": f"example-data/{train_fn_path}/train_{item_dict['id']}.json"
                })
            elif split == "dev":
                dev_fn.write(json.dumps(item_dict) + "\n")
                with open(f"{dev_fn_path}/dev_{item_dict['id']}.json", 'w') as f:
                    json.dump(item_dict, f, indent=4)
                data_paths.append({
                    "id": item_dict["id"],
                    "path": f"example-data/{dev_fn_path}/dev_{item_dict['id']}.json"
                })
            else:
                assert split == "test"
                test_fn.write(json.dumps(item_dict) + "\n")
                with open(f"{test_fn_path}/test_{item_dict['id']}.json", 'w') as f:
                    json.dump(item_dict, f, indent=4)
                data_paths.append({
                    "id": item_dict["id"],
                    "path": f"example-data/{test_fn_path}/test_{item_dict['id']}.json"
                })

            tidx += 1
            
    with open(f"hs_brexit-processed/annotation_split_{split}.json", 'w') as f:
        json.dump(data_paths, f, indent=4)

train_fn.close()
dev_fn.close()
test_fn.close()

create_annotator_split(DATASET_NAME, all_annotator_data, TASK)

with open(f"hs_brexit-processed/annotation_labels.json", 'w') as f:
    json.dump(["hate_speech", "not_hate_speech"], f, indent=4)
print("Number of annotation labels: 2")

with open(f"hs_brexit-processed/stats.json", 'w') as f:
    json.dump(stats, f, indent=4)
print(f"Number of annotators: {len(stats)}")

print(f"task: {TASK}")
