import os
import pandas as pd
import numpy as np
import json
import tqdm
from collections import defaultdict
from copy import deepcopy
from utils import create_annotation_split, create_annotator_split

DATASET_NAME = "md-agreement"
TASK = "offensive"

annotator_data = defaultdict(list)
stats = defaultdict(int)

train_fn_path = f"{DATASET_NAME}-processed/annotation_split_train"
dev_fn_path = f"{DATASET_NAME}-processed/annotation_split_dev"
test_fn_path = f"{DATASET_NAME}-processed/annotation_split_test"
if not os.path.exists(train_fn_path):
    os.makedirs(train_fn_path)
if not os.path.exists(dev_fn_path):
    os.makedirs(dev_fn_path)
if not os.path.exists(test_fn_path):
    os.makedirs(test_fn_path)

skipped_num = 0
idx = 0
all_annotator_data = defaultdict(list)

for split in ["train", "dev", "test"]:
    split_fn = open(f"huggingface-data/{DATASET_NAME}/ann_{split}.jsonl", "a+")
    with open(f"raw-{DATASET_NAME}/MD-Agreement_{split}.json", 'r') as f:
        data = json.load(f)

    if split == "train":
        # we only collect data in the training phase once
        
        for key, value in tqdm.tqdm(iter(data.items())):
            ann_list=value['annotations'].split(',')
            annotator_list=value['annotators'].split(',')
            for annotation, annotator in zip(ann_list, annotator_list):
                annotator_data[annotator].append('offensive_speech' if annotation == '1' else 'not_offensive_speech')

    pdata = []
    for j, (key, value) in tqdm.tqdm(iter(enumerate(data.items()))):
        
        ann_list=value['annotations'].split(',')
        annotator_list=value['annotators'].split(',')
        assert value["number of annotations"] == 5
        for annotation, annotator in zip(ann_list, annotator_list):

            item_dict = {}
            item_dict["sentence"]=value['text']
            item_dict["task"]=value['annotation task']
                    
            item_dict["respondent_id"] = annotator
            stats[annotator] += 1
            item_dict["uid"] = j
            item_dict["original_id"] = key
            item_dict["id"] = idx
            item_dict[TASK] = 'offensive_speech' if annotation == '1' else 'not_offensive_speech'
            item_dict["domain"]=value["other_info"]["domain"]

            all_annotator_data[annotator].append(item_dict)

            if annotator not in annotator_data:
                skipped_num += 1
                continue

            if split == "train":
                label_annotations_except_current_one = deepcopy(annotator_data[annotator])
                label_annotations_except_current_one.remove('offensive_speech' if  annotation == '1' else 'not_offensive_speech')
            else:
                label_annotations_except_current_one = annotator_data[annotator]
            item_dict["anns_except_current_one"]=label_annotations_except_current_one
            pdata.append(deepcopy(item_dict))
            idx += 1  

    pdata_path = []
    for itm in pdata:
        pdata_path.append({
            "id": itm["id"],
            "path": f"example-data/{DATASET_NAME}-processed/annotation_split_{split}/{split}_{itm['id']}.json",
        })
        
        split_fn.write(json.dumps(itm) + "\n")

        with open(f"{DATASET_NAME}-processed/annotation_split_{split}/{split}_{itm['id']}.json", 'w') as f:
            json.dump(itm, f, indent=4)
    with open(f"{DATASET_NAME}-processed/annotation_split_{split}.json", 'w') as f:
        json.dump(pdata_path, f, indent=4)
    print(f"For annotation split: {split}: {len(pdata_path)}")
    split_fn.close()

print(f"For annotation split: skipped {skipped_num}")

create_annotator_split(DATASET_NAME, all_annotator_data, TASK)

with open(f"md-agreement-processed/annotation_labels.json", 'w') as f:
    json.dump(["offensive_speech", "not_offensive_speech"], f, indent=4)

with open(f"md-agreement-processed/stats.json", 'w') as f:
    json.dump(stats, f, indent=4)

