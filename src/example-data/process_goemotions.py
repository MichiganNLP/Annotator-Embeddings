import os
import pandas as pd
import numpy as np
import json
import tqdm
import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy

from utils import create_annotation_split, create_annotator_split

DATASET_NAME = "goemotions"
TASK = "emotion"

emotion_list = [
    'admiration', 'amusement', 'anger', 'annoyance', 
    'approval', 'caring', 'confusion', 
    'curiosity', 'desire', 'disappointment', 
    'disapproval', 'disgust', 'embarrassment', 
    'excitement', 'fear', 'gratitude', 
    'grief', 'joy', 'love', 
    'nervousness', 'optimism', 'pride', 
    'realization', 'relief', 'remorse', 
    'sadness', 'surprise', 'neutral'
]

inv_mapping = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", 
                 "optimism", "caring", "pride", "admiration", "gratitude",
                 "relief", "approval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment",
                 "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    "neutral": ["neutral"]
}

mapping = {ele: k for k, v in inv_mapping.items() for ele in v}

tt_num = 0
except_num = 0
dumped_items = []
annotator_history = defaultdict(list)
annotator_data = defaultdict(list)
stats = defaultdict(int)
itm_idx = 0
for i in [1, 2, 3]:
    with open(f'raw-goemotions/data/full_dataset/goemotions_{i}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for j, row in tqdm.tqdm(iter(enumerate(reader))):
            if j == 0:
                # header of the file
                headers = [col for col in row]
            else:
                tt_num += 1
                ann_dict = {headers[idx]: cell for idx, cell in enumerate(row)}
                ann_dict["sentence"] = ann_dict.pop("text")
                ann_dict["respondent_id"] = ann_dict["rater_id"]
                ann_dict["uid"] = ann_dict["id"]
                ann_dict["id"] = itm_idx
                emo_stat = defaultdict(int)
                for emotion in emotion_list:
                    mapped_emo = mapping[emotion]
                    emo_stat[mapped_emo] += int(ann_dict[emotion])
                non_zero_values = {key: value for key, value in emo_stat.items() if value != 0}
                if len(non_zero_values) != 1:
                    dumped_items.append(ann_dict)
                else:
                    ann_dict["emotion"] = list(non_zero_values.keys())[0]
                    annotator_data[ann_dict["rater_id"]].append(ann_dict)
                    stats[ann_dict["rater_id"]] += 1
                itm_idx += 1

with open(os.path.join("goemotions-processed", "annotation_labels.json"), 'w') as f:
    json.dump(list(inv_mapping.keys()), f, indent=4)      

with open(os.path.join("goemotions-processed", "stats.json"), 'w') as f:
    json.dump(stats, f, indent=4)  

# split data into train, dev and test:
# annotation split
create_annotation_split(DATASET_NAME, annotator_data, TASK)
create_annotator_split(DATASET_NAME, annotator_data, TASK)
    
