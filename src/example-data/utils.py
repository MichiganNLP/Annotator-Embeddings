import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from copy import deepcopy


def create_annotation_split(dataset_name, annotator_data, task, th=None, part = None):
    if th:
        if part:
            train_fn_path = f"{dataset_name}-processed/annotation_split_train_{th}_{part}"
            test_fn_path = f"{dataset_name}-processed/annotation_split_test_{th}_{part}"
        else:
            train_fn_path = f"{dataset_name}-processed/annotation_split_train_{th}"
            test_fn_path = f"{dataset_name}-processed/annotation_split_test_{th}"
    else:
        train_fn_path = f"{dataset_name}-processed/annotation_split_train"
        test_fn_path = f"{dataset_name}-processed/annotation_split_test"

    if not os.path.exists(train_fn_path):
        os.makedirs(train_fn_path)
    if not os.path.exists(test_fn_path):
        os.makedirs(test_fn_path)    

    train_paths = []
    test_paths = []
    if not os.path.exists(f"huggingface-data/{dataset_name}-ann"):
        os.makedirs(f"huggingface-data/{dataset_name}-ann")
    train_fn = open(f"huggingface-data/{dataset_name}-ann/train.jsonl", "a+")
    test_fn = open(f"huggingface-data/{dataset_name}-ann/test.jsonl", "a+")
    for annotator, ann_list in annotator_data.items():
        if len(ann_list) > 1:
            train_ann, test_ann = train_test_split(
                ann_list, test_size=0.3, random_state=42)
        else:
            train_ann = ann_list
            test_ann = []

        label_annotations = [ann[task] for ann in train_ann]
        for ann in train_ann:
            label_annotations_except_current_one = deepcopy(label_annotations)
            label_annotations_except_current_one.remove(ann[task])
            ann["anns_except_current_one"] = label_annotations_except_current_one
            train_fn.write(json.dumps(ann) + "\n")
            with open(f"{train_fn_path}/train_{ann['id']}.json", 'w') as f:
                json.dump(ann, f, indent=4)
            train_paths.append({
                "id": ann["id"],
                "path": f"example-data/{train_fn_path}/train_{ann['id']}.json"
            })
        for ann in test_ann:
            ann["anns_except_current_one"] = label_annotations
            test_fn.write(json.dumps(ann) + "\n")
            with open(f"{test_fn_path}/test_{ann['id']}.json", 'w') as f:
                json.dump(ann, f, indent=4)
            test_paths.append({
                "id": ann["id"],
                "path": f"example-data/{test_fn_path}/test_{ann['id']}.json"
            })

    if th:
        if part:
            with open(os.path.join(f"{dataset_name}-processed", f"annotation_split_train_{th}_{part}.json"), 'w') as f:
                json.dump(train_paths, f, indent=4)

            with open(os.path.join(f"{dataset_name}-processed", f"annotation_split_test_{th}_{part}.json"), 'w') as f:
                json.dump(test_paths, f, indent=4)

            print(f"For annotation split {part} {th}: Train: {len(train_paths)}; Test: {len(test_paths)}")

        else:
            with open(os.path.join(f"{dataset_name}-processed", f"annotation_split_train_{th}.json"), 'w') as f:
                json.dump(train_paths, f, indent=4)

            with open(os.path.join(f"{dataset_name}-processed", f"annotation_split_test_{th}.json"), 'w') as f:
                json.dump(test_paths, f, indent=4)

            print(f"For annotation split {th}: Train: {len(train_paths)}; Test: {len(test_paths)}")

    else:
        with open(os.path.join(f"{dataset_name}-processed", f"annotation_split_train.json"), 'w') as f:
            json.dump(train_paths, f, indent=4)

        with open(os.path.join(f"{dataset_name}-processed", f"annotation_split_test.json"), 'w') as f:
            json.dump(test_paths, f, indent=4)

        print(f"For annotation split: Train: {len(train_paths)}; Test: {len(test_paths)}")
    train_fn.close()
    test_fn.close()


def create_annotator_split(dataset_name, annotator_data, task, th = None, part = None):
    if not os.path.exists(f"huggingface-data/{dataset_name}-atr"):
        os.makedirs(f"huggingface-data/{dataset_name}")
    train_fn = open(f"huggingface-data/{dataset_name}-atr/train.jsonl", "a+")
    test_fn = open(f"huggingface-data/{dataset_name}-atr/test.jsonl", "a+")
    if th:
        if part:
            train_fn_path = f"{dataset_name}-processed/annotator_split_train_{th}_{part}"
            test_fn_path = f"{dataset_name}-processed/annotator_split_test_{th}_{part}"
        else:
            train_fn_path = f"{dataset_name}-processed/annotator_split_train_{th}"
            test_fn_path = f"{dataset_name}-processed/annotator_split_test_{th}"
    else:
        train_fn_path = f"{dataset_name}-processed/annotator_split_train"
        test_fn_path = f"{dataset_name}-processed/annotator_split_test"

    if not os.path.exists(train_fn_path):
        os.makedirs(train_fn_path)
    if not os.path.exists(test_fn_path):
        os.makedirs(test_fn_path)

    train_paths = []
    test_paths = []

    train_annotator, test_annotator = train_test_split(
        list(annotator_data.keys()), test_size=0.3, random_state=42)

    for annotator, anns in annotator_data.items():
        if annotator in train_annotator:
            label_annotations = [ann[task] for ann in anns]
            for ann in anns:
                label_annotations_except_current_one = deepcopy(label_annotations)
                label_annotations_except_current_one.remove(ann[task])
                ann["anns_except_current_one"] = label_annotations_except_current_one
                train_fn.write(json.dumps(ann) + "\n")
                with open(f"{train_fn_path}/train_{ann['id']}.json", 'w') as f:
                    json.dump(ann, f, indent=4)
                train_paths.append({
                    "id": ann["id"],
                    "path": f"example-data/{train_fn_path}/train_{ann['id']}.json"
                })
        else:
            assert annotator in test_annotator
            for ann in anns:
                ann["anns_except_current_one"] = []
                test_fn.write(json.dumps(ann) + "\n")
                with open(f"{test_fn_path}/test_{ann['id']}.json", 'w') as f:
                    json.dump(ann, f, indent=4)
                test_paths.append({
                    "id": ann["id"],
                    "path": f"example-data/{test_fn_path}/test_{ann['id']}.json"
                })

    if th:
        if part:
            with open(os.path.join(f"{dataset_name}-processed", f"annotator_split_train_{th}_{part}.json"), 'w') as f:
                json.dump(train_paths, f, indent=4)

            with open(os.path.join(f"{dataset_name}-processed", f"annotator_split_test_{th}_{part}.json"), 'w') as f:
                json.dump(test_paths, f, indent=4)

            print(f"For annotator split {part} {th}: Train: {len(train_paths)}; Test: {len(test_paths)}")  

        else:
            with open(os.path.join(f"{dataset_name}-processed", f"annotator_split_train_{th}.json"), 'w') as f:
                json.dump(train_paths, f, indent=4)

            with open(os.path.join(f"{dataset_name}-processed", f"annotator_split_test_{th}.json"), 'w') as f:
                json.dump(test_paths, f, indent=4)

            print(f"For annotator split {th}: Train: {len(train_paths)}; Test: {len(test_paths)}")  

    else:
        with open(os.path.join(f"{dataset_name}-processed", f"annotator_split_train.json"), 'w') as f:
            json.dump(train_paths, f, indent=4)

        with open(os.path.join(f"{dataset_name}-processed", f"annotator_split_test.json"), 'w') as f:
            json.dump(test_paths, f, indent=4)

        print(f"For annotator split: Train: {len(train_paths)}; Test: {len(test_paths)}")  
    train_fn.close()
    test_fn.close()


def _filter(text_data, task, th, part = None):
    annotator_data = defaultdict(list)
    for uid, ann_list in text_data.items():
        anns = [itm[task] for itm in ann_list]
        counts = {}
        for item in anns:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        most_common_count = sorted_counts[0][1]
        if not part:
            if most_common_count / len(anns) >= th:
                for item_dict in ann_list:
                    annotator_id = item_dict["respondent_id"]
                    annotator_data[annotator_id].append(item_dict)
        else:
            assert part == "smaller"
            if most_common_count / len(anns) < th:
                for item_dict in ann_list:
                    annotator_id = item_dict["respondent_id"]
                    annotator_data[annotator_id].append(item_dict)

    return annotator_data


def create_th_annotation_split(dataset_name, text_data, task, th, part = None):
    annotator_data = _filter(text_data, task, th, part = part)
    print(f"Number of annotators: {len(annotator_data)}")
    create_annotation_split(dataset_name, annotator_data, task, th, part)


def create_th_annotator_split(dataset_name, text_data, task, th, part = None):
    annotator_data = _filter(text_data, task, th, part = part)
    create_annotator_split(dataset_name, annotator_data, task, th, part)
