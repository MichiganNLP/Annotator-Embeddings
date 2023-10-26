import os
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from src.utils.utils import Task


def example_selection(example_selection_criteria: str, task_predictions: Dict[Task, List[dict]], \
    update_gap: int, pool_size: int) -> Tuple[list, List[dict]]:
    """ Example selection, select the one to annotate """

    probs = defaultdict(float)
    generated = defaultdict(dict)
    for task, predictions in task_predictions.items():
        for prediction in predictions:
            for i, (k, v) in enumerate(prediction.items()):
                if "prob_" in k:
                    probs[(int(k.split("prob_")[1]))] += v
                    assert (int(k.split("prob_")[1])) == prediction["id"][i]
                    generated[(int(k.split("prob_")[1]))][task] = prediction["generated"][i]
    
    assert pool_size >= len(probs)

    if example_selection_criteria == "most-confident":
        probs = sorted(probs.items(), key=lambda x: (x[1], x[0]))[-update_gap:]
    elif example_selection_criteria == "least-confident":
        probs = sorted(probs.items(), key=lambda x: (x[1], x[0]))[:update_gap]
    
    selected_ids = [idx for idx, _ in probs]
    predicted_itms = [generated[idx] for idx in selected_ids]
    return selected_ids, predicted_itms

def forward_prediction(selected_ids: list, predictions: List[dict], data):
    forward_data = []
    predictions = {pred_id: pred for pred_id, pred in zip(selected_ids, predictions)}
    for d in data:
        if d["id"] in selected_ids:
            pred_itm = predictions[d["id"]]
            for task, pred in pred_itm.items():
                d[f"{task.name}_prediction"] = pred
            forward_data.append(d)
    if not os.path.exists("__temp_forward__"):
        os.makedirs("__temp_forward__")
    with open(f"__temp_forward__/forward_predictions.json", 'w') as f:
        json.dump(forward_data, f, indent=4)
