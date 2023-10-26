""" Individual Majority vote baseline models """
import json
import os
from collections import Counter
 
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


SPLIT = "annotation"

class OverallMajorityVote:

    def __init__(self, task, train_path, test_path, output_path):
        self.task = task
        with open(train_path, 'r') as f:
            self.train_data = json.load(f)
        with open(test_path, 'r') as f:
            self.test_data = json.load(f)
        self.output_path = output_path
        
    def train(self):
        self.train_preference = []
        for itm in self.train_data:
            with open(itm["path"], 'r') as f:
                instance = json.load(f)
                label = instance[self.task]
                self.train_preference.append(label)
        
    def test(self, seed, idx):
        predictions = []
        correct_num = 0
        for itm in self.test_data:
            with open(itm["path"], 'r') as f:
                instance = json.load(f)
                respondent_id = instance["respondent_id"]
                gold_label = instance[self.task]
                pred_label = most_frequent(self.train_preference)
                predictions.append({
                    "pred": pred_label,
                    "gold": gold_label,
                    "question": instance["sentence"],
                    "id": itm["id"],
                    "respondent_id": respondent_id
                })
                if pred_label == gold_label:
                    correct_num += 1
        print(f"Acc: {correct_num / len(predictions)}")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        with open(f"{self.output_path}/use_annotator_embed-False-use_annotation_embed-False-pad-True-method-add-test_mode-normal-seed-{seed}-{idx}-{SPLIT}-0.jsonl", 'w') as f:
            f.write("\n".join([json.dumps(itm) for itm in predictions]))


if __name__ == "__main__":
    tasks = {
        "friends_qia": "indirect_ans",
        "pejorative": "pejorative", 
        "hs_brexit": "hs_brexit", 
        "md-agreement": "offensive", 
        "goemotions": "emotion", 
        "humor": "humor", 
        "commitmentbank": "certainty", 
        "sentiment": "sentiment",
        "toxic-ratings": "toxic_score"
    }
    for dataset in ["toxic-ratings"]:
        print(dataset)
        task = tasks[dataset]
        train_path = f"example-data/{dataset}-processed/{SPLIT}_split_train.json"
        test_path = f"example-data/{dataset}-processed/{SPLIT}_split_test.json"
        output_path = f"../experiment-results/{dataset}/overall_majority_vote"
        imv = OverallMajorityVote(task, train_path, test_path, output_path)
        imv.train()
        # for idx, seed in enumerate([32, 42, 52, 62, 72, 82, 92, 102, 112, 122]):
        for idx, seed in enumerate([82, 92, 102, 112, 122]):
            imv.test(seed, idx)

