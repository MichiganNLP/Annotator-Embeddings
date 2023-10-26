import torch
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity



RANDOM_STATE = 32
DATASET = "hs_brexit"
SEED = 32
IDX = 0

def calculate_embed(ann_list, emb_path, emb_type, cidx, label):
    # Embedding file
    # Load embeddings from a PyTorch saved file
    embeddings = torch.load(emb_path)

    embed = torch.cat(embeddings[emb_type], dim=0)

    annotator_embeds = list()

    names = []
    for name, l in ann_list.items():
        idx = l[0]
        annotator_embeds.append(embed[idx].numpy())
        names.append(name)
    # calculate embedding distance between annotators
    annotator_embeds = np.array(annotator_embeds)
    similarity = cosine_similarity(annotator_embeds)
    print("Order of annotators:" + ", ".join(names))
    print(similarity)

   

def main():
    pred_path = f"../experiment-results/{DATASET}/use_annotator_embed-True-use_annotation_embed-False-pad-True-method-add-test_mode-normal-seed-{SEED}-{IDX}-annotation-0.jsonl"
    with open(pred_path, 'r') as f:
        preds = f.readlines()
    preds = [json.loads(d) for d in preds]

    ann_list = defaultdict(list)
    for i, pred in enumerate(preds):
        ann_list[pred["respondent_id"]].append(i)
    calculate_embed(ann_list=ann_list, \
               emb_path = f"../experiment-results/{DATASET}/use_annotator_embed-True-use_annotation_embed-False-pad-True-method-add-test_mode-normal-seed-{SEED}-{IDX}-annotation-0.pt", \
                emb_type="annotator_embed_before_alpha", cidx=0, label=r"Annotator Embedding (E$_a$)")
    calculate_embed(ann_list=ann_list, \
               emb_path = f"../experiment-results/{DATASET}/use_annotator_embed-False-use_annotation_embed-True-pad-True-method-add-test_mode-normal-seed-{SEED}-{IDX}-annotation-0.pt", \
                emb_type="annotation_embed_before_beta", cidx=1, label=r"Annotation Embedding (E$_n$)")

if __name__ == "__main__":
    main()