import torch
import numpy as np
import json
from sklearn.manifold import TSNE
from collections import defaultdict
import matplotlib.pyplot as plt



def plot_embed(ann_list, emb_path, emb_type, RANDOM_STATE):
    # Embedding file
    # Load embeddings from a PyTorch saved file
    embeddings = torch.load(emb_path)

    embed = torch.cat(embeddings[emb_type], dim=0)

    annotator_embeds = list()
    for _, l in ann_list.items():
        idx = l[0]
        annotator_embeds.append(embed[idx].numpy())

    # Apply t-SNE on embeddings
    tsne = TSNE(n_components=2, perplexity=40, random_state=RANDOM_STATE)
    embeddings_tsne = tsne.fit_transform(np.vstack(annotator_embeds))
    # Plot t-SNE results
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], marker='.', s=40)


def main(RANDOM_STATE, DATASET, SEED, IDX):
    # prediction file
    pred_path = f"{DATASET}/use_annotator_embed-True-use_annotation_embed-False-pad-True-method-add-test_mode-normal-seed-{SEED}-{IDX}-annotation-0.jsonl"
    with open(pred_path, 'r') as f:
        preds = f.readlines()
    preds = [json.loads(d) for d in preds]

    ann_list = defaultdict(list)
    for i, pred in enumerate(preds):
        ann_list[pred["respondent_id"]].append(i)
    plot_embed(ann_list=ann_list, \
               emb_path = f"{DATASET}/use_annotator_embed-True-use_annotation_embed-False-pad-True-method-add-test_mode-normal-seed-{SEED}-{IDX}-annotation-0.pt", \
                emb_type="annotator_embed_before_alpha",
                RANDOM_STATE=RANDOM_STATE)
    plot_embed(ann_list=ann_list, \
               emb_path = f"{DATASET}/use_annotator_embed-False-use_annotation_embed-True-pad-True-method-add-test_mode-normal-seed-{SEED}-{IDX}-annotation-0.pt", \
                emb_type="annotation_embed_before_beta",
                RANDOM_STATE=RANDOM_STATE)
    plt.savefig(f"tsne_plots/{DATASET}-annotator-embed-{IDX}-{RANDOM_STATE}.pdf")
    plt.close()


if __name__ == "__main__":
    for DATASET in ["humor", "goemotions", "commitmentbank"]:
        for RANDOM_STATE in [32, 42, 52, 62, 72, 82, 92]:
            for IDX, SEED in zip([0], [32]):
                main(RANDOM_STATE, DATASET, SEED, IDX)
