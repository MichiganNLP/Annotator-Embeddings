# Annotator-Embeddings

Cleaned repo for our paper 
**You Are What You Annotate: Towards Better Models through Annotator Representations** at ***Findings of EMNLP 2023***.

---

### Citation
```
@misc{deng2023annotate,
      title={You Are What You Annotate: Towards Better Models through Annotator Representations}, 
      author={Naihao Deng and Xinliang Frederick Zhang and Siyang Liu and Winston Wu and Lu Wang and Rada Mihalcea},
      year={2023},
      eprint={2305.14663},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---
### What our paper is about

- Rather than aggregating labels, we propose a
setting of training models to directly learn from
data that contains inherent disagreements.

- We propose [**TID-8**](https://huggingface.co/datasets/dnaihao/TID-8), **T**he **I**herent **D**isagreement -
**8** dataset, a benchmark that consists of eight existing language understanding datasets that have
inherent annotator disagreements.

- We propose weighted annotator and annotation
embeddings, which are model-agnostic and improve model performances on six out of the eight
datasets in TID-8.

- We conduct a detailed analysis on the performance variations of our methods and how our methods can be potentially grounded to realworld demographic features.

---
### Structure of this repo


```bash
├── ablation_studies: scripts for ablations
│   ├── annotation_tendencies
│   ├── annotator_accs
│   ├── disagreement_examples
│   ├── heatmaps
│   ├── performance_ablation
│   ├── person_annotation_bars
│   ├── spider_plots
│   └── tsne_plots
├── experiment-results: raw experimental results and the processing script
└── src
    ├── example-data: data for each dataset
    └── src: modeling scripts
        ├── baseline_models
        ├── dataset
        ├── metrics
        ├── tokenization
        ├── training_paradigm
        ├── transformer_models
        └── utils
```

You may create the python environment by using the `environment.yml` file.

---
### Other links to resources for our paper

- [TID-8](https://huggingface.co/datasets/dnaihao/TID-8) dataset on huggingface, here are [more details](src/example-data/README.md) of the processing scripts for each dataset.
- [Project Website](https://lit.eecs.umich.edu/AnnDisagree/)
