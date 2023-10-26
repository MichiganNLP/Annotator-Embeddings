import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union, List

import torch
import sys
from transformers.hf_argparser import DataClassType

# Can't use `from __future__ import annotations` here. See https://github.com/huggingface/transformers/pull/15795
# From the next version of transformers (after v4.17.0) it should be possible.


MODEL_CHOICES = [
    "t5",
    "bert",
    "berttokenclassifier",
    "bertmultichoice",
    "random",
    "roberta-multichoice",
    "deberta-multichoice"
]

TRAINING_PARADIGM = [
    "learn_from_scratch"
]

EXAMPLE_SELECTION_CRITERIA = [
    "most-confident",
    "least-confident"
]

TASK_TYPES = [
    "sequenceclassification",
    "multiplechoice"
]

@dataclass
class TrainAndTestArguments:
    train_data_path: str = "example-data/conll2003-processed/train.json"
    dev_data_path: str = "example_data/conll2003-processed/dev.json"
    test_data_path: str = "example_data/conll2003-processed/test.json"

    # Only Linux OS can support os.sched_getaffinity
    num_workers: int = len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1) \
                    if sys.platform in {"linux1", "linux2"} else 10 // max(torch.cuda.device_count(), 1)
    output_ckpt_dir: Optional[str] = None
    model_name_or_path: str = "t5-base"
    max_seq_length: int = field(
        default=512,
        metadata={"help": "maximum of the text sequence. Truncate if exceeded."}
    )
    beam_size: Optional[int] = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    linear_scheduler: bool = True
    warmup_steps: int = 0
    val_check_interval: float = field(
        default=1.0,
        metadata={"help": '''val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches.
                Default: ``1.0``.'''}
    )
    check_val_every_n_epoch: int = field(
        default=10,
        metadata={"help": '''Check val every n train epochs.
                Default: ``10``. As by default we run for 3 epochs and want to avoid validation in this case.'''}
    )
    train_batch_size: Optional[int] = 32
    eval_batch_size: Optional[int] = 32
    num_train_epochs: int = 100
    gradient_accumulation_steps: Optional[int] = 1
    n_gpu: int = field(default=1, metadata={"help": "number of gpus"})
    early_stop_callback: bool = field(
        default=False,
        metadata={"help": "whether we allow early stop in training."}
    )
    opt_level: Optional[int] = field(
        default=None,
        metadata={"help": "optimization level. you can find out more on optimisation levels here"
                          " https://nvidia.github.io/apex/amp.html#opt-levels-and-properties"}
    )
    fp_16: bool = field(
        default=False,
        metadata={"help": "if you want to enable 16-bit training then install apex and set this to true."}
    )
    max_grad_norm: float = 1.0
    seed: int = 42
    profiler: Optional[str] = None
    use_tpu: bool = False
    test_after_train: bool = True
    wandb_project: str = "annotation-project"
    wandb_name: str = field(
        default=None,
        metadata={"help": "name of this run."}
    )
    wandb_entity: str = field(
        default="annotation-builders",
        metadata={"help": "your account to for wandb."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "if set true, we will not have wandb record online"}
    )
    update_gap: int = field(
        default=10,
        metadata={"help": "after `update_gap` number of examples get annotated, we update the model parameters."}
    )
    pool_size: int = field(
        default=100,
        metadata={"help": "pool size where the examples are selected from"}
    )
    pad_token_id: int = 0
    tasks: List[str] = field(
        default_factory=lambda: ["pos-tags-s", "entities-recognition"],
        metadata={"help": "list of specified tasks for the training"}
    )
    task_types: List[str] = field(
        default_factory=lambda: ["sequenceclassification", "sequenceclassification"],
        metadata={"help": "list of task types for model initialization"}
    )
    add_output_tokens: List[bool] = field(
        default_factory=lambda: [True, True],
        metadata={"help": "whether add the output tokens as special toks"}
    )
    training_paradigm: str = field(
        default="learn_from_scratch"
    )
    num_sanity_val_steps: int = field(
        default=2,
        metadata={"help": "how many validation steps to run before training"}
    )
    drop_last: bool = field(
        default=False,
        metadata={"help": "whether drop the last batch for training"}
    )

    # arguments for demographic feature or modeling the annotators
    use_demographics: bool = field(
        default=False,
        metadata={"help": "whether use demographic features"}
    )
    demographic_features: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "list of demographic features to use; default to an empty list."}
    )
    broadcast_annotator_embedding: bool = field(
        default=False,
        metadata={"help": "whether to broadcast annotator embedding"}
    )
    broadcast_annotation_embedding: bool = field(
        default=False,
        metadata={"help": "whether to broadcast annotation embedding"}
    )
    use_annotator_embed: bool = field(
        default=False,
        metadata={"help": "whether use the annotator embedding"}
    )
    use_annotation_embed: bool = field(
        default= False,
        metadata={"help": "whether use the annotation embedding for annotators"}
    )
    annotator_id_path: str = field(
        default="",
        metadata={"help": "path to the json file of all annotator ids"}
    )
    annotation_label_path: str = field(
        default="",
        metadata={"help": "path to the all of the possible annotation labels"}
    )
    pred_fn_path: str = field(
        default="",
        metadata={"help": "path to the prediction file"}
    )
    annotator_embed_weight: float = field(
        default=1.0,
        metadata={"help": "weight to time annotator embedding: magnify the embedding effect."}
    )
    annotation_embed_weight: float = field(
        default=1.0,
        metadata={"help": "weight to time annotation embedding: magnify the embedding effect."}
    )
    include_pad_annotation: bool = field(
        default=True,
        metadata={"help": "include the pad embedding, or just calculate the embedding \
                  if it is not padding. Note that here the flag is only for annotation \
                  embedding."}
    )
    use_naiive_concat: bool = field(
        default=False,
        metadata={"help": "naiive concatenation of question and annotator id"}
    )
    embed_wo_weight: bool = field(
        default=False,
        metadata={"help": "ablation study of annotator/annotation embedding without weight"}
    )
    method: str = field(
        default="add",
        metadata={"help": "method of incorporating the embeddings; one is by adding to the [CLS] \
                  the other is by concatenating/inserting it into the sequence."}
    ),
    test_mode: str = field(
        default="normal",
        metadata={"help": "normal: annotation split / annotator split; \
                  normal_wo_info: normal mode but annotator information is omitted in test; \
                  test_w_info: trained with question only, in test time, we enable the annotation/annotator embeddings; \
                  cls_only: trained without question tokens, but just the cls token, or with annotator/annotation embedding; \
                  test_cls_only: trained normally, but test time we only provide the cls token"}
    )
    enable_checkpointing: bool = field(
        default=False,
        metadata={"help": "whether to save checkpoints. By default it's False and save nothing."}
    )

    def __post_init__(self):
        assert self.training_paradigm in TRAINING_PARADIGM, f"Wrong training \
            paradigm specified. Please select from {', '.join(TRAINING_PARADIGM)}"
        
        assert self.example_selection_criteria in EXAMPLE_SELECTION_CRITERIA, f"Unsupported \
            example selection criteria. Please select from {', '.join(EXAMPLE_SELECTION_CRITERIA)}"

        assert all(tt in TASK_TYPES for tt in self.task_types), f"Some types \
            in {', '.join(self.task_types)} not in {', '.join(TASK_TYPES)}"

        assert self.method in ["add", "concat"]

        assert self.test_mode in ["normal", "normal_wo_info", "test_w_info", "cls_only", "test_cls_only"]

@dataclass
class Seq2SeqArguments(TrainAndTestArguments):
    constrained_decoding: list = field(
        default_factory=lambda: [True, True],
        metadata={"help": "whether constrain the decoding process and limit the generated tokens to specified tokens"}
    )
    constrained_token_file_paths: list = field(
        default_factory=lambda: [True, True],
        metadata={"help": "the paths to the file contains the special tokens in json format as a list. If not exist, we constrain the output \
        to all the tokens in the original files."}
    )


@dataclass
class EncoderOnlyModelsArguments(TrainAndTestArguments):
    num_classes: List[int] = field(
        default_factory=lambda: [47, 10],
        metadata={"help": "number of classes for each task"}
    )



def model_type_to_dataclass_types(model_type: str) -> Union[DataClassType, Iterable[DataClassType]]:
    assert model_type in MODEL_CHOICES, f"Model type :{model_type} not supported."
    return {
        "t5": Seq2SeqArguments,
        "bert": EncoderOnlyModelsArguments,
        "berttokenclassifier": EncoderOnlyModelsArguments,
        "bertmultichoice": EncoderOnlyModelsArguments,
        "random": EncoderOnlyModelsArguments,
        "roberta-multichoice": EncoderOnlyModelsArguments,
        "deberta-multichoice": EncoderOnlyModelsArguments
    }[model_type]