from __future__ import annotations

import argparse
import json
import math
import os
import warnings
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, NoReturn, Tuple, Union

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.datasets
from cached_path import cached_path
from overrides import overrides
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from src.tokenization import SequenceTokenizer
from src.utils.utils import Task
from src.dataset import BaseDataset
from src.utils.task_utils import PREDICTION_MASK


class EncoderDataset(BaseDataset):

    @overrides
    def __getitem__(self, i: int) -> Mapping[str, Any]:
        instance_path = self.instances[i]
        with open(instance_path["path"], 'r') as f:
            instance = json.load(f)
        # NOTE: hard-coded, modify later!
        if isinstance(instance[self.task.name], list) or isinstance(instance[self.task.name], str):
            answer = instance[self.task.name]
        elif isinstance(instance[self.task.name], int):
            answer = str(instance[self.task.name])
        else:
            raise ValueError(f"type {type(instance[self.task.name])} not supported.")
        
        output = {
            "id": i,
            "data_id": instance["id"],  # the id in the original file
            "question": instance["sentence"].strip(),
            "answer": answer,
            "task_id": self.task.id,
            "annotator_id": instance["respondent_id"],
            "annotations":  instance["anns_except_current_one"] 
        }

        return output

    def collate(self, instances: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        question = batch["question"]
        if self.use_naiive_concat:
            question = [f"{a}</s>{q}" for a, q in zip(batch["annotator_id"], question)]
        tokenization = self.encoder_tokenizer(question, max_length=self.max_len, truncation=True, padding=True,
                                            return_tensors="pt", 
                                            # return_offsets_mapping=True,  # No offset mapping
                                            add_special_tokens=True)
        batch["question_ids"] = tokenization["input_ids"]
        batch["question_mask"] = tokenization["attention_mask"]

        answer = batch["answer"]
        tokenization = self.decoder_tokenizer(answer, max_length=self.max_len, padding=True, return_tensors="pt",
                                                # offset_mappings=tokenization["offset_mapping"]    # No offset
                                                )
        batch["answer_ids"] = tokenization["input_ids"]
        assert self.task.name in PREDICTION_MASK
        if PREDICTION_MASK[self.task.name]:
            batch["prediction_mask"] = tokenization["prediction_mask"]
        batch["origin_annotator_id"] = batch["annotator_id"]
        batch["annotator_id"] = torch.LongTensor([self.annotator_tokenizer[i] for i in batch["annotator_id"]])
        annotations = []
        for l in batch["annotations"]:
            per_annotation = []
            for i in l:
                per_annotation.append(self.annotation_tokenizer[i])
            annotations.append(per_annotation)
        max_len = max([len(ann) for ann in annotations])
        for i, ann in enumerate(annotations):
            for _ in range(max_len - len(ann)):
                ann.append(self.annotation_tokenizer['pad'])    # set pad token as 0
                annotations[i] = ann

        batch["annotations"] = torch.LongTensor(annotations)
        return batch

    def __len__(self) -> int:
        return len(self.instances)

def precision_to_dtype(precision: str | int) -> torch.dtype:
    if precision == 32:
        return torch.float
    elif precision == 64:
        return torch.float64
    elif precision in {16, "mixed"}:
        return torch.float16
    else:
        raise ValueError(f"Unsupported precision value: {precision}")
