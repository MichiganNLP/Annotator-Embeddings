from __future__ import annotations

import argparse
import json
import math
import os
import warnings
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, NoReturn, Tuple

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


class Seq2SeqDataset(Dataset):

    @overrides
    def __getitem__(self, i: int) -> Mapping[str, Any]:
        instance = self.instances[i]

        # NOTE: hard-coded, modify later!
        if isinstance(instance[self.task], list):
            answer = " ".join(instance[self.task])
        elif isinstance(instance[self.task], str):
            answer = instance[self.task]
        else:
            raise ValueError(f"type {type(instance[self.task])} not supported.")

        output = {
            "id": i,
            "data_id": instance["id"],  # the id in the original file
            "question": self.task.name + " " + instance["sentence"].strip(),  # we give one flag token
            "answer": answer,
        }

        return output

    def collate(self, instances: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        for k in keys:
            stack = batch[k]
            first_val = next(iter(stack), None)
            if isinstance(first_val, str) and k not in {"id", "data_id"}:
                tokenizer = self.encoder_tokenizer
                if tokenizer:
                    if self.use_t5_format and k == "question":
                        # It's more efficient to use the private attribute (`_additional_special_tokens`) than the
                        # public one.
                        to_tokenize = [f"{q} {tokenizer._additional_special_tokens[0]}" for q in stack]
                    elif self.use_t5_format and k == "answer":
                        to_tokenize = [f"<extra_id_0> {a} <extra_id_1>" for a in stack]
                    else:
                        to_tokenize = stack

                    tokenization = tokenizer(to_tokenize, max_length=self.max_len, truncation=True, padding=True,
                                                return_tensors="pt")
                    batch[f"{k}_ids"] = tokenization["input_ids"]
                    batch[f"{k}_mask"] = tokenization["attention_mask"]

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
