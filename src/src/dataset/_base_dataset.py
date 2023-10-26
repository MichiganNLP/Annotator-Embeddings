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

class BaseDataset(Dataset):
    def __init__(self, encoder_tokenizer: PreTrainedTokenizerBase | None = None,
                 decoder_tokenizer: Union[PreTrainedTokenizerBase, SequenceTokenizer] | None = None,
                 max_len: int = 512, is_train: bool | None = None, pool_size: int | None = None,
                 ids: list | None = None, data_path: str | None = None, data: dict | None = None,
                 annotated_ids: set | None = None, task: Task | None = None, 
                 annotator_id_path: str | None = None, annotation_label_path: str | None = None,
                 wandb_name: str | None = None,
                 use_naiive_concat: bool | None = None) -> None:
        super().__init__()
        if data_path:
            with open(data_path) as file:
                self.instances = json.load(file)
        else:
            assert data
            self.instances = data
        
        if pool_size:
            raise RuntimeError("Now the selection process is moved to train_and_test.")
        if ids:
            self.instances = [instance for instance in self.instances if instance["id"] in ids]
        if annotated_ids:
            assert all(id not in annotated_ids for id in ids)
        
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_len = max_len  # FIXME: why defining it?
        self.is_train = is_train
        self.task = task
        self.use_naiive_concat = use_naiive_concat
        
        with open(annotator_id_path, 'r') as f:
            annotator_ids = json.load(f)
        self.annotator_tokenizer = {annotator_id: i for i, annotator_id in enumerate(annotator_ids.keys())}
        with open(annotation_label_path, 'r') as f:
            annotation_labels = json.load(f)
        self.annotation_tokenizer = {"pad": 0}
        for i, label in enumerate(annotation_labels):
            self.annotation_tokenizer[label] = i + 1
        if not os.path.exists("__temp__"):
            os.makedirs("__temp__")
        with open(os.path.join("__temp__", wandb_name), 'w') as f:
            # store the mapping of label to ids
            json.dump(self.annotation_tokenizer, f, indent=4)

    def __len__(self) -> int:
        return len(self.instances)


