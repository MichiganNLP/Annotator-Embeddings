from __future__ import annotations

import argparse
import json
import math
import os
import warnings
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, NoReturn, Tuple, Union, Dict

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.datasets
import torch.multiprocessing
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
from src.utils.utils import Task, Tasks
from src.dataset import BaseDataset, model_type_to_dataset, get_decoder_tokenizer

# try to resolve the error of too many files open
torch.multiprocessing.set_sharing_strategy('file_system')

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy('file_system')

def precision_to_dtype(precision: str | int) -> torch.dtype:
    if precision == 32:
        return torch.float
    elif precision == 64:
        return torch.float64
    elif precision in {16, "mixed"}:
        return torch.float16
    else:
        raise ValueError(f"Unsupported precision value: {precision}")


class DataModule(pl.LightningDataModule):  # noqa
    def __init__(self, args: argparse.Namespace, encoder_tokenizer: PreTrainedTokenizerBase | None = None,
                 decoder_tokenizer: Union[PreTrainedTokenizerBase, dict] | None = None, ids: list | None = None,
                 train_data: dict | None = None, dev_data: dict | None = None, test_data: dict | None = None,
                 annotated_ids: set | None = None, tasks: Tasks | None = None, annotator_id_path: str | None = None,
                 annotation_label_path: str | None = None, wandb_name: str | None = None,
                 use_naiive_concat: bool | None = None) -> None:
        super().__init__()
        self.args = args
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizers = {}
        for task in tasks:
            self.decoder_tokenizers[task.name] = get_decoder_tokenizer(args.model_type, decoder_tokenizer, task)
        self.num_workers = args.num_workers
        self.eval_batch_size = getattr(self.args, "eval_batch_size", getattr(self.args, "batch_size", 1))
        self.ids = ids if ids else None
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.annotated_ids = annotated_ids
        self.tasks = tasks
        self.annotator_id_path = annotator_id_path
        self.annotation_label_path = annotation_label_path
        self.wandb_name = wandb_name
        self.use_naiive_concat = use_naiive_concat

    @staticmethod
    def _create_dataset(args: argparse.Namespace,
                        data_path: str | None = None,
                        data: dict | None = None,
                        encoder_tokenizer: PreTrainedTokenizerBase | None = None,
                        decoder_tokenizers: PreTrainedTokenizerBase | Dict[SequenceTokenizer] | None = None,
                        is_train: bool | None = None, pool_size: int | None = None,
                        ids: list | None = None,
                        annotated_ids: set | None = None,
                        tasks: Tasks | None = None,
                        annotator_id_path: str | None = None,
                        annotation_label_path: str | None = None,
                        wandb_name: str | None = None,
                        use_naiive_concat: bool | None = None) -> Dict[BaseDataset]:
        dataset_type = model_type_to_dataset(args.model_type)
        return {task: dataset_type(data_path=data_path, data = data, encoder_tokenizer=encoder_tokenizer,
                                        decoder_tokenizer=decoder_tokenizers[task.name],
                                        max_len=getattr(args, "max_seq_length", 512),
                                        is_train = is_train,
                                        pool_size=pool_size,
                                        ids=ids,
                                        annotated_ids=annotated_ids,
                                        task=task,
                                        annotator_id_path=annotator_id_path,
                                        annotation_label_path=annotation_label_path,
                                        wandb_name = wandb_name,
                                        use_naiive_concat = use_naiive_concat) for task in tasks}
    
    def _create_data_loader(self, data_path: str | None = None, data: dict | None = None, is_train: bool = True, 
                pool_size: int | None = None, ids: list | None = None, annotated_ids: set | None = None) -> Union[DataLoader, Dict[DataLoader]]:
        # if self.trainer:
        #     dtype = precision_to_dtype(self.trainer.precision_plugin.precision)
        datasets = self._create_dataset(encoder_tokenizer=self.encoder_tokenizer,
                                       decoder_tokenizers=self.decoder_tokenizers,
                                       data_path=data_path, is_train=is_train, args=self.args,
                                       pool_size=pool_size, ids=ids, data=data, annotated_ids=annotated_ids,
                                       tasks=self.tasks, annotator_id_path=self.annotator_id_path,
                                       annotation_label_path = self.annotation_label_path,
                                       wandb_name = self.wandb_name,
                                       use_naiive_concat = self.use_naiive_concat)
        if is_train:
            # lightening module only allows dictionary for training
            return {task.name: DataLoader(dataset, batch_size=self.args.train_batch_size if is_train else self.eval_batch_size,
                            drop_last=self.args.drop_last, shuffle=is_train, collate_fn=getattr(dataset, "collate", None),
                            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0, 
                            worker_init_fn=set_worker_sharing_strategy) for task, dataset in datasets.items()}
        return [DataLoader(dataset, batch_size=self.args.train_batch_size if is_train else self.eval_batch_size,
                        drop_last=self.args.drop_last, shuffle=is_train, collate_fn=getattr(dataset, "collate", None),
                        num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0,
                        worker_init_fn=set_worker_sharing_strategy) for task, dataset in datasets.items()]

    @overrides
    def train_dataloader(self) -> Union[DataLoader, Dict[DataLoader]]:
        if self.train_data:
            return self._create_data_loader(data=self.train_data, is_train=True)
        assert self.args.train_data_path
        return self._create_data_loader(data_path=self.args.train_data_path, is_train=True)

    @overrides
    def val_dataloader(self) -> Union[DataLoader, Dict[DataLoader]]:
        if self.dev_data:
            return self._create_data_loader(data=self.dev_data, is_train=False)
        assert self.args.dev_data_path
        return self._create_data_loader(data_path=self.args.dev_data_path, is_train=False)

    @overrides
    def test_dataloader(self) -> Union[DataLoader, Dict[DataLoader]]:
        if self.test_data:
            return self._create_data_loader(data=self.test_data, is_train=False)
        assert self.args.test_data_path
        return self._create_data_loader(data_path=self.args.test_data_path, is_train=False)

