#!/usr/bin/env python
from __future__ import annotations
import argparse
import logging
import os
import json
import random
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer, LogitsProcessorList, PreTrainedTokenizerBase
from collections import defaultdict
from typing import Union
from overrides import override

from src.module import BaseModule
from src.utils.logger_utils import UninitializedWeightsFilter
from src.transformer_models import EncoderModule, Seq2SeqModule
from src.dataset import _read_in_data, _random_sample, DataModule, ActiveSelectionDataModule, \
        ActiveLearningDataModule, ActivePredictionDataModule
from src.utils.utils import set_up_tokenizers, Tasks, Task
from src.utils.generation_utils import ConstraintLogits
from src.training_paradigm import BaseParadigm


class LearnFromScratchParadigm(BaseParadigm):

    def _model_update_module(self, trainer: pl.Trainer, model: BaseModule, encoder_tokenizer: PreTrainedTokenizerBase, \
            decoder_tokenizer: Union[PreTrainedTokenizerBase, dict]) -> None:
            dm = DataModule(self.args, encoder_tokenizer=encoder_tokenizer,
                                                    decoder_tokenizer=decoder_tokenizer,
                                                    tasks=self.args.tasks, annotator_id_path = self.args.annotator_id_path,
                                                    annotation_label_path = self.args.annotation_label_path, wandb_name=self.args.wandb_name,
                                                    use_naiive_concat = self.args.use_naiive_concat)
            trainer.fit(model, datamodule=dm)

    def train_and_test(self) -> None:
        encoder_tokenizer, decoder_tokenizer, \
            trainer, model = super().train_and_test()

        if self.args.model_type not in {"random"}:
            self._model_update_module(trainer=trainer, model=model, encoder_tokenizer=encoder_tokenizer, \
                decoder_tokenizer=decoder_tokenizer)
        
        # Test the final model after all the active learning process
        self._test_module(trainer=trainer, model=model, encoder_tokenizer=encoder_tokenizer, \
                decoder_tokenizer=decoder_tokenizer, annotator_id_path = self.args.annotator_id_path, \
                annotation_label_path = self.args.annotation_label_path, wandb_name=self.args.wandb_name)
