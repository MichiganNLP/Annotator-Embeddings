#!/usr/bin/env python
from __future__ import annotations
import argparse
import logging
import os
import random
import json
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import SingleDeviceStrategy
from transformers import AutoTokenizer, LogitsProcessorList, PreTrainedTokenizerBase
from typing import Union
from overrides import overrides

from src.module import BaseModule
from src.utils.logger_utils import UninitializedWeightsFilter
from src.transformer_models import EncoderModule, Seq2SeqModule, RandomModule
from src.dataset import _read_in_data, _random_sample, DataModule
from src.utils.utils import set_up_tokenizers, Tasks, Task
from src.utils.generation_utils import ConstraintLogits
from src.training_paradigm import example_selection, forward_prediction
from src.utils.model_checkpoint import CustomizedModelCheckpoint


# class CustomizedStrategy(SingleDeviceStrategy):
#     """ For convenience, we only deploy the algorithm on one device """

#     def __init__(self, **kwargs):
#         # NOTE: hardcode
#         kwargs['device'] = torch.device(type='cuda', index=0)
#         super().__init__(**kwargs)

#     @overrides
#     def teardown(self) -> None:
#         # NOTE: nasty fix for datasets with large number of annotators
#         # This is only an issue for Toxic Ratings with more than 50k annotators.
#         pass


class BaseParadigm:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
    
    def _constrained_decoding(self, encoder_tokenizer: PreTrainedTokenizerBase, constrained_tokens: set) -> None:
        if self.args.model_type == "t5":
            logits_processor = LogitsProcessorList([ConstraintLogits(encoder_tokenizer, constrained_tokens)])
            self.args.logits_processor = logits_processor
        
    def _set_unintialized_weights_filter(self) -> type[UninitializedWeightsFilter]:
        uninitialized_weights_filter = UninitializedWeightsFilter()
        logging.getLogger("transformers.modeling_utils").addFilter(uninitialized_weights_filter)
        return uninitialized_weights_filter
    
    def _remove_unintialized_weights_filter(self, uninitialized_weights_filter: UninitializedWeightsFilter | None = None) -> None:
        if uninitialized_weights_filter:
            logging.getLogger("transformers.modeling_utils").removeFilter(uninitialized_weights_filter)
    
    def _set_up_model(self) -> BaseModule:
        model_class = model_type_to_class(self.args.model_type)
        if hasattr(self.args, 'load_ckpt_path') and self.args.load_ckpt_path:
            print("Loading from a checkpoint and continue training...")
            model = model_class.load_from_checkpoint(self.args.load_ckpt_path)
        else:
            model = model_class(**self.args.__dict__)
        return model

    def _set_up_loggers(self) -> list:
        loggers = [
            # WandbLogger(name=self.args.wandb_name, project=self.args.wandb_project, entity=self.args.wandb_entity,
                        # offline=self.args.wandb_offline),
            TensorBoardLogger(save_dir="transformer_models"),
        ]
        return loggers

    def _set_up_callbacks(self) -> list:
        callbacks = [
            TQDMProgressBar()
        ]
        return callbacks

    def _set_up_trainer(self, loggers: list, callbacks: list) -> type[pl.Trainer]:
        # NOTE: here is the nasty fix if we have an annotator pool that is too large.
        # This is only an issue for Toxic Ratings with more than 50k annotators.
        # cs = CustomizedStrategy()
        # trainer = pl.Trainer(strategy=cs, accumulate_grad_batches=self.args.gradient_accumulation_steps, devices=self.args.n_gpu,
        #                     max_epochs=self.args.num_train_epochs, precision=16 if self.args.fp_16 else 32,
        #                     gradient_clip_val=self.args.max_grad_norm, profiler=self.args.profiler,
        #                     log_every_n_steps=1, logger=loggers, callbacks=callbacks, val_check_interval=self.args.val_check_interval,
        #                     check_val_every_n_epoch=self.args.check_val_every_n_epoch, num_sanity_val_steps=self.args.num_sanity_val_steps,
        #                     enable_checkpointing=self.args.enable_checkpointing)
        trainer = pl.Trainer(accumulate_grad_batches=self.args.gradient_accumulation_steps, devices=self.args.n_gpu,
                            max_epochs=self.args.num_train_epochs, precision=16 if self.args.fp_16 else 32,
                            gradient_clip_val=self.args.max_grad_norm, profiler=self.args.profiler,
                            log_every_n_steps=1, logger=loggers, callbacks=callbacks, val_check_interval=self.args.val_check_interval,
                            check_val_every_n_epoch=self.args.check_val_every_n_epoch, num_sanity_val_steps=self.args.num_sanity_val_steps,
                            enable_checkpointing=self.args.enable_checkpointing)
        return trainer

    def _test_module(self, trainer: pl.Trainer, model: BaseModule, encoder_tokenizer: PreTrainedTokenizerBase, \
            decoder_tokenizer: Union[PreTrainedTokenizerBase, dict], annotator_id_path: str | None = None, \
            annotation_label_path: str | None = None, wandb_name: str | None = None):
        for task in self.args.tasks:
            dm = DataModule(self.args, encoder_tokenizer=encoder_tokenizer,
                                                        decoder_tokenizer=decoder_tokenizer,
                                                        annotator_id_path = self.args.annotator_id_path,
                                                        annotation_label_path = self.args.annotation_label_path, 
                                                        wandb_name=self.args.wandb_name,
                                                        tasks = Tasks([(task.id, task.name)]))
            trainer.test(model, datamodule=dm)

    def train_and_test(self) -> None:
        # initialize the task object
        self.args.tasks = Tasks(self.args.tasks)
        # train_data, dev_data, test_data = _read_in_data(self.args)
        with open(self.args.annotation_label_path, 'r') as f:
            annotation_labels = json.load(f)
        encoder_tokenizer, decoder_tokenizer, constrained_tokens = set_up_tokenizers(self.args, \
                annotation_labels=annotation_labels)
        
        self.args.decoder_tokenizer = decoder_tokenizer

        self._constrained_decoding(encoder_tokenizer, constrained_tokens)
        uninitialized_weights_filter = self._set_unintialized_weights_filter()
        model = self._set_up_model()

        if self.args.model_type in {"bert", "bertmultichoice"}:
            model.model.resize_token_embeddings(len(encoder_tokenizer))
        self._remove_unintialized_weights_filter(uninitialized_weights_filter)
        
        loggers = self._set_up_loggers()
        callbacks = self._set_up_callbacks()
        trainer = self._set_up_trainer(loggers, callbacks)
        return encoder_tokenizer, decoder_tokenizer, trainer, model


def model_type_to_class(model_type: str) -> BaseModule:  # noqa
    if "t5" in model_type:
        return Seq2SeqModule
    elif "bert" in model_type:
        return EncoderModule
    elif model_type == "random":
        return RandomModule


