from __future__ import annotations

import inspect
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import json
from collections.abc import Mapping, MutableMapping
from overrides import overrides
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup, LogitsProcessorList
from typing import Any, Optional, Union

from src.metrics.metrics import Perplexity
from src.module import BaseModule, TYPE_BATCH, TYPE_SPLIT


def log_lr(pl_module: pl.LightningModule, **kwargs) -> None:
    for i, optimizer in enumerate(pl_module.trainer.optimizers):
        for j, param_group in enumerate(optimizer.param_groups):
            if (lr := param_group.get("lr")) is not None:
                pl_module.log(f"lr_{i}_group_{j}", lr, **kwargs)


class Module(BaseModule):
    def __init__(self, decoder_tokenizer: Union[PreTrainedTokenizerBase, dict], **kwargs) -> None:  # noqa
        super().__init__(**kwargs)
        with open(self.hparams.train_data_path, 'r') as f:
            self.hparams.train_data_length = len(json.load(f))
        self.decoder_tokenizer = decoder_tokenizer
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    @overrides(check_signature=False)
    def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        output = self._step(batch, split="train")
        loss = output["loss"]
        batch_size = output["batch_size"]
        self.log("batch_size", float(batch_size), batch_size=batch_size)

        log_lr(self, batch_size=batch_size)
        return loss

    def predict_step(self, batch: Any, batch_idx: int):
        output = self._generative_step(batch, step_output=None)
        output["id"] = batch["data_id"]
        return output


    @overrides
    def configure_optimizers(self):
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        # NOTE: not working with parallel GPU setting
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
        #                                             num_training_steps=self.trainer.estimated_stepping_batches)
        if self.hparams.linear_scheduler:
            estimated_stepping_batches = math.ceil(self.hparams.train_data_length / self.hparams.train_batch_size) * max(self.hparams.num_train_epochs, 1)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=estimated_stepping_batches)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return {"optimizer": optimizer}
