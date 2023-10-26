from __future__ import annotations

import torch
import torch.nn as nn
from overrides import overrides

from transformers import BertModel, BertConfig
from src.utils.utils import Task, Tasks


class BERTTokenClassifier(BertModel):
    def __init__(self, config: BertConfig, tasks: Tasks, decoder_tokenizers: dict) -> None:  # noqa
        super().__init__(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)   # randomly zero out some input elements, no need to be shared
        self.tasks = tasks
        self.decoder_tokenizers = decoder_tokenizers
        self.classifiers = nn.ModuleDict({})
        for task in self.tasks:
            # ModuleDict explicitly asks the key to be of type str
            self.classifiers.update({task.name: nn.Linear(config.hidden_size, self.decoder_tokenizers[task].num_labels_plus_pad)})

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, task: Task | None = None, **kwargs) -> Any:
        
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        return self.classifiers[task.name](sequence_output)
        