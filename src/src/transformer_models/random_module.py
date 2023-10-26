from __future__ import annotations

import torch
import torch.nn as nn

from overrides import overrides
from transformers import PreTrainedTokenizerBase, BertModel, BertConfig
from typing import Any, Optional, Union, List
from collections.abc import Mapping, MutableMapping

from src.module import BaseModule, TYPE_BATCH, TYPE_SPLIT
from src.utils.utils import Task
from src.utils.decoding_utils import compute_answer_probs, compute_answer_prob
from src.transformer_models import Module, BERTTokenClassifier, BERTMultiChoice
from src.utils.task_utils import NUM_LABEL_PLUS_ONE, ANSWER_MINUS_ONE, PREDICTION_MASK


class RandomClassifier(nn.Module):
    def __init__(self, label_nums):
        super(RandomClassifier, self).__init__()
        self.label_nums = label_nums

    def forward(self, input_ids, attention_mask=None, task=None, \
                token_type_ids=None, labels=None, **kwargs):
        batch_size, sequence_length = input_ids.shape[:2]
        logits = torch.randn((batch_size, self.label_nums))
        return logits, None, None


class RandomModule(Module):

    def __init__(self, decoder_tokenizer: Union[PreTrainedTokenizerBase, dict], **kwargs) -> None:  # noqa
        super().__init__(decoder_tokenizer=decoder_tokenizer, **kwargs)

        model_kwargs = {}
        self.tasks = self.hparams.tasks
        self.training_paradigm = self.hparams.training_paradigm
        self.model_kwargs = model_kwargs

        model_kwargs["label_nums"] = self.hparams.num_labels
        self.model = RandomClassifier(**model_kwargs)

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, task: Task | None = None, **kwargs) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, task=task, **kwargs)

    def _single_step(self, batch: TYPE_BATCH, task_name: str, **kwargs) -> Tuple[torch.Tensor, int]:
        assert len(set(batch["task_id"])) == 1, "We assume that the data in the same batch come from the same task"
        task = self.tasks[int(batch["task_id"][0])]
        assert (not task_name) or (task.name == task_name), "Mismatched task"
        logits, _, _ = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"], \
                    task=task, annotator_ids=batch["annotator_id"], annotations=batch["annotations"], **kwargs)
        
        answer_ids = batch["answer_ids"]
        assert task.name in NUM_LABEL_PLUS_ONE
        loss = self.cross_entropy_loss(logits.view(-1, self.decoder_tokenizer[task].num_labels), answer_ids)
        return loss, len(batch["question"])
    
    # Don't check the signature here because a transitive dependency has a bug when an argument has a `Literal` type
    # with a string. See https://github.com/bojiang/typing_utils/issues/10
    @overrides(check_signature=False)
    def _step(self, batch: TYPE_BATCH, split: TYPE_SPLIT):
        output = super()._step(batch, split)

        kwargs = {}
        
        if split == "train":
            result = [self._single_step(batch=b, task_name=task_name, **kwargs) for task_name, b in batch.items()]
            output["loss"] = torch.sum(torch.stack([itm[0] for itm in result]))
            batch_size = sum([itm[1] for itm in result])
        else:
            loss, batch_size = self._single_step(batch=batch, task_name=None, **kwargs)
            output["loss"] = loss

        output["batch_size"] = batch_size
        self.log(f"loss/{split}", output["loss"], batch_size=batch_size)

        return output
        

    @overrides
    def _generative_step(self, batch: TYPE_BATCH) -> Mapping[str, Any]:
        output = {}

        assert len(set(batch["task_id"])) == 1, "We assume that the data in the same batch come from the same task"
        task = self.tasks[int(batch["task_id"][0])]

        logits, alpha, beta = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"], task=task, \
                                annotator_ids=batch["annotator_id"], annotations=batch["annotations"])
                
        assert task.name in ANSWER_MINUS_ONE
        
        output["generated_ids"] = logits
        pred_ids = logits.argmax(-1)
        
        assert task.name in PREDICTION_MASK
        output["generated"] = self.decoder_tokenizer[task].batch_decode(pred_ids)
        output["id"] = batch["id"]
        output["question"] = batch["question"]
        output["gold"] = batch["answer"]
        output["respondent_id"] = batch["origin_annotator_id"]
        output["annotator_embed_weight"] = alpha.tolist() if alpha is not None else None
        output["annotation_embed_weight"] = beta.tolist() if beta is not None else None
        return output

    # Don't check the signature here because a transitive dependency has a bug when an argument has a `Literal` type
    # with a string. See https://github.com/bojiang/typing_utils/issues/10
    @overrides(check_signature=False)
    def _update_metrics(self, batch: TYPE_BATCH, \
                        generative_step_output: Mapping[str, Any], split: TYPE_SPLIT) -> None:
        batch_size = len(batch["question"])

        if generated := generative_step_output.get("generated"):
            id_ = batch["id"]
            answer_ids = batch["answer_ids"]
            generated_ids = generative_step_output.get("generated_ids")

            batch_size = generated_ids.shape[0]
            assert len(set(batch["task_id"])) == 1, "We assume that the data in the same batch come from the same task"
            task: Task = self.tasks[int(batch["task_id"][0])]

            preds = torch.argmax(generated_ids, dim=-1)
            golds = answer_ids

            self.accuracies[task.name](preds, golds)
            self.log(f"accuracy_{task.name}/{split}", self.accuracies[task.name], batch_size=batch_size)

            if self.hparams.model_type in {"berttokenclassifier"}:

                self.recall_metric[task.name](preds, golds)
                self.log(f"recall_{task.name}/{split}", self.recall_metric[task.name], batch_size=batch_size)

                self.precision_metric[task.name](preds, golds)
                self.log(f"precision_{task.name}/{split}", self.precision_metric[task.name], batch_size=batch_size)

                self.f1score_metric[task.name](preds, golds)
                self.log(f"f1_{task.name}/{split}", self.f1score_metric[task.name], batch_size=batch_size)

            squad_format_generated = [{"prediction_text": " ".join(generated_instance), "id": id_instance}
                                      for generated_instance, id_instance in zip(generated, id_)]
            squad_format_answers = [{"answers": {"text": answers_instance}, "id": id_instance}
                                    for answers_instance, id_instance in zip(batch["answer"], id_)]
            
            self.squad.update(squad_format_generated, squad_format_answers)

        
