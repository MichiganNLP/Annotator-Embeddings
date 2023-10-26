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
from transformers import PreTrainedTokenizerBase, BertModel, \
        BertConfig, RobertaModel, RobertaConfig, DebertaV2Model, DebertaV2Config
from transformers import get_linear_schedule_with_warmup
from typing import Any, Optional, Union, List

from src.module import BaseModule, TYPE_BATCH, TYPE_SPLIT
from src.transformer_models import Module, BERTTokenClassifier, BERTMultiChoice,\
                                 RobertaMultiChoice, DebertaV2MultiChoice
from src.utils.decoding_utils import compute_answer_probs, compute_answer_prob
from src.tokenization import SequenceTokenizer
from src.utils.utils import Task, Tasks
from src.utils.task_utils import NUM_LABEL_PLUS_ONE, ANSWER_MINUS_ONE, PREDICTION_MASK


def model_type_to_class(model_type: str) -> type[BertModel | RobertaModel | DebertaV2Model]:  # noqa
    return {
        "bert": BERTTokenClassifier,
        "berttokenclassifier": BERTTokenClassifier,
        "bertmultichoice": BERTMultiChoice,
        "roberta-multichoice": RobertaMultiChoice,
        "deberta-multichoice": DebertaV2MultiChoice
    }[model_type]



class EncoderModule(Module):
    def __init__(self, decoder_tokenizer: Union[PreTrainedTokenizerBase, dict], **kwargs) -> None:  # noqa
        super().__init__(decoder_tokenizer=decoder_tokenizer, **kwargs)

        model_class = model_type_to_class(self.hparams.model_type)
        model_kwargs = {}
        self.tasks = self.hparams.tasks
        self.training_paradigm = self.hparams.training_paradigm

        if self.hparams.model_type in {"berttokenclassifier", "bert", "bertmultichoice",
                                       "roberta-multichoice", "deberta-multichoice"}:
            model_kwargs["tasks"] = self.tasks
            model_kwargs["decoder_tokenizers"] = self.decoder_tokenizer
            model_kwargs["num_annotators"] = self.hparams.num_annotators
            model_kwargs["label_nums"] = self.hparams.num_labels
            model_kwargs["broadcast_annotator_embedding"] = self.hparams.broadcast_annotator_embedding
            model_kwargs["broadcast_annotation_embedding"] = self.hparams.broadcast_annotation_embedding
            model_kwargs["annotator_id_path"] = self.hparams.annotator_id_path
            model_kwargs["use_annotator_embed"] = self.hparams.use_annotator_embed
            model_kwargs["use_annotation_embed"] = self.hparams.use_annotation_embed
            model_kwargs["include_pad_annotation"] = self.hparams.include_pad_annotation
            model_kwargs["method"] = self.hparams.method
            if self.hparams.model_type in {"berttokenclassifier", "bert", "bertmultichoice"}:
                model_kwargs["embed_wo_weight"] = self.hparams.embed_wo_weight
        self.model_kwargs = model_kwargs
        self.model = model_class.from_pretrained(self.hparams.model_name_or_path, **model_kwargs)
        self.softmax = nn.Softmax(dim=1)

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, task: Task | None = None, **kwargs) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, task=task, **kwargs)
    
    def _single_step(self, batch: TYPE_BATCH, task_name: str, **kwargs) -> Tuple[torch.Tensor, int]:
        assert len(set(batch["task_id"])) == 1, "We assume that the data in the same batch come from the same task"
        task = self.tasks[int(batch["task_id"][0])]
        assert (not task_name) or (task.name == task_name), "Mismatched task"
        logits, _ = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"], \
                    task=task, annotator_ids=batch["annotator_id"], annotations=batch["annotations"], **kwargs)
        
        answer_ids = batch["answer_ids"]
        assert task.name in NUM_LABEL_PLUS_ONE
        loss = self.cross_entropy_loss(logits.view(-1, self.decoder_tokenizer[task].num_labels), answer_ids)
        return loss, len(batch["question"])

    # Don't check the signature here because a transitive dependency has a bug when an argument has a `Literal` type
    # with a string. See https://github.com/bojiang/typing_utils/issues/10
    @overrides(check_signature=False)
    def _step(self, batch: TYPE_BATCH, split: TYPE_SPLIT) -> MutableMapping[str, torch.Tensor]:
        output = super()._step(batch, split)

        kwargs = {}

        if split != "train":
            kwargs["output_attentions"] = True
            kwargs["output_hidden_states"] = True
        
        if split == "train":
            if self.hparams.test_mode == "test_w_info":
                # in training time, no information is provided
                kwargs["disable_annotator_ids"] = True
                kwargs["disable_annotation"] = True
            elif self.hparams.test_mode == "cls_only":
                kwargs["disable_question"] = True
            result = [self._single_step(batch=b, task_name=task_name, **kwargs) for task_name, b in batch.items()]
            output["loss"] = torch.sum(torch.stack([itm[0] for itm in result]))
            batch_size = sum([itm[1] for itm in result])
        else:
            if split == "test" and self.hparams.test_mode == "normal_wo_info":
                kwargs["disable_annotator_ids"] = True
                kwargs["disable_annotation"] = True
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

        #NOTE: assume the generation happens at the testing time
        if self.hparams.test_mode == "normal_wo_info":
            logits, embedding_output = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"], task=task, \
                                   annotator_ids=None, annotations=None)
        elif self.hparams.test_mode in {"normal", "test_w_info"}:
            logits, embedding_output = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"], task=task, \
                                   annotator_ids=batch["annotator_id"], annotations=batch["annotations"])
        elif self.hparams.test_mode == "cls_only":
            kwargs = {"disable_question": True}
            logits, embedding_output = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"], task=task, \
                                   annotator_ids=batch["annotator_id"], annotations=batch["annotations"], **kwargs)
        elif self.hparams.test_mode == "test_cls_only":
            kwargs = {"disable_question": True}
            logits, embedding_output = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"], task=task, \
                                   annotator_ids=batch["annotator_id"], annotations=batch["annotations"], **kwargs)
        else:
            raise RuntimeError(f"Test mode {self.hparams.test_mode} not supported.")

        # compute probablity
        model_config = self.model.config
        
        assert task.name in ANSWER_MINUS_ONE
        generated_probs = compute_answer_probs(logits, batch["answer_ids"], model_config, ignore_eos_token=True)
        generated_prob = compute_answer_prob(generated_probs)
        assert len(batch["data_id"]) == len(generated_prob)
        
        output["generated_ids"] = logits
        pred_ids = logits.argmax(-1)
        
        assert task.name in PREDICTION_MASK
        output["generated"] = self.decoder_tokenizer[task].batch_decode(pred_ids)
        output["id"] = batch["id"]
        output["question"] = batch["question"]
        output["gold"] = batch["answer"]
        output["respondent_id"] = batch["origin_annotator_id"]
        
        # information about the embedding output
        # For debug purpose: maybe too much information
        # output["annotator_embed_weight"] = embedding_output["alpha"].tolist() if embedding_output["alpha"] is not None else None
        # output["annotation_embed_weight"] = embedding_output["beta"].tolist() if embedding_output["beta"] is not None else None
        # output["full_embeddings"] = embedding_output["embeddings"]
        # output["annotator_embed_before_alpha"] = embedding_output["annotator_embed_before_alpha"]
        # output["annotator_embed_after_alpha"] = embedding_output["annotator_embed_after_alpha"]
        # output["annotation_embed_before_beta"] = embedding_output["annotation_embed_before_beta"]
        # output["annotation_embed_after_beta"] = embedding_output["annotation_embed_after_beta"]
        # output["sentence_embed"] = embedding_output["sentence_embed"]
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

            # Generate the answer and compute metrics based on it:
            model_config = self.model.config

            generated_probs = compute_answer_probs(generated_ids, answer_ids, model_config, ignore_eos_token=True)
            generated_prob = compute_answer_prob(generated_probs)
            self.log(f"generated_prob/{split}", generated_prob.mean(), batch_size=batch_size)
        