from __future__ import annotations

import logging
import json
import os
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Literal, Mapping

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch import nn as nn
from torchmetrics import BLEUScore, Metric, SQuAD, Accuracy
from torchmetrics.classification import Recall, Precision, F1Score
from torchmetrics.text import BERTScore, ROUGEScore

from src.metrics.metrics import normalize_answer, TokenClassificationRecall, \
    TokenClassificationPrecision, TokenClassificationF1
from src.utils.logger_utils import UnusedWeightsFilter
from src.utils.utils import Task

TYPE_SPLIT = Literal["train", "val", "test"]
TYPE_BATCH = Mapping[str, Any]


class BaseModule(pl.LightningModule, ABC):

    def __init__(self, **kwargs) -> None:  # noqa
        super().__init__()

        self.save_hyperparameters()

        self.answer_metrics: Mapping[str, Metric] = nn.ModuleDict({"bleu1": BLEUScore(1), "bleu2": BLEUScore(2),
                                                                   "bleu3": BLEUScore(3)})
        self.rouge = ROUGEScore()
        self.bert_score = BERTScore(model_name_or_path="roberta-large", num_threads=0)
        self.squad = SQuAD()
        self.accuracies: Mapping[Task, Metric] = nn.ModuleDict({task.name: Accuracy() \
                                                                for task in self.hparams.tasks})
        
        if self.hparams.model_type in {"berttokenclassifier"}:
            self.recall_metric: Mapping[Task, Metric] = nn.ModuleDict({task.name: TokenClassificationRecall(O_id=self.hparams.decoder_tokenizer[task].O_id, \
                pad_id=self.hparams.pad_token_id) for task in self.hparams.tasks})
            self.precision_metric: Mapping[Task, Metric] = nn.ModuleDict({task.name: TokenClassificationPrecision(O_id=self.hparams.decoder_tokenizer[task].O_id, \
                pad_id=self.hparams.pad_token_id) for task in self.hparams.tasks})
            self.f1score_metric: Mapping[Task, Metric] = nn.ModuleDict({task.name: TokenClassificationF1(O_id=self.hparams.decoder_tokenizer[task].O_id, \
                pad_id=self.hparams.pad_token_id) for task in self.hparams.tasks})
        
        self.training_step_outputs = []
        self.testing_step_outputs = []

    def _on_eval_start(self) -> None:
        self.bert_score.embedding_device = self.device

    @overrides
    def on_validation_start(self) -> None:
        self._on_eval_start()

    @overrides
    def on_test_start(self) -> None:
        self._on_eval_start()

    def _step(self, batch: TYPE_BATCH, split: TYPE_SPLIT) -> MutableMapping[str, torch.Tensor]:  # noqa
        return {}

    @abstractmethod
    def _generative_step(self, batch: TYPE_BATCH) -> Mapping[str, Any]:
        raise NotImplementedError

    def _update_metrics(self, batch: TYPE_BATCH,  # noqa
                        generative_step_output: Mapping[str, Any], split: TYPE_SPLIT) -> None:
        batch_size = len(batch["question"])

        if generated := generative_step_output.get("generated"):
            id_ = batch["id"]
            answers = batch["answer"]

            # We normalize the generated and the ground truth answers before computing the metrics.
            #
            # Note BLEU, ROUGE, and SQuAD metrics perform different normalizations by default, some of them can't be
            # changed. Still, it doesn't matter because they do basic stuff, as we do, that's good enough for
            # evaluation (similar for the tokenization). But we do something on our end because BLEU doesn't do
            # any normalization.
            normalized_generated = [normalize_answer(generated_instance) for generated_instance in generated]
            normalized_answers = [[normalize_answer(answer_instance) for answer_instance in answers_instance]
                                  for answers_instance in answers]

            for name, metric in self.answer_metrics.items():
                metric(normalized_generated, normalized_answers)
                self.log(f"{name}/{split}", metric, batch_size=batch_size)

            # We handle the following metrics manually by doing `update`, `compute` and `reset` because they return a
            # dictionary of tensors instead of a single tensor, so it can't be done automatically by PL.

            self.rouge.update(normalized_generated, normalized_answers)

            squad_format_generated = [{"prediction_text": generated_instance, "id": id_instance}
                                      for generated_instance, id_instance in zip(normalized_generated, id_)]
            squad_format_answers = [{"answers": {"text": answers_instance}, "id": id_instance}
                                    for answers_instance, id_instance in zip(normalized_answers, id_)]
            
            self.squad.update(squad_format_generated, squad_format_answers)

            # BERTScore doesn't support multiple targets, and we have a variable number of answer.
            # We don't complicate it much and just evaluate the first answer (the original one). It's good enough.
            first_normalized_answer = [normalized_answers_instance[0]
                                       for normalized_answers_instance in normalized_answers]
            # BERTScore does not work on distributed training setting, disabled here
            # self.bert_score.update(normalized_generated, first_normalized_answer)

        if pred_spans := generative_step_output.get("pred_spans"):
            self.iou_f1(pred_spans, batch["evidence"].tolist())
            self.log(f"iou_f1/{split}", self.iou_f1, batch_size=batch_size)

    def _eval_step(self, batch: TYPE_BATCH, split: TYPE_SPLIT) -> None:
        generative_step_output = self._generative_step(batch)
        self._update_metrics(batch, generative_step_output, split)

        # move to cpu so that the CUDA memory will not be piled up
        # NOTE: too much memory consumption, disable for now
        # generative_step_output["annotator_embed_weight"] = self._move2cpu(generative_step_output["annotator_embed_weight"])
        # generative_step_output["annotation_embed_weight"] = self._move2cpu(generative_step_output["annotation_embed_weight"])
        # generative_step_output["full_embeddings"] = self._move2cpu(generative_step_output["full_embeddings"])
        # generative_step_output["annotator_embed_before_alpha"] = self._move2cpu(generative_step_output["annotator_embed_before_alpha"])
        # generative_step_output["annotator_embed_after_alpha"] = self._move2cpu(generative_step_output["annotator_embed_after_alpha"])
        # generative_step_output["annotation_embed_before_beta"] = self._move2cpu(generative_step_output["annotation_embed_before_beta"])
        # generative_step_output["annotation_embed_after_beta"] = self._move2cpu(generative_step_output["annotation_embed_after_beta"])
        # generative_step_output["sentence_embed"] = self._move2cpu(generative_step_output["sentence_embed"])
        return generative_step_output

    @overrides(check_signature=False)
    def validation_step(self, batch: TYPE_BATCH, batch_idx, dataset_idx=None) -> None:
        # add an extra argument for multi-task training
        self._eval_step(batch, split="val")

    @overrides(check_signature=False)
    def test_step(self, batch: TYPE_BATCH, batch_idx, dataset_idx=None) -> None:
        self.testing_step_outputs.append(self._eval_step(batch, split="test"))

    def _eval_epoch_end(self, split: TYPE_SPLIT) -> None:
        instance_count = sum(t.shape[0] for t in self.bert_score.preds_input_ids)

        if instance_count:
            self.log_dict({f"{k}/{split}": v for k, v in self.rouge.compute().items()}, batch_size=instance_count)
            self.rouge.reset()

            for k, v in self.accuracies.items():
                self.log(f"accuracy_{k}/{split}", v.compute(), batch_size=instance_count)
                self.accuracies[k].reset()
            
            for k, v in self.recall_metric.items():
                self.log(f"recall_{k}/{split}", v.compute(), batch_size=instance_count)
                self.recall_metric[k].reset()
            
            for k, v in self.precision_metric.items():
                self.log(f"precision_{k}/{split}", v.compute(), batch_size=instance_count)
                self.precision_metric[k].reset()
            
            for k, v in self.f1score_metric.items():
                self.log(f"f1_{k}/{split}", v.compute(), batch_size=instance_count)
                self.f1score_metric[k].reset()

            self.log_dict({f"{k}/{split}": v / 100 for k, v in self.squad.compute().items()}, batch_size=instance_count)
            self.squad.reset()

        unused_weights_filter = UnusedWeightsFilter()
        logging.getLogger("transformers.modeling_utils").addFilter(unused_weights_filter)
        # self.log_dict({f"bert_score_first_answer_{k}/{split}": sum(v) / len(v)
        #                for k, v in self.bert_score.compute().items()}, batch_size=instance_count)

        logging.getLogger("transformers.modeling_utils").removeFilter(unused_weights_filter)
        # self.bert_score.reset()

    @overrides(check_signature=False)
    def on_validation_epoch_end(self) -> None:
        self._eval_epoch_end(split="val")

    @overrides(check_signature=False)
    def on_test_epoch_end(self) -> None:
        self._eval_epoch_end(split="test")

        # learned annotator embeddings
        outputs = self.testing_step_outputs 

        y_preds = [output["generated"] for output in outputs]
        y_golds = [output["gold"] for output in outputs]
        questions = [output["question"] for output in outputs]
        ids = [output["id"] for output in outputs]
        respondent_ids = [output["respondent_id"] for output in outputs]
        # annotator_embed_weights = [output["annotator_embed_weight"] for output in outputs]
        # annotation_embed_weights = [output["annotation_embed_weight"] for output in outputs]

        def flatten(l):
            pl = []
            for ll in l:
                if isinstance(ll, list):
                    for ele in ll:
                        pl.append(ele)
                elif isinstance(ll, float):
                    pl.append(ele)
                else:
                    raise RuntimeWarning(f"Type {type(ll)} not supported in test processing")
            return pl

        y_preds = flatten(y_preds)
        y_golds = flatten(y_golds)
        questions = flatten(questions)
        ids = flatten(ids)
        respondent_ids = flatten(respondent_ids)
        assert len(y_preds) == len(y_golds) == len(questions) == len(ids) == len(respondent_ids)
        # if annotator_embed_weights is not None and annotator_embed_weights[0] is not None:
        #     annotator_embed_weights = flatten(annotator_embed_weights)
        #     assert len(y_preds) == len(annotator_embed_weights)
        # else:
        #     annotator_embed_weights = [None for _ in y_preds]
        # if annotation_embed_weights is not None and annotation_embed_weights[0] is not None:
        #     annotation_embed_weights = flatten(annotation_embed_weights)
        #     assert len(y_preds) == len(annotation_embed_weights)
        # else:
        #     annotation_embed_weights = [None for _ in y_preds]

        # results = [json.dumps({"pred": pred, "gold": gold, "question": question, "id": id, "respondent_id": respondent_id, \
        #                        "annotator_embed_weight": annotator_embed_weight, "annotation_embed_weight": annotation_embed_weight}) \
        #      for pred, gold, question, id, respondent_id, annotator_embed_weight, annotation_embed_weight \
        #          in zip(y_preds, y_golds, questions, ids, respondent_ids, annotator_embed_weights, annotation_embed_weights)]
        results = [json.dumps({"pred": pred, "gold": gold, "question": question, "id": id, "respondent_id": respondent_id}) \
             for pred, gold, question, id, respondent_id \
                 in zip(y_preds, y_golds, questions, ids, respondent_ids)]
        
        # output prediction file
        if not os.path.exists(os.path.dirname(self.hparams.pred_fn_path)):
            os.makedirs(os.path.dirname(self.hparams.pred_fn_path))
        with open(self.hparams.pred_fn_path, 'w') as f:
            f.write("\n".join(results))

        # # output pt file for the embeddings
        # assert self.hparams.pred_fn_path.endswith(".jsonl")
        # embedding_info = {
        #     "annotator_embed_weight": [output["annotator_embed_weight"] for output in outputs], 
        #     "annotation_embed_weight": [output["annotation_embed_weight"] for output in outputs], 
        #     "full_embeddings": [output["full_embeddings"] for output in outputs], 
        #     "annotator_embed_before_alpha": [output["annotator_embed_before_alpha"] for output in outputs], 
        #     "annotator_embed_after_alpha": [output["annotator_embed_after_alpha"] for output in outputs], 
        #     "annotation_embed_before_beta": [output["annotation_embed_before_beta"] for output in outputs], 
        #     "annotation_embed_after_beta": [output["annotation_embed_after_beta"] for output in outputs],
        #     "sentence_embed": [output["sentence_embed"] for output in outputs],
        #     # learned label embeddings
        #     # annotator_embed: order is in stats.json
        #     # annotation_embed: order is in corresponding files under __temp__/
        #     "annotator_embed": self._move2cpu(self.model.embeddings.annotator_embed.state_dict()),
        #     "label_embed": self._move2cpu(self.model.embeddings.annotation_embed.state_dict())
        # } 
        
        # torch.save(embedding_info, f"{self.hparams.pred_fn_path.split('.jsonl')[0]}.pt")

    def _move2cpu(self, tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                return tensor.to("cpu")
        return tensor