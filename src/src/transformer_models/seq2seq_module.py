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
from typing import Any, Optional

from src.utils.decoding_utils import compute_answer_prob, compute_answer_probs
from src.metrics.metrics import Perplexity
from src.module import BaseModule, TYPE_BATCH, TYPE_SPLIT
from src.transformer_models import Module


def model_type_to_class(model_type: str) -> type[T5ForConditionalGeneration]:  # noqa
    return {
        "t5": T5ForConditionalGeneration,
    }[model_type]


class Seq2SeqModule(Module):
    def __init__(self, decoder_tokenizer: PreTrainedTokenizerBase, generate_kwargs: Mapping[str, Any] | None = None,
                logits_processor: Optional[LogitsProcessorList] = None,
                 **kwargs) -> None:  # noqa
        super().__init__(decoder_tokenizer=decoder_tokenizer, **kwargs)

        model_class = model_type_to_class(self.hparams.model_type)
        model_kwargs = {}

        self.training_paradigm = self.hparams.training_paradigm
        self.model_kwargs = model_kwargs
        self.model = model_class.from_pretrained(self.hparams.model_name_or_path, **model_kwargs)
        self.answers_generation_enabled = isinstance(self.model, T5ForConditionalGeneration)

        self.perplexity = Perplexity()

        self.generate_kwargs = generate_kwargs or {}
        self.logits_processor = logits_processor

        self.generate_kwargs.setdefault("return_dict_in_generate", True)
        self.generate_kwargs.setdefault("output_scores", True)

        # The following are useful to compute the encoder layer output only once.
        self.generate_kwargs.setdefault("output_hidden_states", True)
        self.generate_kwargs.setdefault("output_attentions", True)

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                answer_ids: torch.Tensor | None = None, **kwargs) -> Any:
        if answer_ids is not None:
            answer_ids = answer_ids.clone()
            answer_ids[answer_ids == self.decoder_tokenizer.pad_token_id] = -100  # For the loss computation.

        if self.answers_generation_enabled:
            kwargs["labels"] = answer_ids

        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

    # Don't check the signature here because a transitive dependency has a bug when an argument has a `Literal` type
    # with a string. See https://github.com/bojiang/typing_utils/issues/10
    @overrides(check_signature=False)
    def _step(self, batch: TYPE_BATCH, split: TYPE_SPLIT) -> MutableMapping[str, torch.Tensor]:
        output = super()._step(batch, split)

        kwargs = {}

        if split != "train":
            kwargs["output_attentions"] = True
            kwargs["output_hidden_states"] = True

        model_output = self(input_ids=batch["question_ids"], attention_mask=batch["question_mask"],
                            answer_ids=batch["answer_ids"], **kwargs)

        if self.answers_generation_enabled and split != "train":
            output["encoder_hidden_states"] = model_output.encoder_hidden_states
            output["encoder_attentions"] = model_output.encoder_attentions

        if self.answers_generation_enabled:
            answer_loss = model_output.loss
            output["answer_logits"] = model_output.logits
        else:
            answer_loss = 0

        output["loss"] = answer_loss

        self.log(f"loss/{split}", output["loss"], batch_size=len(batch["question"]))

        return output


    @overrides
    def _generative_step(self, batch: TYPE_BATCH) -> Mapping[str, Any]:
        output = {}

        if self.answers_generation_enabled:
            kwargs = {}

            # FIXME: this doesn't work with all models, because the encoders are different. We should call first the
            #  encoder and save the result. But for it to work, the models need to implement the encoder well.
            # if model_config.is_encoder_decoder:  # Reuse the encoder output to avoid computing it twice.
            #     kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=step_output["encoder_hidden_states"][-1]
            #                                                 hidden_states=step_output["encoder_hidden_states"],
            #                                                 attentions=step_output["encoder_attentions"])
            generated_output = self.model.generate(batch["question_ids"], attention_mask=batch["question_mask"],
                                                    logits_processor=self.logits_processor,
                                                   **self.generate_kwargs, **kwargs)
            output["generated_ids"] = generated_output.sequences
            output["generated"] = self.decoder_tokenizer.batch_decode(output["generated_ids"], skip_special_tokens=True)
            output["generated_scores"] = generated_output.scores

        return output


    # Don't check the signature here because a transitive dependency has a bug when an argument has a `Literal` type
    # with a string. See https://github.com/bojiang/typing_utils/issues/10
    @overrides(check_signature=False)
    def _update_metrics(self, batch: TYPE_BATCH, step_output: MutableMapping[str, torch.Tensor],
                        generative_step_output: Mapping[str, Any], split: TYPE_SPLIT) -> torch.Tensor:
        super()._update_metrics(batch, step_output, generative_step_output, split)

        if (generated_ids := generative_step_output.get("generated_ids")) is not None:
            answer_ids = batch["answer_ids"]

            batch_size = len(answer_ids)

            model_config = self.model.config

            ground_truth_logits = step_output["answer_logits"]
            ground_truth_probs = compute_answer_probs(ground_truth_logits, answer_ids, model_config,
                                                      ignore_eos_token=True)
            ground_truth_prob = compute_answer_prob(ground_truth_probs)
            self.log(f"ground_truth_prob/{split}", ground_truth_prob, batch_size=batch_size)

            perplexity_mask = ((answer_ids != model_config.pad_token_id) & (answer_ids != model_config.eos_token_id))
            self.perplexity(ground_truth_probs, perplexity_mask)
            self.log(f"perplexity/{split}", self.perplexity, batch_size=batch_size)

            # Generate the answer and compute metrics based on it:

            generated_logits = torch.stack(generative_step_output["generated_scores"], dim=1)
            generated_probs = compute_answer_probs(generated_logits, generated_ids, model_config, ignore_eos_token=True)
            generated_prob = compute_answer_prob(generated_probs)
            self.log(f"generated_prob/{split}", generated_prob, batch_size=batch_size)
            assert len(batch["data_id"]) == len(generated_prob)
