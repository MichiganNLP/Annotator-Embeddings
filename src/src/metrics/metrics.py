from __future__ import annotations

import re
import string
from collections.abc import Sequence

import torch
from overrides import overrides
from torchmetrics import Metric


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(?:a|an|the)\b", "", text)


def _remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def normalize_answer(answer: str) -> str:
    return _remove_articles(_remove_punctuation(answer.lower()))


class Perplexity(Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, answer_probs: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if mask is not None:
            answer_probs = answer_probs.clone()
            answer_probs[~mask] = float("NaN")

        # It doesn't matter the log and exp base as long as they are the same because they cancel out.
        self.total += (-answer_probs.log().nanmean(dim=-1)).exp().sum()
        self.count += len(answer_probs)

    @overrides
    def compute(self) -> torch.Tensor:
        return self.total / self.count


class TokenClassificationPrecision(Metric):
    def __init__(self, O_id: int, pad_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.O_id = O_id
        self.pad_id = pad_id
        self.add_state("denom", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.denom += torch.sum((preds != self.O_id) * (preds != self.pad_id))
        self.num += torch.sum((preds == target) * (preds != self.O_id) * (preds != self.pad_id))

    @overrides
    def compute(self) -> float:
        return self.num / self.denom

class TokenClassificationRecall(TokenClassificationPrecision):
    
    @overrides(check_signature=False)
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.denom += torch.sum((target != self.O_id) * (target != self.pad_id))
        self.num += torch.sum((preds == target) * (preds != self.O_id) * (preds != self.pad_id))

    
class TokenClassificationF1(Metric):

    def __init__(self, O_id: int, pad_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.O_id = O_id
        self.pad_id = pad_id
        self.add_state("recall_denom", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("recall_num", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("precision_denom", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("precision_num", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.precision_denom += torch.sum((preds != self.O_id) * (preds != self.pad_id))
        self.precision_num += torch.sum((preds == target) * (preds != self.O_id) * (preds != self.pad_id))
        self.recall_denom += torch.sum((target != self.O_id) * (target != self.pad_id))
        self.recall_num += torch.sum((preds == target) * (preds != self.O_id) * (preds != self.pad_id))

    @overrides
    def compute(self) -> float:
        return (self.recall_num + self.precision_num)/ (self.recall_denom + self.precision_denom)
