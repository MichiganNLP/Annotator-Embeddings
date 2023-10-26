import torch
import numpy as np
from typing import List, Optional, Union


class MultiChoiceTokenizer:

    def __init__(self, labels: Union[list, set], pad_token_id: int = 0) -> None:
        self.labels = labels
        self.id2label_dict = {}
        self.label2id_dict = {}
        self._process_labels()
        self._pad_token_id = pad_token_id
    
    def _process_labels(self) -> None:
        for id, label in enumerate(self.labels):
            self.id2label_dict[id] = label
            self.label2id_dict[label] = id

    def id2label(self, id: Union[int, torch.Tensor]) -> Union[str, ValueError]:
        if torch.is_tensor(id):
            id = int(id)
        if id in self.id2label_dict:
            return self.id2label_dict[id]
        raise ValueError(f"{id} not in the tokenizer")
    
    def label2id(self, label: str) -> Union[int, ValueError]:
        if label in self.label2id_dict:
            return self.label2id_dict[label]
        raise ValueError(f"{label} not in the tokenizer")

    def __call__(self, text: List[str], max_length: Optional[int] = None, \
            truncation: Union[bool, None] = None, \
            padding: Union[bool, None] = None, \
            return_tensors: Union[str, None] = None,
            **kwargs) -> dict:

        input_ids = [self.label2id(label) for label in text]
        assert not truncation and padding
 
        if return_tensors == "pt":
            input_ids = torch.Tensor(input_ids)
            input_ids = input_ids.type(torch.LongTensor)
        else:
            raise ValueError(f"The return tensor type {return_tensors} is not supported.")
        
        return {
            "input_ids": input_ids,
        }
    
    def ignore_mapping(self, mapping):
        return mapping[0] == mapping[1]

    def batch_decode(self, logits: torch.Tensor) -> List[List[str]]:
        """ `logits` has shape (N, L) and dtype float. """
        return [self.id2label(logit) for logit in logits]

    def to(self, device: Union[str, "torch.device"]) -> "MultiChoiceTokenizer":
        """
        Modified from https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/tokenization_utils_base.py#L3374
        Send all values to device by calling `v.to(device)` (PyTorch only).
        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            [`Tokenizer`]: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            RuntimeError(f"Attempting to cast a Tokenizer to type {str(device)}. This is not supported.")
        return self

    @property
    def num_labels(self) -> int:
        return len(self.label2id_dict)
    
    @property
    def num_labels_plus_pad(self) -> int:
        return self.num_labels + 1

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id
