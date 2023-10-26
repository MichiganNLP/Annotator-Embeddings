import torch
import numpy as np
from typing import List, Optional, Union


class SequenceTokenizer:

    def __init__(self, labels: Union[list, set], pad_token_id: int = 0) -> None:
        self.labels = labels
        self.id2label_dict = {}
        self.label2id_dict = {}
        self._process_labels()
        self._pad_token_id = pad_token_id
    
    def _process_labels(self) -> None:
        for id, label in enumerate(self.labels):
            # start from 1 to avoid colliding with the pad_token_id
            self.id2label_dict[id + 1] = label
            self.label2id_dict[label] = id + 1

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

    def __call__(self, text: List[List[str]], max_length: Optional[int] = None, \
            truncation: Union[bool, None] = None, \
            padding: Union[bool, None] = None, \
            return_tensors: Union[str, None] = None,
            offset_mappings: Union[torch.Tensor, None] = None) -> dict:

        input_ids = [self.encode_tags_last(tags, offset_mapping) for tags, offset_mapping in zip(text, offset_mappings)]
        prediction_mask = [self.prediction_mask(offset_mapping) for offset_mapping in offset_mappings]

        assert not truncation and padding
 
        if return_tensors == "pt":
            input_ids = torch.Tensor(input_ids)
            input_ids = input_ids.type(torch.LongTensor)
            prediction_mask = torch.Tensor(prediction_mask)
            prediction_mask = prediction_mask.type(torch.LongTensor)
        else:
            raise ValueError(f"The return tensor type {return_tensors} is not supported.")
        
        return {
            "input_ids": input_ids,
            "prediction_mask": prediction_mask 
        }

    # Copied from https://github.com/AMontgomerie/bulgarian-nlp/blob/master/training/pos_finetuning.ipynb
    # encodes labels in the first token position of each word
    def encode_tags_first(self, pos_tags, offset_mapping):
        labels = [self.label2id(tag) for tag in pos_tags]
        encoded_labels = np.ones(len(offset_mapping), dtype=int) * self.pad_token_id

        for i in range(1, len(offset_mapping)):
            if self.ignore_mapping(offset_mapping[i-1]) or offset_mapping[i-1][-1] != offset_mapping[i][0]:
                if not self.ignore_mapping(offset_mapping[i]):
                    try:
                        encoded_labels[i] = labels.pop(0)
                    except(IndexError):
                        return None
        
        if len(labels) > 0:
            return None

        return encoded_labels.tolist()

    # encodes labels in the last token position of each word
    def encode_tags_last(self, pos_tags, offset_mapping):
        labels = [self.label2id(tag) for tag in pos_tags]
        encoded_labels = np.ones(len(offset_mapping), dtype=int) * self.pad_token_id

        for i in range(1, len(offset_mapping) - 1):
            
            if offset_mapping[i][1] != offset_mapping[i+1][0]:
                if not self.ignore_mapping(offset_mapping[i]):
                    try:
                        encoded_labels[i] = labels.pop(0)
                    except(IndexError):
                        return None
        
        if len(labels) > 0:
            return None

        return encoded_labels.tolist()
    
    # prediction mask
    def prediction_mask(self, offset_mapping):
        prediction_mask = np.ones(len(offset_mapping), dtype=int) * self.pad_token_id

        for i in range(1, len(offset_mapping) - 1):
            
            if offset_mapping[i][1] != offset_mapping[i+1][0]:
                if not self.ignore_mapping(offset_mapping[i]):
                    try:
                        prediction_mask[i] = 1
                    except(IndexError):
                        return None

        return prediction_mask.tolist()
    
    def ignore_mapping(self, mapping):
        return mapping[0] == mapping[1]

    def batch_decode(self, logits: torch.Tensor) -> List[List[str]]:
        """ `logits` has shape (N, L) and dtype float. """
        return [[self.id2label(sid) for sid in sequence if sid != self.pad_token_id] for sequence in logits]

    def to(self, device: Union[str, "torch.device"]) -> "SequenceTokenizer":
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

    @property
    def O_id(self) -> int:
        return self.label2id("O")
