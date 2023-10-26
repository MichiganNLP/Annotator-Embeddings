import random
import json
import argparse
from transformers import PreTrainedTokenizerBase
from typing import Any, Callable, NoReturn, Tuple, Union
from src.dataset import Seq2SeqDataset, EncoderDataset
from src.tokenization import SequenceTokenizer
from src.utils.utils import Task


def _read_in_data(args: argparse.Namespace) -> Tuple[dict, dict, dict]:
    with open(args.train_data_path) as file:
        train_data = json.load(file)
    with open(args.dev_data_path) as file:
        val_data = json.load(file)
    with open(args.test_data_path) as file:
        test_data = json.load(file)
    return train_data, val_data, test_data

def _random_sample(instances, annotated_ids, pool_size):
    pool = [ins["id"] for ins in instances if ins["id"] not in annotated_ids]
    return random.sample(pool, k=min(pool_size, len(pool)))


def model_type_to_dataset(model_type: str) -> Union[Seq2SeqDataset, EncoderDataset]:  # noqa
    if "t5" in model_type:
        return Seq2SeqDataset
    elif "bert" in model_type:
        return EncoderDataset
    elif "random" in model_type:
        return EncoderDataset

def get_decoder_tokenizer(model_type: str, decoder_tokenizer: Union[PreTrainedTokenizerBase, dict], \
    task: Task) -> Union[PreTrainedTokenizerBase, SequenceTokenizer]:
    if "bert" in model_type or "random" in model_type:
        assert isinstance(decoder_tokenizer, dict)
        return decoder_tokenizer[task]
    elif "t5" in model_type:
        assert isinstance(decoder_tokenizer, PreTrainedTokenizerBase)
        return decoder_tokenizer
    else:
        raise ValueError(f"tokenizer for {model_type} not supported.")
