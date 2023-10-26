from __future__ import annotations

import json
import os
from transformers import AutoTokenizer
from src.tokenization import SequenceTokenizer, MultiChoiceTokenizer
from typing import List, Tuple


class Task:

    def __init__(self, task: str, id: int) -> None:
        self._name = task
        self._id = id
    
    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return self._name == other.name, other.id


class Tasks:

    def __init__(self, tasks: List[str] | List[Tuple[int, str]]) -> None:
        self._tasks = []
        first_val = next(iter(tasks), None)
        if isinstance(first_val, str):
            for i, task in enumerate(tasks):
                self._tasks.append(Task(task, i))
        elif isinstance(first_val, tuple):
            for i, task in tasks:
                self._tasks.append(Task(task, i))
        else:
            raise ValueError("Wrong initialization for Tasks. Please check the input of the tasks.")
    
    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, i: int) -> Task:
        return self._tasks[i]



def sequence_output_tokens(data, task: Task):
    possible_output_toks = []
    for instance in data:
        possible_output_toks.extend(instance[task.name])
    return list(set(possible_output_toks))

def multichoice_output_tokens(annotation_labels, task: Task = None):
    return list(set(annotation_labels))

def set_up_tokenizers(args, data=None, annotation_labels=None):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # add task separation special tokens
    tokenizer.add_special_tokens({'eos_token': '</s>'})

    # add new tokens
    if args.model_type == "t5":
        assert len(args.tasks) == len(args.add_output_tokens) == len(args.constrained_decoding) == len(args.constrained_token_file_paths)
        
        toks, constrained_toks = [], []
        for add_flag, task, cons_flag, fn in zip(args.add_output_tokens, args.tasks, \
            args.constrained_decoding, args.constrained_token_file_paths):
            if add_flag:
                toks.extend(sequence_output_tokens(data, task))
            if cons_flag:
                if os.path.exists(fn):
                    with open(fn, 'r') as f:
                        data = json.load(fn)
                    assert isinstance(data, list), "Wrong data format, should be a list in the json file"
                    constrained_toks.extend(data)
                else:
                    # by default we add the output tokens as the constrained tokens
                    constrained_toks.extend(sequence_output_tokens(data, task))

        new_tokens = set(toks) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))
        
        return tokenizer, tokenizer, set(constrained_toks)
    
    elif args.model_type in {"bert", "berttokenclassifier", \
                             "bertmultichoice", "random",
                             "roberta-multichoice", "deberta-multichoice"}:
        toks, gen_tokenizers = [], {}
        for flag, task in zip(args.add_output_tokens, args.tasks):
            if flag:
                if args.model_type in {"bert", "berttokenclassifier"}:
                    task_toks = sequence_output_tokens(data, task)
                    gen_tokenizer = SequenceTokenizer(task_toks, args.pad_token_id)
                elif args.model_type in {"bertmultichoice", "random",
                                         "roberta-multichoice", "deberta-multichoice"}:
                    task_toks = multichoice_output_tokens(annotation_labels, task)
                    gen_tokenizer = MultiChoiceTokenizer(task_toks, args.pad_token_id)
                gen_tokenizers[task] = gen_tokenizer
                toks.extend(task_toks)
        new_tokens = set(toks) - set(tokenizer.vocab.keys())

        return tokenizer, gen_tokenizers, set(toks)
    
    else:
        raise RuntimeError(f"Cannot set up tokenizer. Model type {args.model_type} not supported.")

    