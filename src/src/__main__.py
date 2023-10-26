from __future__ import annotations

import argparse
import sys
import torch
import json
import numpy as np
np.object = object
np.bool = bool    

from transformers import HfArgumentParser
from src.parse_args import model_type_to_dataclass_types
from src.training_paradigm import LearnFromScratchParadigm, ActiveLearningParadigm, \
    ActiveLearningWithInteractionParadigm


def main() -> None:
    torch.autograd.set_detect_anomaly(True)

    model_type = sys.argv[1]
    dataclass_types = model_type_to_dataclass_types(model_type)

    # Don't pass a generator here as it misbehaves. See https://github.com/huggingface/transformers/pull/15758
    parser = HfArgumentParser(dataclass_types)

    args_in_dataclasses_and_extra_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    args_in_dataclasses, extra_args = args_in_dataclasses_and_extra_args[:-1], args_in_dataclasses_and_extra_args[-1]

    extra_args.remove(model_type)
    assert not extra_args, f"Unknown arguments: {extra_args}"

    args = argparse.Namespace(**{k: v for args in args_in_dataclasses for k, v in args.__dict__.items()})

    # Fix the random seed for PyTorch
    torch.manual_seed(args.seed)
    # Fix the random seed for NumPy
    np.random.seed(args.seed)

    with open(args.annotator_id_path, 'r') as f:
        annotator_ids = json.load(f)
    args.num_annotators = len(annotator_ids)

    with open(args.annotation_label_path, 'r') as f:
        annotation_labels = json.load(f)
    args.num_labels = len(annotation_labels)

    args.model_type = model_type

    Paradigm = None
    if args.training_paradigm == "learn_from_scratch":
        Paradigm = LearnFromScratchParadigm
    elif args.training_paradigm == "active_learning":
        Paradigm = ActiveLearningParadigm
    elif args.training_paradigm == "active_learning_with_interaction":
        Paradigm = ActiveLearningWithInteractionParadigm
    training_paradigm = Paradigm(args)
    training_paradigm.train_and_test()


if __name__ == "__main__":
    main()
