
PREDICTION_MASK = {
    "pos-tags-s": True,
    "entities-recognition": True,
    "sentiment": False,
    "hate_speech": False,
    "pejorative": False,
    "certainty": False,
    "indirect_ans": False,
    "emotion": False,
    "abuse": False,
    "hs_brexit": False,
    "offensive": False,
    "humor": False,
    "toxic_score": False,
}

NUM_LABEL_PLUS_ONE = {
    "pos-tags-s": True,
    "entities-recognition": True,
    "sentiment": False,
    "hate_speech": False,
    "pejorative": False,
    "certainty": False,
    "indirect_ans": False,
    "emotion": False,
    "abuse": False,
    "hs_brexit": False,
    "offensive": False,
    "humor": False, 
    "toxic_score": False,
}

ANSWER_MINUS_ONE = {
    "pos-tags-s": False,
    "entities-recognition": False,
    "sentiment": True,
    "hate_speech": True,
    "pejorative": True,
    "certainty": True,
    "indirect_ans": True,
    "emotion": True,
    "abuse": True,
    "hs_brexit": True,
    "offensive": True,
    "humor": True,
    "toxic_score": True,
}