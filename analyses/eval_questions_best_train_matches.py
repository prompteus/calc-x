from typing import Set

import numpy as np
from datasets import load_dataset
# from gensim.parsing.preprocessing import preprocess_string
from gensim.utils import simple_preprocess
import pandas as pd
from tqdm import tqdm

datasets = {
    "gsm": load_dataset("MU-NLPC/Calc-gsm8k"),
    "aqua": load_dataset("MU-NLPC/Calc-aqua_rat"),
    "ape": load_dataset("MU-NLPC/Calc-ape210k"),
    "mathqa": load_dataset("MU-NLPC/Calc-math_qa")
}

dataset_to_keys = {
    "gsm": {
        "question_key": "question",
        "answer_key": "answer",
        "chain_key": "chain",
    },
    "ape": {
        "question_key": "question_english_mt",
        "answer_key": "equation",
        "chain_key": "chain",
    },
    "mathqa": {
        "question_key": "problem",
        "answer_key": "rationale",
        "chain_key": "chain",
    },
    "aqua": {
        "question_key": "question",
        "answer_key": "rationale",
        "chain_key": "chain",
    },
}

train_datasets = ["gsm", "aqua"]
val_datasets = ["gsm", "aqua", "ape", "mathqa"]

# train_datasets = ["gsm"]
# val_datasets = ["gsm"]
val_dataset_firstn = 100

# manually create val split in gsm that we use in training
datasets["gsm"]["validation"] = datasets["gsm"]["test"].select(range(val_dataset_firstn))


def score(wordset1: Set[str], wordset2: Set[str]) -> float:
    return len(wordset1.intersection(wordset2)) / len(wordset1)

for train_d_id in train_datasets:
    datasets[train_d_id]["train"] = datasets[train_d_id]["train"].map(
        lambda row: {"question_wordset": set(simple_preprocess(row[dataset_to_keys[train_d_id]["question_key"]]))}
    )

best_matches = []

for val_d_id in val_datasets:
    for val_q in tqdm(datasets[val_d_id]["validation"][dataset_to_keys[val_d_id]["question_key"]][:val_dataset_firstn],
                      desc=val_d_id):
        val_q_set = set(simple_preprocess(val_q))
        best_match_score = 0
        for train_d_id in train_datasets:
            qs_scores = np.array([score(val_q_set, train_q_set)
                                  for train_q_set in datasets[train_d_id]["train"]["question_wordset"]])
            best_idx = qs_scores.argmax()
            best_score = qs_scores.max()
            best_matching_q = datasets[train_d_id]["train"][dataset_to_keys[train_d_id]["question_key"]][best_idx]
            print("Match score %s\n  val question: '%s', \ntrain question: '%s'" % (best_score, val_q, best_matching_q))
            best_matches.append([val_q, best_matching_q, val_d_id, train_d_id, best_score])

best_matches_df = pd.DataFrame(best_matches, columns=["Q_val", "Q_train", "dataset_val", "dataset_train", "score"])
best_matches_df.to_csv("best_matches_df.csv")
print(best_matches_df[best_matches_df.score > 0.8].groupby(["dataset_val", "dataset_train"]).count().score)
# dataset_val  dataset_train
# ape          ape               54
#              aqua              13
#              gsm               10
#              mathqa            11
# aqua         ape                1
#              aqua              26
#              mathqa             8
# mathqa       ape               12
#              aqua             100
#              gsm                4
#              mathqa            91

# constraints on training datasets:
#   TRAIN       NOT VAL
#   gsm         -
#   aqua        mathqa, aqua
#   ape         ape
#   mathqa      mathqa, aqua

# further eval datasets:
# juletxara/mgsm -- from FLAN
# MU-NLPC/Calc-mawps, MU-NLPC/Calc-svamp, EleutherAI/asdiv -- from Llama