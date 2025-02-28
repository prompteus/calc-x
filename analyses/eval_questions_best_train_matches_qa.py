from typing import Set

import numpy as np
# from gensim.parsing.preprocessing import preprocess_string
from gensim.utils import simple_preprocess
import pandas as pd
from tqdm import tqdm

import cot

datasets = cot.Collection(["commonsense_qa", "strategy_qa", "worldtree", "open_book_qa"])


train_datasets = ["commonsense_qa", "strategy_qa", "worldtree", "open_book_qa"]
val_datasets = ["commonsense_qa", "worldtree", "open_book_qa"]

# train_datasets = ["gsm"]
# val_datasets = ["gsm"]
val_dataset_firstn = 100

def score(wordset1: Set[str], wordset2: Set[str]) -> float:
    return len(wordset1.intersection(wordset2)) / len(wordset1)


for train_d_id in train_datasets:
    datasets[train_d_id]["train"] = datasets[train_d_id]["train"].map(
        lambda row: {"question_wordset": set(simple_preprocess(row["question"]))}
    )

best_matches = []

for val_d_id in val_datasets:
    for val_q, val_a in tqdm(zip(datasets[val_d_id]["validation"]["question"][:val_dataset_firstn],
                                 datasets[val_d_id]["validation"]["answer"][:val_dataset_firstn]), desc=val_d_id):
        val_q_set = set(simple_preprocess(val_q))
        best_match_score = 0
        for train_d_id in train_datasets:
            qs_scores = np.array([score(val_q_set, train_q_set)
                                  for train_q_set in datasets[train_d_id]["train"]["question_wordset"]])
            best_idx = qs_scores.argmax()
            best_score = qs_scores.max()
            best_matching_q = datasets[train_d_id]["train"]["question"][best_idx]
            matching_a = datasets[train_d_id]["train"]["answer"][best_idx]

            print("Match score %s\n  val question: '%s', answer: `%s` \ntrain question: '%s', answer: '%s'"
                  % (best_score, val_q, val_a, best_matching_q, matching_a))
            best_matches.append([val_q, best_matching_q, matching_a, val_a, val_d_id, train_d_id, best_score])

best_matches_df = pd.DataFrame(best_matches,
                               columns=["Q_val", "Q_train", "A_val", "A_train", "dataset_val", "dataset_train", "score"])
best_matches_df.to_csv("best_matches_df.csv")
print(best_matches_df[best_matches_df.score > 0.8].groupby(["dataset_val", "dataset_train"]).count().score)

# dataset_val     dataset_train
# commonsense_qa  commonsense_qa     6
#                 open_book_qa       1
# open_book_qa    open_book_qa      12
#                 strategy_qa        2
#                 worldtree          7
# worldtree       open_book_qa       2
#                 worldtree          2

# dataset_test    dataset_train
# commonsense_qa  commonsense_qa     9
# open_book_qa    commonsense_qa     1
#                 open_book_qa       7
#                 worldtree          3
# worldtree       worldtree         10
