import os
from typing import List, Dict

import pandas as pd

teabreac_dir = "../learn_to_prime/training/data"


def _get_answer(ans_object: List[Dict[str, str]]) -> str:
    # list([{'number': '121', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': []}])
    return " ".join(ans['number'] if ans['number']
                    else " ".join(ans['date'].values()) if "".join(ans['date'].values())
                    else " ".join(ans['spans']) for ans in ans_object)


tea_train = pd.read_json(os.path.join(teabreac_dir, "teabreac_v1.0_multihop_qa_train.jsonl"), lines=True)
orig_len = len(tea_train)

tea_train = tea_train[tea_train["context_text"].apply(lambda text: len(text) < 1000)]
print("Reduced to %s percent of original samples by length." % (len(tea_train) / orig_len) * 100)

tea_train["context_text"] = tea_train["context_text"].apply(lambda c: c.replace(" -> ", ". "))
tea_train["answers_text"] = tea_train["answers_objects"].apply(lambda ans_obj: _get_answer(ans_obj))

tea_val = pd.read_json(os.path.join(teabreac_dir, "teabreac_v1.0_multihop_qa_dev.jsonl"), lines=True)
tea_val["context_text"] = tea_val["context_text"].apply(lambda c: c.replace(" -> ", ". "))
tea_val["answers_text"] = tea_val["answers_objects"].apply(lambda ans_obj: _get_answer(ans_obj))

tea_val = tea_val[tea_val["answers_text"].apply(lambda ans: ans is not None and isinstance(ans, str) and len(ans.strip()) > 0)]
