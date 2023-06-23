import re


def preprocessing_factory(tokenizer, question_key, answer_key, chain_key):
    def preprocess_fn(sample):
        inputs = tokenizer(sample[question_key], truncation=True)
        labels = tokenizer(text_target=sample[chain_key], truncation=True)
        return {
            "question": sample[question_key],
            "answer": sample[answer_key],
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels_old": labels.input_ids,
            "chain": sample[chain_key],
        }

    return preprocess_fn


dataset_to_keys = {
    "Calc-ape210k": {
        "question_key": "question_english_mt",
        "answer_key": "equation",
        "chain_key": "chain",
    },
    "Calc-gsm8k": {
        "question_key": "question",
        "answer_key": "answer",
        "chain_key": "chain",
    },
    "Calc-math_qa": {
        "question_key": "problem",
        "answer_key": "rationale",
        "chain_key": "chain",
    },
    "Calc-aqua_rat": {
        "question_key": "question",
        "answer_key": "rationale",
        "chain_key": "chain",
    },
}


def gsm8k_prep(sample):
    res = re.sub('<<[^>]*>>','', sample['answer'])
    split = res.split('\n')
    label = '\n'.join(split[:-1])
    result_numbers = re.findall('\d+', split[-1])
    assert len(result_numbers) == 1
    label += f'\n. The final result is {result_numbers[0]}.'
    return label

def ape210k_prep(sample):
    result_numbers = re.findall('<result>(.*?)</result>', sample['chain'])
    assert len(result_numbers) == 1
    label = sample['answer']+f'\n. The final result is {result_numbers[0]}.'
    return label

def math_qa_prep(sample):
    answer = sample['answer'].strip('"')
    patterns = [
        '(?:option\s:\s[a-z])',
        '(?:option\s[a-z]$)',
        '(?:answer\s:\s[a-z]$)',
        '(?:answer : option [a-z])',
        '(?:imo option [a-z])',
        '(?:imo [a-z])',
        '(?:the answer is [a-z])',
        # '(?:[a-z]$)',
        '(?:choice [a-z])',
        '(?:answer will be [a-z])',
        '(?:answer is option [a-z])',
        '(?:the answer is \, therefore \, [a-z])',
        '(?:ans\soption [a-z])',
        
        
    ]
    pattern = '|'.join(patterns)
    matches = re.findall(pattern, answer)
    
    if(len(matches) == 1):
        new = f". The final result is {sample['options'][matches[0][-1]].rstrip()}."
        return new.join(answer.rsplit(matches[0], 1))
        
        # return answer.replace(matches[0], f". The final result is {sample['options'][matches[0][-1]].rstrip()}.")
        
    patterns = [
        '(?:correct answer is ([a-z]).*$)',
        '(?:([a-z]) is the answer$)',
        '(?:\' ([a-z]) \' is the answer$)',
        '(?:\s([a-z])$)',
        '(?:answer is ([a-z])\s\.)',
        '(?:answer is ([a-z])\s\=.*)',
        '(?:\s([a-z])\s\)$)',
        '(?:([a-z]) is thus the correct answer)',
        '(?:option \' ([a-z]) \')',
        '(?:answer :\s([a-z])\s.*)',
        '(?:ans\s\( ([a-z]) \))',
        '(?:so answer is ([a-z]) .*)',
        '(?:answer ([a-z]) \.$)',
        '(?:answer \= \( ([a-z]) \) \.$)',
        '(?:([a-z]) is the correct answer \.$)',
        '(?:hence answer will be \( ([a-z]) \) .*)',
        '(?:so our answer is \( ([a-z]) \) .*)',
        '(?:answer ([a-z]) for me)',
        '(?:correct answer ([a-z]).*)',
        '(?:thus , i think ([a-z]) would be the correct answer.*)',
        '(?:answer ([a-z]) [\d%+])',
        '(?:option ([a-z]))',
        '(?:([a-z]) is correct)',
        '(?:answer \: ([a-z]))',
    ]
    for pattern in patterns:
        matches = re.search(pattern, answer)
        if(matches):
            new = f". The final result is {sample['options'][matches.group(1)]}."
            #Replace only last occurence of match
            return new.join(answer.rsplit(matches[0], 1))
        
            # return answer.replace(matches[0], f". The final result is {sample['options'][matches.group(1)]}.")
        
    patterns = [
        '(?:^([a-z]) \$)', 
        '(?:^solution : ([a-z]))',
        '(?:^([abcde])\s)',
        '(?:^answer is \( ([a-z]) \))',
    ]
    for pattern in patterns:
        matches = re.search(pattern, answer)
        if(matches):
            answer = answer.replace(matches[0],"")
            answer = answer+f". The final result is {sample['options'][matches.group(1)]}."
            
            return answer
    patterns = [
        '(?:correct option : (\d+))', 
        '(?:ans\s.\s(\d+))', 
        
    ]
    for pattern in patterns:
        matches = re.search(pattern, answer)
        if(matches):
            new = f". The final result is {matches.group(1)}."
            return new.join(answer.rsplit(matches[0], 1))
            
            # return answer.replace(matches[0], f". The final result is {matches.group(1)}.")
    
    raise Exception() 
    
def aqua_rat_prep(sample):
    res = re.sub('<<[^>]*>>','', sample['answer'])
    split = res.split('\n')
    label = '\n'.join(split[:-1])
    split[-1] = split[-1].rstrip('.')
    result_letters = re.search('(([ABCDE])$)', split[-1])
    # print(result_letters.group(1))
    if(not result_letters):
        # print(split[-1])
        raise Exception()
    result_letters = [result_letters.group(1)]
    assert len(result_letters) == 1, result_letters
    matches = []
    for option in sample['options']:
        if result_letters[0] in option:
            matches.append(option)
    assert len(matches) == 1, matches
    label += f'\n. The final result is {matches[0][2:]}.'
    return label



def labeling_factory(tokenizer, labeler_fn, question_key):
    def preprocess_fn(sample):
        inputs = tokenizer(sample[question_key], truncation=True)
        try:
            text_label = labeler_fn(sample)
            labels = tokenizer(text_target=text_label, truncation=True)  
            return {
                "question":sample["question"],
                "answer":text_label,
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": labels.input_ids,
                
            }
        except:
            return {
                "question":sample["question"],
                "answer":sample["answer"],
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": None,
            }
            
    return preprocess_fn
            
                  
dataset_to_labeler = {
    'Calc-ape210k':ape210k_prep,
    'Calc-gsm8k':gsm8k_prep,
    'Calc-math_qa':math_qa_prep,
    'Calc-aqua_rat':aqua_rat_prep,
}
