from typing import Tuple, List
import random
from string import Template

import pandas as pd
import numpy as np

import random
import pandas as pd
from sklearn.datasets import make_regression

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from io import StringIO
from contextlib import redirect_stdout

from gadget import Calculator
from data_iterators.iterator import DataIterator
from names_dataset import NameDataset
import inflect


def make_dataset(col_names):
    n_samples = random.randint(100, 1000)
    n_features = random.randint(4, 10)
    n_informative = random.randint(1, min(n_features, 5))
    bias = 2 ** random.uniform(-10, 10)
    effective_rank = random.randint(1, n_features)
    tail_strength = random.uniform(0, 1)
    noise = 0

    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, bias=bias,
                           effective_rank=effective_rank, tail_strength=tail_strength, noise=noise)
    for i in range(X.shape[1]):
        X[:, i] *= 2 ** random.uniform(-2, 30)
    random.shuffle(col_names)
    df = pd.DataFrame(X, columns=col_names[:X.shape[1]])
    target_col_name = col_names[-1]
    df[target_col_name] = y

    filetype = random.choice(['csv', 'tsv', 'json'])  # TODO: choose csv, tsv, json
    filename = f'{col_names[-2]}.{filetype}'
    if filetype == 'csv':
        df.to_csv(filename, index=False)
        read_cmd = f'df = pd.read_csv("{filename}")'
    elif filetype == 'tsv':
        df.to_csv(filename, index=False, sep='\t')
        read_cmd = f'df = pd.read_csv("{filename}", sep="\t")'
    else:
        df.to_json(filename, orient='records')
        read_cmd = f'df = pd.read_json("{filename}")'

    task = f'''
Open the dataset {filename}. When trying to predict feature {target_col_name} what are relevant features?
'''

    description = f'''
Regression dataset with {n_samples} rows, {n_features} features.
There are {n_informative} informative features: {df.columns[:n_informative].tolist()}.
    '''

    model_types = [RandomForestRegressor, LinearRegression, Ridge, Lasso, DecisionTreeRegressor]
    model_type = random.choice(model_types)

    if model_type == RandomForestRegressor:
        model_cmd = f'model = RandomForestRegressor(n_estimators={random.randint(10, 100)}, max_depth={random.randint(1, 10)})'
    elif model_type == LinearRegression:
        model_cmd = f'model = LinearRegression()'
    elif model_type == Ridge:
        model_cmd = f'model = Ridge(alpha={random.uniform(0, 10)})'
    elif model_type == Lasso:
        model_cmd = f'model = Lasso(alpha={random.uniform(0, 10)})'
    elif model_type == SVR:
        model_cmd = f"model = SVR(kernel='{random.choice(['linear', 'poly', 'rbf', 'sigmoid'])}', C={random.uniform(0, 10)}, epsilon={random.uniform(0, 10)})"
    else:
        model_cmd = f"model = DecisionTreeRegressor(max_depth={random.randint(1, 10)})"

    if model_type == RandomForestRegressor or model_type == DecisionTreeRegressor:
        importance_cmd = f'feature_importance = model.feature_importances_'
    if model_type == LinearRegression or model_type == Ridge or model_type == Lasso:
        importance_cmd = f'feature_importance = abs(model.coef_)'
    else:
        importance_cmd = f'feature_importance = np.ones(shape={n_features})'
        importance_cmd = f'''
rfe = RFE(model, n_features_to_select={n_features // 2})
rfe.fit(X_train, y_train)
feature_importance = rfe.ranking_
'''

    steps = [
        read_cmd,  # load based on filetype
        # gather dataset information
        'print(df.head(2))',
        'print(df.describe())',
        'print(df.info())',
        # imputation/removal
        'df.fillna(df.mean(), inplace=True)',  # impute NaNs and infs with means
        'df = df.replace([np.inf, -np.inf], np.nan)',  # replace NaNs and infs with NaNs
        'df.dropna(inplace=True)',  # remove corrupted lines
        f'print(df.corr().sort_values("{target_col_name}").{target_col_name})',  # get correlation with target column
        # split df into train, test
        f'''
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
X_train = train.drop("{target_col_name}", axis=1)
y_train = train["{target_col_name}"]
''',
        model_cmd,  # instantiates the scikit model
        # train a `model` on train
        f'''
model.fit(X_train, y_train)
print("R^2:", model.score(test.drop("{target_col_name}", axis=1), test["{target_col_name}"]))''',
        importance_cmd,
        f'''
feature_importance = dict(zip(df.columns, feature_importance))
feature_importance = pd.Series(feature_importance).sort_values()
print(feature_importance)
    '''
    ]
    return task, steps, description


def make_viable_dataset(col_names):
    while True:
        try:
            task, steps, description = make_dataset(col_names)
            prompt = task

            f = StringIO()
            with redirect_stdout(f):
                for step in steps:
                    print(f"<gadget id=ipython>")
                    print(f'{step}')
                    print(f"</gadget>")
                    print(f'<output>')
                    exec(step)
                    print(f'</output>')
            chain = f.getvalue()

            final_answer = f'''
            <result>
            {description}
            </result>'''

            return prompt, chain, final_answer
        except Exception as e:
            print(e)


class DataMiningIterator(DataIterator):
    """Takes 0.4 s to generate one example"""

    def __init__(self):
        self.col_names = list(
            pd.read_csv('https://raw.githubusercontent.com/dwyl/english-words/master/words.txt', sep=' ', header=None,
                        encoding='utf-8')[0])

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, str, str]:
        return make_viable_dataset(self.col_names)


if __name__ == "__main__":
    ite = DataMiningIterator()
    for _ in range(5):
        print(next(ite))
