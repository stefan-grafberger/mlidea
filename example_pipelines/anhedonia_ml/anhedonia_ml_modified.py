import time

import pandas as pd
from scikeras.wrappers import KerasClassifier
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, label_binarize

from example_pipelines.anhedonia_ml.pipeline_utils import initialize_environment, create_model
from mlidea.utils import get_project_root

initialize_environment()

initial_start = time.time()

def load_train_data(user_location, tweet_location, included_countries):
    # pylint: disable=redefined-outer-name
    users = pd.read_parquet(user_location)
    users = users[users['country'].isin(included_countries)]
    tweets = pd.read_parquet(tweet_location)
    return users.merge(tweets, on='user_id')


def weak_labeling(data):
    data['anhedonia'] = ((data['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
                          | data['tweet'].str.contains('(lose|losing|lost).{0,15} (pleasure|motivation)', regex=True))
                         & ~(data['tweet'].str.contains('recover.{0,15} from (0|no|zero) (motivation|interest)', regex=True))
                         )
    return data


def encode_features():
    #model = SentenceTransformer('mrm8488/bert-tiny-finetuned-squadv2')  # model named is changed for time and computation gians :)
    model = SentenceTransformer('all-MiniLM-L6-v2')  # model from the tutorial page
    embedder = FunctionTransformer(lambda item: model.encode(item))  # pylint: disable=unnecessary-lambda
    preprocessor = ColumnTransformer(transformers=[('embedder', embedder, 'tweet')])
    return preprocessor

user_location = f'{str(get_project_root())}/example_pipelines/anhedonia_ml/data/users.pqt'
tweet_location = f'{str(get_project_root())}/example_pipelines/anhedonia_ml/data/tweets.pqt'
nl_be = ['NL', 'BE']
test_location = f'{str(get_project_root())}/example_pipelines/anhedonia_ml/data/expert_labeled.pqt'

train = load_train_data(user_location, tweet_location, included_countries=nl_be)
train = weak_labeling(train)
test = pd.read_parquet(test_location)

estimator = Pipeline([
    ('features', encode_features()),
    ('learner', KerasClassifier(model=create_model, epochs=3, batch_size=32, verbose=0,
                                hidden_layer_sizes=(9, 9,), loss="binary_crossentropy"))])

estimator.fit(train[['tweet']], label_binarize(train['anhedonia'], classes=[True, False]))
accuracy = estimator.score(test[['tweet']], label_binarize(test['anhedonia'], classes=[True, False]))
print(f'Test accuracy is: {accuracy}')

initial_end = time.time()
print(f"initial: {(initial_end - initial_start) * 1000}")
