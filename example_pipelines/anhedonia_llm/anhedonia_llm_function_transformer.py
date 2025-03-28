from functools import partial
import pandas as pd
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # pylint: disable=no-name-in-module
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import FunctionTransformer
import asyncio
import nest_asyncio
from googletrans import Translator
import time


from example_pipelines.anhedonia_llm.pipeline_utils import initialize_environment, \
    wait_llm_call, get_langchain_rag_binary_classification
from mlidea.utils import get_project_root

nest_asyncio.apply()
translator = Translator()

initialize_environment()

initial_start = time.time()

boolean_dictionary = {True: 'anhedonia', False: 'regular'}

def load_train_data(user_location, tweet_location, included_countries):
    # pylint: disable=redefined-outer-name
    users = pd.read_parquet(user_location)
    users = users[users['country'].isin(included_countries)]
    tweets = pd.read_parquet(tweet_location)
    return users.merge(tweets, on='user_id')


def weak_labeling(data):
    data['anhedonia'] = ((data['tweet'].str.contains('(0|no|zero) (motivation|interest)', regex=True)
                          | data['tweet'].str.contains('(lose|losing|lost).{0,15} (interest|pleasure|motivation)', regex=True))
                         & ~(data['tweet'].str.contains('recover.{0,15} from (0|no|zero) (motivation|interest)', regex=True))
                         )
    # Addition to help the llm
    data['label'] = data['anhedonia'].replace(boolean_dictionary)
    return data

user_location = f'{str(get_project_root())}/example_pipelines/anhedonia_ml/data/users.pqt'
tweet_location = f'{str(get_project_root())}/example_pipelines/anhedonia_ml/data/tweets.pqt'
nl_be = ['NL', 'BE']
test_location = f'{str(get_project_root())}/example_pipelines/anhedonia_ml/data/expert_labeled.pqt'

train = load_train_data(user_location, tweet_location, included_countries=nl_be)
train = weak_labeling(train)
test = pd.read_parquet(test_location)

def translate(series):
    if isinstance(series, pd.Series):
        series = [result.text for result in asyncio.run(translator.translate(series.to_list()))]
    else:
        series = [result.text for result in asyncio.run(translator.translate(series))]
    return series

translate_transformer = FunctionTransformer(translate)
train['tweet'] = translate_transformer.fit_transform(train['tweet'])
test['tweet'] = translate_transformer.transform(test['tweet'])

# pylint: disable=no-member
vectorstore = Chroma.from_texts(texts=train['tweet'].to_list(), metadatas=train[['label']].to_dict('records'),
                                embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

# remember to reset this if we want to update the index on more than the first test set prediction call
rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

y_predicted = wait_llm_call(partial(rag_chain.batch, test['tweet'].to_list()), test)
y_test_binarized = label_binarize(test['anhedonia'], classes=[True, False])
accuracy = accuracy_score(y_test_binarized, y_predicted)
print(f'Test accuracy is: {accuracy}')

initial_end = time.time()
print(f"initial: {(initial_end - initial_start) * 1000}")
