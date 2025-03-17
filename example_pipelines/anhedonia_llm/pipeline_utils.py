"""
Some useful utils for the project
"""
from __future__ import annotations
# import getpass
import os
import random
import time
from inspect import cleandoc

import numpy
import pandas
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sklearn.exceptions import NotFittedError

from example_pipelines.anhedonia_llm.gensim_wrapper import W2VTransformer
from mlidea.utils import get_project_root


class MyW2VTransformer(W2VTransformer):
    """Some custom w2v transformer."""

    def partial_fit(self, X):
        super().partial_fit([X])

    def fit(self, X, y=None):
        X = X.iloc[:, 0].tolist()
        return super().fit([X], y)

    def transform(self, words):
        words = words.iloc[:, 0].tolist()
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        vectors = []
        for word in words:
            if word in self.gensim_model.wv:
                vectors.append(self.gensim_model.wv[word])
            else:
                vectors.append(numpy.zeros(self.size))
        return numpy.reshape(numpy.array(vectors), (len(words), self.size))


def get_langchain_rag_binary_classification(classes, retriever):
    # Prompt template taken from skllm
    FEW_SHOT_CLF_PROMPT_TEMPLATE = cleandoc(f"""
        You will be provided with the following information:
        1. An arbitrary text sample. The sample is delimited with triple backticks.
        2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.
        3. Examples of text samples and their assigned categories. The examples are delimited with triple backticks. The assigned categories are enclosed in a list-like structure. These examples are to be used as training data.

        Perform the following tasks:
        1. Identify to which category the provided text belongs to with the highest probability.
        2. Assign the provided text to that category.
        3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.

        List of categories: {classes}

        Training data:
        {{context}}

        Text sample: ```{{question}}```

        Your JSON response:
        """)
    prompt = ChatPromptTemplate.from_template(FEW_SHOT_CLF_PROMPT_TEMPLATE)
    # Make sure this pipeline is executable in Github Actions, but also uses a real LLM locally if needed
    # if os.getenv("GITHUB_ACTIONS") != "true":
    # FIXME: Quick and ugly way to develop w/o unneeded LLM calls
    if False and os.getenv("GITHUB_ACTIONS") != "true":  # pylint: disable=condition-evals-to-constant
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    else:
        llm = FakeListChatModel(responses=[f"{{\"label\": \"{classes[0]}\"}}""", f"{{\"label\": \"{classes[1]}\"}}"""])

    def format_docs(docs):
        retrieved_formatted = "\n\n".join(
            f"```{doc.page_content}```\nassigned category: ['{doc.metadata['label']}']" for doc in
            docs)  # Here we can also use label!
        # TODO: Create a copy of these functions and then modify the metadata in the llm mislabel pipeline to
        #  also contain the document id in the metadata. We don't have access to the id here, but as metadata
        #  we can access a static variable with the index datastructure
        #  We can infer the query id based on the order that we call the langchain with, I guess these are not
        #  randomly shuffled or anything like that. But we should verify this.
        #  However, we might not always want to have this additional id assign overhead, but maybe its fine
        return retrieved_formatted

    def format_response(json_object):
        if not isinstance(json_object, dict) or not "label" in json_object:
            return random.choice([0, 1])
        assigned_label = json_object["label"]
        if assigned_label not in classes:
            return random.choice([0, 1])
        return classes.index(assigned_label)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | JsonOutputParser()
            | format_response
    )
    return rag_chain


def wait_llm_call(partial, test_df_for_size_calculation):
    # this needs a wait because it is cache-only in our experiments
    # groq mixtral: https://groq.com/?model_id=mixtral-8x7b-32768
    # 500T/s
    # For mistral: "A word is generally 2-3 tokens" https://replicate.com/mistralai/mistral-7b-instruct-v0.1/api
    # Let us calculate with 3 to be conservative
    # how many words do we have here?
    # avg_word_count = sum(map(lambda x: len(x.split()), test['tweet'].tolist())) / len(test['tweet'].tolist())
    # print(avg_word_count)
    # 12.95
    # calculation: (batch_size * 13 * 3) / 500
    # TODO: slow/fast LLM at some point in the future?
    if isinstance(test_df_for_size_calculation, pandas.DataFrame):
        test_size = test_df_for_size_calculation.shape[0]
    elif isinstance(test_df_for_size_calculation, list):
        test_size = len(test_df_for_size_calculation)
    else:
        test_size = test_df_for_size_calculation.size
    realistic_wait_time_calculation = (test_size * 13 * 3) / 500
    langchain_start = time.time()
    result = partial()
    langchain_end = time.time()
    time.sleep(max(realistic_wait_time_calculation - (langchain_end - langchain_start) / 1000, 0))
    additional_sleep = max(realistic_wait_time_calculation - (langchain_end - langchain_start) / 1000, 0)
    print(f"Sleeping an additional {additional_sleep}s to simulate real API call when cache was hit "
          f"({realistic_wait_time_calculation} - {(langchain_end - langchain_start) / 1000})!")
    # langchain batch using ChatGPT 3: 29,649606943130493s
    # estimated wait time for it: 7.8s

    return result


def initialize_environment():
    seed = 42
    numpy.random.seed(seed)
    random.seed(seed)
    set_llm_cache(SQLiteCache(database_path=f"{str(get_project_root())}/example_pipelines/anhedonia_llm/"
                                            f"offline/.langchain.db"))
    # os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
    os.environ["OPENAI_API_KEY"] = "not_required_because_of_sqlite_caching"
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
