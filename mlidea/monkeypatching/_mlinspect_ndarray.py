"""
Monkey patching for numpy
"""
from typing import Any

import numpy
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import PrivateAttr


class MlinspectList(list):
    """A list wrapper that can store mlinspect annotations"""
    _mlinspect_dag_node = None
    _mlinspect_annotation = None


class MlinspectDict(dict):
    """A list wrapper that can store mlinspect annotations"""
    _mlinspect_dag_node = None
    _mlinspect_annotation = None

class MlinspectNdarray(numpy.ndarray):
    """
    A wrapper for numpy ndarrays to store our additional annotations.
    See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    """

    def __new__(cls, input_array, _mlinspect_dag_node=None, _mlinspect_annotation=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj._mlinspect_dag_node = _mlinspect_dag_node
        obj._mlinspect_annotation = _mlinspect_annotation
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._mlinspect_dag_node = getattr(obj, '_mlinspect_dag_node', None)
        self._mlinspect_annotation = getattr(obj, '_mlinspect_annotation', None)

    def ravel(self, order='C'):
        result = super().ravel(order)
        assert isinstance(result, MlinspectNdarray)
        result._mlinspect_dag_node = self._mlinspect_dag_node  # pylint: disable=protected-access
        result._mlinspect_annotation = self._mlinspect_annotation  # pylint: disable=protected-access
        return result

class MlideaChromaVectorStoreRetrieverPlaceHolder(VectorStoreRetriever):
    retrieval_corpus_X: Any
    retrieval_corpus_y: Any
    embedding: Any
    _mlinspect_dag_node: Any = PrivateAttr(None)  # Why this is necessary: https://stackoverflow.com/a/75712642
    precomputed_result: Any
    def __init__(self, retrieval_corpus_X: list[str], retrieval_corpus_y: list[dict[str, any]], embedding: Embeddings,
                 **kwargs: any):
        # TODO: This is ugly, but we want a placeholder class can be used as part of the declarative langchain
        #  definition without actually executing something expensive. There is for sure a better way to do this,
        #  but this can be cleaned up later
        super().__init__(**kwargs, vectorstore=Chroma(), tags=None)
        self.retrieval_corpus_X = retrieval_corpus_X
        self.retrieval_corpus_y = retrieval_corpus_y
        self.embedding = embedding

    def invoke(self, *args: Any, **kwargs: Any):
        return self.precomputed_result.invoke(*args, **kwargs)

    def precompute_results(self):
        self.precomputed_result = Chroma.from_texts(texts=self.retrieval_corpus_X, metadatas=self.retrieval_corpus_y,
                                                    embedding=self.embedding).as_retriever()
        # FIXME: Not sure yet how we can replace the individual invokes with a batch invoke or how else
        #  we should handle this. Should we complete replace the langchain batch implementation?

    def as_retriever(self):
        return self

    def columns(self):
        return ["texts", *list(self.retrieval_corpus_y[0].keys())]
