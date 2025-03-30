"""
Monkey patching for sklearn
"""
from __future__ import annotations

import dataclasses
from functools import partial
from typing import cast, TypedDict

import gorilla
import numpy
import pandas
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore
from langchain_community import vectorstores as community_vectorstores
from langchain_community.embeddings import huggingface
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.beta.runnables.context import config_with_context
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.load.dump import dumpd
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import base, RunnableParallel, RunnableSequence
from langchain_core.runnables.config import (
    RunnableConfig,
    get_config_list, patch_config,
)
from langchain_core.runnables.utils import (
    Input, Output,
)

from mlidea.instrumentation._operator_types import FunctionInfo, OperatorType
from mlidea.execution._func_executor import capture_optimizer_info
from mlidea.execution._pipeline_executor import singleton
from mlidea.instrumentation._dag_node import DagNode, BasicCodeLocation, DagNodeDetails, OperatorContext, \
    CodeReference
from mlidea.instrumentation._operator_call_info import OperatorCallInfo
from mlidea.monkeypatching._mlinspect_ndarray import MlideaChromaVectorStoreRetrieverPlaceHolder, MlinspectList
from mlidea.monkeypatching._monkey_patching_utils import get_optional_code_info_or_none, \
    FunctionCallResult, add_dag_node, get_input_info, \
    add_test_data_dag_node, add_train_data_node, add_train_label_node, execute_patched_func_indirect_allowed, \
    execute_patched_func_no_op_id, wrap_in_mlinspect_array_if_necessary


class LangchainCallInfo:
    """ Contains info like lineno from the current Transformer so indirect utility function calls can access it """
    # pylint: disable=too-few-public-methods
    runnable_sequence_active: bool = False


call_info_singleton = LangchainCallInfo()


def execute_embedding_similarity_join(retrieval_corpus_X, retrieval_corpus_y, embedding, inputs: list[Input]):
    # pylint: disable=too-many-locals
    Chroma(collection_name=Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME).delete_collection()
    document_indices = [str(index) for index in range(len(retrieval_corpus_X))]
    if singleton.prov_enabled is False:
        retrieval_corpus_y = retrieval_corpus_y.copy()
        for row, index in zip(retrieval_corpus_y, document_indices):
            row['_metadata_ids'] = index
        filled_vectorstore = Chroma.from_texts(texts=retrieval_corpus_X, metadatas=retrieval_corpus_y,
                                               embedding=embedding, ids=document_indices
                                               ).as_retriever()
    else:
        # TODO: Make this more general, what if it isn't a dict with only one entry
        assert (hasattr(retrieval_corpus_X, "_mlinspect_provenance") and
                retrieval_corpus_X._mlinspect_provenance is not None)
                # and len(retrieval_corpus_X._mlinspect_provenance.items()) == 1)

        prov_str_dict = {}
        for prov_key, prov_value in list(retrieval_corpus_X._mlinspect_provenance.items()):
            prov_value_str_list = list(map(str, prov_value))  # pylint: disable=bad-builtin
            prov_str_dict[prov_key] = prov_value_str_list

        # all_prov_value_str = []
        # for row_prov_id in range(len(list(prov_str_dict.items())[0][1])):
        #     new_prov_value_str = ""
        #     for prov_key, prov_value in list(prov_str_dict.items()):
        #         new_prov_value_str += f"{prov_key}: {prov_value[row_prov_id]};"
        #     all_prov_value_str.append(new_prov_value_str)
        # TODO: Improve performance here
        metadatas_with_prov = []
        for row_metadatas, row_prov_id in zip(retrieval_corpus_y, range(len(retrieval_corpus_X))):
            new_dict_for_row = row_metadatas
            for prov_key, prov_value in list(prov_str_dict.items()):
                new_dict_for_row = new_dict_for_row | {prov_key: prov_value[row_prov_id]}
            metadatas_with_prov.append(new_dict_for_row)
        for row, index in zip(metadatas_with_prov, document_indices):
            row['_metadata_ids'] = index
        filled_vectorstore = Chroma.from_texts(texts=retrieval_corpus_X, metadatas=metadatas_with_prov,
                                               embedding=embedding,
                                               ids=document_indices).as_retriever()
    results = filled_vectorstore.batch(inputs)
    results = wrap_in_mlinspect_array_if_necessary(results)
    results._mlinspect_provenance = {}
    for prov_key in list(prov_str_dict.keys()):
        if singleton.prov_enabled is True:
            prov_ids = [[] for _ in results[0]]
            for result in results:
                for doc_index, doc in enumerate(result):
                    prov_ids[doc_index].append(int(doc.metadata[prov_key]))
            data_source, index_to_deduplicate = prov_key.rsplit('_', 1)
            index_to_deduplicate = int(index_to_deduplicate)
            prov_id_names = []
            for _ in results[0]:
                prov_id_names.append(f"{data_source}_{index_to_deduplicate}")
                index_to_deduplicate += 1

            for prov_id_name, prov_id_value in zip(prov_id_names, prov_ids):
                results._mlinspect_provenance[prov_id_name] = numpy.array(prov_id_value)
            results._mlinspect_provenance = (results._mlinspect_provenance | inputs._mlinspect_provenance)
    # Without this there are some re-execution issues
    results._mlinspect_vectorstore_ref = filled_vectorstore.vectorstore

    retrieval_index = numpy.zeros((len(results), 4), dtype=int)
    for prediction_index, result in enumerate(results):
        retrieval_index[prediction_index, :] = [doc.metadata['_metadata_ids'] for doc in result]
    results._mlinspect_retrieval_index = retrieval_index
    return results


@gorilla.patches(base.RunnableSequence)
class RunnableSequencePatching:
    """ Patches for sklearn """

    @gorilla.name('batch')
    @gorilla.settings(allow_hit=True)
    def patched_batch(self, inputs: list[Input], config: [list[RunnableConfig] | RunnableConfig | None] = None,
                      *, return_exceptions: bool = False, **kwargs: any):
        original = gorilla.get_original_attribute(base.RunnableSequence, 'batch')
        if call_info_singleton.runnable_sequence_active is False:
            def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
                """ Execute inspections, add DAG node """
                # pylint: disable=too-many-locals,no-member
                call_info_singleton.runnable_sequence_active = True
                # TODO: It is a bit unclear if it is better to use the vectorstore code location info here or the LLM
                #  info for the first part. For now, going wiht the vectorstore
                function_info_if_error = FunctionInfo('langchain_core.runnables.base', 'batch')
                retriever_with_info = self.find_retriever()
                input_info_a = get_input_info(retriever_with_info[3], caller_filename, lineno, function_info_if_error,
                                              optional_code_reference, optional_source_code)

                _, test_data_node, test_data_result = add_test_data_dag_node(
                    inputs, input_info_a.dag_node.operator_info.function_info, lineno, optional_code_reference,
                    optional_source_code, caller_filename)

                non_data_kwargs = {'chain': str(retriever_with_info[2].to_json()),
                                   'return_exceptions': return_exceptions, **kwargs}
                operator_context_rag = OperatorContext(OperatorType.RAG_JOIN,
                                                   input_info_a.dag_node.operator_info.function_info,
                                                   non_data_kwargs)
                operator_call_info_rag = OperatorCallInfo(operator_context_rag, [input_info_a.dag_node, test_data_node])

                processing_func = partial(RunnableSequencePatching.execute_retriever, retriever_with_info)
                optimizer_info, result = capture_optimizer_info(singleton, operator_call_info_rag,
                                                                processing_func, [retriever_with_info[3],
                                                                        test_data_result])
                description = "Embedding similarity join"
                dag_node_rag = DagNode(singleton.get_next_op_id(operator_call_info_rag),
                                       input_info_a.dag_node.code_location,
                                       operator_context_rag,
                                       DagNodeDetails(description, input_info_a.dag_node.details.columns,
                                                      optimizer_info),
                                       input_info_a.dag_node.optional_code_info,
                                       processing_func)
                function_call_result = FunctionCallResult(result)
                add_dag_node(dag_node_rag, [input_info_a.dag_node, test_data_node], function_call_result)
                embedding_join_result = function_call_result.function_result

                function_info = FunctionInfo('langchain_core.runnables.base', 'batch')

                processing_func_predict = partial(
                    RunnableSequencePatching.execute_langchain_batch_with_preexecuted_retriever,
                    self, config, return_exceptions)
                non_data_kwargs = {'prompt': str(self.get_prompts()), 'config': config,
                                   'return_exceptions': return_exceptions, **kwargs}
                operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info, non_data_kwargs)
                operator_call_info_predict = OperatorCallInfo(operator_context_predict,
                                                              [dag_node_rag])
                optimizer_info_predict, result_predict = capture_optimizer_info(singleton, operator_call_info_predict,
                                                                                processing_func_predict,
                                                                                        [embedding_join_result])
                dag_node_predict = DagNode(singleton.get_next_op_id(operator_call_info_predict),
                                           BasicCodeLocation(caller_filename, lineno),
                                           operator_context_predict,
                                           DagNodeDetails("LLM", [], optimizer_info_predict),
                                           get_optional_code_info_or_none(optional_code_reference,
                                                                          optional_source_code),
                                           processing_func_predict)
                function_call_result = FunctionCallResult(result_predict)
                add_dag_node(dag_node_predict, [dag_node_rag], function_call_result)
                llm_result = function_call_result.function_result

                call_info_singleton.runnable_sequence_active = False
                return llm_result

            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            new_result = original(self, inputs, config, return_exceptions=return_exceptions, **kwargs)
        return new_result

    @staticmethod
    def execute_retriever(retriever_steps, retriever_concat_result, inputs):
        retriever_step_index, retriever_sub_step_name, retriever_sub_step, _ = retriever_steps
        retrieval_results = inputs
        rag_provenance = None
        if retrieval_results:
            for child_sequence_step in retriever_sub_step.steps:
                if isinstance(child_sequence_step, BaseRetriever):
                    retrieval_results = execute_embedding_similarity_join(
                        retriever_concat_result.retrieval_corpus_X, retriever_concat_result.retrieval_corpus_y,
                        retriever_concat_result.embedding, retrieval_results)
                    rag_provenance = retrieval_results._mlinspect_provenance
                    vectorstore_ref = retrieval_results._mlinspect_vectorstore_ref
                    retrieval_index = retrieval_results._mlinspect_retrieval_index
                else:
                    retrieval_results = child_sequence_step.batch(retrieval_results)
        found_retriever = (retriever_step_index, retriever_sub_step_name, retrieval_results, inputs, rag_provenance,
                           vectorstore_ref, retrieval_index, retriever_steps)
        return found_retriever

    @staticmethod
    @gorilla.name('execute_rag_join_diff')
    @gorilla.settings(allow_hit=True)
    def execute_rag_join_diff(retriever_steps, inputs, filled_vectorstore):
        _, _, retriever_sub_step, _ = retriever_steps
        retrieval_results = inputs
        retrieval_index_update = numpy.zeros((len(inputs), 4), dtype=int)
        if retrieval_results:
            for child_sequence_step in retriever_sub_step.steps:
                if isinstance(child_sequence_step, BaseRetriever):
                    retrieval_results = filled_vectorstore.as_retriever().batch(inputs)
                    retrieval_results = wrap_in_mlinspect_array_if_necessary(retrieval_results)
                    for prediction_index, result in enumerate(retrieval_results):
                        retrieval_index_update[prediction_index, :] = [doc.metadata['_metadata_ids'] for doc in result]
                else:
                    retrieval_results = child_sequence_step.batch(retrieval_results)
        return retrieval_results, retrieval_index_update

    def find_retriever(self):
        # pylint: disable=no-member,too-many-nested-blocks
        found_retriever = None
        for step_index, step in enumerate(self.steps):
            if isinstance(step, BaseRetriever):
                raise NotImplementedError("Only VectorStoreRetriever that appear nested in a step are supported "
                                          "currently!")
            if isinstance(step, RunnableParallel):
                child_retrievers = [(step_name, step_content) for (step_name, step_content) in step.steps.items()
                                    if isinstance(step_content, BaseRetriever)]
                if len(child_retrievers) >= 1:
                    raise NotImplementedError("Retriever steps that appear directly as a runnable child without a "
                                              "formatting function are not supported right now!")
                child_sequences = [(step_name, step_content) for (step_name, step_content) in step.steps.items()
                                   if isinstance(step_content, RunnableSequence)]
                for child_sequence in child_sequences:
                    for child_sequence_step in child_sequence[1].steps:
                        if isinstance(child_sequence_step, BaseRetriever):
                            if step_index != 0:
                                raise NotImplementedError(
                                    "Only Retrievers at the beginning of langchain pipelines are supported currently!")
                            return step_index, child_sequence[0], child_sequence[1], child_sequence_step
                raise ValueError("Only langchain pipelines with a retrieval step are supported currently!")
        return found_retriever

    @staticmethod
    def execute_langchain_batch_with_preexecuted_retriever(runnable_sequence, config, return_exceptions,
                                                           found_retriever):
        # pylint: disable=no-member
        # TODO: Clean this up
        retriever_step_num, retriever_step_name, retriever_step_result, inputs, provenance, _, _, _ = found_retriever
        if not inputs:
            return []
        configs, run_managers = RunnableSequencePatching.do_langchain_batch_setup(runnable_sequence,
                                                                                  config, inputs, return_exceptions)
        for i, step in enumerate(runnable_sequence.steps):
            if i is not retriever_step_num:
                inputs = step.batch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        patch_config(
                            config, callbacks=rm.get_child(f"seq:step:{i + 1}")
                        )
                        for rm, config in zip(run_managers, configs)
                    ],
                )
            elif i is retriever_step_num and isinstance(step, RunnableParallel):
                result_dict = {}
                for step_name, step in step.steps.items():
                    if step_name is not retriever_step_name:
                        result_dict[step_name] = step.batch(inputs)
                    else:
                        result_dict[step_name] = retriever_step_result
                result_dict_as_list_of_dicts = pandas.DataFrame(result_dict).to_dict("records")
                inputs = result_dict_as_list_of_dicts
            else:
                raise NotImplementedError("TODO: Add support for langchain pipelines not following this pattern"
                                          " if necessary")
        new_result = cast(list[Output], inputs)
        new_result = MlinspectList(new_result)
        new_result._mlinspect_provenance = provenance
        return new_result

    @staticmethod
    def do_langchain_batch_setup(runnable_sequence, config, inputs, return_exceptions):
        if return_exceptions is True:
            raise NotImplementedError("Exception propagation not supported currently")
        configs = [
            config_with_context(c, runnable_sequence.steps)  # pylint: disable=no-member
            for c in get_config_list(config, len(inputs))
        ]
        callback_managers = [
            CallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        # start the root runs, one per input
        run_managers = [
            cm.on_chain_start(
                dumpd(runnable_sequence),
                input,
                name=config.get("run_name") or runnable_sequence.get_name(),  # pylint: disable=no-member
                run_id=config.pop("run_id", None),
            )
            for cm, input, config in zip(callback_managers, inputs, configs)
        ]
        return configs, run_managers


@dataclasses.dataclass
class CallerInfo:
    mlinspect_caller_filename: str
    mlinspect_lineno: int
    mlinspect_optional_code_reference: CodeReference or None
    mlinspect_optional_source_code: str or None


@gorilla.patches(community_vectorstores.Chroma)
class ChromaPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('from_texts')
    @gorilla.settings(allow_hit=True)
    @staticmethod
    def patched_from_texts(texts, metadatas=None, embedding=None, **kwargs):
        # We might not want to patch this one directly, only catch the batch call above
        original = gorilla.get_original_attribute(community_vectorstores.Chroma, 'from_texts')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            function_info = FunctionInfo('langchain_community.vectorstores.Chroma', 'from_texts')
            if metadatas is None:
                raise NotImplementedError("Vectorstore only supported in LLM+RAG scenarios with labels currently!")

            caller_info = CallerInfo(caller_filename, lineno, optional_code_reference, optional_source_code)
            _, train_data_node, train_data_result = add_train_data_node(caller_info, texts, function_info)
            _, train_labels_node, train_labels_result = add_train_label_node(caller_info, metadatas,
                                                                             function_info)

            input_dag_nodes = [train_data_node, train_labels_node]

            operator_context = OperatorContext(OperatorType.CONCATENATION, function_info, {'embedding': embedding})
            operator_call_info = OperatorCallInfo(operator_context, input_dag_nodes)

            # We want to always cache embeddings, even if they are used on updated inputs
            embedding_cache_lookup_key = OperatorCallInfo(operator_context, [])
            if embedding_cache_lookup_key in singleton.reuse_info.cached_embedding_func:
                cached_embedding = singleton.reuse_info.cached_embedding_func[embedding_cache_lookup_key]
            else:
                cached_embedding = CacheBackedEmbeddings.from_bytes_store(embedding, InMemoryByteStore())
                singleton.reuse_info.cached_embedding_func[embedding_cache_lookup_key] = cached_embedding

            # input_annotated_dfs = [input_info.annotated_dfobject for input_info in input_infos]
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            def processing_func(*input_dfs):
                assert isinstance(input_dfs[0], list) and isinstance(input_dfs[0][0], str)
                assert isinstance(input_dfs[1], list) and isinstance(input_dfs[1][0], dict)
                new_result = MlideaChromaVectorStoreRetrieverPlaceHolder(input_dfs[0], input_dfs[1], cached_embedding)
                return new_result

            optimizer_info, result = capture_optimizer_info(singleton, operator_call_info, processing_func,
                                                            [train_data_result, train_labels_result])

            dag_node = DagNode(singleton.get_next_op_id(operator_call_info),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(None,
                                              train_data_node.details.columns + train_labels_node.details.columns,
                                              optimizer_info),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, input_dag_nodes, function_call_result)
            new_result = function_call_result.function_result

            # For us, the actual embedding similarity join will happen in the langchain LLM chain
            # initial_func = partial(original, self, texts=texts, metadatas=metadatas, embedding=embedding, **kwargs)
            # optimizer_info, result = capture_optimizer_info(initial_func)

            return new_result

        return execute_patched_func_no_op_id(original, execute_inspections, texts=texts, metadatas=metadatas,
                                             embedding=embedding, **kwargs)


@gorilla.patches(huggingface.HuggingFaceEmbeddings)
class HuggingFaceEmbeddingsPatching:
    # TODO: Not sure yet if we also want to have the embeddings as a separate step in the DAG at some point, or if
    #  we want to avoid recomputations in a different way by doing everything in an incremental update of the
    #  embedding similarity join

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        original = gorilla.get_original_attribute(huggingface.HuggingFaceEmbeddings, '__init__')
        original(self, *args, **kwargs)

    @gorilla.name('embed_documents')
    @gorilla.settings(allow_hit=True)
    def patched_embed_documents(self, *args, **kwargs) -> list[list[float]]:
        # TODO: There are also async version of these functions, we might want to support them at some point
        # Here, it is a bit unclear as of now whether we want to present this as an extra node
        #  or have it as part of the vectorstore node
        original = gorilla.get_original_attribute(huggingface.HuggingFaceEmbeddings, 'embed_documents')
        new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.name('embed_query')
    @gorilla.settings(allow_hit=True)
    def patched_embed_query(self, *args, **kwargs) -> list[float]:
        # TODO: There are also async version of these functions, we might want to support them at some point
        # No batching is used for this one! Need to potentially find some workarounds to increase efficiency
        # We might not want to patch this one directly, only catch the batch call above
        original = gorilla.get_original_attribute(huggingface.HuggingFaceEmbeddings, 'embed_query')
        new_result = original(self, *args, **kwargs)
        return new_result
