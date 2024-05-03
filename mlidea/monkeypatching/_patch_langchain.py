"""
Monkey patching for sklearn
"""
from functools import partial
from typing import (
    Any,
    List,
    Optional,
    Union, cast,
)

import gorilla
import pandas
from langchain_community import vectorstores as community_vectorstores
from langchain_community.embeddings import huggingface
from langchain_community.vectorstores.chroma import Chroma
from langchain_core import vectorstores as core_vectorstores
from langchain_core.language_models import BaseChatModel
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
from langchain_core.beta.runnables.context import config_with_context
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.vectorstores import VectorStoreRetriever

from execution._pipeline_executor import singleton
from execution._stat_tracking import capture_optimizer_info
from mlidea import DagNode, BasicCodeLocation, DagNodeDetails, FunctionInfo, OperatorContext, OperatorType
from monkeypatching._mlinspect_ndarray import MlideaChromaVectorStoreRetrieverPlaceHolder
from monkeypatching._monkey_patching_utils import execute_patched_func, get_optional_code_info_or_none, \
    FunctionCallResult, add_dag_node, get_input_info, execute_patched_func_indirect_allowed, \
    execute_patched_func_indirect_allowed_with_op_id


class LangchainCallInfo:
    """ Contains info like lineno from the current Transformer so indirect utility function calls can access it """
    # pylint: disable=too-few-public-methods
    runnable_sequence_active: bool = False


call_info_singleton = LangchainCallInfo()


def execute_embedding_similarity_join(retrieval_corpus_X, retrieval_corpus_y, embedding, inputs: list[Input]):
    filled_vectorstore = Chroma.from_texts(texts=retrieval_corpus_X, metadatas=retrieval_corpus_y,
                                           embedding=embedding).as_retriever()
    return filled_vectorstore.batch(inputs)


@gorilla.patches(base.RunnableSequence)
class RunnableSequencePatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods
    @gorilla.name('batch')
    @gorilla.settings(allow_hit=True)
    def patched_batch(self, inputs: List[Input], config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
                      *, return_exceptions: bool = False, **kwargs: Optional[Any]):
        original = gorilla.get_original_attribute(base.RunnableSequence, 'batch')
        if call_info_singleton.runnable_sequence_active is False:
            def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
                """ Execute inspections, add DAG node """
                call_info_singleton.runnable_sequence_active = True
                # TODO: Maybe use vectorstore info here and not the LLM info
                function_info = FunctionInfo('langchain_core.runnables.base', 'batch')
                retriever_with_info = self.find_retriever()
                input_info_a = get_input_info(retriever_with_info[3], caller_filename, lineno, function_info,
                                              optional_code_reference, optional_source_code)
                input_info_b = get_input_info(inputs, caller_filename, lineno, function_info,
                                              optional_code_reference, optional_source_code)
                operator_context = OperatorContext(OperatorType.JOIN, function_info)

                processing_func = partial(self.execute_retriever, inputs, retriever_with_info)
                optimizer_info, result = capture_optimizer_info(processing_func)
                description = "Embedding similarity join"
                dag_node = DagNode(op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails(description, ["array"], optimizer_info),
                                   get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                   processing_func)
                function_call_result = FunctionCallResult(result)
                add_dag_node(dag_node, [input_info_a.dag_node, input_info_b.dag_node], function_call_result)
                embedding_join_result = function_call_result.function_result

                # TODO: Create second LLM node
                new_result = self.execute_langchain_batch_with_preexecuted_retriever(embedding_join_result, config,
                                                                                     inputs, return_exceptions)
                call_info_singleton.runnable_sequence_active = False

                return new_result

            new_result = execute_patched_func_indirect_allowed_with_op_id(execute_inspections)

        else:
            new_result = original(self, inputs, config, return_exceptions=return_exceptions, **kwargs)
        return new_result

    @staticmethod
    def execute_retriever(inputs, retriever_with_info):
        retriever_step_index, retriever_sub_step_name, retriever_sub_step, _ = retriever_with_info
        retrieval_results = inputs
        if retrieval_results:
            for child_sequence_step in retriever_sub_step.steps:
                if isinstance(child_sequence_step, BaseRetriever):
                    retrieval_results = execute_embedding_similarity_join(
                        child_sequence_step.retrieval_corpus_X, child_sequence_step.retrieval_corpus_y,
                        child_sequence_step.embedding, retrieval_results)
                else:
                    retrieval_results = child_sequence_step.batch(retrieval_results)
        found_retriever = (retriever_step_index, retriever_sub_step_name, retrieval_results)
        return found_retriever

    def find_retriever(self):
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
                                    "Only Retrievers at the beginning of langchain pipeliens are supported currently!")
                            return step_index, child_sequence[0], child_sequence[1], child_sequence_step
                raise ValueError("Only langchain pipelines with a retrieval step are supported currently!")
        return found_retriever

    def execute_langchain_batch_with_preexecuted_retriever(self, found_retriever, config, inputs, return_exceptions):
        if not inputs:
            return []
        retriever_step_num, retriever_step_name, retriever_step_result = found_retriever
        configs, run_managers = self.do_langchain_batch_setup(config, inputs, return_exceptions)
        for i, step in enumerate(self.steps):
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
        new_result = cast(List[Output], inputs)
        return new_result

    def do_langchain_batch_setup(self, config, inputs, return_exceptions):
        if return_exceptions is True:
            raise NotImplementedError("Exception propagation not supported currently")
        configs = [
            config_with_context(c, self.steps)
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
                dumpd(self),
                input,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for cm, input, config in zip(callback_managers, inputs, configs)
        ]
        return configs, run_managers


@gorilla.patches(community_vectorstores.Chroma)
class ChromaPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('from_texts')
    @gorilla.settings(allow_hit=True)
    def patched_from_texts(texts, metadatas=None, embedding=None, **kwargs):
        # pylint: disable=no-self-argument
        # We might not want to patch this one directly, only catch the batch call above
        original = gorilla.get_original_attribute(community_vectorstores.Chroma, 'from_texts')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            function_info = FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')
            input_infos = []
            if metadatas is None:
                raise NotImplementedError("Vectorstore only supported in LLM+RAG scenarios with labels currently!")
            input_info_X_train = get_input_info(texts, caller_filename, lineno, function_info,
                                                optional_code_reference, optional_source_code)
            input_infos.append(input_info_X_train)
            input_info_y_train = get_input_info(metadatas, caller_filename, lineno, function_info,
                                                optional_code_reference, optional_source_code)
            input_infos.append(input_info_y_train)

            operator_context = OperatorContext(OperatorType.CONCATENATION, function_info)

            # input_annotated_dfs = [input_info.annotated_dfobject for input_info in input_infos]
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            def processing_func(*input_dfs):
                assert isinstance(input_dfs[0], list) and isinstance(input_dfs[0][0], str)
                assert isinstance(input_dfs[1], list) and isinstance(input_dfs[1][0], dict)
                new_result = MlideaChromaVectorStoreRetrieverPlaceHolder(input_dfs[0], input_dfs[1], embedding)
                return new_result

            initial_func = partial(processing_func, texts, metadatas, **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)

            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(None, result.columns(), optimizer_info),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)
            input_dag_nodes = [input_info.dag_node for input_info in input_infos]
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, input_dag_nodes, function_call_result)
            new_result = function_call_result.function_result

            # For us, the actual embedding similarity join will happen in the langchain LLM chain
            # initial_func = partial(original, self, texts=texts, metadatas=metadatas, embedding=embedding, **kwargs)
            # optimizer_info, result = capture_optimizer_info(initial_func)

            return new_result

        return execute_patched_func(original, execute_inspections, texts=texts, metadatas=metadatas,
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
    def patched_embed_documents(*args, **kwargs) -> list[list[float]]:
        # TODO: There are also async version of these functions, we might want to support them at some point
        # Here, it is a bit unclear as of now whether we want to present this as an extra node
        #  or have it as part of the vectorstore node
        original = gorilla.get_original_attribute(huggingface.HuggingFaceEmbeddings, 'embed_documents')
        new_result = original(*args, **kwargs)
        return new_result

    @gorilla.name('embed_query')
    @gorilla.settings(allow_hit=True)
    def patched_embed_query(*args, **kwargs) -> list[float]:
        # TODO: There are also async version of these functions, we might want to support them at some point
        # No batching is used for this one! Need to potentially find some workarounds to increase efficiency
        # We might not want to patch this one directly, only catch the batch call above
        original = gorilla.get_original_attribute(huggingface.HuggingFaceEmbeddings, 'embed_query')
        new_result = original(*args, **kwargs)
        return new_result
