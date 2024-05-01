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
from langchain_community import vectorstores as community_vectorstores
from langchain_community.embeddings import huggingface
from langchain_core import vectorstores as core_vectorstores
from langchain_core.language_models import BaseChatModel
from langchain_core.load.dump import dumpd
from langchain_core.runnables import base, RunnableParallel, RunnableSequence
from langchain_core.runnables.config import (
    RunnableConfig,
    get_config_list, patch_config,
)
from langchain_core.runnables.utils import (
    Input, Output,
)
from langchain_core.vectorstores import VectorStoreRetriever

from execution._stat_tracking import capture_optimizer_info
from mlidea import DagNode, BasicCodeLocation, DagNodeDetails, FunctionInfo, OperatorContext, OperatorType
from monkeypatching._mlinspect_ndarray import MlideaChromaVectorStoreRetrieverPlaceHolder
from monkeypatching._monkey_patching_utils import execute_patched_func, get_optional_code_info_or_none, \
    FunctionCallResult, add_dag_node, get_input_info


class LangchainCallInfo:
    """ Contains info like lineno from the current Transformer so indirect utility function calls can access it """
    # pylint: disable=too-few-public-methods
    runnable_sequence_active: bool = False


call_info_singleton = LangchainCallInfo()


@gorilla.patches(base.RunnableSequence)
class RunnableSequencePatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods
    @gorilla.name('batch')
    @gorilla.settings(allow_hit=True)
    def patched_batch(self, inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any]):
        original = gorilla.get_original_attribute(base.RunnableSequence, 'batch')
        if call_info_singleton.runnable_sequence_active is False:
            call_info_singleton.runnable_sequence_active = True

            if return_exceptions is True:
                raise NotImplementedError("Exception propagation not supported currently")


            # Setup code
            from langchain_core.beta.runnables.context import config_with_context
            from langchain_core.callbacks.manager import CallbackManager

            if not inputs:
                return []

            # setup callbacks and context
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
            # End setup
            # TODO: Now we can look for the retrieval step and create a DAG node for it and precompute the result
            retriever_step = None
            for i, step in enumerate(self.steps):
                if isinstance(step, VectorStoreRetriever):
                    raise NotImplementedError("Only VectorStoreRetriever that appear nested in a step are supported "
                                              "currently!")
                if isinstance(step, RunnableParallel):
                    child_retrievers = [(step_name, step_content) for (step_name, step_content) in step.steps.items()
                                        if isinstance(step_content, VectorStoreRetriever)]
                    if len(child_retrievers) >= 1:
                        raise NotImplementedError("Retriever steps that appear directly as a runnable child without a "
                                                  "formatting function are not supported right now!")
                    child_sequences = [(step_name, step_content) for (step_name, step_content) in step.steps.items()
                                       if isinstance(step_content, RunnableSequence)]
                    for child_sequence in child_sequences:
                        for child_sequence_step in child_sequence[1].steps:
                            if isinstance(child_sequence_step, VectorStoreRetriever):
                                print("retriever step found")
                                retriever_step = i
                                print(step)
                                child_sequence_step.precompute_results()
            if retriever_step != 0:
                print(retriever_step)
                raise NotImplementedError("Only Retrievers at the beginning of langchain pipeliens are supported right "
                                          "now!")
            # TODO: Now we can execute the rest of the langchain pipeline while making sure to reuse the computed result
            for i, step in enumerate(self.steps):
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
            new_result = cast(List[Output], inputs)

            # def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            #     """ Execute inspections, add DAG node """
            #     function_info = FunctionInfo('sklearn.preprocessing._label', 'label_binarize')
            #     input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
            #                                 optional_source_code)
            #
            #     operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            #     initial_func = partial(original, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            #     optimizer_info, result = capture_optimizer_info(initial_func)
            #     processing_func = lambda df: original(df, *args[1:], **kwargs)
            #
            #     classes = kwargs['classes']
            #     description = f"label_binarize, classes: {classes}"
            #     dag_node = DagNode(op_id,
            #                        BasicCodeLocation(caller_filename, lineno),
            #                        operator_context,
            #                        DagNodeDetails(description, ["array"], optimizer_info),
            #                        get_optional_code_info_or_none(optional_code_reference, optional_source_code),
            #                        processing_func)
            #     function_call_result = FunctionCallResult(result)
            #     add_dag_node(dag_node, [input_info.dag_node], function_call_result)
            #     new_result = function_call_result.function_result
            #
            #     return new_result

            call_info_singleton.runnable_sequence_active = False
        else:
            new_result = original(self, inputs, config, return_exceptions=return_exceptions, **kwargs)
        return new_result


@gorilla.patches(core_vectorstores.VectorStoreRetriever)
class VectorStoreRetrieverPatching:
    """ Patches for sklearn """

    @gorilla.name('invoke')
    @gorilla.settings(allow_hit=True)
    def patched_invoke(*args, **kwargs):
        """ Patch for ('langchain_core.vectorstores', 'VectorStoreRetriever') """
        # pylint: disable=no-self-argument
        # We might not want to patch this one directly, only catch the batch call above
        original = gorilla.get_original_attribute(core_vectorstores.VectorStoreRetriever, 'invoke')
        new_result = original(*args, **kwargs)
        return new_result


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
                               BasicCodeLocation(caller_filename,lineno),
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
