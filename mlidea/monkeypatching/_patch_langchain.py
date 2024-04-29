"""
Monkey patching for sklearn
"""
import copy
import dataclasses
import warnings
from collections.abc import Callable
from functools import partial

import gorilla
from langchain_community.embeddings import huggingface
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import base, RunnableParallel, RunnableSequence
from langchain_community import vectorstores as community_vectorstores
from langchain_core import vectorstores as core_vectorstores
from langchain_core.vectorstores import VectorStoreRetriever

from execution._pipeline_executor import singleton


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
    def patched_batch(self, *args, **kwargs):
        # pylint: disable=no-self-argument
        original = gorilla.get_original_attribute(base.RunnableSequence, 'batch')

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

        # return execute_patched_func(original, execute_inspections, *args, **kwargs)

        if call_info_singleton.runnable_sequence_active is False:
            for step in self.steps:
                if isinstance(step, VectorStoreRetriever):
                    print("retriever step found")
                    print(step)
                if isinstance(step, RunnableParallel):
                    child_retrievers = [(step_name, step_content) for (step_name, step_content) in step.steps.items()
                                        if isinstance(step_content, VectorStoreRetriever)]
                    if len(child_retrievers) >= 1:
                        print("retriever step found")
                        print(step)
                    child_sequences = [(step_name, step_content) for (step_name, step_content) in step.steps.items()
                                       if isinstance(step_content, RunnableSequence)]
                    # TODO: Beware of recursive calls, these are the same as the current class. Introduce a singleton
                    #  with a boolean again to make sure that only the parent one is patched?
                    retrieval_corpus_input_dag_node_id = None
                    for child_sequence in child_sequences:
                        for child_sequence_step in child_sequence[1].steps:
                            if isinstance(child_sequence_step, VectorStoreRetriever):
                                print("retriever step found")
                                print(step)
                                retrieval_corpus_input_dag_node_id = child_sequence_step.vectorstore._mlinspect_input_dag_node
                    # This child_sequence here is the one that ultimately becomes the DAG node!

            for step in self.steps:
                if isinstance(step, BaseChatModel):
                    print("llm step found")
                    print(step)

            call_info_singleton.runnable_sequence_active = True
            new_result = original(self, *args, **kwargs)
            call_info_singleton.runnable_sequence_active = False
        else:
            new_result = original(self, *args, **kwargs)
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
    def patched_from_texts(*args, **kwargs):
        # pylint: disable=no-self-argument
        # We might not want to patch this one directly, only catch the batch call above
        original = gorilla.get_original_attribute(community_vectorstores.Chroma, 'from_texts')
        new_result = original(*args, **kwargs)
        # FIXME: Make sure that to_list forwards the mlinspect_dag_node and does not loose it
        new_result._mlinspect_input_dag_node = "test input forwarding"
        # new_result._mlinspect_input_dag_node = kwargs['texts']._mlinspect_dag_node
        # Actually use add_dag_node etc
        return new_result


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
