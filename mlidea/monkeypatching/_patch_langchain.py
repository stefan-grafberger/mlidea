"""
Monkey patching for sklearn
"""
import copy
import dataclasses
import warnings
from collections.abc import Callable
from functools import partial

import gorilla
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import base, RunnableParallel, RunnableSequence
from langchain_core import vectorstores
from langchain_core.vectorstores import VectorStoreRetriever


@gorilla.patches(base.RunnableSequence)
class RunnableSequencePatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('batch')
    @gorilla.settings(allow_hit=True)
    def patched_batch(self, *args, **kwargs):
        """ Patch for ('langchain_core.runnables.base', 'RunnableSequence') """
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
                for child_sequence in child_sequences:
                    for child_sequence_step in child_sequence[1].steps:
                        if isinstance(child_sequence_step, VectorStoreRetriever):
                            print("retriever step found")
                            print(step)

        for step in self.steps:
            if isinstance(step, BaseChatModel):
                print("llm step found")
                print(step)

        new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(vectorstores.VectorStoreRetriever)
class VectorStoreRetrieverPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('invoke')
    @gorilla.settings(allow_hit=True)
    def patched_invoke(*args, **kwargs):
        """ Patch for ('langchain_core.vectorstores', 'VectorStoreRetriever') """
        # pylint: disable=no-self-argument
        original = gorilla.get_original_attribute(vectorstores.VectorStoreRetriever, 'invoke')

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
        new_result = original(*args, **kwargs)
        return new_result