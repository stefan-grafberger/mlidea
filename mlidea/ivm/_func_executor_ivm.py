"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""

from mlidea.ivm._func_executor_change_detection import determine_parents_compared_to_previous_dag
from mlidea.instrumentation._dag_node import OperatorContext
from mlidea.instrumentation._operator_call_info import OperatorCallInfo, OperatorOutputChange, OutputChangeType
from mlidea.instrumentation._operator_types import OperatorType, ConditionalResult


def execute_with_partial_reuse(current_dag_node, estimator_transformer_state, instrumented_function_call,
                               operator_call_info, singleton, stop_signal_received):
    parent_nodes_from_previous_run = determine_parents_compared_to_previous_dag(operator_call_info, singleton)
    updated_operator_call_info = OperatorCallInfo(OperatorContext(operator_call_info.operator,
                                                                  operator_call_info.function_info,
                                                                  operator_call_info.non_data_kwargs),
                                                  parent_nodes_from_previous_run)
    if updated_operator_call_info in singleton.reuse_info.operator_call_info_to_dag_node:
        # Incrementally update the old result
        # First load the old result

        old_dag_node = singleton.reuse_info.operator_call_info_to_dag_node[updated_operator_call_info]
        old_result = singleton.reuse_info.cached_intermediates[old_dag_node]  # pylint: disable=unused-variable

        # FIXME: Make sure we get the function to call here without having already bound arguments so we can
        #  call it on different arguments as needed

        # FIXME: Then update old result
        # TODO: Look at changes. Certain kind of changes are also compatible and mergeable, especially if there are
        #  just multiple different row-level changes.
        # TODO: if extract_or_conditional is True, we always need to reexecute because of the label extraction
        if stop_signal_received is False:
            result = instrumented_function_call()
            if estimator_transformer_state is not None:
                result._mlinspect_annotation = estimator_transformer_state
        else:
            result = ConditionalResult.STOP_EXECUTION

        # Extract results are always a final node and only appear after first creating a DAG node, so no need to
        #  mark them as transitive here
        if updated_operator_call_info.operator != OperatorType.EXTRACT_RESULT:
            # FIXME: At this point, we should know what changed
            singleton.reuse_info.unprocessed_call_info_transitive_change_only[operator_call_info] = (
                updated_operator_call_info, OperatorOutputChange(OutputChangeType.TOO_MUCH_CHANGED))
        else:
            old_dag_node = singleton.reuse_info.operator_call_info_to_dag_node[updated_operator_call_info]
            singleton.reuse_info.operator_transitive.add(current_dag_node)
            singleton.reuse_info.new_node_to_old_node[current_dag_node] = (
                old_dag_node, OperatorOutputChange(OutputChangeType.TOO_MUCH_CHANGED))
    else:
        # TODO: Here we have a real change then. Maybe we want to count those?
        if stop_signal_received is False:
            result = instrumented_function_call()
            if estimator_transformer_state is not None:
                result._mlinspect_annotation = estimator_transformer_state
        else:
            result = ConditionalResult.STOP_EXECUTION
        if operator_call_info.operator != OperatorType.MISSING_OP:
            # This can happen, e.g., for the grid search operation in sklearn that we do not want to capture in the
            #  DAG currently
            singleton.reuse_info.undetermined_new_nodes.add(operator_call_info)

    return result
