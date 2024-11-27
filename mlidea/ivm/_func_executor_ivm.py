"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""
from copy import copy
from functools import partial

import duckdb
import numpy
import pandas

from mlidea.ivm._func_executor_change_detection import determine_parents_compared_to_previous_dag
from mlidea.instrumentation._dag_node import OperatorContext
from mlidea.instrumentation._operator_call_info import OperatorCallInfo, OperatorOutputChange, OutputChangeType
from mlidea.instrumentation._operator_types import OperatorType, ConditionalResult, FunctionInfo
from mlidea.monkeypatching._mlinspect_ndarray import MlinspectNdarray, MlinspectList, MlinspectDict, MlinspectTuple


def fix_data_diff_detection_mask_only(input_df, corrupted_result):
    if isinstance(input_df, (pandas.Series, pandas.DataFrame)):
        input_df = input_df.reset_index(drop=True)
    elif isinstance(input_df, list):
        input_df = numpy.array(input_df)
    if isinstance(corrupted_result, (pandas.Series, pandas.DataFrame)):
        corrupted_result = corrupted_result.reset_index(drop=True)
    elif isinstance(corrupted_result, list):
        corrupted_result = numpy.array(corrupted_result)
    if isinstance(input_df, pandas.Series):
        corrupt_diff_mask = (corrupted_result != input_df).to_numpy()
    elif len(input_df.shape) == 2:
        if input_df.shape == corrupted_result.shape:
            corrupt_diff_mask = numpy.any(corrupted_result != input_df, axis=1)
        else:
            # FIXME: This case can happen if the shape mismatches. Ideally, in the case of filters, we know that
            #  a filter happened and we immediately find out which rows are added/removed and do not need to run this
            #  function at all. Otherwise, we would have to try to understand the changes here. But maybe that
            #  is also okay, e.g., with a DuckDB join. But then we should really only run this function if the
            #  runtime of the operation we do not want to fully execute is sufficiently high.
            corrupt_diff_mask = numpy.ones(corrupted_result.shape, dtype=bool)
    else:
        corrupt_diff_mask = corrupted_result != input_df
    return corrupt_diff_mask


def fix_data_mask_to_indices(corrupt_diff_mask):
    changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
    return changed_indices_corrupt


def wrap_in_mlinspect_array_if_necessary(df_object):
    """
    Makes sure annotations can be stored in a df_object. For example, numpy arrays need a wrapper for this.
    """
    prov = None
    if hasattr(df_object, "_mlinspect_provenance"):
        # Not really sure yet why this is necessary, we should clean this up in the future
        prov = df_object._mlinspect_provenance
    if isinstance(df_object, numpy.ndarray) and not isinstance(df_object, MlinspectNdarray):
        df_object = MlinspectNdarray(df_object)
    elif isinstance(df_object, list):
        df_object = MlinspectList(df_object)
    elif isinstance(df_object, dict):
        df_object = MlinspectDict(df_object)
    elif isinstance(df_object, tuple):
        df_object = MlinspectTuple(df_object)
    if prov is not None:
        df_object._mlinspect_provenance = prov
    return df_object


def apply_diff_filter(input_df, corrupted_index):
    # TODO
    if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
        input_df = input_df.reset_index(drop=True)
    if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
        corrupted_diff = input_df.iloc[corrupted_index]
    elif isinstance(input_df, list):
        corrupted_diff = numpy.array(input_df)[corrupted_index]
    elif isinstance(input_df, tuple) and len(input_df) == 8:  # RAG Join Result
        corrupted_diff_list = list(copy(input_df))
        corrupted_diff_list[2] = list(numpy.array(corrupted_diff_list[2])[corrupted_index])
        corrupted_diff_list[3] = list(numpy.array(corrupted_diff_list[3])[corrupted_index])
        corrupted_diff_list[6] = corrupted_diff_list[6][corrupted_index, :]
        corrupted_diff = tuple(corrupted_diff_list)
    else:
        corrupted_diff = input_df[corrupted_index]
    if isinstance(corrupted_diff, (pandas.Series, pandas.DataFrame)):
        corrupted_diff = corrupted_diff.reset_index(drop=True)
    corrupted_diff = wrap_in_mlinspect_array_if_necessary(corrupted_diff)
    corrupted_diff._mlinspect_provenance = None

    return corrupted_diff


def changed_data_diff_detection(input_df, corrupted_result):
    corrupt_diff_mask = fix_data_diff_detection_mask_only(input_df, corrupted_result)
    changed_indices_corrupt = fix_data_mask_to_indices(corrupt_diff_mask)
    return changed_indices_corrupt


def execute_with_partial_reuse(current_dag_node, estimator_transformer_state, instrumented_function_call,
                               instrumented_function_call_args, operator_call_info, singleton, stop_signal_received):
    if instrumented_function_call is not None:
        original_func_call_with_args = partial(instrumented_function_call, *instrumented_function_call_args)
    else:
        original_func_call_with_args = None

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
        if stop_signal_received is False and operator_call_info.function_info in {
            FunctionInfo(
                'sklearn.preprocessing_function_transformer', 'FunctionTransformer'),
            FunctionInfo('example_pipelines.healthcare.healthcare_utils',
                                                                'MyW2VTransformer')}:
            result = function_transformer_ivm(estimator_transformer_state, instrumented_function_call,
                                              instrumented_function_call_args, old_result, original_func_call_with_args,
                                              parent_nodes_from_previous_run, singleton)
        elif ((not isinstance(parent_nodes_from_previous_run[-1], ConditionalResult) or
            parent_nodes_from_previous_run[-1] != ConditionalResult.STOP_EXECUTION) and stop_signal_received is False and
              operator_call_info.operator == OperatorType.PROJECTION_MODIFY_SUBSET):
            result = projection_modify_subset_ivm(instrumented_function_call, instrumented_function_call_args,
                                                  old_result, original_func_call_with_args,
                                                  parent_nodes_from_previous_run, singleton)
        # FIXME: Rag Join and LLM Calls
        elif stop_signal_received is False:
            result = original_func_call_with_args()
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
            result = original_func_call_with_args()
            if estimator_transformer_state is not None:
                result._mlinspect_annotation = estimator_transformer_state
        else:
            result = ConditionalResult.STOP_EXECUTION
        if operator_call_info.operator != OperatorType.MISSING_OP:
            # This can happen, e.g., for the grid search operation in sklearn that we do not want to capture in the
            #  DAG currently
            singleton.reuse_info.undetermined_new_nodes.add(operator_call_info)

    return result


def projection_modify_subset_ivm(instrumented_function_call, instrumented_function_call_args, old_result,
                                 original_func_call_with_args, parent_nodes_from_previous_run, singleton):
    # FIXME: What if they have a different length?
    #  Use a duckdb join like in the shadow pipeline experiments to determine what to recompute and
    #  construct the final result.
    data_arg = instrumented_function_call_args[0]
    data_arg_indices = instrumented_function_call_args[1]
    if not isinstance(old_result, ConditionalResult) or old_result != ConditionalResult.STOP_EXECUTION:
        # TODO: Compare old input with new input
        parent_data_node = parent_nodes_from_previous_run[0]
        parent_data_result = singleton.reuse_info.cached_intermediates[parent_data_node]
        parent_data_indices_node = parent_nodes_from_previous_run[1]
        parent_data_indices = singleton.reuse_info.cached_intermediates[parent_data_indices_node]
        parent_filtered_input = apply_diff_filter(parent_data_result, parent_data_indices)
        data_arg_filtered_input = apply_diff_filter(data_arg, data_arg_indices)
        filtered_old_dag_node_output = apply_diff_filter(old_result, parent_data_indices)
        already_fixed_output = filtered_old_dag_node_output.copy()
        if isinstance(already_fixed_output, pandas.DataFrame):
            was_df = True
            was_series = False
            was_numpy = False
        elif isinstance(already_fixed_output, pandas.Series):
            was_df = False
            was_series = True
            was_numpy = False
            series_column_name = already_fixed_output.name
            if series_column_name is None:
                series_column_name = "column"
            already_fixed_output = pandas.DataFrame({series_column_name: already_fixed_output})

            assert isinstance(data_arg_filtered_input, pandas.Series)
            series_column_name = data_arg_filtered_input.name
            data_arg_filtered_input = pandas.DataFrame({series_column_name: data_arg_filtered_input})
        elif isinstance(already_fixed_output, (numpy.ndarray, list)):
            was_df = False
            was_series = False
            was_numpy = True
            series_column_name = "column"
            already_fixed_output = pandas.DataFrame({series_column_name: already_fixed_output})
            data_arg_filtered_input = pandas.DataFrame({series_column_name: data_arg_filtered_input})
        else:
            raise NotImplementedError("TODO")
        columns = list(already_fixed_output.columns)
        assert len(columns) == 1
        column = columns[0]
        already_fixed_output["before_fix"] = parent_filtered_input
        # This cache should be stored in a singleton. Also, it should be cleared once it becomes too big
        #  maybe we can add a rounds id there and clear the oldest round once memory consumption becomes too big
        new_indices_df = pandas.DataFrame({"test_id": data_arg_indices, "before_fix": data_arg_filtered_input[column]})
        not_fixed_yet = duckdb.sql("""
                        SELECT n.test_id
                        FROM new_indices_df n ANTI JOIN already_fixed_output a ON n.before_fix = a.before_fix
                    """).df()
        already_fixed = duckdb.sql(f"""
                        SELECT n.test_id, a.{column}
                        FROM new_indices_df n JOIN already_fixed_output a ON n.before_fix = a.before_fix
                    """).df()
        if not_fixed_yet.shape[0] != 0:
            updated_args = [data_arg, not_fixed_yet['test_id']]
            diff_func_call_with_args = partial(instrumented_function_call, *updated_args)
            updated_result = diff_func_call_with_args()
        else:
            updated_result = data_arg.copy()
        if isinstance(updated_result, (pandas.DataFrame, pandas.Series)):
            updated_result = updated_result.reset_index(drop=True)

        if was_df and already_fixed.shape[0] != 0:
            updated_result.iloc[already_fixed['test_id'].values, 0] = already_fixed[column].values
        elif was_series and already_fixed.shape[0] != 0:
            updated_result.iloc[already_fixed['test_id'].values] = already_fixed[column].values
        elif was_numpy and already_fixed.shape[0] != 0:
            if isinstance(updated_result, list):
                updated_result = wrap_in_mlinspect_array_if_necessary(numpy.ravel(updated_result))
            updated_result[already_fixed['test_id'].values] = already_fixed[column].values
        elif already_fixed.shape[0] == 0:
            pass
        else:
            raise NotImplementedError("Can this happen?")
        result = updated_result
    else:
        result = original_func_call_with_args()
    result._mlinspect_provenance = data_arg._mlinspect_provenance
    return result


def function_transformer_ivm(estimator_transformer_state, instrumented_function_call, instrumented_function_call_args,
                             old_result, original_func_call_with_args, parent_nodes_from_previous_run, singleton):
    # TODO: Compare old input with new input
    if len(parent_nodes_from_previous_run) == 1:
        parent_data_node = parent_nodes_from_previous_run[0]
        data_arg = instrumented_function_call_args[0]
    else:
        parent_data_node = parent_nodes_from_previous_run[1]
        data_arg = instrumented_function_call_args[1]
    parent_data_input = singleton.reuse_info.cached_intermediates[parent_data_node]
    # FIXME: What if they have a different length?
    # FIXME: Also use a cache here maybe?
    if ((not isinstance(parent_data_input, ConditionalResult) or
         parent_data_input != ConditionalResult.STOP_EXECUTION)
            and len(parent_data_input) == len(data_arg)):
        parent_diff_index = changed_data_diff_detection(parent_data_input, data_arg)
        result = old_result.copy()
        if len(parent_diff_index) != 0:
            diff_df = apply_diff_filter(data_arg, parent_diff_index)
            if len(parent_nodes_from_previous_run) == 1:
                updated_args = [diff_df]
            else:
                updated_args = [instrumented_function_call_args[0], diff_df]
            diff_func_call_with_args = partial(instrumented_function_call, *updated_args)
            diff_result = diff_func_call_with_args()
            result[parent_diff_index] = diff_result
        if estimator_transformer_state is not None:
            result._mlinspect_annotation = estimator_transformer_state
    else:
        # FIXME: len(parent_data_input) != len(data_arg))
        result = original_func_call_with_args()
        if estimator_transformer_state is not None:
            result._mlinspect_annotation = estimator_transformer_state
    result._mlinspect_provenance = data_arg._mlinspect_provenance
    return result
