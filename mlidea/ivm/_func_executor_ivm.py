"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""
from functools import partial

import duckdb
import numpy
import pandas

from mlidea.ivm._utils import fix_data_diff_detection_mask_only, fix_data_mask_to_indices, apply_diff_filter, \
    changed_data_diff_detection, rag_join_update, _get_rag_join_results_to_rerun, wrap_in_mlinspect_array_if_necessary, \
    update_prediction_diff
from mlidea.instrumentation._dag_node import OperatorContext
from mlidea.instrumentation._operator_call_info import OperatorCallInfo, OperatorOutputChange, OutputChangeType
from mlidea.instrumentation._operator_types import OperatorType, ConditionalResult, FunctionInfo
from mlidea.ivm._func_executor_change_detection import determine_parents_compared_to_previous_dag
from mlidea.utils._utils import get_sorted_parent_nodes


def execute_with_partial_reuse(current_dag_node, estimator_transformer_state, instrumented_function_call,
                               instrumented_function_call_args, operator_call_info, singleton, stop_signal_received):
    if instrumented_function_call is not None:
        original_func_call_with_args = partial(instrumented_function_call, *instrumented_function_call_args)
    else:
        original_func_call_with_args = None

    parent_nodes_from_previous_run = determine_parents_compared_to_previous_dag(operator_call_info, singleton)
    parent_node_result_from_previous_run = [singleton.reuse_info.cached_intermediates[node]
                                            for node in parent_nodes_from_previous_run]
    updated_operator_call_info = OperatorCallInfo(OperatorContext(operator_call_info.operator,
                                                                  operator_call_info.function_info,
                                                                  operator_call_info.non_data_kwargs),
                                                  parent_nodes_from_previous_run)
    new_parent_nodes = [singleton.get_dag_node_for_id(node_id) for node_id in operator_call_info.parent_node_ids]
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
        elif ((not isinstance(parent_node_result_from_previous_run[-1], ConditionalResult) or
            parent_node_result_from_previous_run[-1] != ConditionalResult.STOP_EXECUTION) and stop_signal_received is False and
              operator_call_info.operator == OperatorType.PROJECTION_MODIFY_SUBSET):
            result = projection_modify_subset_ivm(instrumented_function_call, instrumented_function_call_args,
                                                  old_result, original_func_call_with_args,
                                                  parent_nodes_from_previous_run, singleton)
        elif ((not isinstance(parent_node_result_from_previous_run[-1], ConditionalResult) or
               parent_node_result_from_previous_run[
                   -1] != ConditionalResult.STOP_EXECUTION) and stop_signal_received is False and
              operator_call_info.operator == OperatorType.RAG_JOIN and operator_call_info.function_info ==
              FunctionInfo('langchain_community.vectorstores.Chroma', 'from_texts')):
            result = rag_join_ivm(instrumented_function_call_args, old_result, old_dag_node, current_dag_node,
                                  original_func_call_with_args, new_parent_nodes, parent_nodes_from_previous_run,
                                  singleton, operator_call_info, updated_operator_call_info)
        elif ((not isinstance(parent_node_result_from_previous_run[-1], ConditionalResult) or
               parent_node_result_from_previous_run[
                   -1] != ConditionalResult.STOP_EXECUTION) and stop_signal_received is False and
              operator_call_info.operator == OperatorType.PREDICT and operator_call_info.function_info ==
              FunctionInfo('langchain_core.runnables.base', 'batch')):
            result = llm_predict_ivm(instrumented_function_call, instrumented_function_call_args, old_result,
                                     old_dag_node, current_dag_node, original_func_call_with_args, new_parent_nodes,
                                     parent_nodes_from_previous_run, singleton)
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
            #  If it is in the map, we already know what changed
            if operator_call_info not in singleton.reuse_info.unprocessed_call_info_transitive_change_only:
                singleton.reuse_info.unprocessed_call_info_transitive_change_only[operator_call_info] = (
                    updated_operator_call_info, OperatorOutputChange(OutputChangeType.UNKNOWN))
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


def rag_join_ivm(instrumented_function_call_args,
                 old_result, old_dag_node, current_dag_node, original_func_call_with_args, new_parent_nodes,
                 parent_nodes_from_previous_run, singleton, new_operator_call_info, old_operator_call_info):
    # TODO: Compare old input with new input
    train_side_node_new = new_parent_nodes[0]
    train_side_node_old = parent_nodes_from_previous_run[0]
    inference_side_new = new_parent_nodes[1]
    inference_side_old = parent_nodes_from_previous_run[1]
    train_side_changed = train_side_node_new != train_side_node_old
    inference_side_changed = inference_side_new != inference_side_old

    train_side_corpus_new = instrumented_function_call_args[0]
    train_side_corpus_old = singleton.reuse_info.cached_intermediates[parent_nodes_from_previous_run[0]]
    inference_side_rows_new = instrumented_function_call_args[1]
    inference_side_rows_old = singleton.reuse_info.cached_intermediates[parent_nodes_from_previous_run[1]]
    if train_side_changed:
        if len(train_side_corpus_new.retrieval_corpus_X) == len(train_side_corpus_old.retrieval_corpus_X):
            old_dag = singleton.global_old_dag
            new_dag = singleton.global_new_dag
            concat_parent_X_new, concat_parent_y_new = get_sorted_parent_nodes(new_dag, train_side_node_new)
            concat_parent_X_old, concat_parent_y_old = get_sorted_parent_nodes(old_dag, train_side_node_old)
            X_changed = concat_parent_X_new != concat_parent_X_old
            y_changed = concat_parent_y_new != concat_parent_y_old
            diff_mask_combined = numpy.zeros(len(train_side_corpus_new.retrieval_corpus_X), dtype=bool)
            vectorstore = old_result[5]
            if X_changed:

                diff_mask_X = fix_data_diff_detection_mask_only(
                    train_side_corpus_new.retrieval_corpus_X, train_side_corpus_old.retrieval_corpus_X)
                diff_mask_combined = diff_mask_combined | diff_mask_X

                diff_indices_X = fix_data_mask_to_indices(diff_mask_X)

                # Update the labels in the vectorstore
                if len(diff_indices_X) > 0:
                    diff_X = apply_diff_filter(train_side_corpus_new.retrieval_corpus_y, diff_indices_X)
                    vectorstore_ids = [str(index) for index in diff_indices_X]
                    old_entries = vectorstore.get(ids=vectorstore_ids, include=["embeddings", "metadatas"])
                    vectorstore._collection.update(vectorstore_ids, old_entries['embeddings'], old_entries['metadatas'],
                                                   list(diff_X))
            if y_changed:
                assert y_changed
                diff_mask_y = fix_data_diff_detection_mask_only(
                    train_side_corpus_new.retrieval_corpus_y, train_side_corpus_old.retrieval_corpus_y)
                diff_mask_combined = diff_mask_combined | diff_mask_y
                diff_indices_y = fix_data_mask_to_indices(diff_mask_y)

                # Update the labels in the vectorstore
                if len(diff_indices_y) > 0:
                    diff_y = apply_diff_filter(train_side_corpus_new.retrieval_corpus_y, diff_indices_y)
                    vectorstore_ids = [str(index) for index in diff_indices_y]
                    old_entries = vectorstore.get(ids=vectorstore_ids, include=["embeddings", "documents"])
                    vectorstore._collection.update(vectorstore_ids, old_entries['embeddings'], list(diff_y),
                                                   old_entries['documents'])

            corpus_changed_diff_index = fix_data_mask_to_indices(diff_mask_combined)
            # We redo the lookups even if there is only a label change for simplicity with langchain, but since the
            #  embeddings are cached the costs for this should be negligible
            parent_diff_index = _get_rag_join_results_to_rerun(old_result, corpus_changed_diff_index)
            singleton.reuse_info.unprocessed_call_info_transitive_change_only[new_operator_call_info] = (
                old_operator_call_info,
                OperatorOutputChange(OutputChangeType.ROWS_UPDATED, rows_updated=parent_diff_index))

            result = rag_join_update(old_result, inference_side_rows_new, vectorstore, parent_diff_index)
            # We do not need to revert the changes here since the original pipeline is always changed after this
        else:
            result = original_func_call_with_args()
    else:
        assert inference_side_changed
        if len(inference_side_rows_new) == len(inference_side_rows_old):
            parent_diff_index = changed_data_diff_detection(inference_side_rows_new, inference_side_rows_old)

            vectorstore = old_result[5]
            result = rag_join_update(old_result, inference_side_rows_new, vectorstore, parent_diff_index)
            singleton.reuse_info.unprocessed_call_info_transitive_change_only[new_operator_call_info] = (
                old_operator_call_info,
                OperatorOutputChange(OutputChangeType.ROWS_UPDATED, rows_updated=parent_diff_index))
        else:
            result = original_func_call_with_args()

    # FIXME: Tell LLM Predict which rows need to be rerun
    #  We can use the change maps here that we don't use as much as we should yet

    return result


def llm_predict_ivm(instrumented_function_call, instrumented_function_call_args,
                    old_result, current_dag_node, old_dag_node, original_func_call_with_args, new_parent_nodes,
                    parent_nodes_from_previous_run, singleton):
    # FIXME: What if they have a different length?
    #  Use a duckdb join like in the shadow pipeline experiments to determine what to recompute and
    #  construct the final result.
    input_arg = instrumented_function_call_args[0]
    old_predict_input_node = parent_nodes_from_previous_run[0]
    new_predict_input_node = new_parent_nodes[0]
    _, predict_input_diff = singleton.reuse_info.new_node_to_old_node[new_predict_input_node]
    old_predict_input_object = singleton.reuse_info.cached_intermediates[old_predict_input_node]
    old_predict_input_data = old_predict_input_object[3]
    new_predict_input_data = instrumented_function_call_args[0][3]
    if (predict_input_diff.change_type == OutputChangeType.ROWS_UPDATED and len(old_predict_input_data) ==
            len(new_predict_input_data)):
        rows_modified = predict_input_diff.rows_updated
        filtered_predict_input = apply_diff_filter(instrumented_function_call_args[0], rows_modified)
        diff_predict_result = instrumented_function_call(filtered_predict_input)
        result = update_prediction_diff(old_result, diff_predict_result, rows_modified)
        result = wrap_in_mlinspect_array_if_necessary(result)
    else:
        result = original_func_call_with_args()
    # FIXME: Also use caching in-between iterations additionally

    # FIXME: Need to check why this is happening and why the provenance gets lost without this
    if hasattr(new_predict_input_data, "_mlinspect_provenance"):
        result._mlinspect_provenance = new_predict_input_data._mlinspect_provenance
    else:
        result._mlinspect_provenance = input_arg[4]
    # FIXME: Cache LLM results across LLM execs?
    # data_arg_indices = instrumented_function_call_args[1]
    # if not isinstance(old_result, ConditionalResult) or old_result != ConditionalResult.STOP_EXECUTION:
    #     # TODO: Compare old input with new input
    #     parent_data_node = parent_nodes_from_previous_run[0]
    #     parent_data_result = singleton.reuse_info.cached_intermediates[parent_data_node]
    #     parent_data_indices_node = parent_nodes_from_previous_run[1]
    #     parent_data_indices = singleton.reuse_info.cached_intermediates[parent_data_indices_node]
    #     parent_filtered_input = apply_diff_filter(parent_data_result, parent_data_indices)
    #     data_arg_filtered_input = apply_diff_filter(data_arg, data_arg_indices)
    #     filtered_old_dag_node_output = apply_diff_filter(old_result, parent_data_indices)
    #     already_fixed_output = filtered_old_dag_node_output.copy()
    #     if isinstance(already_fixed_output, pandas.DataFrame):
    #         was_df = True
    #         was_series = False
    #         was_numpy = False
    #     elif isinstance(already_fixed_output, pandas.Series):
    #         was_df = False
    #         was_series = True
    #         was_numpy = False
    #         series_column_name = already_fixed_output.name
    #         if series_column_name is None:
    #             series_column_name = "column"
    #         already_fixed_output = pandas.DataFrame({series_column_name: already_fixed_output})
    #
    #         assert isinstance(data_arg_filtered_input, pandas.Series)
    #         series_column_name = data_arg_filtered_input.name
    #         data_arg_filtered_input = pandas.DataFrame({series_column_name: data_arg_filtered_input})
    #     elif isinstance(already_fixed_output, (numpy.ndarray, list)):
    #         was_df = False
    #         was_series = False
    #         was_numpy = True
    #         series_column_name = "column"
    #         already_fixed_output = pandas.DataFrame({series_column_name: already_fixed_output})
    #         data_arg_filtered_input = pandas.DataFrame({series_column_name: data_arg_filtered_input})
    #     else:
    #         raise NotImplementedError("TODO")
    #     columns = list(already_fixed_output.columns)
    #     assert len(columns) == 1
    #     column = columns[0]
    #     already_fixed_output["before_fix"] = parent_filtered_input
    #     # This cache should be stored in a singleton. Also, it should be cleared once it becomes too big
    #     #  maybe we can add a rounds id there and clear the oldest round once memory consumption becomes too big
    #     new_indices_df = pandas.DataFrame({"test_id": data_arg_indices, "before_fix": data_arg_filtered_input[column]})
    #     not_fixed_yet = duckdb.sql("""
    #                     SELECT n.test_id
    #                     FROM new_indices_df n ANTI JOIN already_fixed_output a ON n.before_fix = a.before_fix
    #                 """).df()
    #     already_fixed = duckdb.sql(f"""
    #                     SELECT n.test_id, a.{column}
    #                     FROM new_indices_df n JOIN already_fixed_output a ON n.before_fix = a.before_fix
    #                 """).df()
    #     if not_fixed_yet.shape[0] != 0:
    #         updated_args = [data_arg, not_fixed_yet['test_id']]
    #         diff_func_call_with_args = partial(instrumented_function_call, *updated_args)
    #         updated_result = diff_func_call_with_args()
    #     else:
    #         updated_result = data_arg.copy()
    #     if isinstance(updated_result, (pandas.DataFrame, pandas.Series)):
    #         updated_result = updated_result.reset_index(drop=True)
    #
    #     if was_df and already_fixed.shape[0] != 0:
    #         updated_result.iloc[already_fixed['test_id'].values, 0] = already_fixed[column].values
    #     elif was_series and already_fixed.shape[0] != 0:
    #         updated_result.iloc[already_fixed['test_id'].values] = already_fixed[column].values
    #     elif was_numpy and already_fixed.shape[0] != 0:
    #         if isinstance(updated_result, list):
    #             updated_result = wrap_in_mlinspect_array_if_necessary(numpy.ravel(updated_result))
    #         updated_result[already_fixed['test_id'].values] = already_fixed[column].values
    #     elif already_fixed.shape[0] == 0:
    #         pass
    #     else:
    #         raise NotImplementedError("Can this happen?")
    #     result = updated_result
    # else:
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
