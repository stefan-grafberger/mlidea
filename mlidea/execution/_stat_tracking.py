"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""
import sys
import time
from functools import partial

import keras
from dill import dumps
import numpy
import pandas
import sklearn
from fairlearn.metrics import MetricFrame
from scikeras import wrappers
from scipy.sparse import csr_matrix

from mlidea.instrumentation._operator_call_info import OperatorCallInfo, OperatorOutputChange, OutputChangeType
from mlidea.instrumentation._dag_node import OptimizerInfo, OperatorContext
from mlidea.monkeypatching._mlinspect_ndarray import MlideaChromaVectorStoreRetrieverPlaceHolder
from mlidea.utils._utils import get_sorted_parent_nodes


def capture_optimizer_info(singleton, operator_call_info, instrumented_function_call: partial,
                           obj_for_inplace_ops: any or None = None,
                           estimator_transformer_state: any or None = None,
                           keras_batch_size: int or None = None,
                           force_disable_reuse=False) \
        -> tuple[OptimizerInfo, any]:
    """Function to measure the runtime of instrumented user function calls and get output metadata"""
    execution_start = time.time()
    not_a_constructor = (obj_for_inplace_ops is None or estimator_transformer_state is not None)
    if (not_a_constructor and
            operator_call_info in singleton.operator_call_info_to_dag_node
            and singleton.enable_cache_reuse is True and force_disable_reuse is False):
        dag_node = singleton.operator_call_info_to_dag_node[operator_call_info]
        result = singleton.cached_intermediates[dag_node]
        singleton.new_node_to_old_node[dag_node] = dag_node, OperatorOutputChange(OutputChangeType.NOTHING_CHANGED)
    # Maybe reuse
    elif (not_a_constructor and singleton.enable_cache_reuse is True and force_disable_reuse is False and
          singleton.old_dag is not None):

        parent_nodes_from_previous_run = []
        changes = []

        for parent_index, parent_op_id in enumerate(operator_call_info.parent_node_ids):
            new_dag_parent_node = [node for node in singleton.analysis_results.original_dag.nodes if node.node_id == parent_op_id][0]

            # Create operator call info
            new_dag_parent_operator_call_info = dag_node_to_operator_call_info(
                singleton.analysis_results.original_dag, new_dag_parent_node)
            unprocessed_transitive_change = new_dag_parent_operator_call_info in singleton.unprocessed_call_info_transitive_change_only
            is_undetermined = new_dag_parent_operator_call_info in singleton.undetermined_new_nodes
            assert unprocessed_transitive_change is False or is_undetermined is False
            if unprocessed_transitive_change:  # We need to delay processing them to ensure consecutive dag node ids
                old_operator_call_info, change_type = singleton.unprocessed_call_info_transitive_change_only[
                    new_dag_parent_operator_call_info]
                old_dag_node = singleton.get_next_op_id(old_operator_call_info)
                singleton.new_node_to_old_node[new_dag_parent_node] = old_dag_node, change_type
                singleton.unprocessed_call_info_transitive_change_only.pop(new_dag_parent_operator_call_info)
                singleton.operator_transitive.add(new_dag_parent_node)
                if change_type.change_type == OutputChangeType.TOO_MUCH_CHANGED:
                    singleton.operator_too_many_changes.add(new_dag_parent_node)
            elif is_undetermined:
                singleton.undetermined_new_nodes.remove(new_dag_parent_operator_call_info)

                old_dag = singleton.old_dag
                new_dag = singleton.analysis_results.original_dag

                # Maybe don't run all of this code if the change type is already found
                is_replacement, node_being_replaced = determine_is_replacement(new_dag,
                                                                               new_dag_parent_node, old_dag,
                                                                               operator_call_info, parent_index,
                                                                               singleton.operator_call_info_to_dag_node)
                is_addition, node_being_added_to = determine_is_addition(new_dag, new_dag_parent_node, operator_call_info,
                                                    singleton.operator_call_info_to_dag_node)
                is_deletion, deleted_node = determine_is_deletion(new_dag, new_dag_parent_node, old_dag)

                change_diff = OperatorOutputChange(OutputChangeType.TOO_MUCH_CHANGED)  # FIXME: We also need to compute the actual changes!
                if is_replacement:
                    singleton.operator_replacement.add(new_dag_parent_node)
                    singleton.new_node_to_old_node[new_dag_parent_node] = node_being_replaced, change_diff
                elif is_addition:
                    singleton.operator_addition.add(new_dag_parent_node)
                    singleton.new_node_to_old_node[new_dag_parent_node] = node_being_added_to, change_diff
                elif is_deletion:
                    singleton.operator_deletion.add(new_dag_parent_node)
                    singleton.new_node_to_old_node[new_dag_parent_node] = deleted_node, change_diff
                else:
                    singleton.operator_too_many_changes.add(new_dag_parent_node)

            assert new_dag_parent_node in singleton.new_node_to_old_node
            corresponding_node_in_old_dag, change_diff = singleton.new_node_to_old_node[new_dag_parent_node]
            parent_nodes_from_previous_run.append(corresponding_node_in_old_dag)
            changes.append(change_diff)

        if len([change for change in changes if change.change_type != OutputChangeType.NOTHING_CHANGED]) == 0:
            result = instrumented_function_call()
            if estimator_transformer_state is not None:
                result._mlinspect_annotation = estimator_transformer_state
            singleton.undetermined_new_nodes.add(operator_call_info)
        elif (len([change for change in changes if change.change_type == OutputChangeType.TOO_MUCH_CHANGED]) > 0 or
                len([change for change in changes if change.change_type != OutputChangeType.NOTHING_CHANGED]) > 1):
            # TODO: Certain kind of changes are also compatible and mergeable, especially if there are just
            #  multiple different row-level changes.
            singleton.unprocessed_call_info_transitive_change_only[operator_call_info] = (
                operator_call_info, OperatorOutputChange(OutputChangeType.TOO_MUCH_CHANGED))
            result = instrumented_function_call()
            if estimator_transformer_state is not None:
                result._mlinspect_annotation = estimator_transformer_state
        else:
            # Incrementally update the old result
            # First load the old result
            updated_operator_call_info = OperatorCallInfo(OperatorContext(operator_call_info.operator,
                                                                          operator_call_info.function_info,
                                                                          operator_call_info.non_data_kwargs),
                                                          parent_nodes_from_previous_run)
            old_dag_node = singleton.operator_call_info_to_dag_node[updated_operator_call_info]
            old_result = singleton.cached_intermediates[old_dag_node]
            # FIXME: Then update old result
            result = instrumented_function_call()
            if estimator_transformer_state is not None:
                result._mlinspect_annotation = estimator_transformer_state
            # FIXME: At this point, we should know what changed
            singleton.unprocessed_call_info_transitive_change_only[operator_call_info] = (
                updated_operator_call_info, OperatorOutputChange(OutputChangeType.TOO_MUCH_CHANGED))
    elif (not_a_constructor is False and operator_call_info in singleton.operator_call_info_to_dag_node
            and singleton.enable_cache_reuse is True and force_disable_reuse is False):
        result = instrumented_function_call()
        if estimator_transformer_state is not None:
            result._mlinspect_annotation = estimator_transformer_state
        # TODO: The node does not actually get reused yet. However, this is tricky with constructors
        dag_node = singleton.operator_call_info_to_dag_node[operator_call_info]
        singleton.new_node_to_old_node[dag_node] = dag_node, OperatorOutputChange(OutputChangeType.NOTHING_CHANGED)

    else: # Actually execute it
        result = instrumented_function_call()
        if estimator_transformer_state is not None:
            result._mlinspect_annotation = estimator_transformer_state
        singleton.undetermined_new_nodes.add(operator_call_info)
    execution_duration = time.time() - execution_start
    execution_duration_in_ms = execution_duration * 1000
    if result is not None:
        result_or_inplace_obj = result
    else:
        result_or_inplace_obj = obj_for_inplace_ops

    shape = get_df_shape(result_or_inplace_obj)
    size = get_df_memory(result_or_inplace_obj, estimator_transformer_state, keras_batch_size)
    return OptimizerInfo(execution_duration_in_ms, shape, size), result


def determine_is_deletion(new_dag, new_dag_parent_node, old_dag):
    # We can check the old DAG: if new_dag_parent_node is in the old DAG, but has a parent that does
    #  not exist in the new DAG, but if the parent parent exists in the new DAG
    is_deletion = False
    deleted_node = None
    # Step 1: Confirm the node exists in both DAGs
    if new_dag_parent_node in old_dag:
        # Step 2: Check each parent in the old DAG
        for parent in old_dag.predecessors(new_dag_parent_node):
            # If the parent is missing in the new DAG
            if parent not in new_dag:
                # Check if the grandparent exists in the new DAG
                for grandparent in old_dag.predecessors(parent):
                    if grandparent in new_dag:
                        is_deletion = True  # Found a deleted node's child with an existing grandparent
                        deleted_node = parent
    return is_deletion, deleted_node


def determine_is_addition(new_dag, new_dag_parent_node, operator_call_info, operator_call_info_to_dag_node):
    # I can check if a operator call info constructed based on the current node and the previous node
    # parents exists in the old dag
    parent_parents = get_sorted_parent_nodes(new_dag, new_dag_parent_node)
    if len(parent_parents) == 1:
        test_addition_operator_call_info = OperatorCallInfo(
            OperatorContext(operator_call_info.operator,
                            operator_call_info.function_info,
                            operator_call_info.non_data_kwargs),
            parent_parents
        )
        is_addition = test_addition_operator_call_info in operator_call_info_to_dag_node
        node_being_added_to = parent_parents[0]
        result = is_addition, node_being_added_to
    else:
        result = False, None  # Fast updates for addition of operations like joins is not supported currently
    return result


def determine_is_replacement(new_dag, new_dag_parent_node, old_dag, operator_call_info, parent_index,
                             operator_call_info_to_dag_node):
    # to compute is_replacement, we check the parents to new_dag_parent_operator_call_info and operator
    #  type and look in the old DAG if we can find a similar operation there with the same child and
    #  the same parents

    # Gather attributes and relationships for the new node
    is_replacement = False  # Default: No replacement found
    new_node_type = new_dag_parent_node.operator_info.operator
    new_parents = set(new_dag.predecessors(new_dag_parent_node))
    # Search for a similar node in the old DAG
    for old_node in old_dag.nodes:
        # Check if operator type matches
        if old_node.operator_info.operator == new_node_type:
            # Check if parents and children match
            old_parents = set(old_dag.predecessors(old_node))
            old_children = set(old_dag.successors(old_node))

            # TODO: Theoretically, there can be situations with simultaneous changes in the form of multiple
            #  replacement-like node inserts, then we need to be careful with naive replacement maps
            old_children_contains_current_node = OperatorCallInfo(
                OperatorContext(operator_call_info.operator, operator_call_info.function_info,
                                operator_call_info.non_data_kwargs),
                list(operator_call_info.parent_node_ids)[:parent_index] + [old_node.node_id] +
                list(operator_call_info.parent_node_ids)[parent_index + 1:]
            ) in operator_call_info_to_dag_node
            if new_parents == old_parents and old_children_contains_current_node:
                is_replacement = True  # Found a 1-to-1 replacement in the old DAG
                node_being_replaced = old_node
    return is_replacement, node_being_replaced


def dag_node_to_operator_call_info(dag, node):
    new_dag_parent_parents = get_sorted_parent_nodes(dag, node)
    new_dag_parent_operator_call_info = OperatorCallInfo(node.operator_info, new_dag_parent_parents)
    return new_dag_parent_operator_call_info


def get_df_memory(result_or_inplace_obj, estimator_transformer_state: any or None = None,
                  keras_batch_size: int or None = None, memory_calc_too_expensive=True):
    """Get the size in bytes of a df-like object"""
    # Just using sys.getsize of is not sufficient. See this section of its documentation:
    #  Only the memory consumption directly attributed to the object is accounted for,
    #  not the memory consumption of objects it refers to.
    if memory_calc_too_expensive is True:
        return 0  # FIXME: Do this properly with flag in the main API etc and not use 0 here
    if isinstance(result_or_inplace_obj, pandas.DataFrame):
        # For pandas, sizeof seems to work as expected
        size = sys.getsizeof(result_or_inplace_obj)
    elif isinstance(result_or_inplace_obj, numpy.ndarray):
        system_size = sys.getsizeof(result_or_inplace_obj)
        numpy_self_report = result_or_inplace_obj.nbytes
        if system_size >= numpy_self_report:
            size = system_size
        else:
            size = system_size + numpy_self_report
    elif isinstance(result_or_inplace_obj, csr_matrix):
        size = result_or_inplace_obj.data.nbytes + sys.getsizeof(result_or_inplace_obj)
    elif isinstance(result_or_inplace_obj, (tuple, list)):
        size = sum(get_df_memory(elem) for elem in result_or_inplace_obj)
    else:
        size = sys.getsizeof(result_or_inplace_obj)
    if estimator_transformer_state is not None:
        if isinstance(estimator_transformer_state, (sklearn.base.BaseEstimator, sklearn.base.TransformerMixin)):
            size += sys.getsizeof(dumps(estimator_transformer_state))
        elif isinstance(estimator_transformer_state, wrappers.KerasClassifier):
            size += get_model_memory_usage_in_bytes(estimator_transformer_state.model, keras_batch_size)
        else:
            raise NotImplementedError(f"Measuring the memory size of {type(estimator_transformer_state).__name__} is "
                                      f"not supported yet!")
    return size


def get_model_memory_usage_in_bytes(model, batch_size):
    """Function to get the memory size of a Keras model"""
    # Based on https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage_in_bytes(batch_size, layer)
        single_layer_mem = 1
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for dimension in out_shape:
            if dimension is None:
                continue
            single_layer_mem *= dimension
        shapes_mem_count += single_layer_mem

    trainable_count = numpy.sum([keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = numpy.sum([keras.backend.count_params(p) for p in model.non_trainable_weights])

    number_size = 4
    if keras.backend.floatx() == 'float16':
        number_size = 2
    if keras.backend.floatx() == 'float64':
        number_size = 8

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    return round(total_memory + internal_model_mem_count)


def get_df_shape(result_or_inplace_obj):
    """Get the shape of a df-like object"""
    if isinstance(result_or_inplace_obj, pandas.DataFrame):
        shape = result_or_inplace_obj.shape
    elif isinstance(result_or_inplace_obj, numpy.ndarray):
        if result_or_inplace_obj.ndim == 2:
            shape = result_or_inplace_obj.shape
        elif result_or_inplace_obj.ndim == 1:
            shape = len(result_or_inplace_obj), 1
        elif 3 <= result_or_inplace_obj.ndim <= 4:
            shape = result_or_inplace_obj.shape  # Image pipeline
        else:
            raise NotImplementedError("Currently only numpy arrays with 1-4 dims are supported!")
    elif isinstance(result_or_inplace_obj, pandas.Series):
        shape = len(result_or_inplace_obj), 1
    elif isinstance(result_or_inplace_obj, pandas.core.groupby.generic.DataFrameGroupBy):
        shape = (result_or_inplace_obj.ngroups, result_or_inplace_obj.ndim)
    elif isinstance(result_or_inplace_obj, list):
        # A few operations like train_test_split return a list
        if len(result_or_inplace_obj) > 1 and isinstance(result_or_inplace_obj[0], str):
            shape = (len(result_or_inplace_obj), 1)
        elif isinstance(result_or_inplace_obj[0], numpy.ndarray) and result_or_inplace_obj[0].ndim == 1:
            shape = (len(result_or_inplace_obj), len(result_or_inplace_obj[0]))
        elif isinstance(result_or_inplace_obj, list) and isinstance(result_or_inplace_obj[0], dict):
            shape = (len(result_or_inplace_obj), len(list(result_or_inplace_obj[0].keys())))
        elif isinstance(result_or_inplace_obj, list) and not isinstance(result_or_inplace_obj[0],
                                                                        (list, numpy.ndarray)):
            shape = (len(result_or_inplace_obj), 1)
        else:
            assert len(result_or_inplace_obj) == 2
            shape_a = get_df_shape(result_or_inplace_obj[0])
            shape_b = get_df_shape(result_or_inplace_obj[1])
            assert shape_a[1] == shape_b[1]
            shape = shape_a[0] + shape_b[0], shape_a[1]
    elif isinstance(result_or_inplace_obj, csr_matrix):
        # Here we use the csr_matrix column count as width instead of treating it as 1 column only as we do logically.
        #  This is because we might need it for optimisation purposes to choose whether dense or sparse matrices are
        #  better. We might want to potentially change this in the future.
        shape = result_or_inplace_obj.shape
    elif isinstance(result_or_inplace_obj, (float, MetricFrame)):
        # E.g., a score metric output from estimator.score
        shape = (1, 1)
    elif isinstance(result_or_inplace_obj, dict) and isinstance(list(result_or_inplace_obj.values())[0], dict):
        # E.g., pandas dataframe to_dict output
        shape = (len(list(result_or_inplace_obj.values())[0]), len(result_or_inplace_obj))
    elif isinstance(result_or_inplace_obj, MlideaChromaVectorStoreRetrieverPlaceHolder):
        shape = (len(result_or_inplace_obj.retrieval_corpus_X), len(result_or_inplace_obj.retrieval_corpus_y[0]) + 1)
    else:
        shape = None
    return shape
