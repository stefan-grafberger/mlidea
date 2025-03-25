"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""
import sys
import time
from functools import partial

import keras
import numpy
import pandas
import sklearn
from dill import dumps
from fairlearn.metrics import MetricFrame
from scikeras import wrappers
from scipy.sparse import csr_matrix

from mlidea.ivm._func_executor_ivm import execute_with_partial_reuse
from mlidea.instrumentation._dag_node import OptimizerInfo, DagNode
from mlidea.instrumentation._operator_call_info import OperatorOutputChange, OutputChangeType
from mlidea.instrumentation._operator_types import OperatorType, ConditionalResult
from mlidea.monkeypatching._mlinspect_ndarray import MlideaChromaVectorStoreRetrieverPlaceHolder


def capture_optimizer_info(singleton, operator_call_info, instrumented_function_call: partial or None,
                           instrumented_function_call_args: list[any] or None,
                           obj_for_inplace_ops: any or None = None,
                           estimator_transformer_state: any or None = None,
                           keras_batch_size: int or None = None,
                           extract_or_conditional=False,
                           stop_signal_received=False,
                           current_dag_node: DagNode or None=None) \
        -> tuple[OptimizerInfo, any]:
    """Function to measure the runtime of instrumented user function calls and get output metadata"""
    # pylint: disable=too-many-arguments
    if instrumented_function_call_args is None:
        instrumented_function_call_args = []
    execution_start = time.time()
    not_a_constructor = (obj_for_inplace_ops is None or estimator_transformer_state is not None
                         or operator_call_info.operator == OperatorType.PROJECTION_MODIFY)
    if singleton.enable_cache_reuse is True:
        result = try_ivm_reuse_using_cache(current_dag_node, estimator_transformer_state, extract_or_conditional,
                                           not_a_constructor, operator_call_info, instrumented_function_call,
                                           instrumented_function_call_args, singleton, stop_signal_received)

    else:
        if instrumented_function_call is not None:
            original_func_call_with_args = partial(instrumented_function_call, *instrumented_function_call_args)
        else:
            original_func_call_with_args = None
        result = execute_function(estimator_transformer_state, original_func_call_with_args, stop_signal_received)

    optimizer_info = get_optimizer_info(estimator_transformer_state, execution_start, keras_batch_size,
                                        obj_for_inplace_ops, result, stop_signal_received)
    return optimizer_info, result


def try_ivm_reuse_using_cache(current_dag_node, estimator_transformer_state, extract_or_conditional, not_a_constructor,
                              operator_call_info, instrumented_function_call, instrumented_function_call_args,
                              singleton, stop_signal_received):
    if instrumented_function_call is not None:
        original_func_call_with_args = partial(instrumented_function_call, *instrumented_function_call_args)
    else:
        original_func_call_with_args = None
    # Guaranteed reuse
    if (not_a_constructor and
            operator_call_info in singleton.reuse_info.operator_call_info_to_dag_node
            and extract_or_conditional is False):
        dag_node = singleton.reuse_info.operator_call_info_to_dag_node[operator_call_info]
        result = singleton.reuse_info.cached_intermediates[dag_node]
        if dag_node not in singleton.reuse_info.new_node_to_old_node:
            singleton.reuse_info.new_node_to_old_node[dag_node] = dag_node, OperatorOutputChange(
                OutputChangeType.NOTHING_CHANGED)
    # We have to re-execute conditionals
    elif (not_a_constructor and
          operator_call_info in singleton.reuse_info.operator_call_info_to_dag_node
          and extract_or_conditional is True):
        dag_node = singleton.reuse_info.operator_call_info_to_dag_node[operator_call_info]
        result = original_func_call_with_args()
        if estimator_transformer_state is not None:
            result._mlinspect_annotation = estimator_transformer_state
        if dag_node not in singleton.reuse_info.new_node_to_old_node:
            singleton.reuse_info.new_node_to_old_node[dag_node] = dag_node, OperatorOutputChange(
                OutputChangeType.NOTHING_CHANGED)
    # Constructors cannot be reused currently
    elif (not_a_constructor is False and operator_call_info in singleton.reuse_info.operator_call_info_to_dag_node):
        result = original_func_call_with_args()
        if estimator_transformer_state is not None:
            result._mlinspect_annotation = estimator_transformer_state
        # TODO: The node does not actually get reused yet. However, this is tricky with constructors
        dag_node = singleton.reuse_info.operator_call_info_to_dag_node[operator_call_info]
        singleton.reuse_info.new_node_to_old_node[dag_node] = dag_node, OperatorOutputChange(
            OutputChangeType.NOTHING_CHANGED)
    # Maybe reuse
    # TODO: Do we want to get rid of operator_call_info is not None? This is currently required because of the
    #  pandas groupby operation that gets executed before agg is called after. We also cannot reuse intermediates
    #  for that operation currently.
    elif singleton.old_dag is not None and operator_call_info is not None:
        result = execute_with_partial_reuse(current_dag_node, estimator_transformer_state, instrumented_function_call,
                                            instrumented_function_call_args, operator_call_info, singleton, stop_signal_received)
    # Fallback
    else:
        result = execute_function(estimator_transformer_state, original_func_call_with_args, stop_signal_received)
    return result


def execute_function(estimator_transformer_state, original_func_call_with_args, stop_signal_received):
    if stop_signal_received is False:  # Actually execute it
        result = original_func_call_with_args()
        if estimator_transformer_state is not None:
            result._mlinspect_annotation = estimator_transformer_state
    else:
        result = ConditionalResult.STOP_EXECUTION
    return result


def get_optimizer_info(estimator_transformer_state, execution_start, keras_batch_size, obj_for_inplace_ops, result,
                       stop_signal_received):
    if stop_signal_received is False:
        execution_duration = time.time() - execution_start
        execution_duration_in_ms = execution_duration * 1000
        if result is not None:
            result_or_inplace_obj = result
        else:
            result_or_inplace_obj = obj_for_inplace_ops
        if not isinstance(result_or_inplace_obj, list) or len(result_or_inplace_obj) > 0:
            shape = get_df_shape(result_or_inplace_obj)
        else:
            shape = None
        size = get_df_memory(result_or_inplace_obj, estimator_transformer_state, keras_batch_size)
        optimizer_info = OptimizerInfo(execution_duration_in_ms, shape, size)
    else:
        optimizer_info = OptimizerInfo(None, None, None)
    return optimizer_info


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
