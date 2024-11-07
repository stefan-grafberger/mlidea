"""
Functions for the implementation for the monkey patched functions
"""
import ast
import dataclasses
import sys
from functools import partial

import numpy
from langchain_core.retrievers import BaseRetriever
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

from mlidea.instrumentation._operator_call_info import OperatorCallInfo
from mlidea.execution import _pipeline_executor
from mlidea.execution._pipeline_executor import singleton
from mlidea.execution._stat_tracking import get_df_shape, get_df_memory, capture_optimizer_info
from mlidea.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo
from mlidea.instrumentation._operator_types import OperatorContext, OperatorType
from mlidea.monkeypatching._mlinspect_ndarray import MlinspectNdarray, MlinspectList, MlinspectDict, \
    MlinspectTuple
from mlidea import monkeypatching


@dataclasses.dataclass(frozen=False)
class FunctionCallResult:
    """ The annotated dataframe and the annotations for the current DAG node """
    function_result: any or None
    other: any = None  # TODO: input/output cardinality


@dataclasses.dataclass(frozen=True)
class AnnotatedDfObject:
    """ A dataframe-like object and its annotations """
    result_data: any
    result_annotation: any


@dataclasses.dataclass(frozen=True)
class InputInfo:
    """ WIP experiments """
    dag_node: DagNode
    annotated_dfobject: AnnotatedDfObject


def execute_patched_internal_func_with_depth(original_func, execute_inspections_func, depth, *args, **kwargs):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    # Performance aspects: https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
    # CPython implementation detail: This function should be used for internal and specialized purposes only.
    #  It is not guaranteed to exist in all implementations of Python.
    #  inspect.getcurrentframe() also only does return `sys._getframe(1) if hasattr(sys, "_getframe") else None`
    #  We can execute one hasattr check right at the beginning of the mlidea execution

    caller_filename = sys._getframe(depth).f_code.co_filename

    if caller_filename != singleton.source_code_path:
        result = original_func(*args, **kwargs)
    elif singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                              singleton.col_offset_next_call_or_subscript,
                                              singleton.end_lineno_next_call_or_subscript,
                                              singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(-1, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
    else:
        caller_lineno = sys._getframe(2).f_lineno
        result = execute_inspections_func(-1, caller_filename, caller_lineno, None, None)
    return result


def execute_patched_func_no_op_id(original_func, execute_inspections_func, *args, **kwargs):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    caller_filename = sys._getframe(2).f_code.co_filename

    if caller_filename != singleton.source_code_path:
        result = original_func(*args, **kwargs)
    elif singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                              singleton.col_offset_next_call_or_subscript,
                                              singleton.end_lineno_next_call_or_subscript,
                                              singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(-1, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
    else:
        caller_lineno = sys._getframe(2).f_lineno
        result = execute_inspections_func(-1, caller_filename, caller_lineno, None, None)
    return result


def execute_patched_func_indirect_allowed(execute_inspections_func):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    # Performance aspects: https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
    # CPython implementation detail: This function should be used for internal and specialized purposes only.
    #  It is not guaranteed to exist in all implementations of Python.
    #  inspect.getcurrentframe() also only does return `sys._getframe(1) if hasattr(sys, "_getframe") else None`
    #  We can execute one hasattr check right at the beginning of the mlidea execution

    frame = sys._getframe(2)
    while frame.f_code.co_filename != singleton.source_code_path:
        frame = frame.f_back

    caller_filename = frame.f_code.co_filename

    if singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                              singleton.col_offset_next_call_or_subscript,
                                              singleton.end_lineno_next_call_or_subscript,
                                              singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(-1, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
    else:
        caller_lineno = sys._getframe(2).f_lineno
        result = execute_inspections_func(-1, caller_filename, caller_lineno, None, None)
    return result


def get_input_info(df_object, caller_filename, lineno, function_info, optional_code_reference, optional_source_code) \
        -> InputInfo:
    """
    Uses the patched _mlinspect_dag_node attribute and the singleton.op_id_to_dag_node map to find the parent DAG node
    for the DAG node we want to insert in the next step.
    """
    # pylint: disable=unused-argument
    columns = get_column_names(df_object)
    if hasattr(df_object, "_mlinspect_dag_node"):
        input_op_id = df_object._mlinspect_dag_node
        input_dag_node = singleton.op_id_to_dag_node[input_op_id]
        input_info = InputInfo(input_dag_node, AnnotatedDfObject(df_object, None))  # TODO: Remove annotation stuff
    else:
        if optional_code_reference:
            code_reference = f"({optional_source_code})"
        else:
            code_reference = ""
        description = (f"Warning! Operator {caller_filename}:{lineno} {code_reference} encountered a DataFrame "
                       f"resulting from an operation without mlidea support!")
        missing_op_id = singleton.get_next_missing_op_id()
        input_dag_node = DagNode(missing_op_id,
                                 BasicCodeLocation(caller_filename, lineno),
                                 OperatorContext(OperatorType.MISSING_OP, None, {}),
                                 DagNodeDetails(description, columns,
                                                OptimizerInfo(None, get_df_shape(df_object), get_df_memory(df_object))),
                                 OptionalCodeInfo(optional_code_reference, optional_source_code))
        function_call_result = FunctionCallResult(df_object)
        add_dag_node(input_dag_node, [], function_call_result)
        input_info = InputInfo(input_dag_node, AnnotatedDfObject(df_object, None))  # TODO: Remove annotation stuff
        if singleton.prov_enabled is True:
            monkeypatching._provenance_propagation.generate_and_add_provenance_data_source(
                input_info.annotated_dfobject.result_data, missing_op_id)
    return input_info


def get_column_names(df_object):
    """Get column names for a dataframe ojbect"""
    if isinstance(df_object, DataFrame):
        columns = list(df_object.columns)  # TODO: Update this for numpy arrays etc. later
    elif isinstance(df_object, Series):
        columns = [df_object.name]
    elif isinstance(df_object, (csr_matrix, numpy.ndarray, list, MlinspectTuple)):
        columns = ['array']
    elif isinstance(df_object, BaseRetriever):
        columns = df_object.columns()
    else:
        raise NotImplementedError(f"TODO: Type: '{type(df_object)}' still is not supported!")
    return columns


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


def get_dag_node_copy_with_optimizer_info(dag_node: DagNode, optimizer_info: OptimizerInfo):
    """Because DagNodes are immutable, we need this fuction to create a copy where the optimizer_info is set"""
    new_node = DagNode(dag_node.node_id,
                       dag_node.code_location,
                       dag_node.operator_info,
                       DagNodeDetails(dag_node.details.description, dag_node.details.columns,
                                      optimizer_info),
                       dag_node.optional_code_info,
                       dag_node.processing_func)
    return new_node


def add_dag_node(dag_node: DagNode, dag_node_parents: list[DagNode], function_call_result: FunctionCallResult):
    """
    Inserts a new node into the DAG
    """
    # pylint: disable=protected-access
    # print("")
    # print("{}:{}: {}".format(dag_node.caller_filename, dag_node.lineno, dag_node.module))

    # print("source code: {}".format(dag_node.optional_source_code))
    if function_call_result.function_result is not None and dag_node.operator_info.operator != OperatorType.SCORE:
        function_call_result.function_result = wrap_in_mlinspect_array_if_necessary(
            function_call_result.function_result)
        function_call_result.function_result._mlinspect_dag_node = dag_node.node_id
    elif dag_node.operator_info.operator == OperatorType.SCORE:
        result_label = f"original_L{dag_node.code_location.lineno}"
        result_value = function_call_result.function_result
        singleton.original_pipeline_labels_to_extracted_plan_results[result_label] = result_value
    if dag_node_parents:
        for parent_index, parent in enumerate(dag_node_parents):
            singleton.analysis_results.original_dag.add_edge(parent, dag_node, arg_index=parent_index)
            # TODO: This duplication is not that clean
            singleton.global_new_dag.add_edge(parent, dag_node, arg_index=parent_index)
    else:
        singleton.analysis_results.original_dag.add_node(dag_node)
        # TODO: This duplication is not that clean
        singleton.global_new_dag.add_node(dag_node)
    singleton.op_id_to_dag_node[dag_node.node_id] = dag_node

    if singleton.enable_caching is True:
        singleton.reuse_info.operator_call_info_to_dag_node[OperatorCallInfo(dag_node.operator_info, dag_node_parents)] = dag_node
        # # TODO: Is this copy really necessary? Without it, the columns sometimes mismatch with cached dfs that
        # #  get updated later on during the original pipeline
        if isinstance(function_call_result.function_result, DataFrame):
            df_result = function_call_result.function_result
            df_result_copy = df_result.copy()
            if hasattr(df_result, "_mlinspect_provenance"):
                df_result_copy._mlinspect_provenance = df_result._mlinspect_provenance
            # FIXME: Do we need to manually forward other attributes as well?
            singleton.reuse_info.cached_intermediates[dag_node] = df_result_copy
        else:
            singleton.reuse_info.cached_intermediates[dag_node] = function_call_result.function_result
        # if dag_node.operator_info.operator != OperatorType.PROJECTION_MODIFY:
        #     singleton.reuse_info.cached_intermediates[dag_node] = function_call_result.function_result
        # else:
        #     singleton.reuse_info.cached_intermediates[dag_node] = function_call_result.function_result.copy()
    # if function_call_result.other is not None:
    # singleton.inspection_results.dag_node_to_inspection_results[dag_node] = backend_result.dag_node_annotation
    # TODO: Do we want to capture other meta information here? Or as part of the DAG node?


def get_dag_node_for_id(dag_node_id: int):
    """
    Get a DAG node by id
    """
    return singleton.op_id_to_dag_node[dag_node_id]


def get_optional_code_info_or_none(optional_code_reference: CodeReference or None,
                                   optional_source_code: str or None) -> OptionalCodeInfo or None:
    """
    If code reference tracking is enabled, return OptionalCodeInfo, otherwise None
    """
    if singleton.track_code_references:
        assert optional_code_reference is not None
        assert optional_source_code is not None
        code_info_or_none = OptionalCodeInfo(optional_code_reference, optional_source_code)
    else:
        assert optional_code_reference is None
        assert optional_source_code is None
        code_info_or_none = None
    return code_info_or_none


def add_train_label_node(estimator, train_label_arg, function_info):
    """Add a Train Data DAG Node for a estimator.fit call"""
    input_info_train_labels = get_input_info(train_label_arg, estimator.mlinspect_caller_filename,
                                             estimator.mlinspect_lineno, function_info,
                                             estimator.mlinspect_optional_code_reference,
                                             estimator.mlinspect_optional_source_code)
    columns = input_info_train_labels.dag_node.details.columns
    operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info, {})
    operator_call_info = OperatorCallInfo(operator_context,
                                          [input_info_train_labels.dag_node])
    train_label_op_id = _pipeline_executor.singleton.get_next_op_id(operator_call_info)
    process_func = lambda df_object: df_object
    initial_func = partial(process_func, train_label_arg)
    _, result = capture_optimizer_info(singleton, operator_call_info, initial_func)
    train_labels_dag_node = DagNode(train_label_op_id,
                                    BasicCodeLocation(estimator.mlinspect_caller_filename, estimator.mlinspect_lineno),
                                    operator_context,
                                    DagNodeDetails(None, columns, OptimizerInfo(0, get_df_shape(train_label_arg),
                                                                                  get_df_memory(train_label_arg))),
                                    get_optional_code_info_or_none(estimator.mlinspect_optional_code_reference,
                                                                   estimator.mlinspect_optional_source_code),
                                    process_func)
    function_call_result = FunctionCallResult(result)
    add_dag_node(train_labels_dag_node, [input_info_train_labels.dag_node], function_call_result)
    train_labels_result = function_call_result.function_result
    return function_call_result, train_labels_dag_node, train_labels_result


def add_train_data_node(estimator, train_data_arg, function_info):
    """Add a Train Label DAG Node for a estimator.fit call"""
    input_info_train_data = get_input_info(train_data_arg, estimator.mlinspect_caller_filename,
                                           estimator.mlinspect_lineno, function_info,
                                           estimator.mlinspect_optional_code_reference,
                                           estimator.mlinspect_optional_source_code)
    columns = input_info_train_data.dag_node.details.columns
    operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info, {})
    operator_call_info = OperatorCallInfo(operator_context,
                                          [input_info_train_data.dag_node])
    train_data_op_id = _pipeline_executor.singleton.get_next_op_id(operator_call_info)
    process_func = lambda df_object: df_object

    initial_func = partial(process_func, train_data_arg)
    _, result = capture_optimizer_info(singleton, operator_call_info, initial_func)
    train_data_dag_node = DagNode(train_data_op_id,
                                  BasicCodeLocation(estimator.mlinspect_caller_filename, estimator.mlinspect_lineno),
                                  operator_context,
                                  DagNodeDetails(None, columns, OptimizerInfo(0, get_df_shape(train_data_arg),
                                                                                get_df_memory(train_data_arg))),
                                  get_optional_code_info_or_none(estimator.mlinspect_optional_code_reference,
                                                                 estimator.mlinspect_optional_source_code),
                                  process_func)
    function_call_result = FunctionCallResult(result)
    add_dag_node(train_data_dag_node, [input_info_train_data.dag_node], function_call_result)
    train_data_result = function_call_result.function_result
    return function_call_result, train_data_dag_node, train_data_result


def add_test_data_dag_node(test_data_arg, function_info, lineno, optional_code_reference, optional_source_code,
                           caller_filename):
    """Add a Test Data DAG Node for a estimator.score call"""
    input_info_test_data = get_input_info(test_data_arg, caller_filename, lineno, function_info,
                                          optional_code_reference, optional_source_code)
    columns = input_info_test_data.dag_node.details.columns
    operator_context = OperatorContext(OperatorType.TEST_DATA, function_info, {})
    operator_call_info = OperatorCallInfo(operator_context,
                                          [input_info_test_data.dag_node])
    test_data_op_id = _pipeline_executor.singleton.get_next_op_id(operator_call_info)
    process_func = lambda df_object: df_object
    initial_func = partial(process_func, test_data_arg)
    _, result = capture_optimizer_info(singleton, operator_call_info, initial_func)
    test_data_dag_node = DagNode(test_data_op_id,
                                 BasicCodeLocation(caller_filename, lineno),
                                 operator_context,
                                 DagNodeDetails(None, columns,
                                                OptimizerInfo(0, get_df_shape(test_data_arg),
                                                              get_df_memory(test_data_arg))),
                                 get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                 process_func)
    function_call_result = FunctionCallResult(result)
    add_dag_node(test_data_dag_node, [input_info_test_data.dag_node], function_call_result)
    test_data_result = function_call_result.function_result
    return function_call_result, test_data_dag_node, test_data_result


def add_test_label_node(test_label_arg, caller_filename, function_info, lineno, optional_code_reference,
                        optional_source_code):
    """Add a Test Label DAG Node for a estimator.score call"""
    input_info_test_labels = get_input_info(test_label_arg, caller_filename, lineno, function_info,
                                            optional_code_reference, optional_source_code)
    operator_context = OperatorContext(OperatorType.TEST_LABELS, function_info, {})
    operator_call_info = OperatorCallInfo(operator_context,
                                          [input_info_test_labels.dag_node])
    columns = input_info_test_labels.dag_node.details.columns
    test_label_op_id = _pipeline_executor.singleton.get_next_op_id(operator_call_info)
    process_func = lambda df_object: df_object
    initial_func = partial(process_func, test_label_arg)
    _, result = capture_optimizer_info(singleton, operator_call_info, initial_func)
    test_labels_dag_node = DagNode(test_label_op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails(None, columns,
                                                  OptimizerInfo(0, get_df_shape(test_label_arg),
                                                                get_df_memory(test_label_arg))),
                                   get_optional_code_info_or_none(optional_code_reference,
                                                                  optional_source_code),
                                   process_func)
    function_call_result = FunctionCallResult(result)
    add_dag_node(test_labels_dag_node, [input_info_test_labels.dag_node], function_call_result)
    test_labels_result = function_call_result.function_result
    return function_call_result, test_labels_dag_node, test_labels_result


def get_simple_non_data_kwargs(*args, except_indices=None, except_kws=None, **kwargs):
    # We need a non_data_kwarg dict to check if a function has been called with the same non-data arguments before
    # TODO: This is a quick hack that saves a lot of time for now
    #  Ideally, unnamed args should not exist. But this requires going through every single monkey patch, which we
    #  do not want to do right now.
    kwargs = kwargs.copy()
    if except_kws is not None:
        for kw in except_kws:
            kwargs.pop(kw)
    for arg_index, arg_value in enumerate(args):
        kwargs[str(arg_index)] = arg_value
    if except_indices is not None:
        for index in except_indices:
            kwargs.pop(str(index))
    return kwargs



