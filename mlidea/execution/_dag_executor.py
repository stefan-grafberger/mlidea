"""
The place where the DAG execution happens
"""
import dataclasses
from copy import copy
from functools import partial

import networkx

from mlidea.instrumentation._operator_call_info import OperatorCallInfo
from mlidea.execution._stat_tracking import capture_optimizer_info
from mlidea.instrumentation._operator_types import OperatorType, ConditionalResult
from mlidea.instrumentation._dag_node import DagNode, OptimizerInfo
from mlidea.utils._utils import get_sorted_parent_nodes


@dataclasses.dataclass(frozen=True)
class DagNodeResult:
    """ Holds the result from a processing_func after a DagNode has been executed """
    node_id: int = dataclasses.field(default=None)
    dag_node: DagNode = dataclasses.field(default=None)
    result_df: any = dataclasses.field(hash=False, default=None)


class DagExecutor:
    """ Executes given DAGs using the processing_funcs started with each DagNode """

    def __init__(self, pipeline_executor):
        self.pipeline_executor = pipeline_executor

    def execute(self, dag: networkx.DiGraph, use_dfs_exec_strategy: bool = False):
        """ Execute a given input DAG """
        # TODO: Currently, this returns the final result from some DagNode without children but in the future,
        #  we want to have a mechanism to store the results from selected DagNodes with a label in some result map
        dag = dag.copy()

        def execute_node(current_node: DagNode):
            if current_node.operator_info.operator == OperatorType.MISSING_OP:
                raise NotImplementedError(f"Missing Ops not supported currently! The operator: {current_node}")
            parent_nodes = get_sorted_parent_nodes(dag, current_node)
            inputs = self.get_required_values(dag, current_node, parent_nodes)
            operator_call_info = OperatorCallInfo(current_node.operator_info, parent_nodes)
            stop_signal_received = False
            for input_index, input_val in enumerate(inputs):
                if isinstance(input_val, ConditionalResult):
                    if input_val == ConditionalResult.STOP_EXECUTION:
                        stop_signal_received = True
                    else:
                        # A ConditionalResult.CONTINUE should never be propagated further, and can only occur directly
                        #  from conditional nodes. However, conditional nodes are always the input node with the
                        #  highest arg_index, the last argument of some other node
                        assert input_index == len(inputs) - 1
                        inputs = inputs[:-1]
            # This is necessary because these two node types extract results
            if stop_signal_received is False:
                executable_processing_func = partial(current_node.processing_func, *inputs)
                extract_or_conditional = current_node.operator_info.operator in {
                    OperatorType.EXTRACT_RESULT, OperatorType.CONDITIONAL_STOP}
                optimizer_info, result_df = capture_optimizer_info(self.pipeline_executor, operator_call_info,
                                                                   executable_processing_func,
                                                                   force_disable_reuse=extract_or_conditional)
            elif current_node.operator_info.operator == OperatorType.EXTRACT_RESULT:
                executable_processing_func = partial(current_node.processing_func, ConditionalResult.STOP_EXECUTION)
                _, result_df = capture_optimizer_info(self.pipeline_executor, operator_call_info,
                                                      executable_processing_func,
                                                      force_disable_reuse=True)
                optimizer_info = OptimizerInfo(None, None, None)  # We want to avoid the DAG from being confusing
            else:
                optimizer_info = OptimizerInfo(None, None, None)
                result_df = ConditionalResult.STOP_EXECUTION
            self.pipeline_executor.operators_to_runtime_during_analysis[copy(current_node)] = optimizer_info

            if self.pipeline_executor.enable_caching is True:
                self.pipeline_executor.operator_context_parents_to_result[
                    OperatorCallInfo(current_node.operator_info, parent_nodes)] = current_node
                self.pipeline_executor.cached_intermediates[current_node] = result_df

            result = self.replace_node_with_result(dag, current_node, result_df)
            return result

        if use_dfs_exec_strategy is False:
            self.traverse_graph_and_process_nodes_bfs(dag, execute_node)
        else:
            self.traverse_graph_and_process_nodes_dfs(dag, execute_node)

    @staticmethod
    def traverse_graph_and_process_nodes_bfs(graph: networkx.DiGraph, func):
        """
        Traverse the DAG node by node from top to bottom
        """
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
        processed_nodes = set()
        while len(current_nodes) != 0:
            node = current_nodes.pop(0)
            processed_nodes.add(node.node_id)
            result_node = func(node)
            if result_node is not None:
                children = list(graph.successors(result_node))
                # Nodes can have multiple parents, only want to process them once we processed all parents
                for child in children:
                    if child.node_id not in processed_nodes:
                        predecessors = [predecessor.node_id for predecessor in graph.predecessors(child)]
                        if processed_nodes.issuperset(predecessors):
                            current_nodes.append(child)

        return graph

    @staticmethod
    def traverse_graph_and_process_nodes_dfs(graph: networkx.DiGraph, func):
        """
        Traverse the DAG node by node from top to bottom
        """
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
        processed_nodes = set()
        while len(current_nodes) != 0:
            node = current_nodes.pop(-1)
            processed_nodes.add(node.node_id)
            result_node = func(node)
            if result_node is not None:
                children = list(graph.successors(result_node))
                # Nodes can have multiple parents, only want to process them once we processed all parents
                for child in children:
                    if child.node_id not in processed_nodes:
                        predecessors = [predecessor.node_id for predecessor in graph.predecessors(child)]
                        if processed_nodes.issuperset(predecessors):
                            current_nodes.append(child)

        return graph

    @staticmethod
    def replace_node_with_result(sub_dag, dag_node: DagNode, result_df):
        """ This replaces a DAG node with the result from its processing_func """
        new_value_node = DagNodeResult(dag_node.node_id, dag_node, result_df)
        sub_dag.add_node(new_value_node)
        for parent_node in sub_dag.predecessors(dag_node):
            edge_data = sub_dag.get_edge_data(parent_node, dag_node)
            sub_dag.add_edge(parent_node, new_value_node, **edge_data)
        for child_node in sub_dag.successors(dag_node):
            edge_data = sub_dag.get_edge_data(dag_node, child_node)
            sub_dag.add_edge(new_value_node, child_node, **edge_data)
        sub_dag.remove_node(dag_node)
        return new_value_node

    @staticmethod
    def get_required_values(sub_dag: networkx.DiGraph, current_node: DagNode, parent_nodes: list[DagNode]):
        """
        This gets all required input values for the processing_func of a dag_node from its DagNode parents.
        Deletes results from parents that are no longer required.
        """
        required_df_values = []
        for parent_node in parent_nodes:
            assert isinstance(parent_node, DagNodeResult)
            df_value = parent_node.result_df
            sub_dag.remove_edge(parent_node, current_node)
            # We want to enable garbage collection of value_node if we no longer need to keep the value around
            if not list(sub_dag.successors(parent_node)):
                sub_dag.remove_node(parent_node)
            required_df_values.append(df_value)
        return required_df_values
