import dataclasses

from mlidea.instrumentation._operator_types import OperatorType, FunctionInfo
from mlidea.instrumentation._dag_node import OperatorContext


@dataclasses.dataclass(unsafe_hash=True)
class OperatorCallInfo:
    """
    A DAG Node
    """

    operator: OperatorType
    function_info: FunctionInfo
    non_data_kwargs: tuple[str, any]
    parent_node_ids: tuple[int] or None = None

    def __init__(self, operator_context: OperatorContext, parent_nodes: list[any]):
        self.operator = operator_context.operator
        self.function_info = operator_context.function_info
        self.non_data_kwargs = tuple(operator_context.non_data_kwargs.items())
        parent_node_ids = []
        for parent in parent_nodes:
            if hasattr(parent, 'dag_node'):
                parent_node_ids.append(parent.dag_node.node_id)
            elif hasattr(parent, 'node_id'):
                parent_node_ids.append(parent.node_id)
            else:
                raise NotImplementedError("TODO")
        self.parent_node_ids = tuple(parent_node_ids)
