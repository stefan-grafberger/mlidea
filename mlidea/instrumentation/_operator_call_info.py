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
        hashable_non_data_kwargs = []
        for kwarg_key, kwarg_value in operator_context.non_data_kwargs.items():
            # if isinstance(kwarg_value, list):
            #     kwarg_value = tuple(kwarg_value)
            # if isinstance(kwarg_value, tuple) and isinstance(kwarg_value[0], slice):
            #     kwarg_value = (kwarg_value[0].start, kwarg_value[0].step, kwarg_value[0].stop),
            hashable_non_data_kwargs.append((kwarg_key, str(kwarg_value)))
        self.non_data_kwargs = tuple(hashable_non_data_kwargs)
        parent_node_ids = []
        for parent in parent_nodes:
            if hasattr(parent, 'dag_node'):
                parent_node_ids.append(parent.dag_node.node_id)
            elif hasattr(parent, 'node_id'):
                parent_node_ids.append(parent.node_id)
            else:
                raise NotImplementedError("TODO")
        self.parent_node_ids = tuple(parent_node_ids)
