import dataclasses
from enum import Enum

from mlidea.instrumentation._operator_types import OperatorType, FunctionInfo
from mlidea.instrumentation._dag_node import OperatorContext


@dataclasses.dataclass(unsafe_hash=True)
class OperatorCallInfo:
    """
    A DAG Node
    """

    operator: OperatorType
    function_info: FunctionInfo
    non_data_kwargs: tuple[str, str]
    parent_node_ids: tuple[int] or None = None

    def __init__(self, operator_context: OperatorContext, parent_nodes: list[any]):
        self.operator = operator_context.operator
        self.function_info = operator_context.function_info
        hashable_non_data_kwargs = []
        if isinstance(operator_context.non_data_kwargs, dict):
            for kwarg_key, kwarg_value in operator_context.non_data_kwargs.items():
                # if isinstance(kwarg_value, list):
                #     kwarg_value = tuple(kwarg_value)
                # if isinstance(kwarg_value, tuple) and isinstance(kwarg_value[0], slice):
                #     kwarg_value = (kwarg_value[0].start, kwarg_value[0].step, kwarg_value[0].stop),
                hashable_non_data_kwargs.append((kwarg_key, str(kwarg_value)))
        elif isinstance(operator_context.non_data_kwargs, tuple):
            hashable_non_data_kwargs = operator_context.non_data_kwargs
        else:
            raise NotImplementedError("TODO")
        self.non_data_kwargs = tuple(hashable_non_data_kwargs)
        parent_node_ids = []
        for parent in parent_nodes:
            if hasattr(parent, 'dag_node'):
                parent_node_ids.append(parent.dag_node.node_id)
            elif hasattr(parent, 'node_id'):
                parent_node_ids.append(parent.node_id)
            elif isinstance(parent, int):
                parent_node_ids.append(parent)
            else:
                raise NotImplementedError("TODO")
        self.parent_node_ids = tuple(parent_node_ids)

    def __eq__(self, __value):
        # FIXME: No idea why this is necessary
        return (self.operator == __value.operator and self.function_info == __value.function_info
                and self.non_data_kwargs == __value.non_data_kwargs and self.parent_node_ids == __value.parent_node_ids)

class OutputChangeType(Enum):
    """
    The different operator types in our DAG
    """
    TOO_MUCH_CHANGED = "Too much changed"
    COLUMNS_CHANGED = "Changed columns"
    ROWS_UPDATED = "Updated rows"
    ROWS_ADDED = "Added rows"
    ROWS_REMOVED = "Removed rows"
    NOTHING_CHANGED = "No change"
    UNKNOWN = "Unknown"


@dataclasses.dataclass
class OperatorOutputChange:
    change_type: OutputChangeType
    columns_changed: list[str] or None = None
    rows_updated: list[int] or None = None
    rows_added: list[int] or None = None
    rows_removed: list[int] or None = None
