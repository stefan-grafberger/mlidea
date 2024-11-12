"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""

import networkx

from mlidea.instrumentation._dag_node import OperatorContext
from mlidea.instrumentation._operator_call_info import OperatorCallInfo, OperatorOutputChange, OutputChangeType
from mlidea.instrumentation._operator_types import OperatorType
from mlidea.utils._utils import get_sorted_parent_nodes


def determine_parents_compared_to_previous_dag(operator_call_info, singleton):
    parent_nodes_from_previous_run = []
    changes = []
    for parent_index, parent_op_id in enumerate(operator_call_info.parent_node_ids):
        old_dag = singleton.global_old_dag
        new_dag = singleton.global_new_dag

        new_dag_parent_node = [node for node in new_dag.nodes if node.node_id == parent_op_id][0]

        # Create operator call info
        new_dag_parent_operator_call_info = dag_node_to_operator_call_info(
            new_dag, new_dag_parent_node)
        unprocessed_transitive_change = new_dag_parent_operator_call_info in singleton.reuse_info.unprocessed_call_info_transitive_change_only
        is_undetermined = new_dag_parent_operator_call_info in singleton.reuse_info.undetermined_new_nodes
        assert unprocessed_transitive_change is False or is_undetermined is False
        if unprocessed_transitive_change:  # We need to delay processing them to ensure consecutive dag node ids
            process_transitive_change(new_dag_parent_node, new_dag_parent_operator_call_info, old_dag, singleton)
        elif is_undetermined:
            determine_parent_change_type(new_dag, new_dag_parent_node, new_dag_parent_operator_call_info, old_dag,
                                         operator_call_info, parent_index, singleton)

        assert new_dag_parent_node in singleton.reuse_info.new_node_to_old_node
        corresponding_node_in_old_dag, change_diff = singleton.reuse_info.new_node_to_old_node[new_dag_parent_node]
        parent_nodes_from_previous_run.append(corresponding_node_in_old_dag)
        changes.append(change_diff)
    return parent_nodes_from_previous_run


def process_transitive_change(new_dag_parent_node, new_dag_parent_operator_call_info, old_dag, singleton):
    old_operator_call_info, change_type = singleton.reuse_info.unprocessed_call_info_transitive_change_only[
        new_dag_parent_operator_call_info]
    if (old_operator_call_info in singleton.reuse_info.operator_call_info_to_dag_node and
            len([node for node in old_dag.nodes
                 if node.node_id == singleton.get_next_op_id(old_operator_call_info)]) > 0):
        old_dag_node_id = singleton.get_next_op_id(old_operator_call_info)
        dag_node_to_map_to = [node for node in old_dag.nodes if node.node_id == old_dag_node_id][0]
        singleton.reuse_info.operator_transitive.add(new_dag_parent_node)
    else:
        dag_node_to_map_to = new_dag_parent_node  # Mapping to old DAG failed
        singleton.reuse_info.operator_too_many_changes.add(new_dag_parent_node)
    singleton.reuse_info.new_node_to_old_node[new_dag_parent_node] = dag_node_to_map_to, change_type
    singleton.reuse_info.unprocessed_call_info_transitive_change_only.pop(new_dag_parent_operator_call_info)


def determine_parent_change_type(new_dag, new_dag_parent_node, new_dag_parent_operator_call_info, old_dag,
                                 operator_call_info, parent_index, singleton):
    singleton.reuse_info.undetermined_new_nodes.remove(new_dag_parent_operator_call_info)
    # Actually re-executed without any chance of IVM
    singleton.reuse_info.operator_reexecuted.add(new_dag_parent_node)

    # Determine the type of change
    is_replacement, node_being_replaced = determine_is_replacement(new_dag,
                                                                   new_dag_parent_node, old_dag,
                                                                   operator_call_info, parent_index,
                                                                   singleton.reuse_info.operator_call_info_to_dag_node,
                                                                   singleton.reuse_info.new_node_to_old_node)

    if not is_replacement:
        is_addition, node_being_added_to = determine_is_addition(new_dag, new_dag_parent_node,
                                                                 operator_call_info, parent_index,
                                                                 singleton.reuse_info.operator_call_info_to_dag_node,
                                                                 singleton.reuse_info.new_node_to_old_node)
    else:
        is_addition, node_being_added_to = False, None
    if not is_replacement and not is_addition:
        is_deletion, deleted_node_child = determine_is_deletion(new_dag, new_dag_parent_node, old_dag)
    else:
        is_deletion, deleted_node_child = False, None

    if is_replacement:
        singleton.reuse_info.operator_replacement.add(new_dag_parent_node)
        change_diff = OperatorOutputChange(
            OutputChangeType.TOO_MUCH_CHANGED)  # FIXME: We also need to compute the actual changes!
        singleton.reuse_info.new_node_to_old_node[new_dag_parent_node] = node_being_replaced, change_diff
    elif is_addition:
        singleton.reuse_info.operator_addition.add(new_dag_parent_node)
        change_diff = OperatorOutputChange(
            OutputChangeType.TOO_MUCH_CHANGED)  # FIXME: We also need to compute the actual changes!
        singleton.reuse_info.new_node_to_old_node[new_dag_parent_node] = node_being_added_to, change_diff
    elif is_deletion:
        singleton.reuse_info.operator_deletion.add(new_dag_parent_node)
        change_diff = OperatorOutputChange(
            OutputChangeType.TOO_MUCH_CHANGED)  # FIXME: We also need to compute the actual changes!
        singleton.reuse_info.new_node_to_old_node[new_dag_parent_node] = deleted_node_child, change_diff
    else:
        singleton.reuse_info.operator_too_many_changes.add(new_dag_parent_node)
        singleton.reuse_info.new_node_to_old_node[
            new_dag_parent_node] = new_dag_parent_node, OperatorOutputChange(
            OutputChangeType.TOO_MUCH_CHANGED)


def determine_is_deletion(new_dag, new_dag_parent_node, old_dag):
    # We can check the old DAG: if new_dag_parent_node is in the old DAG, but has a parent that does
    #  not exist in the new DAG, but if the parent parent exists in the new DAG
    is_deletion = False
    deleted_node_child = None
    # FIXME: Think about using replacement map here
    # Step 1: Confirm the node exists in both DAGs
    # candidates for current node, ignoring the node id
    candidates = [node for node in old_dag.nodes if (node.operator_info == new_dag_parent_node.operator_info and
                                                     node.details == new_dag_parent_node.details)]

    for candidate in candidates:
        # Step 2: Check each parent in the old DAG if the parent is missing in the new DAG
        candidate_old_parents_not_in_new_dag = [parent for parent in old_dag.predecessors(candidate)
                                                if parent not in new_dag]
        for old_parent in candidate_old_parents_not_in_new_dag:
            # Check if the grandparent exists in the new DAG
            grand_parents_in_both_dags = [grandparent for grandparent in old_dag.predecessors(old_parent)
                                        if grandparent in new_dag and
                                        grandparent.operator_info.operator != OperatorType.SUBSCRIPT]
            for grandparent in grand_parents_in_both_dags:
                # However, maybe we want to make sure to look at all nodes in-between
                simple_paths = list(networkx.all_simple_paths(old_dag, grandparent, candidate))
                if len(simple_paths) != 0:
                    nodes_in_paths = set(node for path in simple_paths for node in path)
                    nodes_in_paths.remove(grandparent)
                    nodes_in_paths.remove(old_parent)
                    nodes_in_paths.remove(candidate)
                    if len([node for node in nodes_in_paths if
                            node.operator_info.operator not in {OperatorType.SUBSCRIPT,
                                                                OperatorType.PROJECTION}]) == 0:
                        is_deletion = True  # Found a deleted node's child with an existing grandparent
                        # deleted_node = parent
                        # deleted_node_parent = grandparent
                        deleted_node_child = candidate
    return is_deletion, deleted_node_child


def determine_is_addition(new_dag, new_dag_parent_node, operator_call_info, parent_index,
                          operator_call_info_to_dag_node, new_node_to_old_node):
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
    elif (len(parent_parents) == 2 and new_dag_parent_node.operator_info.operator in
        {OperatorType.PROJECTION_MODIFY, OperatorType.SELECTION}):
        is_addition = False

        before_addition_parent_candidate = networkx.lowest_common_ancestor(new_dag, parent_parents[0],
                                                                           parent_parents[1])
        before_addition_parent = None
        if before_addition_parent_candidate is not None:
            simple_paths = list(networkx.all_simple_paths(new_dag, before_addition_parent_candidate, new_dag_parent_node))
            nodes_in_paths = set(node for path in simple_paths for node in path)
            nodes_in_paths.discard(before_addition_parent_candidate)
            nodes_in_paths.discard(new_dag_parent_node)
            if len([node for node in nodes_in_paths if node.operator_info.operator not in {OperatorType.SUBSCRIPT,
                                                                                       OperatorType.PROJECTION,
                                                                                       OperatorType.PROJECTION_MODIFY}]) == 0:
                before_addition_parent = before_addition_parent_candidate

        if before_addition_parent is not None:
            old_parent_node_ids = []
            for arg_index, parent_node_id in enumerate(list(operator_call_info.parent_node_ids)):
                parent_node = [node for node in new_dag.nodes if node.node_id == parent_node_id][0]
                if arg_index == parent_index:
                    old_parent_node_ids.append(before_addition_parent.node_id)
                elif parent_node in new_node_to_old_node:
                    old_parent_node_ids.append(new_node_to_old_node[parent_node][0].node_id)
                else:
                    old_parent_node_ids.append(parent_node_id)

            test_addition_operator_call_info = OperatorCallInfo(
                OperatorContext(operator_call_info.operator,
                                operator_call_info.function_info,
                                operator_call_info.non_data_kwargs),
                old_parent_node_ids
            )
            is_addition = test_addition_operator_call_info in operator_call_info_to_dag_node
        result = is_addition, before_addition_parent
    else:
        result = False, None  # Fast updates for addition of operations like joins is not supported currently
    return result


def determine_is_replacement(new_dag, new_dag_parent_node, old_dag, operator_call_info, parent_index,
                             operator_call_info_to_dag_node, new_node_to_old_node):
    # to compute is_replacement, we check the parents to new_dag_parent_operator_call_info and operator
    #  type and look in the old DAG if we can find a similar operation there with the same child and
    #  the same parents

    # Gather attributes and relationships for the new node
    is_replacement = False  # Default: No replacement found
    node_being_replaced = None  # Default: No replacement found
    new_node_type = new_dag_parent_node.operator_info.operator
    new_parents = {new_node_to_old_node[node][0] if node in new_node_to_old_node else node
                   for node in new_dag.predecessors(new_dag_parent_node)}
    old_nodes_already_matched = {old_node for old_node, _ in new_node_to_old_node.values()}
    # Search for a similar node in the old DAG
    for old_node in old_dag.nodes:
        # Check if operator type matches
        if old_node.operator_info.operator == new_node_type and old_node not in old_nodes_already_matched:
            # Check if parents and children match
            old_parents = set(old_dag.predecessors(old_node))
            # Not needed for now because old_children_contains_current_node does this part of the check
            # old_children = set(old_dag.successors(old_node))

            old_parent_node_ids = []
            for arg_index, parent_node_id in enumerate(list(operator_call_info.parent_node_ids)):
                parent_node = [node for node in new_dag.nodes if node.node_id == parent_node_id][0]
                if arg_index == parent_index:
                    old_parent_node_ids.append(old_node.node_id)
                elif parent_node in new_node_to_old_node:
                    old_parent_node_ids.append(new_node_to_old_node[parent_node][0].node_id)
                else:
                    old_parent_node_ids.append(parent_node_id)
            old_children_contains_current_node = OperatorCallInfo(
                OperatorContext(operator_call_info.operator, operator_call_info.function_info,
                                operator_call_info.non_data_kwargs),
                old_parent_node_ids
            ) in operator_call_info_to_dag_node
            if new_parents == old_parents and old_children_contains_current_node:
                is_replacement = True  # Found a 1-to-1 replacement in the old DAG
                node_being_replaced = old_node
                break
    return is_replacement, node_being_replaced


def dag_node_to_operator_call_info(dag, node):
    new_dag_parent_parents = get_sorted_parent_nodes(dag, node)
    new_dag_parent_operator_call_info = OperatorCallInfo(node.operator_info, new_dag_parent_parents)
    return new_dag_parent_operator_call_info
