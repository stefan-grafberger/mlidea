from mlidea import DagNode, OperatorContext, OperatorType, DagNodeDetails


def get_intermediate_extraction_node(singleton, dag_node, label: str):
    """Add a new node behind some given node to extract the intermediate result of that given node"""

    def extract_intermediate(intermediate_value):
        singleton.labels_to_extracted_plan_results[label] = intermediate_value
        return intermediate_value

    new_extraction_node = DagNode(singleton.get_next_op_id(),
                                  dag_node.code_location,
                                  OperatorContext(OperatorType.EXTRACT_RESULT, None),
                                  DagNodeDetails(None, dag_node.details.columns),
                                  None,
                                  extract_intermediate)
    return new_extraction_node


def copy_node_with_new_id(singleton, dag_node):
    result = DagNode(singleton.get_next_op_id(),
                     dag_node.code_location,
                     dag_node.operator_info,
                     dag_node.details,
                     dag_node.optional_code_info,
                     dag_node.processing_func,
                     dag_node.make_classifier_func)
    return result