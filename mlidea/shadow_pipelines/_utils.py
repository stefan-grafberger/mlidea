import warnings
from functools import partial

import duckdb
import networkx
import numpy
from autocorrect import Speller
from sklearn.preprocessing import FunctionTransformer

from mlidea.instrumentation._operator_types import ConditionalResult
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


def get_sorted_parent_nodes(dag: networkx.DiGraph, first_op_requiring_corruption):
    """Get the parent nodes of a node sorted by arg_index"""
    operator_parent_nodes = list(dag.predecessors(first_op_requiring_corruption))
    parent_nodes_with_arg_index = [(parent_node, dag.get_edge_data(parent_node, first_op_requiring_corruption))
                                   for parent_node in operator_parent_nodes]
    parent_nodes_with_arg_index = sorted(parent_nodes_with_arg_index, key=lambda x: x[1]['arg_index'])
    operator_parent_nodes = [node for (node, _) in parent_nodes_with_arg_index]
    return operator_parent_nodes


def find_nodes_by_type(new_dag: networkx.DiGraph, operator_type: OperatorType):
    """Find DagNodes in the DAG by OperatorType"""
    return [node for node in new_dag.nodes if node.operator_info.operator == operator_type]


def find_train_or_test_pipeline_part_end(dag, train_not_test):
    """We want to start at the end of the pipeline to find the relevant train or test operations"""
    if train_not_test is True:
        search_start_nodes = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
        if len(search_start_nodes) == 0:
            search_start_nodes = find_nodes_by_type(dag, OperatorType.RAG_JOIN)
            if len(search_start_nodes) != 1:
                raise NotImplementedError(
                    "Currently, DataCorruption only supports pipelines with exactly one estimator or RAG!")
            search_start_node = search_start_nodes[0]
            search_start_node = get_sorted_parent_nodes(dag, search_start_node)[0]
        elif len(search_start_nodes) != 1:
            raise NotImplementedError("Currently, DataCorruption only supports pipelines with exactly one estimator "
                                      "or RAG!")
        else:
            search_start_node = search_start_nodes[0]
    else:
        search_start_nodes = find_nodes_by_type(dag, OperatorType.RAG_JOIN)
        if len(search_start_nodes) == 1:
            search_start_node = search_start_nodes[0]
            search_start_node = get_sorted_parent_nodes(dag, search_start_node)[-1]
        else:
            search_start_nodes = find_nodes_by_type(dag, OperatorType.PREDICT)
            if len(search_start_nodes) != 1:
                raise NotImplementedError("Currently, DataCorruption only supports pipelines with exactly one predict "
                                          "call for the test set or RAG!")
            search_start_node = search_start_nodes[0]
    return search_start_node


def get_typo_adder(column):
    fraction_to_typo = 0.1

    def add_typos(df):
        indices = numpy.arange(len(df))
        numpy.random.shuffle(indices)
        num_values_to_typo = int(len(df) * fraction_to_typo)
        indices_to_typo = indices[:num_values_to_typo]
        # df.loc[indices_to_typo, 'tweet'] = df.loc[indices_to_typo, 'tweet'].apply(lambda txt: typo_augmenter.augment(txt)[0])
        data_to_corrupt = df[[column]].iloc[indices_to_typo]
        data_to_corrupt['row_id'] = list(range(data_to_corrupt.shape[0]))
        # corrupted_data = duckdb.query("""
        #     SELECT regexp_replace(tweet, '')
        #     FROM data_to_corrupt
        # """).df()['tweet']
        corrupted_data = duckdb.query(f"""
                SELECT
                CASE
                WHEN random() < 0.3 THEN (
                    SELECT
                        REPLACE(
                            REPLACE(
                                REPLACE(
                                    REPLACE({column}, 'n', 'm'),
                                'b', 'v'),
                            't', 'r'),
                        'o', 'p'),
                )
                -- Introduce substitution errors
                WHEN random() < 0.3 THEN (
                    SELECT
                        STRING_AGG(
                            CASE
                                WHEN random() < 0.03 THEN chr(65 + (abs(ASCII(character)) + CAST(random() * 25 AS INT)) % 26)  -- Substitute with random character
                                ELSE character
                            END, ''
                        )
                    FROM
                        UNNEST(SPLIT({column}, '')) AS t(character)
                )
                -- Introduce insertion errors
                WHEN random() < 0.3 THEN (
                    SELECT
                        STRING_AGG(
                            CASE WHEN random() < 0.02 THEN CONCAT(character, chr(65 + (abs(ASCII(character)) + CAST(random() * 25 AS INT)) % 26))
                            ELSE character END,
                            ''
                        )
                    FROM
                        UNNEST(SPLIT({column}, '')) AS t(character)
                )
                -- Introduce deletion errors
                ELSE (
                    SELECT
                        STRING_AGG(
                            character,
                            ''
                        )
                    FROM
                        UNNEST(SPLIT({column}, '')) AS t(character)
                    WHERE
                        random() > 0.02
                )
                -- Introduce transposition errors
                -- WHEN random() < 0.1 THEN (
                --     SELECT
                --         STRING_AGG(
                --             CONCAT(
                --                 LEAST(character1, character2),
                --                 GREATEST(character1, character2)
                --             ),
                --             ''
                --         )
                --     FROM
                --         UNNEST(SPLIT({column}, '')) AS t(character1)
                --     LEFT JOIN
                --         UNNEST(SPLIT({column}, '')) AS u(character2)
                --     ON
                --         random() < 0.001
                -- )
                -- ELSE {column}
            END AS {column}
            FROM data_to_corrupt
            ORDER BY row_id
            """).df()[column]
        df[column].iloc[indices_to_typo] = corrupted_data
        return df

    warnings.filterwarnings('ignore')
    typo_adder = FunctionTransformer(add_typos)

    # typo_transformation = WordSwapQWERTY(random_one=False)
    # typo_transformation = CompositeTransformation(
    #     [WordSwapRandomCharacterDeletion(), WordSwapQWERTY()]
    # )
    # typo_transformation = WordSwapRandomCharacterSubstitution(random_one=True)
    # typo_augmenter = Augmenter(transformation=typo_transformation, fast_augment=True, transformations_per_example=4,
    #                            pct_words_to_swap=0.8)
    # def add_typos(df):
    #     indices = numpy.arange(len(df))
    #     numpy.random.shuffle(indices)
    #     num_values_to_typo = int(len(df) * fraction_to_typo)
    #     indices_to_typo = indices[:num_values_to_typo]
    #     df.loc[indices_to_typo, 'tweet'] = df.loc[indices_to_typo, 'tweet'].apply(lambda txt: typo_augmenter.augment(txt)[0])
    #     return df
    # warnings.filterwarnings('ignore')
    # typo_adder = FunctionTransformer(add_typos)

    # corrupted_tweet = BrokenCharacters(column='tweet', fraction=.2).transform(test)
    return typo_adder


def get_typo_fixer(column):
    # def fix_typos(df):
    #     df['tweet'] = df['tweet'].map(lambda txt: str(TextBlob(txt).correct()))
    #     # TODO: This is very slow. We might need to look into different libraries.
    #     return df
    # warnings.filterwarnings('ignore')
    # typo_fixer = FunctionTransformer(fix_typos)
    spell = Speller()

    def fix_typos(df):
        # df['tweet'] = df['tweet'].map(lambda txt: str(TextBlob(txt).correct()))
        df[column] = df[column].map(lambda txt: spell(txt))
        # TODO: This spellchecker is much faster. However, I am not entirely sure how good it is
        return df

    warnings.filterwarnings('ignore')
    typo_fixer = FunctionTransformer(fix_typos)
    return typo_fixer


def duplicate_descendants(original_dag, new_dag, original_node, modified_copy, singleton):
    # Create a mapping of old nodes to new nodes
    mapping = {original_node: modified_copy}

    # Get all descendants of the original node (children and their children recursively)
    descendants = networkx.descendants(original_dag, original_node)

    # Create a queue to process each node in topological order (to handle dependencies)
    queue = list(networkx.topological_sort(original_dag.subgraph(descendants)))
    # Iterate through all descendants and create a duplicate for each using your method
    for node in queue:
        if node.operator_info.operator != OperatorType.EXTRACT_RESULT:
            new_node = copy_node_with_new_id(singleton, node)

            # Store the mapping of original to duplicate
            mapping[node] = new_node

    # Copy the edges from the original subgraph to the duplicate subgraph, maintaining edge attributes
    for node in queue:
        if node.operator_info.operator != OperatorType.EXTRACT_RESULT:
            new_node = mapping[node]

            # Replicate edges from the original parents to the new duplicate nodes, preserving edge attributes
            for parent in original_dag.predecessors(node):
                if parent in mapping:
                    edge_data = original_dag.get_edge_data(parent, node)
                    new_dag.add_edge(mapping[parent], new_node, **edge_data)
                else:
                    edge_data = original_dag.get_edge_data(parent, node)
                    new_dag.add_edge(parent, new_node, **edge_data)

    return set(mapping.keys()), set(mapping.values())


def get_conditional_stop_node(singleton, condition_func, description, parent_node):
    def check_condition(condition_func, *inputs):
        condition_bool = condition_func(*inputs)
        if condition_bool is True:
            result = ConditionalResult.CONTINUE_EXECUTION
        else:
            result = ConditionalResult.STOP_EXECUTION
        return result

    processing_func = partial(check_condition, condition_func=condition_func)

    new_extraction_node = DagNode(singleton.get_next_op_id(),
                                  parent_node.code_location,
                                  OperatorContext(OperatorType.EXTRACT_RESULT, None),
                                  DagNodeDetails(description, parent_node.details.columns),
                                  None,
                                  processing_func)
    return new_extraction_node
