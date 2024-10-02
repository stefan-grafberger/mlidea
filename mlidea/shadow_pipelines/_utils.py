import warnings
from copy import copy
from enum import Enum
from functools import partial

import duckdb
import networkx
import numpy
import pandas
from autocorrect import Speller
from deep_translator import GoogleTranslator
from fairlearn.metrics import MetricFrame
from sklearn.preprocessing import FunctionTransformer

from mlidea.instrumentation._operator_types import ConditionalResult
from mlidea import DagNode, OperatorContext, OperatorType, DagNodeDetails
from mlidea.shadow_pipelines.cached_text_transformer import CachedTextTransformer
from mlidea.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


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


def add_typos(column, fraction_to_typo, df):
    indices = numpy.arange(len(df))
    numpy.random.shuffle(indices)
    num_values_to_typo = int(len(df) * fraction_to_typo)
    indices_to_typo = indices[:num_values_to_typo]
    df = df.reset_index(drop=True)
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


def get_typo_adder(column, fraction_to_typo):

    warnings.filterwarnings('ignore')
    processing_func = partial(add_typos, column, fraction_to_typo)
    typo_adder = FunctionTransformer(processing_func)

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

    def fix_typos(bound_column, bound_spell, df):
        # df['tweet'] = df['tweet'].map(lambda txt: str(TextBlob(txt).correct()))
        df[bound_column] = df[column].map(bound_spell)
        return df

    processing_func = partial(fix_typos, column, spell)
    warnings.filterwarnings('ignore')
    typo_fixer = FunctionTransformer(processing_func)
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

    ordered_new_scores = [new_node for _, new_node in
                          sorted(list(mapping.items()), key=lambda old_new_tuple: old_new_tuple[0].node_id)
                          if new_node.operator_info.operator == OperatorType.SCORE]

    return set(mapping.keys()), set(mapping.values()), ordered_new_scores


def filter_estimator_transformer_edges(parent, child):
    """Filter edges that are not relevant to the actual data flow but only to estimator/transformer state"""
    is_transformer_edge = ((parent.operator_info.operator == OperatorType.TRANSFORMER
                            and ": fit_transform" in parent.details.description
                            and child.operator_info.operator == OperatorType.TRANSFORMER
                            and ": transform" in child.details.description) or
                           (parent.operator_info.operator == OperatorType.ESTIMATOR
                            and child.operator_info.operator == OperatorType.PREDICT))
    return not is_transformer_edge


def get_conditional_stop_node(singleton, condition_func, label, description, parent_node):
    def check_condition(bound_condition_func, bound_label, *inputs):
        condition_bool = bound_condition_func(*inputs)
        if condition_bool is True:
            result = ConditionalResult.CONTINUE_EXECUTION
        else:
            result = ConditionalResult.STOP_EXECUTION
        singleton.labels_to_extracted_plan_results[bound_label] = condition_bool
        return result

    processing_func = partial(check_condition, condition_func, label)

    new_extraction_node = DagNode(singleton.get_next_op_id(),
                                  parent_node.code_location,
                                  OperatorContext(OperatorType.CONDITIONAL_STOP, None),
                                  DagNodeDetails(description, parent_node.details.columns),
                                  None,
                                  processing_func)
    return new_extraction_node


class DataType(Enum):
    """
    The different data types that we base our error detection techniques on
    """
    NUM = "numerical"
    CAT = "categorical"
    TEXT = "text"


TRANSFORMER_TO_DATA_TYPES = {
    "One-Hot": DataType.CAT,
    "Word2Vec": DataType.TEXT,
    "Standard Scaler": DataType.NUM
}


def get_transformer_operators_to_test(dag):
    """
    For now, we will ignore project modifies and focus on selections and transformers.
    This is because for transformers it is easy to find the corresponding test set operation and for the
    selection we do not need to worry about finding corresponding test set operations.
    """
    # This only works for traditional ML of course and not LLMs
    # pylint: disable=redefined-variable-type
    search_start_node = find_train_or_test_pipeline_part_end(dag, False)
    nodes_to_search = set(networkx.ancestors(dag, search_start_node))
    # Maybe start with outliers and text typos
    transformers_to_test = [node for node in nodes_to_search if
                            node.operator_info.operator == OperatorType.TRANSFORMER
                            and ": transform" in node.details.description
                            ]
    data_parent_and_data_type = []
    for transformer in transformers_to_test:
        for transformer_desc, data_type in TRANSFORMER_TO_DATA_TYPES.items():
            if transformer_desc in transformer.details.description:
                data_parent = get_sorted_parent_nodes(dag, transformer)[1]
                data_parent_and_data_type.append((data_parent, transformer, data_type))

    # A simple heuristic for now to detect embedding operations in FunctionTransformers in pipelines like
    #  anhedonia_ml
    function_transformers = [node for node in nodes_to_search if
                             node.operator_info.operator == OperatorType.TRANSFORMER
                             and "Function Transformer: transform" in node.details.description]
    for function_transformer in function_transformers:
        data_parent = get_sorted_parent_nodes(dag, function_transformer)[1]
        if (data_parent.details.optimizer_info.shape[1] == 1 and
                function_transformer.details.optimizer_info.shape[1] >= 100):
            data_parent_and_data_type.append((data_parent, function_transformer, DataType.TEXT))
    return data_parent_and_data_type


def get_translate_transformer(column, database_path):
    translator = GoogleTranslator(source='auto', target='en')

    # translator = MyMemoryTranslator(source='auto', target='en-US')
    # Could also use HuggingFace, but then it would be even slower probably
    # https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
    def translate(df, bound_column):
        # df['tweet'] = df['tweet'].map(lambda txt: translator.translate(txt))
        if isinstance(df, pandas.DataFrame):
            df[bound_column] = translator.translate_batch(df[bound_column].to_list())
        else:
            df = translator.translate_batch(df)
        # TODO: Is this fast enough?
        return df

    translate = partial(translate, bound_column=column)
    warnings.filterwarnings('ignore')
    translate_transformer = FunctionTransformer(translate)
    # TODO: What to do with this? Where to store this savefile?
    translate_transformer = CachedTextTransformer(translate_transformer,
                                                  database_path=database_path)
    return translate_transformer


def get_relative_score_change(*old_scores_and_new_scores, max_not_min=True):
    # This function compares all scores of the original pipeline and the changed pipeline
    # So the number of scores in both pipeline variants should be equal
    assert len(old_scores_and_new_scores) % 2 == 0
    number_of_scores_each = int(len(old_scores_and_new_scores) / 2)
    score_differences = []
    for score_index in range(number_of_scores_each):
        if isinstance(old_scores_and_new_scores[score_index], float):
            score_difference = (old_scores_and_new_scores[score_index + number_of_scores_each] /
                                old_scores_and_new_scores[score_index]
                                if old_scores_and_new_scores[score_index] else 0)
        elif isinstance(old_scores_and_new_scores[score_index], MetricFrame):
            score_difference = (old_scores_and_new_scores[score_index + number_of_scores_each].overall /
                                old_scores_and_new_scores[score_index].overall
                                if old_scores_and_new_scores[score_index].overall else 0)
        else:
            raise NotImplementedError("TODO")
        score_differences.append(score_difference)
    if max_not_min is True:
        result = max(score_differences)
    else:
        result = min(score_differences)
    return result


def add_orig_score_extraction_nodes(singleton, new_dag, score_operators):
    for score_index, score_operator in enumerate(score_operators):
        orig_extraction_node = get_intermediate_extraction_node(singleton, score_operator,
                                                                f"orig-{score_index}")
        new_dag.add_edge(score_operator, orig_extraction_node, arg_index=0)


def apply_diff_filter(input_df, corrupted_index):
    # TODO
    if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
        input_df = input_df.reset_index(drop=True)
    if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
        corrupted_diff = input_df.iloc[corrupted_index]
    elif isinstance(input_df, list):
        corrupted_diff = numpy.array(input_df)[corrupted_index]
    elif isinstance(input_df, tuple) and len(input_df) == 8:  # RAG Join Result
        corrupted_diff = list(copy(input_df))
        corrupted_diff[2] = list(numpy.array(corrupted_diff[2])[corrupted_index])
        corrupted_diff[3] = list(numpy.array(corrupted_diff[3])[corrupted_index])
        corrupted_diff[6] = corrupted_diff[6][corrupted_index, :]
        corrupted_diff = tuple(corrupted_diff)
    else:
        corrupted_diff = input_df[corrupted_index]
    if isinstance(corrupted_diff, (pandas.Series, pandas.DataFrame)):
        corrupted_diff = corrupted_diff.reset_index(drop=True)
    corrupted_diff = wrap_in_mlinspect_array_if_necessary(corrupted_diff)
    corrupted_diff._mlinspect_provenance = None

    return corrupted_diff


def projection(column_names, input):
    # TODO: What if not all inputs are pandas dfs?
    result = input[column_names]
    result = wrap_in_mlinspect_array_if_necessary(result)
    return result


def changed_data_diff_detection(input_df, corrupted_result):
    if isinstance(input_df, (pandas.Series, pandas.DataFrame)):
        input_df = input_df.reset_index(drop=True)
    if isinstance(corrupted_result, (pandas.Series, pandas.DataFrame)):
        corrupted_result = corrupted_result.reset_index(drop=True)
    if isinstance(input_df, pandas.Series):
        corrupt_diff_mask = (corrupted_result != input_df).to_numpy()
    elif isinstance(input_df, list):
        corrupt_diff_mask = numpy.array(corrupted_result) != numpy.array(input_df)
    else:
        corrupt_diff_mask = numpy.any(corrupted_result != input_df, axis=1)
    changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
    return changed_indices_corrupt
