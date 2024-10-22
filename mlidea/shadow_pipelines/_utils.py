import warnings
from copy import copy
from enum import Enum
from functools import partial
from inspect import getframeinfo, currentframe

import duckdb
import networkx
import numpy
import pandas
from autocorrect import Speller
from deep_translator import GoogleTranslator
from fairlearn.metrics import MetricFrame
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer

from mlidea.instrumentation._dag_node import DagNode, OperatorContext, DagNodeDetails, BasicCodeLocation
from mlidea.instrumentation._operator_call_info import OperatorCallInfo
from mlidea.instrumentation._operator_types import ConditionalResult, FunctionInfo
from mlidea.instrumentation._operator_types import OperatorType
from mlidea.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary
from mlidea.monkeypatching._patch_langchain import RunnableSequencePatching
from mlidea.monkeypatching._provenance_propagation import wrap_projection_func
from mlidea.shadow_pipelines.cached_text_transformer import CachedTextTransformer


def get_intermediate_extraction_node(singleton, dag, dag_node, label: str):
    """Add a new node behind some given node to extract the intermediate result of that given node"""

    def extract_intermediate(intermediate_value):
        singleton.labels_to_extracted_plan_results[label] = intermediate_value
        return intermediate_value

    new_extraction_node = DagNode(singleton.get_next_op_id(None),
                                  dag_node.code_location,
                                  OperatorContext(OperatorType.EXTRACT_RESULT, None, {}),
                                  DagNodeDetails(None, dag_node.details.columns),
                                  None,
                                  extract_intermediate)
    dag.add_edge(dag_node, new_extraction_node, arg_index=0)
    return new_extraction_node


def copy_node_with_new_id(singleton, dag, dag_node_to_copy, new_parents):
    operator_call_info = OperatorCallInfo(dag_node_to_copy.operator_info, new_parents)
    result = DagNode(singleton.get_next_op_id(operator_call_info),
                     dag_node_to_copy.code_location,
                     dag_node_to_copy.operator_info,
                     dag_node_to_copy.details,
                     dag_node_to_copy.optional_code_info,
                     dag_node_to_copy.processing_func,
                     dag_node_to_copy.make_classifier_func)
    add_parent_node_edges(dag, result, new_parents)
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


def duplicate_descendants_and_filter_concat_inputs(singleton, original_dag, new_dag, original_node, modified_copy,
                                                   changed_indices_node, conditional_node, shadow_pipeline_name):
    # Create a mapping of old nodes to new nodes
    mapping = {original_node: modified_copy}
    all_new_nodes = {modified_copy}

    # Get all descendants of the original node (children and their children recursively)
    descendants = networkx.descendants(original_dag, original_node)

    # Create a queue to process each node in topological order (to handle dependencies)
    queue = list(networkx.topological_sort(original_dag.subgraph(descendants)))
    # Iterate through all descendants and create a duplicate for each
    for node in queue:
        if node.operator_info.operator != OperatorType.EXTRACT_RESULT:
            new_parents = []
            if node.operator_info.operator == OperatorType.CONCATENATION:
                for concat_parent in get_sorted_parent_nodes(original_dag, node):
                    if concat_parent not in all_new_nodes:  # Old nodes need to be filtered first
                        parents = [concat_parent, changed_indices_node, conditional_node]
                        new_concat_parent_filter_node = get_diff_filter_node(singleton, new_dag, shadow_pipeline_name,
                                                                             parents)
                        new_parents.append(new_concat_parent_filter_node)
                    else:  # New node is already filtered
                        new_parents.append(concat_parent)
            elif node.operator_info.operator not in {OperatorType.EXTRACT_RESULT, OperatorType.SCORE}:
                for parent in get_sorted_parent_nodes(original_dag, node):
                    if parent in mapping:
                        new_parents.append(mapping[parent])
                    else:
                        new_parents.append(parent)
            new_node = copy_node_with_new_id(singleton, new_dag, node, new_parents)
            mapping[node] = new_node
            all_new_nodes.add(new_node)

    return set(mapping.keys()), set(mapping.values())


def filter_estimator_transformer_edges(parent, child):
    """Filter edges that are not relevant to the actual data flow but only to estimator/transformer state"""
    is_transformer_edge = ((parent.operator_info.operator == OperatorType.TRANSFORMER
                            and ": fit_transform" in parent.details.description
                            and child.operator_info.operator == OperatorType.TRANSFORMER
                            and ": transform" in child.details.description) or
                           (parent.operator_info.operator == OperatorType.ESTIMATOR
                            and child.operator_info.operator == OperatorType.PREDICT))
    return not is_transformer_edge


def get_basic_code_location_for_current_line():
    frame_info = getframeinfo(currentframe().f_back)
    return BasicCodeLocation(frame_info.filename, frame_info.lineno)


def get_conditional_stop_node(singleton, dag, condition_func, function_info,
                              label, description, parent_nodes, udf_kwargs=None):
    if udf_kwargs is None:
        udf_kwargs = {}

    def check_condition(bound_condition_func, bound_label, *inputs):
        condition_bool = bound_condition_func(*inputs)
        if condition_bool is True:
            result = ConditionalResult.CONTINUE_EXECUTION
        else:
            result = ConditionalResult.STOP_EXECUTION
        singleton.labels_to_extracted_plan_results[bound_label] = condition_bool
        return result

    processing_func = partial(check_condition, condition_func, label)
    non_data_kwargs = {'label': label, 'description': description, **udf_kwargs}
    operator_context = OperatorContext(OperatorType.CONDITIONAL_STOP, function_info, non_data_kwargs)
    operator_call_info = OperatorCallInfo(operator_context, parent_nodes)
    new_conditional_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                   get_basic_code_location_for_current_line(),
                                   operator_context,
                                   DagNodeDetails(description, None),
                                   None,
                                   processing_func)
    add_parent_node_edges(dag, new_conditional_node, parent_nodes)
    return new_conditional_node


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


def get_transformer_parents_with_data_types(dag):
    """
    For now, we will ignore project modifies and focus on selections and transformers.
    This is because for transformers it is easy to find the corresponding test set operation and for the
    selection we do not need to worry about finding corresponding test set operations.
    """
    # This only works for traditional ML of course and not LLMs
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
                data_parent_and_data_type.append((data_parent, data_type))

    # A simple heuristic for now to detect embedding operations in FunctionTransformers in pipelines like
    #  anhedonia_ml
    function_transformers = [node for node in nodes_to_search if
                             node.operator_info.operator == OperatorType.TRANSFORMER
                             and "Function Transformer: transform" in node.details.description]
    for function_transformer in function_transformers:
        data_parent = get_sorted_parent_nodes(dag, function_transformer)[1]
        if (data_parent.details.optimizer_info.shape[1] == 1 and
                function_transformer.details.optimizer_info.shape[1] >= 100):
            data_parent_and_data_type.append((data_parent, DataType.TEXT))
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
    caching_translate_transformer = CachedTextTransformer(translate_transformer, database_path=database_path)
    return caching_translate_transformer


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
        _ = get_intermediate_extraction_node(singleton, new_dag, score_operator, f"orig-{score_index}")


def apply_diff_filter(input_df, corrupted_index):
    # TODO
    if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
        input_df = input_df.reset_index(drop=True)
    if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
        corrupted_diff = input_df.iloc[corrupted_index]
    elif isinstance(input_df, list):
        corrupted_diff = numpy.array(input_df)[corrupted_index]
    elif isinstance(input_df, tuple) and len(input_df) == 8:  # RAG Join Result
        corrupted_diff_list = list(copy(input_df))
        corrupted_diff_list[2] = list(numpy.array(corrupted_diff_list[2])[corrupted_index])
        corrupted_diff_list[3] = list(numpy.array(corrupted_diff_list[3])[corrupted_index])
        corrupted_diff_list[6] = corrupted_diff_list[6][corrupted_index, :]
        corrupted_diff = tuple(corrupted_diff_list)
    else:
        corrupted_diff = input_df[corrupted_index]
    if isinstance(corrupted_diff, (pandas.Series, pandas.DataFrame)):
        corrupted_diff = corrupted_diff.reset_index(drop=True)
    corrupted_diff = wrap_in_mlinspect_array_if_necessary(corrupted_diff)
    corrupted_diff._mlinspect_provenance = None

    return corrupted_diff


def projection(column_names, input_df):
    # TODO: What if not all inputs are pandas dfs?
    result = input_df[column_names]
    result = wrap_in_mlinspect_array_if_necessary(result)
    return result


def changed_data_diff_detection(input_df, corrupted_result):
    corrupt_diff_mask = fix_data_diff_detection_mask_only(input_df, corrupted_result)
    changed_indices_corrupt = fix_data_mask_to_indices(corrupt_diff_mask)
    return changed_indices_corrupt


def fix_data_diff_detection_mask_only(input_df, corrupted_result):
    if isinstance(input_df, (pandas.Series, pandas.DataFrame)):
        input_df = input_df.reset_index(drop=True)
    elif isinstance(input_df, list):
        input_df = numpy.array(input_df)
    if isinstance(corrupted_result, (pandas.Series, pandas.DataFrame)):
        corrupted_result = corrupted_result.reset_index(drop=True)
    elif isinstance(corrupted_result, list):
        corrupted_result = numpy.array(corrupted_result)
    if isinstance(input_df, pandas.Series):
        corrupt_diff_mask = (corrupted_result != input_df).to_numpy()
    elif len(input_df.shape) == 2:
        corrupt_diff_mask = numpy.any(corrupted_result != input_df, axis=1)
    else:
        corrupt_diff_mask = corrupted_result != input_df
    return corrupt_diff_mask


def fix_data_mask_to_indices(corrupt_diff_mask):
    changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
    return changed_indices_corrupt


def update_prediction_diff(old_predictions, prediction_diff, prediction_index):
    updated_predictions = numpy.array(old_predictions.copy())
    updated_predictions[prediction_index] = prediction_diff
    return updated_predictions


def rag_join_update(rag_join_result, inputs):
    vectorstore = rag_join_result[5]

    diff_rag_result, diff_retrieval_index = RunnableSequencePatching.execute_rag_join_diff(
        rag_join_result[7], list(inputs), vectorstore)
    new_rag_join_text_result = list(diff_rag_result)

    # TODO: Should we propagate provenance here? Might be important for explanations later
    new_rag_join_result = (rag_join_result[0], rag_join_result[1], new_rag_join_text_result, list(inputs),
                           None, rag_join_result[5], diff_retrieval_index, rag_join_result[7])
    return new_rag_join_result


def prov_join_with_data_source(intermediate_df, data_source):
    # TODO: What if not all inputs are pandas dfs?
    assert data_source._mlinspect_provenance
    assert intermediate_df._mlinspect_provenance
    data_source_prov = data_source._mlinspect_provenance
    intermediate_df_prov = intermediate_df._mlinspect_provenance

    data_source = data_source.copy()
    assert len(data_source_prov) == 1
    target_data_source_id, _ = list(data_source_prov.keys())[0].rsplit('_', 1)
    target_data_source_id = int(target_data_source_id)
    prov_value = list(data_source_prov.values())[0]
    data_source['join_prov'] = prov_value

    intermediate_df_prov_columns = []
    intermediate_df_prov_dict = {}
    for prov_key, prov_values in intermediate_df_prov.items():
        current_data_source, _ = prov_key.rsplit('_', 1)
        if int(current_data_source) == target_data_source_id:
            intermediate_df_prov_columns.append(prov_key)
            intermediate_df_prov_dict[prov_key] = prov_values
    assert len(intermediate_df_prov_dict) == 1
    intermediate_df_prov_df = pandas.DataFrame({'join_prov': list(intermediate_df_prov.values())[0]})

    result = pandas.merge(intermediate_df_prov_df, data_source, how="inner", on="join_prov")
    assert len(result) == len(list(intermediate_df_prov.values())[0])
    result = result.drop("join_prov", axis=1)
    result = wrap_in_mlinspect_array_if_necessary(result)
    result._mlinspect_provenance = intermediate_df_prov
    return result


def get_diff_filter_node(singleton, dag, shadow_pipeline_name, parents):
    description = "Filter for diff only"
    operator_context = OperatorContext(OperatorType.SELECTION, None, {'description': description,
                                                                      'func': apply_diff_filter})
    operator_call_info = OperatorCallInfo(operator_context, parents)
    new_fix_diff_filter_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                       BasicCodeLocation(shadow_pipeline_name, None),
                                       operator_context,
                                       DagNodeDetails(
                                           description,
                                           None),
                                       None,
                                       apply_diff_filter)
    add_parent_node_edges(dag, new_fix_diff_filter_node, parents)
    return new_fix_diff_filter_node


def add_parent_node_edges(dag, node_with_parents, parents):
    for arg_index, parent in enumerate(parents):
        dag.add_edge(parent, node_with_parents, arg_index=arg_index)


def get_changed_indices_node(singleton, dag, shadow_pipeline_name, parent_nodes):
    description = "Detect changed indices"
    non_data_kwargs = {'description': description,
                       'func': changed_data_diff_detection}
    operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, None, non_data_kwargs)
    operator_call_info = OperatorCallInfo(operator_context, parent_nodes)
    new_changed_indices_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                       BasicCodeLocation(shadow_pipeline_name, None),
                                       operator_context,
                                       DagNodeDetails(description, None),
                                       None,
                                       changed_data_diff_detection)
    add_parent_node_edges(dag, new_changed_indices_node, parent_nodes)
    return new_changed_indices_node


def merge_prediction_diff_with_old_predictions(singleton, dag, shadow_pipeline_name, parent_nodes):
    description = "Merge prediction diff with old predictions"
    operator_context = OperatorContext(OperatorType.SELECTION, None,
                                       {'description': description,
                                        'func': update_prediction_diff})
    operator_call_info = OperatorCallInfo(operator_context, parent_nodes)
    new_fix_predict_diff_update_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                               BasicCodeLocation(shadow_pipeline_name, None),
                                               operator_context,
                                               DagNodeDetails(description, None),
                                               None,
                                               update_prediction_diff)
    add_parent_node_edges(dag, new_fix_predict_diff_update_node, parent_nodes)
    return new_fix_predict_diff_update_node


def add_new_score_and_score_extraction_nodes(singleton, new_dag, new_predict_node, score_operators, label_prefix):
    new_score_nodes = []
    for score_index, score_operator in enumerate(score_operators):
        new_score_node = copy_node_with_new_id(singleton, new_dag, score_operator,
                                               [new_predict_node,
                                                *get_sorted_parent_nodes(new_dag, score_operator)[1:]])
        new_score_nodes.append(new_score_node)
        _ = get_intermediate_extraction_node(singleton, new_dag, new_score_node, f"{label_prefix}-{score_index}")
    return new_score_nodes


def assert_standard_llm_shape(dag, shadow_pipeline_name):
    predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
    score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
    rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)
    train_data_operators = find_nodes_by_type(dag, OperatorType.TRAIN_DATA)
    train_labels_operators = find_nodes_by_type(dag, OperatorType.TRAIN_LABELS)
    test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
    test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)
    # pylint: disable=too-many-boolean-expressions
    if len(predict_operators) != 1 or len(score_operators) < 1 or len(rag_join_operators) != 1 \
            or len(train_data_operators) != 1 or len(train_labels_operators) != 1 \
            or len(test_data_operators) != 1 or len(test_labels_operators) < 1:
        raise NotImplementedError(f"Currently, {shadow_pipeline_name} only supports pipelines following a "
                                  f"very specific pattern!")


def assert_standard_ml_shape(dag, shadow_pipeline_name):
    predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
    score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
    model_operators = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
    train_data_operators = find_nodes_by_type(dag, OperatorType.TRAIN_DATA)
    train_labels_operators = find_nodes_by_type(dag, OperatorType.TRAIN_LABELS)
    test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
    test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)
    # pylint: disable=too-many-boolean-expressions
    if len(predict_operators) != 1 or len(score_operators) < 1 or len(model_operators) != 1 \
            or len(train_data_operators) != 1 or len(train_labels_operators) != 1 \
            or len(test_data_operators) != 1 or len(test_labels_operators) < 1:
        raise NotImplementedError(f"Currently, {shadow_pipeline_name} only supports pipelines following a very "
                                  f"specific pattern!")


def concat_func(*inputs):
    # TODO: What if not all inputs are pandas dfs?
    result = pandas.concat(inputs, axis=1)
    result = wrap_in_mlinspect_array_if_necessary(result)
    # Not sure if this might be necessary at some point
    # result._mlinspect_provenance = ...
    return result


def prov_join_node_with_data_sources(singleton, data_sources_with_sensitive_columns, new_dag,
                                     node_requiring_side_info):
    dag_to_consider = networkx.subgraph_view(new_dag, filter_edge=filter_estimator_transformer_edges)
    data_sources_concat = {}
    data_sources_prov_join = {}
    for data_source, columns in list(data_sources_with_sensitive_columns.items()):
        paths = list(networkx.all_simple_paths(dag_to_consider, source=data_source, target=node_requiring_side_info))
        if len(paths) != 0:
            nodes_in_paths = set(node for path in paths for node in path)
            if len([node for node in nodes_in_paths if
                    node.operator_info.operator in {OperatorType.SELECTION, OperatorType.JOIN}]) == 0:
                data_sources_concat[data_source] = columns
            else:
                data_sources_prov_join[data_source] = columns

    nodes_to_concat = []
    for data_source, column_names in data_sources_concat.items():
        projection_processing_func = wrap_projection_func(
            partial(projection, column_names))

        description = f"Select sensitive attributes: {column_names}"
        operator_context = OperatorContext(OperatorType.PROJECTION, None, {'description': description,
                                                                           'func': projection,
                                                                           'column_names': column_names})
        parents = [data_source]
        operator_call_info = OperatorCallInfo(operator_context, parents)
        projection_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                  BasicCodeLocation("Fairness Slices", None),
                                  operator_context,
                                  DagNodeDetails(description, None),
                                  None,
                                  projection_processing_func)
        new_dag.add_edge(data_source, projection_node, arg_index=0)
        nodes_to_concat.append(projection_node)
    for data_source, column_names in data_sources_prov_join.items():
        projection_processing_func = wrap_projection_func(
            partial(projection, column_names))
        description = f"Select sensitive attributes: {column_names}"
        operator_context = OperatorContext(OperatorType.PROJECTION, None, {'description': description,
                                                                           'func': projection_processing_func})
        operator_call_info = OperatorCallInfo(operator_context, [data_source])
        projection_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                  BasicCodeLocation("Fairness Slices", None),
                                  operator_context,
                                  DagNodeDetails(description, None),
                                  None,
                                  projection_processing_func)
        new_dag.add_edge(data_source, projection_node, arg_index=0)

        description = "Join on provenance"
        operator_context = OperatorContext(OperatorType.JOIN, None, {'description': description,
                                                                     'func': prov_join_with_data_source})
        operator_call_info = OperatorCallInfo(operator_context, [node_requiring_side_info, projection_node])
        join_node = DagNode(singleton.get_next_op_id(operator_call_info),
                            BasicCodeLocation("Fairness Slices", None),
                            operator_context,
                            DagNodeDetails(description, None),
                            None,
                            prov_join_with_data_source)
        new_dag.add_edge(node_requiring_side_info, join_node, arg_index=0)
        new_dag.add_edge(projection_node, join_node, arg_index=1)

        nodes_to_concat.append(join_node)

    description = "Concat sensitive attributes"
    operator_context = OperatorContext(OperatorType.CONCATENATION, None, {'description': description,
                                                                          'func': concat_func})
    operator_call_info = OperatorCallInfo(operator_context, nodes_to_concat)
    concat_node = DagNode(singleton.get_next_op_id(operator_call_info),
                          BasicCodeLocation("Data Errors", None),
                          operator_context,
                          DagNodeDetails(description, None),
                          None,
                          concat_func)
    for arg_index, node in enumerate(nodes_to_concat):
        new_dag.add_edge(node, concat_node, arg_index=arg_index)
    return concat_node


def get_proxy_model_node(executor_singleton, dag, parent_nodes):
    non_data_kwargs = {'loss': 'log_loss', 'max_iter': 30, 'n_jobs': 1}
    model_function = partial(SGDClassifier, **non_data_kwargs)
    new_processing_func = partial(_fit_model_variant, make_classifier_func=model_function)
    new_description = "Fast proxy model"
    operator_context = OperatorContext(OperatorType.ESTIMATOR,
                                       FunctionInfo('sklearn.linear_model._stochastic_gradient', 'SGDClassifier'),
                                       non_data_kwargs)
    operator_call_info = OperatorCallInfo(operator_context, parent_nodes)
    new_estimator_node = DagNode(executor_singleton.get_next_op_id(operator_call_info),
                                 get_basic_code_location_for_current_line(),
                                 operator_context,
                                 DagNodeDetails(new_description, None, None),
                                 None,
                                 new_processing_func)
    add_parent_node_edges(dag, new_estimator_node, parent_nodes)
    return new_estimator_node


def _fit_model_variant(train_data, train_labels, make_classifier_func):
    """Create the classifier and fit it"""
    estimator = make_classifier_func()
    estimator.fit(train_data, train_labels)
    return estimator


def get_top_n_df_rows(corruption_diff_fix_df, sample_size):
    if isinstance(corruption_diff_fix_df, (pandas.DataFrame, pandas.Series)):
        corruption_diff_fix_df_sample = corruption_diff_fix_df.head(sample_size)
    elif (isinstance(corruption_diff_fix_df, numpy.ndarray) and
          corruption_diff_fix_df.ndim == 2):
        corruption_diff_fix_df_sample = corruption_diff_fix_df[:sample_size, :]
    elif (isinstance(corruption_diff_fix_df, numpy.ndarray) and
          corruption_diff_fix_df.ndim == 1):
        corruption_diff_fix_df_sample = corruption_diff_fix_df[:sample_size]
    else:
        raise NotImplementedError("TODO")
    return corruption_diff_fix_df_sample


def df_or_array_non_empty(df):
    return len(df) != 0


def df_or_array_non_empty_func_info():
    return FunctionInfo('mlidea.shadow_pipelines._utils', 'df_or_array_non_empty')
