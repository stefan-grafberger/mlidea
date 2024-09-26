# 2. Fairness with slice finder: does not exist in mlwhatif. first, we need a new patch type on the outputs of the
# model test set predictions joined with the input data. this join kind of requires provenance alreday.
# does it???? we can just rely on the order not changing between featurisation and before. however, attributes might
# be removed during this process. or do we assume they don't?
#  once we have a problematic slice, we need to go back and apply it as a filter again, which again requires
#  provenance. maybe we need to add basic provenance support first? or not!! this filter can be computed using
#  the unfeaturised state of the data by relying on the order not changing. then we can just compute a mask and
#  apply it to the unfeaturised data.
from collections import defaultdict
# First locate test data
# then look at provenance and find out which source tables to look at
# then look at those source tables and find all operators in-between the two operators
# then also get all source table columns and look for potentially sensitive columns
# then filter source tables with provenance connection to source tables with potentially sensitive columns
# then look
# for each of those, apply projections first to the relevant columns


from enum import Enum
from functools import partial

import duckdb
import networkx
import numpy
import pandas
from fairlearn.metrics import MetricFrame
from jenga.corruptions.generic import MissingValues
from jenga.corruptions.numerical import Scaling
from jenga.corruptions.text import BrokenCharacters
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sliceline import Slicefinder

from mlidea.analysis._cleaning_methods import OutlierCleaner
from mlidea.execution._pipeline_executor import singleton
from mlidea.analysis._analysis_utils import find_nodes_by_type
from mlidea import OperatorType, DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline
from mlidea.shadow_pipelines._utils import get_intermediate_extraction_node, copy_node_with_new_id, \
    get_sorted_parent_nodes, find_train_or_test_pipeline_part_end, get_typo_adder, duplicate_descendants, \
    get_typo_fixer, get_conditional_stop_node, filter_estimator_transformer_edges
from mlidea.monkeypatching._patch_langchain import RunnableSequencePatching
from mlidea.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary
from monkeypatching._provenance_propagation import wrap_projection_func


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


class FairnessSlices(ShadowPipeline):
    """
    The Data Error Robustness Shadow Pipeline
    """

    def check_rebuilding_necessary(self, extracted_plan_results: dict[str, any]) -> any:
        return False

    def __init__(self, additional_column_names=None):
        if additional_column_names is None:
            additional_column_names = []
        self._additional_column_names = additional_column_names
        self._shadow_pipeline_id = (tuple(additional_column_names),)
        self.score_operator_count = 0
        self.sensitive_column_count = 0

    @property
    def shadow_pipeline_id(self):
        return self._shadow_pipeline_id

    @property
    def simple_name(self):
        return "data_errors"

    @staticmethod
    def is_column_sensitive(column_name, additional_column_names):
        # TODO: There are many different ways to do this to explore in the future, e.g., using LLMs
        return column_name in {"race", "gender", "age", "lang", "country", "sex"}.union(additional_column_names)

    def generate_shadow_pipeline_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        # pylint: disable=too-many-locals,too-many-statements
        # TODO: Maybe it would be better to delete all unrelated DAG nodes here that are not specifically mentioned
        #  below. But this only works once intermediate resutl caching is implemented

        data_sources_concat, data_sources_prov_join = FairnessSlices.get_data_sources_to_sensitive_columns(
            dag, self._additional_column_names)
        self.sensitive_column_count = len(data_sources_concat) + len(data_sources_prov_join)

        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)

        if len(rag_join_operators) == 0:
            new_dag = self.get_traditional_ml_dag(dag, data_sources_concat, data_sources_prov_join)
        else:
            new_dag = self.get_llm_rag_dag(dag, data_sources_concat, data_sources_prov_join)

        return new_dag

    def get_traditional_ml_dag(self, dag, data_sources_concat, data_sources_prov_join):
        new_dag = dag.copy()

        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        model_operators = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
        train_data_operators = find_nodes_by_type(dag, OperatorType.TRAIN_DATA)
        train_labels_operators = find_nodes_by_type(dag, OperatorType.TRAIN_LABELS)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)
        if len(predict_operators) != 1 or len(score_operators) < 1 or len(model_operators) != 1 \
                or len(train_data_operators) != 1 or len(train_labels_operators) != 1 \
                or len(test_data_operators) != 1 or len(test_labels_operators) < 1:
            raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                      "pattern!")
        FairnessSlices.add_orig_score_extraction_nodes(new_dag, score_operators)
        self.score_operator_count = len(score_operators)

        if len(data_sources_concat) == 0 and len(data_sources_prov_join) == 0:
            return new_dag

        def concat_processing_func(*inputs):
            # TODO: What if not all inputs are pandas dfs?
            result = pandas.concat(inputs, axis=1)
            result = wrap_in_mlinspect_array_if_necessary(result)
            # Not sure if this might be necessary at some point
            # result._mlinspect_provenance = ...
            return result

        concat_node = DagNode(singleton.get_next_op_id(),
                              BasicCodeLocation("Data Errors", None),
                              OperatorContext(OperatorType.CONCATENATION, None),
                              DagNodeDetails(
                                  f"Concat sensitive attributes", None),
                              None,
                              concat_processing_func)

        for data_source, column_names in data_sources_concat.items():
            projection_processing_func = wrap_projection_func(
                partial(FairnessSlices.projection_processing_func, column_names))

            projection_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("Fairness Slices", None),
                                      OperatorContext(OperatorType.PROJECTION, None),
                                      DagNodeDetails(
                                          f"Select sensitive attributes", None),
                                      None,
                                      projection_processing_func)
            new_dag.add_edge(data_source, projection_node, arg_index=0)
            new_dag.add_edge(projection_node, concat_node, arg_index=0)

        for data_source, column_names in data_sources_prov_join.items():
            projection_processing_func = wrap_projection_func(
                partial(FairnessSlices.projection_processing_func, column_names))

            projection_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("Fairness Slices", None),
                                      OperatorContext(OperatorType.PROJECTION, None),
                                      DagNodeDetails(
                                          f"Select sensitive attributes", None),
                                      None,
                                      projection_processing_func)
            new_dag.add_edge(data_source, projection_node, arg_index=0)

            join_processing_func = partial(FairnessSlices.prov_join_processing_func, data_source.node_id)
            join_node = DagNode(singleton.get_next_op_id(),
                                BasicCodeLocation("Fairness Slices", None),
                                OperatorContext(OperatorType.JOIN, None),
                                DagNodeDetails(
                                    f"Join on provenance", None),
                                None,
                                join_processing_func)
            new_dag.add_edge(test_data_operators[0], join_node, arg_index=0)
            new_dag.add_edge(projection_node, join_node, arg_index=0)

            new_dag.add_edge(join_node, concat_node, arg_index=0)

        # TODO: The prov join version where all need a projection and join before connecting it to the concat

        new_slice_finder_node = DagNode(singleton.get_next_op_id(),
                                        BasicCodeLocation("Fairness Slices", None),
                                        OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                        DagNodeDetails(
                                            f"Run Slice Finder", None),
                                        None,
                                        FairnessSlices.get_slice_finder_mask_and_slice)
        new_dag.add_edge(concat_node, new_slice_finder_node, arg_index=0)
        new_dag.add_edge(test_labels_operators[0], new_slice_finder_node, arg_index=1)
        new_dag.add_edge(predict_operators[0], new_slice_finder_node, arg_index=2)
        extraction_node = get_intermediate_extraction_node(singleton, new_slice_finder_node,
                                                           "fairness-slices-slice-line-result")
        new_dag.add_edge(new_slice_finder_node, extraction_node, arg_index=0)

        problematic_slice_found_func = lambda slice_finder_result: (slice_finder_result[0] is not None and
                                                                    slice_finder_result[1] is not None)
        conditional_corruption_made_changes_node = get_conditional_stop_node(
            singleton, problematic_slice_found_func, f"fairness-slices-slice-line-problematic-slice-found",
            "Check if corrupt function made changes", new_slice_finder_node)
        new_dag.add_edge(new_slice_finder_node, conditional_corruption_made_changes_node, arg_index=0)

        # The first step is to concat the test

        # data_parent_transformer_and_data_type = FairnessSlices._get_transformer_operators_to_test(dag)
        # self.transformer_inputs_to_check_count = len(data_parent_transformer_and_data_type)
        #
        # for data_type_index, (data_parent, transformer, data_type) in enumerate(data_parent_transformer_and_data_type):
        #     processing_func = partial(FairnessSlices.corrupt_data,
        #                               data_type=data_type,
        #                               corruption_fraction=self._corruption_fraction)
        #     new_corruption_node = DagNode(singleton.get_next_op_id(),
        #                                   BasicCodeLocation("Data Errors", None),
        #                                   OperatorContext(OperatorType.PROJECTION_MODIFY, None),
        #                                   DagNodeDetails(
        #                                       f"Corrupt {self._corruption_fraction} of {data_type.value} values", None),
        #                                   None,
        #                                   processing_func)
        #     new_dag.add_edge(data_parent, new_corruption_node, arg_index=0)
        #
        #     new_corruption_diff_node = DagNode(singleton.get_next_op_id(),
        #                                        BasicCodeLocation("Data Errors", None),
        #                                        OperatorContext(OperatorType.GROUP_BY_AGG, None),
        #                                        DagNodeDetails(
        #                                            f"Detect changed indices from corrupting", None),
        #                                        None,
        #                                        FairnessSlices.corrupt_data_diff_detection)
        #     new_dag.add_edge(data_parent, new_corruption_diff_node, arg_index=0)
        #     new_dag.add_edge(new_corruption_node, new_corruption_diff_node, arg_index=1)
        #
        #     condition_corrupt_function = lambda np_array: len(np_array) != 0
        #     conditional_corruption_made_changes_node = get_conditional_stop_node(
        #         singleton, condition_corrupt_function, f"data-errors-corruption-made-changes-{data_type_index}",
        #         "Check if corrupt function made changes", new_corruption_diff_node)
        #     new_dag.add_edge(new_corruption_diff_node, conditional_corruption_made_changes_node, arg_index=1)
        #
        #     new_corruption_diff_filter_node = DagNode(singleton.get_next_op_id(),
        #                                               BasicCodeLocation("Data Errors", None),
        #                                               OperatorContext(OperatorType.SELECTION, None),
        #                                               DagNodeDetails(
        #                                                   f"Filter for diff only",
        #                                                   None),
        #                                               None,
        #                                               FairnessSlices.apply_diff_filter)
        #     new_dag.add_edge(new_corruption_node, new_corruption_diff_filter_node, arg_index=0)
        #     new_dag.add_edge(new_corruption_diff_node, new_corruption_diff_filter_node, arg_index=1)
        #     new_dag.add_edge(conditional_corruption_made_changes_node, new_corruption_diff_filter_node, arg_index=2)
        #
        #     extraction_node = get_intermediate_extraction_node(singleton, new_corruption_diff_filter_node,
        #                                                        f"data-errors-corruption-diff-{data_type_index}")
        #     new_dag.add_edge(new_corruption_diff_filter_node, extraction_node, arg_index=0)
        #
        #     # Evaluate with corrupted data
        #     old_copied_nodes, new_nodes, new_score_nodes = duplicate_descendants(
        #         dag, new_dag, data_parent, new_corruption_diff_filter_node, singleton)
        #
        #     # Now apply filter to all other concatenation inputs
        #     concats = [node for node in new_nodes if node.operator_info.operator == OperatorType.CONCATENATION]
        #     if len(concats) >= 1:
        #         if len(concats) != 1:
        #             raise NotImplementedError(
        #                 "Currently, Label Errors only supports pipelines following a very specific "
        #                 "pattern!")
        #         for concat in concats:
        #             concat_parents = get_sorted_parent_nodes(new_dag, concat)
        #             for concat_parent in concat_parents:
        #                 if concat_parent not in new_nodes:
        #                     edge_data = new_dag.get_edge_data(concat_parent, concat)
        #                     new_dag.remove_edge(concat_parent, concat)
        #                     new_concat_parent_filter_node = DagNode(singleton.get_next_op_id(),
        #                                                             BasicCodeLocation("Data Errors", None),
        #                                                             OperatorContext(OperatorType.SELECTION, None),
        #                                                             DagNodeDetails(
        #                                                                 f"Filter for diff only",
        #                                                                 None),
        #                                                             None,
        #                                                             FairnessSlices.apply_diff_filter)
        #                     new_dag.add_edge(concat_parent, new_concat_parent_filter_node, arg_index=0)
        #                     new_dag.add_edge(new_corruption_diff_node, new_concat_parent_filter_node, arg_index=1)
        #                     new_dag.add_edge(conditional_corruption_made_changes_node, new_concat_parent_filter_node,
        #                                      arg_index=2)
        #                     new_dag.add_edge(new_concat_parent_filter_node, concat, **edge_data)
        #     test_predict = [node for node in new_nodes
        #                     if node.operator_info.operator == OperatorType.PREDICT][0]
        #     old_predict = [node for node in old_copied_nodes
        #                    if node.operator_info.operator == OperatorType.PREDICT][0]
        #
        #     new_corrupt_predict_diff_update_node = DagNode(singleton.get_next_op_id(),
        #                                                    BasicCodeLocation("Data Errors", None),
        #                                                    OperatorContext(OperatorType.SELECTION, None),
        #                                                    DagNodeDetails(
        #                                                        f"Merge corruption diff with old predictions",
        #                                                        None),
        #                                                    None,
        #                                                    FairnessSlices.update_prediction_diff)
        #     new_dag.add_edge(old_predict, new_corrupt_predict_diff_update_node, arg_index=0)
        #     new_dag.add_edge(test_predict, new_corrupt_predict_diff_update_node, arg_index=1)
        #     new_dag.add_edge(new_corruption_diff_node, new_corrupt_predict_diff_update_node, arg_index=2)
        #     new_dag.add_edge(conditional_corruption_made_changes_node, new_corrupt_predict_diff_update_node,
        #                      arg_index=3)
        #
        #     if len(new_score_nodes) < 1:
        #         raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
        #                                   "pattern!")
        #     for score_index, score_operator in enumerate(new_score_nodes):
        #         edge_data = new_dag.get_edge_data(test_predict, score_operator)
        #         new_dag.remove_edge(test_predict, score_operator)
        #         new_dag.add_edge(new_corrupt_predict_diff_update_node, score_operator, **edge_data)
        #
        #         extraction_node = get_intermediate_extraction_node(singleton, score_operator,
        #                                                            f"data-errors-corrupt-{score_index}-{data_type_index}")
        #         new_dag.add_edge(score_operator, extraction_node, arg_index=0)
        #
        #     condition_processing_func = partial(FairnessSlices.condition_corruption_significant_function,
        #                                         self._corruption_significant_relative_threshold)
        #     conditional_corruption_significant_node = get_conditional_stop_node(
        #         singleton, condition_processing_func, f"data-errors-corruption-significant-{data_type_index}",
        #         "Check if fix function made changes", new_score_nodes[0])
        #     for score_index, score_operator in enumerate(score_operators):
        #         new_dag.add_edge(score_operator, conditional_corruption_significant_node,
        #                          arg_index=score_index)
        #     for score_index, score_operator in enumerate(new_score_nodes):
        #         new_dag.add_edge(score_operator, conditional_corruption_significant_node,
        #                          arg_index=score_index + self.score_operator_count)
        #     # End evaluate
        #
        #     if data_type == DataType.TEXT:
        #         fix_input_node = new_corruption_diff_filter_node
        #     else:
        #         fix_input_node = new_corruption_node
        #
        #     processing_func = partial(FairnessSlices.fix_data, data_type=data_type)
        #     new_fix_node = DagNode(singleton.get_next_op_id(),
        #                            BasicCodeLocation("Data Errors", None),
        #                            OperatorContext(OperatorType.PROJECTION_MODIFY, None),
        #                            DagNodeDetails(
        #                                f"Fix {self._corruption_fraction} of {data_type.value} values", None),
        #                            None,
        #                            processing_func)
        #     new_dag.add_edge(fix_input_node, new_fix_node, arg_index=0)
        #     new_dag.add_edge(conditional_corruption_significant_node, new_fix_node, arg_index=1)
        #
        #     if data_type == DataType.TEXT:
        #         fix_node_to_extract = new_fix_node
        #     else:
        #         new_fix_with_corruption_change_filter_node = DagNode(singleton.get_next_op_id(),
        #                                                              BasicCodeLocation("Data Errors", None),
        #                                                              OperatorContext(OperatorType.SELECTION, None),
        #                                                              DagNodeDetails(
        #                                                                  f"Filter for diff only",
        #                                                                  None),
        #                                                              None,
        #                                                              FairnessSlices.apply_diff_filter)
        #         new_dag.add_edge(new_fix_node, new_fix_with_corruption_change_filter_node, arg_index=0)
        #         new_dag.add_edge(new_corruption_diff_node, new_fix_with_corruption_change_filter_node, arg_index=1)
        #         new_dag.add_edge(conditional_corruption_made_changes_node, new_fix_with_corruption_change_filter_node,
        #                          arg_index=2)
        #         fix_node_to_extract = new_fix_with_corruption_change_filter_node
        #     extraction_node = get_intermediate_extraction_node(singleton, new_fix_node,
        #                                                        f"data-errors-corruption-diff-fix-{data_type_index}")
        #     new_dag.add_edge(fix_node_to_extract, extraction_node, arg_index=0)
        #
        #     new_fix_diff_mask_node = DagNode(singleton.get_next_op_id(),
        #                                      BasicCodeLocation("Data Errors", None),
        #                                      OperatorContext(OperatorType.PROJECTION_MODIFY, None),
        #                                      DagNodeDetails(
        #                                          f"Compute change mask from fixing", None),
        #                                      None,
        #                                      FairnessSlices.fix_data_diff_detection_mask_only)
        #     new_dag.add_edge(fix_input_node, new_fix_diff_mask_node, arg_index=0)
        #     new_dag.add_edge(new_fix_node, new_fix_diff_mask_node, arg_index=1)
        #
        #     new_fix_diff_indices_node = DagNode(singleton.get_next_op_id(),
        #                                         BasicCodeLocation("Data Errors", None),
        #                                         OperatorContext(OperatorType.GROUP_BY_AGG, None),
        #                                         DagNodeDetails(
        #                                             f"Compute changed indices from fixing", None),
        #                                         None,
        #                                         FairnessSlices.fix_data_mask_to_indices)
        #     new_dag.add_edge(new_fix_diff_mask_node, new_fix_diff_indices_node, arg_index=0)
        #
        #     condition_fix_function = lambda np_array: len(np_array) != 0
        #     conditional_fixes_changed_something_node = get_conditional_stop_node(
        #         singleton, condition_fix_function, f"data-errors-corruption-diff-fix-not-empty-{data_type_index}",
        #         "Check if fix function made changes", new_fix_diff_indices_node)
        #     new_dag.add_edge(new_fix_diff_indices_node, conditional_fixes_changed_something_node, arg_index=0)
        #
        #     new_fix_diff_filter_node = DagNode(singleton.get_next_op_id(),
        #                                        BasicCodeLocation("Data Errors", None),
        #                                        OperatorContext(OperatorType.SELECTION, None),
        #                                        DagNodeDetails(
        #                                            f"Filter for diff only",
        #                                            None),
        #                                        None,
        #                                        FairnessSlices.apply_diff_filter)
        #     new_dag.add_edge(new_fix_node, new_fix_diff_filter_node, arg_index=0)
        #     new_dag.add_edge(new_fix_diff_indices_node, new_fix_diff_filter_node, arg_index=1)
        #     new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_diff_filter_node, arg_index=2)
        #
        #     # Evaluate with fixed data
        #     old_copied_nodes, new_nodes, new_score_nodes = duplicate_descendants(
        #         dag, new_dag, data_parent, new_fix_diff_filter_node, singleton)
        #
        #     # Now apply filter to all other concatenation inputs
        #     concats = [node for node in new_nodes if node.operator_info.operator == OperatorType.CONCATENATION]
        #     if len(concats) >= 1:
        #         if len(concats) != 1:
        #             raise NotImplementedError(
        #                 "Currently, Label Errors only supports pipelines following a very specific "
        #                 "pattern!")
        #         for concat in concats:
        #             concat_parents = get_sorted_parent_nodes(new_dag, concat)
        #             for concat_parent in concat_parents:
        #                 if concat_parent not in new_nodes:
        #                     edge_data = new_dag.get_edge_data(concat_parent, concat)
        #                     new_dag.remove_edge(concat_parent, concat)
        #                     new_concat_parent_filter_node = DagNode(singleton.get_next_op_id(),
        #                                                             BasicCodeLocation("Data Errors", None),
        #                                                             OperatorContext(OperatorType.SELECTION, None),
        #                                                             DagNodeDetails(
        #                                                                 f"Filter for diff only",
        #                                                                 None),
        #                                                             None,
        #                                                             FairnessSlices.apply_diff_filter)
        #                     new_dag.add_edge(concat_parent, new_concat_parent_filter_node, arg_index=0)
        #                     new_dag.add_edge(new_fix_diff_indices_node, new_concat_parent_filter_node, arg_index=1)
        #                     new_dag.add_edge(new_concat_parent_filter_node, concat, **edge_data)
        #     test_predict = [node for node in new_nodes
        #                     if node.operator_info.operator == OperatorType.PREDICT][0]
        #
        #     if data_type == DataType.TEXT:
        #         new_indices_before_corruption_node = DagNode(singleton.get_next_op_id(),
        #                                                      BasicCodeLocation("Data Errors", None),
        #                                                      OperatorContext(OperatorType.SELECTION, None),
        #                                                      DagNodeDetails(
        #                                                          f"Compute indices relative to before corrupting and fixing",
        #                                                          None),
        #                                                      None,
        #                                                      FairnessSlices.fix_data_diff_indices_before_corruption)
        #         new_dag.add_edge(new_corruption_diff_node, new_indices_before_corruption_node, arg_index=0)
        #         new_dag.add_edge(new_fix_diff_mask_node, new_indices_before_corruption_node, arg_index=1)
        #         new_dag.add_edge(conditional_corruption_made_changes_node, new_indices_before_corruption_node,
        #                          arg_index=2)
        #
        #         prediction_filter_index_node = new_indices_before_corruption_node
        #     else:
        #         prediction_filter_index_node = new_fix_diff_indices_node
        #
        #     new_fix_predict_diff_update_node = DagNode(singleton.get_next_op_id(),
        #                                                BasicCodeLocation("Data Errors", None),
        #                                                OperatorContext(OperatorType.SELECTION, None),
        #                                                DagNodeDetails(
        #                                                    f"Merge fix diff with old predictions",
        #                                                    None),
        #                                                None,
        #                                                FairnessSlices.update_prediction_diff)
        #     new_dag.add_edge(new_corrupt_predict_diff_update_node, new_fix_predict_diff_update_node, arg_index=0)
        #     new_dag.add_edge(test_predict, new_fix_predict_diff_update_node, arg_index=1)
        #     new_dag.add_edge(prediction_filter_index_node, new_fix_predict_diff_update_node, arg_index=2)
        #     new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_predict_diff_update_node, arg_index=3)
        #
        #     if len(new_score_nodes) < 1:
        #         raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
        #                                   "pattern!")
        #     for score_index, score_operator in enumerate(new_score_nodes):
        #         edge_data = new_dag.get_edge_data(test_predict, score_operator)
        #         new_dag.remove_edge(test_predict, score_operator)
        #         new_dag.add_edge(new_fix_predict_diff_update_node, score_operator, **edge_data)
        #
        #         extraction_node = get_intermediate_extraction_node(singleton, score_operator,
        #                                                            f"data-errors-corrupt-fix-{score_index}-{data_type_index}")
        #         new_dag.add_edge(score_operator, extraction_node, arg_index=0)
        # End evaluate
        return new_dag

    def get_llm_rag_dag(self, dag, data_sources_concat, data_sources_prov_join):
        new_dag = dag.copy()

        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)
        train_data_operators = find_nodes_by_type(dag, OperatorType.TRAIN_DATA)
        train_labels_operators = find_nodes_by_type(dag, OperatorType.TRAIN_LABELS)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)
        if len(predict_operators) != 1 or len(score_operators) < 1 or len(rag_join_operators) != 1 \
                or len(train_data_operators) != 1 or len(train_labels_operators) != 1 \
                or len(test_data_operators) != 1 or len(test_labels_operators) < 1:
            raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                      "pattern!")
        FairnessSlices.add_orig_score_extraction_nodes(new_dag, score_operators)
        self.score_operator_count = len(score_operators)
        self.transformer_inputs_to_check_count = 1

        data_parent = test_data_operators[0]
        data_type = DataType.TEXT

        processing_func = partial(FairnessSlices.corrupt_data,
                                  data_type=data_type,
                                  corruption_fraction=self._corruption_fraction)
        new_corruption_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("Data Errors", None),
                                      OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                      DagNodeDetails(
                                          f"Corrupt {self._corruption_fraction} of {data_type.value} values", None),
                                      None,
                                      processing_func)
        new_dag.add_edge(data_parent, new_corruption_node, arg_index=0)

        new_corruption_diff_node = DagNode(singleton.get_next_op_id(),
                                           BasicCodeLocation("Data Errors", None),
                                           OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                           DagNodeDetails(
                                               f"Detect changed indices from corrupting", None),
                                           None,
                                           FairnessSlices.corrupt_data_diff_detection)
        new_dag.add_edge(data_parent, new_corruption_diff_node, arg_index=0)
        new_dag.add_edge(new_corruption_node, new_corruption_diff_node, arg_index=1)

        condition_corrupt_function = lambda np_array: len(np_array) != 0
        conditional_corruption_made_changes_node = get_conditional_stop_node(
            singleton, condition_corrupt_function, f"data-errors-corruption-made-changes-0",
            "Check if problematic slice was found", new_corruption_diff_node)
        new_dag.add_edge(new_corruption_diff_node, conditional_corruption_made_changes_node, arg_index=1)

        new_corruption_diff_filter_node = DagNode(singleton.get_next_op_id(),
                                                  BasicCodeLocation("Data Errors", None),
                                                  OperatorContext(OperatorType.SELECTION, None),
                                                  DagNodeDetails(
                                                      f"Filter for diff only",
                                                      None),
                                                  None,
                                                  FairnessSlices.apply_diff_filter)
        new_dag.add_edge(new_corruption_node, new_corruption_diff_filter_node, arg_index=0)
        new_dag.add_edge(new_corruption_diff_node, new_corruption_diff_filter_node, arg_index=1)
        new_dag.add_edge(conditional_corruption_made_changes_node, new_corruption_diff_filter_node, arg_index=2)

        extraction_node = get_intermediate_extraction_node(singleton, new_corruption_diff_filter_node,
                                                           f"data-errors-corruption-diff-0")
        new_dag.add_edge(new_corruption_diff_filter_node, extraction_node, arg_index=0)

        # Evaluate with corrupted data
        # Operator to get the rag join results
        # TODO: Actually implement the function used here
        new_rag_join_update_node = DagNode(singleton.get_next_op_id(),
                                           BasicCodeLocation("Data Errors", None),
                                           OperatorContext(OperatorType.RAG_JOIN, None),
                                           DagNodeDetails("RAG join for test set diff", None),
                                           None,
                                           FairnessSlices.rag_join_update)
        new_dag.add_edge(rag_join_operators[0], new_rag_join_update_node, arg_index=0)
        new_dag.add_edge(new_corruption_diff_filter_node, new_rag_join_update_node, arg_index=1)
        # Duplicate predict operator and connect with rag join result update and prediction update
        test_predict = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_rag_join_update_node, test_predict, arg_index=0)
        old_predict = predict_operators[0]

        new_corrupt_predict_diff_update_node = DagNode(singleton.get_next_op_id(),
                                                       BasicCodeLocation("Data Errors", None),
                                                       OperatorContext(OperatorType.SELECTION, None),
                                                       DagNodeDetails(
                                                           f"Merge corruption diff with old predictions",
                                                           None),
                                                       None,
                                                       FairnessSlices.update_prediction_diff)
        new_dag.add_edge(old_predict, new_corrupt_predict_diff_update_node, arg_index=0)
        new_dag.add_edge(test_predict, new_corrupt_predict_diff_update_node, arg_index=1)
        new_dag.add_edge(new_corruption_diff_node, new_corrupt_predict_diff_update_node, arg_index=2)
        new_dag.add_edge(conditional_corruption_made_changes_node, new_corrupt_predict_diff_update_node,
                         arg_index=3)

        new_score_nodes = []
        for score_index, score_operator in enumerate(score_operators):
            new_score_node = copy_node_with_new_id(singleton, score_operator)
            new_score_nodes.append(new_score_node)
            new_dag.add_edge(new_corrupt_predict_diff_update_node, new_score_node, arg_index=0)
            for parent in get_sorted_parent_nodes(dag, score_operator)[1:]:
                edge_data = new_dag.get_edge_data(parent, score_operator)
                new_dag.add_edge(parent, new_score_node, **edge_data)
            extraction_node = get_intermediate_extraction_node(singleton, score_operator,
                                                               f"data-errors-corrupt-{score_index}-0")
            new_dag.add_edge(score_operator, extraction_node, arg_index=0)

        condition_processing_func = partial(FairnessSlices.condition_corruption_significant_function,
                                            self._corruption_significant_relative_threshold)
        conditional_corruption_significant_node = get_conditional_stop_node(
            singleton, condition_processing_func, f"data-errors-corruption-significant-0",
            "Check if fix function made changes", new_score_nodes[0])
        for score_index, score_operator in enumerate(score_operators):
            new_dag.add_edge(score_operator, conditional_corruption_significant_node,
                             arg_index=score_index)
        for score_index, score_operator in enumerate(new_score_nodes):
            new_dag.add_edge(score_operator, conditional_corruption_significant_node,
                             arg_index=score_index + self.score_operator_count)
        # End evaluate

        fix_input_node = new_corruption_diff_filter_node

        processing_func = partial(FairnessSlices.fix_data, data_type=data_type)
        new_fix_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation("Data Errors", None),
                               OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                               DagNodeDetails(
                                   f"Fix {self._corruption_fraction} of {data_type.value} values", None),
                               None,
                               processing_func)
        new_dag.add_edge(fix_input_node, new_fix_node, arg_index=0)
        new_dag.add_edge(conditional_corruption_significant_node, new_fix_node, arg_index=1)

        fix_node_to_extract = new_fix_node

        extraction_node = get_intermediate_extraction_node(singleton, new_fix_node,
                                                           f"data-errors-corruption-diff-fix-0")
        new_dag.add_edge(fix_node_to_extract, extraction_node, arg_index=0)

        new_fix_diff_mask_node = DagNode(singleton.get_next_op_id(),
                                         BasicCodeLocation("Data Errors", None),
                                         OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                         DagNodeDetails(
                                             f"Compute change mask from fixing", None),
                                         None,
                                         FairnessSlices.fix_data_diff_detection_mask_only)
        new_dag.add_edge(fix_input_node, new_fix_diff_mask_node, arg_index=0)
        new_dag.add_edge(new_fix_node, new_fix_diff_mask_node, arg_index=1)

        new_fix_diff_indices_node = DagNode(singleton.get_next_op_id(),
                                            BasicCodeLocation("Data Errors", None),
                                            OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                            DagNodeDetails(
                                                f"Compute changed indices from fixing", None),
                                            None,
                                            FairnessSlices.fix_data_mask_to_indices)
        new_dag.add_edge(new_fix_diff_mask_node, new_fix_diff_indices_node, arg_index=0)

        condition_fix_function = lambda np_array: len(np_array) != 0
        conditional_fixes_changed_something_node = get_conditional_stop_node(
            singleton, condition_fix_function, f"data-errors-corruption-diff-fix-not-empty-0",
            "Check if fix function made changes", new_fix_diff_indices_node)
        new_dag.add_edge(new_fix_diff_indices_node, conditional_fixes_changed_something_node, arg_index=0)

        new_fix_diff_filter_node = DagNode(singleton.get_next_op_id(),
                                           BasicCodeLocation("Data Errors", None),
                                           OperatorContext(OperatorType.SELECTION, None),
                                           DagNodeDetails(
                                               f"Filter for diff only",
                                               None),
                                           None,
                                           FairnessSlices.apply_diff_filter)
        new_dag.add_edge(new_fix_node, new_fix_diff_filter_node, arg_index=0)
        new_dag.add_edge(new_fix_diff_indices_node, new_fix_diff_filter_node, arg_index=1)
        new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_diff_filter_node, arg_index=2)

        # Evaluate with fixed data
        # Operator to get the rag join results
        # TODO: Actually implement the function used here
        new_rag_join_update_node = DagNode(singleton.get_next_op_id(),
                                           BasicCodeLocation("Data Errors", None),
                                           OperatorContext(OperatorType.RAG_JOIN, None),
                                           DagNodeDetails("RAG join for test set diff", None),
                                           None,
                                           FairnessSlices.rag_join_update)
        new_dag.add_edge(rag_join_operators[0], new_rag_join_update_node, arg_index=0)
        new_dag.add_edge(new_fix_diff_filter_node, new_rag_join_update_node, arg_index=1)
        # Duplicate predict operator and connect with rag join result update and prediction update
        test_predict = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_rag_join_update_node, test_predict, arg_index=0)

        new_indices_before_corruption_node = DagNode(singleton.get_next_op_id(),
                                                     BasicCodeLocation("Data Errors", None),
                                                     OperatorContext(OperatorType.SELECTION, None),
                                                     DagNodeDetails(
                                                         f"Compute indices relative to before corrupting and fixing",
                                                         None),
                                                     None,
                                                     FairnessSlices.fix_data_diff_indices_before_corruption)
        new_dag.add_edge(new_corruption_diff_node, new_indices_before_corruption_node, arg_index=0)
        new_dag.add_edge(new_fix_diff_mask_node, new_indices_before_corruption_node, arg_index=1)
        new_dag.add_edge(conditional_corruption_made_changes_node, new_indices_before_corruption_node,
                         arg_index=2)

        prediction_filter_index_node = new_indices_before_corruption_node

        new_fix_predict_diff_update_node = DagNode(singleton.get_next_op_id(),
                                                   BasicCodeLocation("Data Errors", None),
                                                   OperatorContext(OperatorType.SELECTION, None),
                                                   DagNodeDetails(
                                                       f"Merge fix diff with old predictions",
                                                       None),
                                                   None,
                                                   FairnessSlices.update_prediction_diff)
        new_dag.add_edge(new_corrupt_predict_diff_update_node, new_fix_predict_diff_update_node, arg_index=0)
        new_dag.add_edge(test_predict, new_fix_predict_diff_update_node, arg_index=1)
        new_dag.add_edge(prediction_filter_index_node, new_fix_predict_diff_update_node, arg_index=2)
        new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_predict_diff_update_node, arg_index=3)

        for score_index, score_operator in enumerate(score_operators):
            new_score_node = copy_node_with_new_id(singleton, score_operator)
            new_dag.add_edge(new_fix_predict_diff_update_node, new_score_node, arg_index=0)
            for parent in get_sorted_parent_nodes(dag, score_operator)[1:]:
                edge_data = new_dag.get_edge_data(parent, score_operator)
                new_dag.add_edge(parent, new_score_node, **edge_data)
            extraction_node = get_intermediate_extraction_node(singleton, score_operator,
                                                               f"data-errors-corrupt-fix-{score_index}-0")
            new_dag.add_edge(score_operator, extraction_node, arg_index=0)
        # End evaluate
        return new_dag

    @staticmethod
    def get_data_sources_to_sensitive_columns(dag, additional_column_names):
        data_sources_to_columns = defaultdict(list)
        data_sources = find_nodes_by_type(dag, OperatorType.DATA_SOURCE)
        for data_source in data_sources:
            for column_name in data_source.details.columns:
                if FairnessSlices.is_column_sensitive(column_name, additional_column_names) is True:
                    data_sources_to_columns[data_source].append(column_name)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        dag_to_consider = networkx.subgraph_view(dag, filter_edge=filter_estimator_transformer_edges)
        data_sources_concat = {}
        data_sources_prov_join = {}
        for data_source, columns in list(data_sources_to_columns.items()):
            paths = list(networkx.all_simple_paths(dag_to_consider, source=data_source, target=test_data_operators[0]))
            if len(paths) != 0:
                nodes_in_paths = set(node for path in paths for node in path)
                if len([node for node in nodes_in_paths if
                        node.operator_info.operator in {OperatorType.SELECTION, OperatorType.JOIN}]) == 0:
                    data_sources_concat[data_source] = columns
                else:
                    data_sources_prov_join[data_source] = columns

        return data_sources_concat, data_sources_prov_join

    @staticmethod
    def add_orig_score_extraction_nodes(new_dag, score_operators):
        for score_index, score_operator in enumerate(score_operators):
            orig_extraction_node = get_intermediate_extraction_node(singleton, score_operator,
                                                                    f"label-errors-orig-{score_index}")
            new_dag.add_edge(score_operator, orig_extraction_node, arg_index=0)

    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        report = ""
        orig_result = []
        for score_index in range(self.score_operator_count):
            orig_result.append(extracted_plan_results[f"label-errors-orig-{score_index}"])
        report += f"The original result was {orig_result}.\n"
        if self.sensitive_column_count == 0:
            report += "Slice finding could not be applied since no sensitive column could be found!"
        elif extracted_plan_results["fairness-slices-slice-line-problematic-slice-found"] is False:
            report += "No problematic slice could be found by the slice finder"
        else:
            slice_line_result = extracted_plan_results["fairness-slices-slice-line-result"]
            report += f"The problematic slice that was found is {slice_line_result[0]}."
        # for transformer_index in range(self.transformer_inputs_to_check_count):
        #     report += (f"Issue {transformer_index}\n-\n")
        #     if extracted_plan_results[f"data-errors-corruption-made-changes-{transformer_index}"] is False:
        #         report += "The corruption function did not make any changes."
        #     else:
        #         corruption_diff_df = extracted_plan_results[
        #             f"data-errors-corruption-diff-{transformer_index}"]
        #         if isinstance(corruption_diff_df, (pandas.DataFrame, pandas.Series)):
        #             corruption_diff_df_sample = corruption_diff_df.head(20)
        #         elif isinstance(corruption_diff_df, numpy.ndarray) and corruption_diff_df.ndim == 1:
        #             corruption_diff_df_sample = corruption_diff_df[:20]
        #         elif isinstance(corruption_diff_df, numpy.ndarray) and corruption_diff_df.ndim == 2:
        #             corruption_diff_df_sample = corruption_diff_df[:20, :]
        #         else:
        #             raise NotImplementedError("TODO")
        #         corrupt_result = []
        #         for score_index in range(self.score_operator_count):
        #             corrupt_result.append(
        #                 extracted_plan_results[f"data-errors-corrupt-{score_index}-{transformer_index}"])
        #         report += (
        #             f"The original result was {orig_result}. After corrupting {self._corruption_fraction} of rows, "
        #             f"the pipeline metric was {corrupt_result}, indicating robustness problems. A sample of the corrupted "
        #             f"rows: {str(corruption_diff_df_sample)}. ")
        #
        #         if extracted_plan_results[f"data-errors-corruption-significant-{transformer_index}"] is True:
        #             corruption_diff_fix_df = extracted_plan_results[
        #                 f"data-errors-corruption-diff-fix-{transformer_index}"]
        #             if isinstance(corruption_diff_fix_df, (pandas.DataFrame, pandas.Series)):
        #                 corruption_diff_fix_df_sample = corruption_diff_fix_df.head(20)
        #             elif (isinstance(corruption_diff_fix_df, numpy.ndarray) and
        #                   corruption_diff_fix_df.ndim == 2):
        #                 corruption_diff_fix_df_sample = corruption_diff_fix_df[:20, :]
        #             elif (isinstance(corruption_diff_fix_df, numpy.ndarray) and
        #                   corruption_diff_fix_df.ndim == 1):
        #                 corruption_diff_fix_df_sample = corruption_diff_fix_df[:20]
        #             else:
        #                 raise NotImplementedError("TODO")
        #             score_after_fixing = []
        #             for score_index in range(self.score_operator_count):
        #                 score_after_fixing.append(
        #                     extracted_plan_results[f"data-errors-corrupt-fix-{score_index}-{transformer_index}"])
        #         else:
        #             report += (
        #                 "Fortunately, corruption function was not able to significantly affect the performance beyond "
        #                 "the configured acceptable threshold.")
        #         if (extracted_plan_results[f"data-errors-corruption-significant-{transformer_index}"] is True and
        #                 extracted_plan_results[
        #                     f"data-errors-corruption-diff-fix-not-empty-{transformer_index}"] is True):
        #             report += (f"After adding a fix method, the pipeline metric was "
        #                        f"{score_after_fixing}. A sample of the fixed rows: {str(corruption_diff_fix_df_sample)}")
        #         elif extracted_plan_results[f"data-errors-corruption-significant-{transformer_index}"] is True:
        #             report += "Unfortunately, the fix method was not able to automatically address the corrupted rows."
        #     report += "\n"
        return report

    @staticmethod
    def corrupt_data(input_df, data_type, corruption_fraction):
        corrupted_result = input_df.copy()
        if data_type == DataType.TEXT:
            if isinstance(corrupted_result, pandas.DataFrame):
                for column in corrupted_result.columns:
                    corrupted_result = get_typo_adder(column, corruption_fraction).fit_transform(corrupted_result)
            elif isinstance(corrupted_result, pandas.Series):
                corrupted_result = pandas.DataFrame(corrupted_result)
                corrupted_result = get_typo_adder(list(corrupted_result.columns)[0], corruption_fraction).fit_transform(
                    corrupted_result)
                corrupted_result = corrupted_result.iloc[:, 0]
            elif isinstance(corrupted_result, list):
                corrupted_result = pandas.DataFrame({"text": corrupted_result})
                corrupted_result = get_typo_adder("text", corruption_fraction).fit_transform(corrupted_result)
                corrupted_result = corrupted_result["text"].to_list()
            else:
                raise NotImplementedError("TODO")
        elif data_type == DataType.CAT:
            # TODO: Broken Characters is pretty slow, maybe do not use it
            """Corrupt broken characters that may be in a pandas df, but may also be in a different format"""
            if isinstance(corrupted_result, pandas.DataFrame):
                for column in corrupted_result.columns:
                    corrupted_result = MissingValues(column=column, fraction=corruption_fraction, na_value="0"
                                                     ).transform(corrupted_result)
            elif isinstance(corrupted_result, list):
                corrupted_result = pandas.DataFrame({"column": corrupted_result})
                corrupted_result = MissingValues(column="column", fraction=corruption_fraction, na_value="0"
                                                 ).transform(corrupted_result)
            elif isinstance(corrupted_result, numpy.ndarray):
                corrupted_result = pandas.DataFrame(corrupted_result)
                for column in corrupted_result.columns:
                    corrupted_result = MissingValues(column=column, fraction=corruption_fraction, na_value="0"
                                                     ).transform(corrupted_result)
            else:
                raise NotImplementedError("TODO")
        elif data_type == DataType.NUM:
            for column in corrupted_result.columns:
                corrupted_result = Scaling(column=column, fraction=corruption_fraction).transform(corrupted_result)
        else:
            raise NotImplementedError(f"TODO: Add support for datatype {DataType.value}!")
        corrupted_result = wrap_in_mlinspect_array_if_necessary(corrupted_result)
        corrupted_result._mlinspect_provenance = None
        return corrupted_result

    @staticmethod
    def apply_diff_filter(input_df, corrupted_index):
        # TODO
        if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
            input_df = input_df.reset_index(drop=True)
        if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
            corrupted_diff = input_df.iloc[corrupted_index]
        elif isinstance(input_df, list):
            corrupted_diff = numpy.array(input_df)[corrupted_index]
        else:
            corrupted_diff = input_df[corrupted_index]
        if isinstance(corrupted_diff, (pandas.Series, pandas.DataFrame)):
            corrupted_diff = corrupted_diff.reset_index(drop=True)
        corrupted_diff = wrap_in_mlinspect_array_if_necessary(corrupted_diff)
        corrupted_diff._mlinspect_provenance = None

        return corrupted_diff

    @staticmethod
    def update_prediction_diff(old_predictions, prediction_diff, prediction_index):
        updated_predictions = numpy.array(old_predictions.copy())
        updated_predictions[prediction_index] = prediction_diff
        return updated_predictions

    @staticmethod
    def fix_data(input_df, data_type):
        fixed_corrupted = input_df.copy()
        if data_type == DataType.TEXT:
            was_series = False
            was_numpy = False
            if isinstance(fixed_corrupted, pandas.Series):
                fixed_corrupted = pandas.DataFrame(fixed_corrupted)
                was_series = True
            elif isinstance(fixed_corrupted, numpy.ndarray):
                fixed_corrupted = pandas.DataFrame({"column": fixed_corrupted})
                was_numpy = True
            for column in fixed_corrupted.columns:
                if fixed_corrupted[column].dtype == object:
                    typo_fixer = get_typo_fixer(column)
                    fixed_corrupted = typo_fixer.fit_transform(fixed_corrupted)
            if was_series is True:
                fixed_corrupted = fixed_corrupted[column]
            elif was_numpy is True:
                fixed_corrupted = fixed_corrupted["column"].to_numpy()
        elif data_type == DataType.CAT:
            fixed_corrupted = input_df
            for column in fixed_corrupted.columns:
                # TODO: There are also smarter ways to do this
                # This should always be the case if the conditional nodes didn't already abort the execution
                assert len(fixed_corrupted[column]) != 0
                if (fixed_corrupted[column].dtype == object and not isinstance(fixed_corrupted[column][0], bool) and
                        not isinstance(input_df[column][0], int)):
                    fixed_corrupted = SimpleImputer(strategy="most_frequent", copy=True, missing_values="0"
                                                    ).fit_transform(fixed_corrupted)
        elif data_type == DataType.NUM:
            # FIXME: This doesn't work that well because the OutlierCleaner never sees clean rows this way
            #  ALso, this might not perform any changes. In these cases, the shadow pipeline shouldn't crash
            for column in fixed_corrupted.columns:
                is_int = fixed_corrupted[column].dtype == int
                fixed_corrupted = OutlierCleaner.fit_transform_all(fixed_corrupted, detection_strategy='IQR',
                                                                   repair_strategy='mean', column=column)
                if is_int:
                    fixed_corrupted[column] = fixed_corrupted[column].astype(int)
            # fixed_corrupted = MinMaxScaler(feature_range=(0, 10)).fit_transform(input_df)
        else:
            raise NotImplementedError(f"TODO: Add support for datatype {DataType.value}!")

        fixed_corrupted = wrap_in_mlinspect_array_if_necessary(fixed_corrupted)
        fixed_corrupted._mlinspect_provenance = None

        return fixed_corrupted

    @staticmethod
    def corrupt_data_diff_detection(input_df, corrupted_result):
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

    @staticmethod
    def fix_data_diff_detection_mask_only(input_df, corrupted_result):
        if isinstance(input_df, (pandas.Series, pandas.DataFrame)):
            input_df = input_df.reset_index(drop=True)
        if isinstance(corrupted_result, (pandas.Series, pandas.DataFrame)):
            corrupted_result = corrupted_result.reset_index(drop=True)
        if isinstance(input_df, pandas.Series):
            corrupt_diff_mask = (corrupted_result != input_df).to_numpy()
        elif len(input_df.shape) == 2:
            corrupt_diff_mask = numpy.any(corrupted_result != input_df, axis=1)
        else:
            corrupt_diff_mask = corrupted_result != input_df
        return corrupt_diff_mask

    @staticmethod
    def fix_data_mask_to_indices(corrupt_diff_mask):
        changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
        return changed_indices_corrupt

    @staticmethod
    def fix_data_diff_indices_before_corruption(corrupted_diff_index, corrupt_fix_diff_mask):
        if isinstance(corrupted_diff_index, (pandas.Series, pandas.DataFrame)):
            corrupted_diff_index = corrupted_diff_index.reset_index(drop=True)
        changed_indices_fix_corrupt = corrupted_diff_index[corrupt_fix_diff_mask]
        return changed_indices_fix_corrupt

    @staticmethod
    def rag_join_update(rag_join_result, inputs):
        vectorstore = rag_join_result[5]
        retrieval_index = rag_join_result[6]

        pandas_retrieval_index_df = pandas.DataFrame(retrieval_index,
                                                     columns=['train_retrieved_1', 'train_retrieved_2',
                                                              'train_retrieved_3', 'train_retrieved_4'])
        pandas_retrieval_index_df['prediction_id'] = list(range(len(rag_join_result[2])))
        diff_inputs = list(numpy.array(inputs))

        diff_rag_result, diff_retrieval_index = RunnableSequencePatching.execute_rag_join_diff(
            rag_join_result[7], diff_inputs, vectorstore)
        new_rag_join_text_result = list(diff_rag_result)

        # TODO: Should we propagate provenance here? Might be important for explanations later
        new_rag_join_result = (rag_join_result[0], rag_join_result[1], new_rag_join_text_result, list(inputs),
                               None, rag_join_result[5], diff_retrieval_index, rag_join_result[7])
        return new_rag_join_result

    @staticmethod
    def add_new_score_and_score_extraction_nodes(new_dag, new_predict_node, score_operators,
                                                 test_labels_operators):
        for score_index, score_operator in enumerate(score_operators):
            new_score_node = copy_node_with_new_id(singleton, score_operator)
            new_dag.add_edge(new_predict_node, new_score_node, arg_index=0)
            new_dag.add_edge(test_labels_operators[0], new_score_node, arg_index=1)
            parents = get_sorted_parent_nodes(new_dag, score_operator)[2:]
            for parent_index, parent in enumerate(parents):
                # TODO: There might be shadow pipeline edge cases where this does not work without further work
                new_dag.add_edge(parent, new_score_node, arg_index=parent_index + 2)

            retrain_extraction_node = get_intermediate_extraction_node(singleton, new_score_node,
                                                                       f"label-errors-flip-retrain-{score_index}")
            new_dag.add_edge(new_score_node, retrain_extraction_node, arg_index=0)

    @staticmethod
    def _get_transformer_operators_to_test(dag):
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
                data_parent_and_data_type.append((data_parent, transformer, DataType.TEXT))
        return data_parent_and_data_type

    @staticmethod
    def condition_corruption_significant_function(corruption_significant_relative_threshold, *scores):
        # This function compares all scores of the original pipeline and the corrupted pipeline
        # So the number of scores in both pipeline variants should be equal
        assert len(scores) % 2 == 0
        number_of_scores_each = int(len(scores) / 2)
        scores_different_enough = False
        for score_index in range(number_of_scores_each):
            # TODO: More sophisticated handling of FairLearn MetricFrames
            if ((isinstance(scores[score_index], float) and scores[
                score_index] * corruption_significant_relative_threshold >=
                 scores[score_index + number_of_scores_each]) or
                    (isinstance(scores[score_index], MetricFrame) and
                     scores[score_index].overall * corruption_significant_relative_threshold >=
                     scores[score_index + number_of_scores_each].overall)
            ):
                scores_different_enough = True
        return scores_different_enough

    @staticmethod
    def get_slice_finder_mask_and_slice(side_info_df, encoded_test_labels, predicted_test_labels):
        # TODO: In the documentation, it is only used on train and with known float loss
        sf = Slicefinder(
            alpha=0.95,
            k=1,
            max_l=2,
            min_sup=1,
            verbose=True,
        )
        if isinstance(encoded_test_labels, pandas.Series):
            encoded_test_labels = encoded_test_labels.to_numpy()

        side_info_df = side_info_df.fillna("nan")
        sf_result = sf.fit(side_info_df, (encoded_test_labels.reshape(-1, ) != predicted_test_labels.reshape(-1, )))
        if len(sf_result.top_slices_) == 0:
            return None, None
        top_slice = sf_result.top_slices_[0]
        # Make sure that our experiment consistently hits the case where something needs to be translated
        test_mask = numpy.ones(shape=(len(side_info_df)), dtype=bool)
        for column_index, column_value in enumerate(top_slice):
            if column_value is not None:
                test_mask = test_mask & (side_info_df.iloc[:, column_index] == column_value).to_numpy()
        return top_slice, test_mask

    @staticmethod
    def projection_processing_func(column_names, input):
        # TODO: What if not all inputs are pandas dfs?
        result = input[column_names]
        result = wrap_in_mlinspect_array_if_necessary(result)
        return result

    @staticmethod
    def prov_join_processing_func(target_data_source_id, left, right):
        # TODO: What if not all inputs are pandas dfs?
        assert left._mlinspect_provenance
        assert right._mlinspect_provenance
        left_prov = left._mlinspect_provenance
        right_prov = right._mlinspect_provenance

        right_prov_columns = []
        right_prov_dict = {}
        for prov_key, prov_values in right_prov.items():
            current_data_source, index_to_deduplicate = prov_key.rsplit('_', 1)
            if int(current_data_source) == target_data_source_id:
                right_prov_columns.append(prov_key)
                right_prov_dict[prov_key] = prov_values
        assert len(right_prov_dict) == 1
        right_prov_df = pandas.DataFrame({'join_prov': list(right_prov.values())[0]})

        left_copy = left.copy()
        assert len(left_prov) == 1
        prov_value = list(left_prov.values())[0]
        left_copy['join_prov'] = prov_value

        result = pandas.merge(right_prov_df, left_copy, how="inner", on="join_prov")
        assert len(result) == len(list(right_prov.values())[0])
        result = result.drop("join_prov", axis=1)
        result = wrap_in_mlinspect_array_if_necessary(result)
        result._mlinspect_provenance = right_prov
        return result
