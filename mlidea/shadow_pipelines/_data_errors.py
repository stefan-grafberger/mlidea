# 3. robustness:
# we can just run a corruption udf on everything that changes a small part of the data, then we run a diff
# detection and use it to create a mask. only on the changed ones do we need to try a correction. this needs to
# be modeled in the DAG, since we don't want to run the correction on all data in this first step.
# however, not perfectly accurate results
#
# also, the provenance part should be possible to turn off for performance comparisons. in general, we do need
# provenance, but with enough simplyfying assumptions about the order not changing and all data being available
# until right before the featurisation, we can get away without. maybe I wasted a day today... or we still build
# it to have better explanations?
# 1. Mislabel: in mlwhatif, two approaches, shapley and cleanlab. for mlidea workshop paper we only used shapley.
# in general, for LLM+RAG, we need the embeddings, that we don't have specifically in the DAG right now.
# Do we need to update the DAG? Or use some hack like letting the RAG join output the embeddings next to the text?
# but might have a big of added performance overhead. then, conditional operator depending on how many mislabels
# found. but maybe not that problematic here. but maybe for this we do want to use the provenance since the labeling
# might not be the final step in the data preprocessing and there might be filte
from enum import Enum
from functools import partial

import duckdb
import networkx
import numpy
import pandas
from jenga.corruptions.numerical import Scaling
from jenga.corruptions.text import BrokenCharacters
from sklearn.preprocessing import MinMaxScaler

from mlidea.analysis._cleaning_methods import OutlierCleaner
from mlidea.execution._pipeline_executor import singleton
from mlidea.analysis._analysis_utils import find_nodes_by_type
from mlidea import OperatorType, DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline
from mlidea.shadow_pipelines._utils import get_intermediate_extraction_node, copy_node_with_new_id, \
    get_sorted_parent_nodes, find_train_or_test_pipeline_part_end, get_typo_adder, duplicate_descendants, \
    get_typo_fixer, get_conditional_stop_node
from mlidea.monkeypatching._patch_langchain import RunnableSequencePatching
from mlidea.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


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


class DataErrorRobustness(ShadowPipeline):
    """
    The Data Error Robustness Shadow Pipeline
    """

    def check_rebuilding_necessary(self, extracted_plan_results: dict[str, any]) -> any:
        return False

    def __init__(self, corruption_fraction=.1, corruption_significant_relative_threshold=0.99):
        self._corruption_fraction = corruption_fraction
        self._shadow_pipeline_id = (corruption_fraction, corruption_significant_relative_threshold)
        self.score_operator_count = 0
        self._corruption_significant_relative_threshold = corruption_significant_relative_threshold

    @property
    def shadow_pipeline_id(self):
        return self._shadow_pipeline_id

    @property
    def simple_name(self):
        return "data_errors"

    def generate_shadow_pipeline_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        # pylint: disable=too-many-locals,too-many-statements
        # TODO: Maybe it would be better to delete all unrelated DAG nodes here that are not specifically mentioned
        #  below. But this only works once intermediate resutl caching is implemented

        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)

        if len(rag_join_operators) == 0:
            new_dag = self.get_traditional_ml_dag(dag)
        else:
            new_dag = self.get_llm_rag_dag(dag)

        return new_dag

    def get_traditional_ml_dag(self, dag):
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
        DataErrorRobustness.add_orig_score_extraction_nodes(new_dag, score_operators)
        self.score_operator_count = len(score_operators)

        data_parent_transformer_and_data_type = DataErrorRobustness._get_transformer_operators_to_test(dag)

        for data_parent, transformer, data_type, in data_parent_transformer_and_data_type:
            processing_func = partial(DataErrorRobustness.corrupt_data,
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
                                               DataErrorRobustness.corrupt_data_diff_detection)
            new_dag.add_edge(data_parent, new_corruption_diff_node, arg_index=0)
            new_dag.add_edge(new_corruption_node, new_corruption_diff_node, arg_index=1)

            new_corruption_diff_filter_node = DagNode(singleton.get_next_op_id(),
                                                      BasicCodeLocation("Data Errors", None),
                                                      OperatorContext(OperatorType.SELECTION, None),
                                                      DagNodeDetails(
                                                          f"Filter for diff only",
                                                          None),
                                                      None,
                                                      DataErrorRobustness.apply_diff_filter)
            new_dag.add_edge(new_corruption_node, new_corruption_diff_filter_node, arg_index=0)
            new_dag.add_edge(new_corruption_diff_node, new_corruption_diff_filter_node, arg_index=1)

            extraction_node = get_intermediate_extraction_node(singleton, new_corruption_diff_filter_node,
                                                               "data-errors-corruption-diff")
            new_dag.add_edge(new_corruption_diff_filter_node, extraction_node, arg_index=0)

            # Evaluate with corrupted data
            old_copied_nodes, new_nodes = duplicate_descendants(dag, new_dag, data_parent,
                                                                new_corruption_diff_filter_node, singleton)
            new_score_nodes = [node for node in new_nodes if node.operator_info.operator == OperatorType.SCORE]

            # Now apply filter to all other concatenation inputs
            concats = [node for node in new_nodes if node.operator_info.operator == OperatorType.CONCATENATION]
            if len(concats) >= 1:
                if len(concats) != 1:
                    raise NotImplementedError(
                        "Currently, Label Errors only supports pipelines following a very specific "
                        "pattern!")
                for concat in concats:
                    concat_parents = get_sorted_parent_nodes(new_dag, concat)
                    for concat_parent in concat_parents:
                        if concat_parent not in new_nodes:
                            edge_data = new_dag.get_edge_data(concat_parent, concat)
                            new_dag.remove_edge(concat_parent, concat)
                            new_concat_parent_filter_node = DagNode(singleton.get_next_op_id(),
                                                                    BasicCodeLocation("Data Errors", None),
                                                                    OperatorContext(OperatorType.SELECTION, None),
                                                                    DagNodeDetails(
                                                                        f"Filter for diff only",
                                                                        None),
                                                                    None,
                                                                    DataErrorRobustness.apply_diff_filter)
                            new_dag.add_edge(concat_parent, new_concat_parent_filter_node, arg_index=0)
                            new_dag.add_edge(new_corruption_diff_node, new_concat_parent_filter_node, arg_index=1)
                            new_dag.add_edge(new_concat_parent_filter_node, concat, **edge_data)
            test_score = [node for node in new_nodes
                          if node.operator_info.operator == OperatorType.SCORE][0]
            test_predict = [node for node in new_nodes
                            if node.operator_info.operator == OperatorType.PREDICT][0]
            old_predict = [node for node in old_copied_nodes
                           if node.operator_info.operator == OperatorType.PREDICT][0]
            edge_data = new_dag.get_edge_data(test_predict, test_score)
            new_dag.remove_edge(test_predict, test_score)

            new_corrupt_predict_diff_update_node = DagNode(singleton.get_next_op_id(),
                                                   BasicCodeLocation("Data Errors", None),
                                                   OperatorContext(OperatorType.SELECTION, None),
                                                   DagNodeDetails(
                                                       f"Merge corruption diff with old predictions",
                                                       None),
                                                   None,
                                                   DataErrorRobustness.update_prediction_diff)
            new_dag.add_edge(old_predict, new_corrupt_predict_diff_update_node, arg_index=0)
            new_dag.add_edge(test_predict, new_corrupt_predict_diff_update_node, arg_index=1)
            new_dag.add_edge(new_corruption_diff_node, new_corrupt_predict_diff_update_node, arg_index=2)
            new_dag.add_edge(new_corrupt_predict_diff_update_node, test_score, **edge_data)

            if len(new_score_nodes) < 1:
                raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                          "pattern!")
            for score_index, score_operator in enumerate(new_score_nodes):
                extraction_node = get_intermediate_extraction_node(singleton, score_operator,
                                                                   f"label-errors-corrupt-{score_index}")
                new_dag.add_edge(score_operator, extraction_node, arg_index=0)

            def condition_corruption_significant_function(corruption_significant_relative_threshold, *scores):
                # This function compares all scores of the original pipeline and the corrupted pipeline
                # So the number of scores in both pipeline variants should be equal
                assert len(scores) % 2 == 0
                number_of_scores_each = int(len(scores) / 2)
                scores_different_enough = False
                for score_index in range(number_of_scores_each):
                    if (scores[score_index] * corruption_significant_relative_threshold >=
                            scores[score_index + number_of_scores_each]):
                        scores_different_enough = True
                return scores_different_enough
            condition_processing_func = partial(condition_corruption_significant_function,
                                      self._corruption_significant_relative_threshold)
            conditional_corruption_significant_node = get_conditional_stop_node(
                singleton, condition_processing_func, "data-errors-corruption-significant",
                "Check if fix function made changes", new_score_nodes[0])
            for score_index, score_operator in enumerate(score_operators):
                new_dag.add_edge(score_operator, conditional_corruption_significant_node,
                                 arg_index=score_index)
            for score_index, score_operator in enumerate(new_score_nodes):
                new_dag.add_edge(score_operator, conditional_corruption_significant_node,
                                 arg_index=score_index + self.score_operator_count)
            # End evaluate

            if data_type == DataType.TEXT:
                fix_input_node = new_corruption_diff_filter_node
            else:
                fix_input_node = new_corruption_node

            processing_func = partial(DataErrorRobustness.fix_data, data_type=data_type)
            new_fix_node = DagNode(singleton.get_next_op_id(),
                                   BasicCodeLocation("Data Errors", None),
                                   OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                   DagNodeDetails(
                                       f"Fix {self._corruption_fraction} of {data_type.value} values", None),
                                   None,
                                   processing_func)
            new_dag.add_edge(fix_input_node, new_fix_node, arg_index=0)
            new_dag.add_edge(conditional_corruption_significant_node, new_fix_node, arg_index=1)

            if data_type == DataType.TEXT:
                fix_node_to_extract = new_fix_node
            else:
                new_fix_with_corruption_change_filter_node = DagNode(singleton.get_next_op_id(),
                                                                     BasicCodeLocation("Data Errors", None),
                                                                     OperatorContext(OperatorType.SELECTION, None),
                                                                     DagNodeDetails(
                                                                         f"Filter for diff only",
                                                                         None),
                                                                     None,
                                                                     DataErrorRobustness.apply_diff_filter)
                new_dag.add_edge(new_fix_node, new_fix_with_corruption_change_filter_node, arg_index=0)
                new_dag.add_edge(new_corruption_diff_node, new_fix_with_corruption_change_filter_node, arg_index=1)
                fix_node_to_extract = new_fix_with_corruption_change_filter_node
            extraction_node = get_intermediate_extraction_node(singleton, new_fix_node,
                                                               "data-errors-corruption-diff-fix")
            new_dag.add_edge(fix_node_to_extract, extraction_node, arg_index=0)

            new_fix_diff_mask_node = DagNode(singleton.get_next_op_id(),
                                             BasicCodeLocation("Data Errors", None),
                                             OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                             DagNodeDetails(
                                                 f"Compute change mask from fixing", None),
                                             None,
                                             DataErrorRobustness.fix_data_diff_detection_mask_only)
            new_dag.add_edge(fix_input_node, new_fix_diff_mask_node, arg_index=0)
            new_dag.add_edge(new_fix_node, new_fix_diff_mask_node, arg_index=1)

            new_fix_diff_indices_node = DagNode(singleton.get_next_op_id(),
                                                BasicCodeLocation("Data Errors", None),
                                                OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                                DagNodeDetails(
                                                    f"Compute changed indices from fixing", None),
                                                None,
                                                DataErrorRobustness.fix_data_mask_to_indices)
            new_dag.add_edge(new_fix_diff_mask_node, new_fix_diff_indices_node, arg_index=0)

            condition_fix_function = lambda np_array: len(np_array) != 0
            conditional_fixes_changed_something_node = get_conditional_stop_node(
                singleton, condition_fix_function, "data-errors-corruption-diff-fix-not-empty",
                "Check if fix function made changes", new_fix_diff_indices_node)
            new_dag.add_edge(new_fix_diff_indices_node, conditional_fixes_changed_something_node, arg_index=0)

            new_fix_diff_filter_node = DagNode(singleton.get_next_op_id(),
                                               BasicCodeLocation("Data Errors", None),
                                               OperatorContext(OperatorType.SELECTION, None),
                                               DagNodeDetails(
                                                   f"Filter for diff only",
                                                   None),
                                               None,
                                               DataErrorRobustness.apply_diff_filter)
            new_dag.add_edge(new_fix_node, new_fix_diff_filter_node, arg_index=0)
            new_dag.add_edge(new_fix_diff_indices_node, new_fix_diff_filter_node, arg_index=1)
            new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_diff_filter_node, arg_index=2)

            # Evaluate with corrupted data
            old_copied_nodes, new_nodes = duplicate_descendants(dag, new_dag, data_parent,
                                                                new_fix_diff_filter_node, singleton)
            new_score_nodes = [node for node in new_nodes if node.operator_info.operator == OperatorType.SCORE]

            # Now apply filter to all other concatenation inputs
            concats = [node for node in new_nodes if node.operator_info.operator == OperatorType.CONCATENATION]
            if len(concats) >= 1:
                if len(concats) != 1:
                    raise NotImplementedError(
                        "Currently, Label Errors only supports pipelines following a very specific "
                        "pattern!")
                for concat in concats:
                    concat_parents = get_sorted_parent_nodes(new_dag, concat)
                    for concat_parent in concat_parents:
                        if concat_parent not in new_nodes:
                            edge_data = new_dag.get_edge_data(concat_parent, concat)
                            new_dag.remove_edge(concat_parent, concat)
                            new_concat_parent_filter_node = DagNode(singleton.get_next_op_id(),
                                                                    BasicCodeLocation("Data Errors", None),
                                                                    OperatorContext(OperatorType.SELECTION, None),
                                                                    DagNodeDetails(
                                                                        f"Filter for diff only",
                                                                        None),
                                                                    None,
                                                                    DataErrorRobustness.apply_diff_filter)
                            new_dag.add_edge(concat_parent, new_concat_parent_filter_node, arg_index=0)
                            new_dag.add_edge(new_fix_diff_indices_node, new_concat_parent_filter_node, arg_index=1)
                            new_dag.add_edge(new_concat_parent_filter_node, concat, **edge_data)
            test_score = [node for node in new_nodes
                          if node.operator_info.operator == OperatorType.SCORE][0]
            test_predict = [node for node in new_nodes
                            if node.operator_info.operator == OperatorType.PREDICT][0]
            edge_data = new_dag.get_edge_data(test_predict, test_score)
            new_dag.remove_edge(test_predict, test_score)

            if data_type == DataType.TEXT:
                new_indices_before_corruption_node = DagNode(singleton.get_next_op_id(),
                                                             BasicCodeLocation("Data Errors", None),
                                                             OperatorContext(OperatorType.SELECTION, None),
                                                             DagNodeDetails(
                                                                 f"Compute indices relative to before corrupting and fixing",
                                                                 None),
                                                             None,
                                                             DataErrorRobustness.fix_data_diff_indices_before_corruption)
                new_dag.add_edge(new_corruption_diff_node, new_indices_before_corruption_node, arg_index=0)
                new_dag.add_edge(new_fix_diff_mask_node, new_indices_before_corruption_node, arg_index=1)

                prediction_filter_index_node = new_indices_before_corruption_node
            else:
                prediction_filter_index_node = new_fix_diff_indices_node

            new_fix_predict_diff_update_node = DagNode(singleton.get_next_op_id(),
                                                   BasicCodeLocation("Data Errors", None),
                                                   OperatorContext(OperatorType.SELECTION, None),
                                                   DagNodeDetails(
                                                       f"Merge fix diff with old predictions",
                                                       None),
                                                   None,
                                                   DataErrorRobustness.update_prediction_diff)
            new_dag.add_edge(new_corrupt_predict_diff_update_node, new_fix_predict_diff_update_node, arg_index=0)
            new_dag.add_edge(test_predict, new_fix_predict_diff_update_node, arg_index=1)
            new_dag.add_edge(prediction_filter_index_node, new_fix_predict_diff_update_node, arg_index=2)
            new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_predict_diff_update_node, arg_index=3)
            new_dag.add_edge(new_fix_predict_diff_update_node, test_score, **edge_data)

            if len(new_score_nodes) < 1:
                raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                          "pattern!")
            for score_index, score_operator in enumerate(new_score_nodes):
                extraction_node = get_intermediate_extraction_node(singleton, score_operator,
                                                                   f"label-errors-corrupt-fix-{score_index}")
                new_dag.add_edge(score_operator, extraction_node, arg_index=0)
            # End evaluate
        return new_dag

    def get_llm_rag_dag(self, dag):
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
        label_encoder_operators = list(new_dag.predecessors(test_labels_operators[0]))
        if len(label_encoder_operators) != 1 or "label_binarize" not in label_encoder_operators[0].details.description:
            raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                      "pattern!")

        self.score_operator_count = len(score_operators)

        train_labels_dict_conversion = list(new_dag.predecessors(train_labels_operators[0]))[0]
        train_labels_before_dict = list(new_dag.predecessors(train_labels_dict_conversion))[0]

        DataErrorRobustness.add_orig_score_extraction_nodes(new_dag, score_operators)

        processing_func = partial(DataErrorRobustness.shapley_top_k_func_llm,
                                  train_fraction_to_consider=self._train_fraction_to_consider,
                                  test_fraction_to_consider=self._test_fraction_to_consider,
                                  cleaning_batch_size=self._cleaning_batch_size,
                                  label_encoding_op=label_encoder_operators[0])
        new_shapley_node = DagNode(singleton.get_next_op_id(),
                                   BasicCodeLocation("Label Errors", None),
                                   OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                   DagNodeDetails(
                                       f"Top {self._cleaning_batch_size} Shapley values", None),
                                   None,
                                   processing_func)
        new_dag.add_edge(rag_join_operators[0], new_shapley_node, arg_index=0)
        new_dag.add_edge(train_labels_before_dict, new_shapley_node, arg_index=1)
        new_dag.add_edge(test_data_operators[0], new_shapley_node, arg_index=2)
        new_dag.add_edge(test_labels_operators[0], new_shapley_node, arg_index=3)
        extraction_node = get_intermediate_extraction_node(singleton, new_shapley_node, "label-errors-shapley-values")
        new_dag.add_edge(new_shapley_node, extraction_node, arg_index=0)

        new_label_flip_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("Label Errors", None),
                                      OperatorContext(OperatorType.PROJECTION, None),
                                      DagNodeDetails(
                                          f"Flip {self._cleaning_batch_size} most likely incorrect labels", None),
                                      None,
                                      DataErrorRobustness.label_flip_processing_func_llm)
        new_dag.add_edge(rag_join_operators[0], new_label_flip_node, arg_index=0)
        new_dag.add_edge(train_labels_operators[0], new_label_flip_node, arg_index=1)
        new_dag.add_edge(extraction_node, new_label_flip_node, arg_index=2)
        new_dag.add_edge(test_data_operators[0], new_label_flip_node, arg_index=3)

        new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_label_flip_node, new_predict_node, arg_index=0)
        DataErrorRobustness.add_new_score_and_score_extraction_nodes(new_dag, new_predict_node, score_operators,
                                                                     test_labels_operators)
        return new_dag

    @staticmethod
    def add_orig_score_extraction_nodes(new_dag, score_operators):
        for score_index, score_operator in enumerate(score_operators):
            orig_extraction_node = get_intermediate_extraction_node(singleton, score_operator,
                                                                    f"label-errors-orig-{score_index}")
            new_dag.add_edge(score_operator, orig_extraction_node, arg_index=0)

    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        orig_result = []
        for score_index in range(self.score_operator_count):
            orig_result.append(extracted_plan_results[f"label-errors-orig-{score_index}"])
        corruption_diff_df_sample = extracted_plan_results["data-errors-corruption-diff"].head(20)
        corrupt_result = []
        for score_index in range(self.score_operator_count):
            corrupt_result.append(extracted_plan_results[f"label-errors-corrupt-{score_index}"])
        report = (f"The original result was {orig_result}. After corrupting {self._corruption_fraction} of rows, "
                  f"the pipeline metric was {corrupt_result}, indicating robustness problems. A sample of the corrupted "
                  f"rows: {str(corruption_diff_df_sample)}. ")

        if extracted_plan_results["data-errors-corruption-significant"] is True:
            corruption_diff_fix_df = extracted_plan_results["data-errors-corruption-diff-fix"]
            if isinstance(corruption_diff_fix_df, (pandas.DataFrame, pandas.Series)):
                corruption_diff_fix_df_sample = corruption_diff_fix_df.head(20)
            else:
                corruption_diff_fix_df_sample = corruption_diff_fix_df[:20, :]
            score_after_fixing = []
            for score_index in range(self.score_operator_count):
                score_after_fixing.append(extracted_plan_results[f"label-errors-corrupt-fix-{score_index}"])
        else:
            report += ("Fortunately, corruption function was not able to significantly affect the performance beyond "
                       "the configured acceptable threshold.")
        if (extracted_plan_results["data-errors-corruption-significant"] is True and
                extracted_plan_results["data-errors-corruption-diff-fix-not-empty"] is True):
            report += (f"After adding a fix method, the pipeline metric was "
                       f"{score_after_fixing}. A sample of the fixed rows: {str(corruption_diff_fix_df_sample)}")
        elif extracted_plan_results["data-errors-corruption-significant"] is True:
            report += "Unfortunately, the fix method was not able to automatically address the corrupted rows."
        return report

    @staticmethod
    def shapley_top_k_func_llm(rag_join_result, train_labels_before_dict, encoded_test_data, encoded_test_labels,
                               train_fraction_to_consider, test_fraction_to_consider, cleaning_batch_size,
                               label_encoding_op):
        # TODO: Should we propagate provenance here? Might be important for explanations later
        indices = numpy.arange(len(train_labels_before_dict))
        numpy.random.shuffle(indices)

        num_values_to_typo = int(len(train_labels_before_dict) * train_fraction_to_consider)
        train_indices_to_consider = indices[:num_values_to_typo]

        vectorstore = rag_join_result[5]
        train_data_sample = numpy.array(vectorstore.get(
            ids=list(map(str, train_indices_to_consider)), include=["embeddings"])['embeddings'])
        to_label_encode = train_labels_before_dict.iloc[train_indices_to_consider, 0]
        to_label_encode._mlinspect_provenance = None
        train_label_sample = label_encoding_op.processing_func(to_label_encode)

        indices = numpy.arange(len(encoded_test_labels))
        num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
        test_indices_to_consider = indices[:num_values_to_typo]
        test_data_sample = numpy.array(vectorstore.embeddings.embed_documents(
            numpy.array(encoded_test_data)[test_indices_to_consider]))
        test_label_sample = encoded_test_labels[test_indices_to_consider]

        shapley_values = DataErrorRobustness._compute_shapley_values(train_data_sample,
                                                                     numpy.squeeze(train_label_sample),
                                                                     test_data_sample, numpy.squeeze(test_label_sample))
        df_with_id_and_shapley_value = pandas.DataFrame(
            {"train_id": train_indices_to_consider, "shapley_value": shapley_values})

        rows_to_fix = df_with_id_and_shapley_value.nsmallest(cleaning_batch_size, "shapley_value")
        return rows_to_fix

    @staticmethod
    def corrupt_data(input_df, data_type, corruption_fraction):
        if data_type == DataType.TEXT:
            if isinstance(input_df, pandas.DataFrame):
                for column in input_df.columns:
                    corrupted_result = get_typo_adder(column).fit_transform(input_df)
            elif isinstance(input_df, pandas.Series):
                pandas_df = pandas.DataFrame({input_df.name: input_df})
                corrupted_result = get_typo_adder(input_df.name).fit_transform(pandas_df)
                corrupted_result = corrupted_result[input_df.name]
            else:
                raise NotImplementedError("TODO")
        elif data_type == DataType.CAT:
                # TODO: Broken Characters is pretty slow, maybe do not use it
                """Corrupt broken characters that may be in a pandas df, but may also be in a different format"""
                if isinstance(input_df, pandas.DataFrame):
                    for column in input_df.columns:
                        corrupted_result = BrokenCharacters(column=column, fraction=corruption_fraction).transform(input_df)
                elif isinstance(input_df, list):
                    pandas_df = pandas.DataFrame({"column": input_df})
                    corrupted_result = BrokenCharacters(column="column", fraction=corruption_fraction).transform(
                        pandas_df)
                elif isinstance(input_df, numpy.ndarray):
                    pandas_df = pandas.DataFrame(input_df)
                    for column in pandas_df.columns:
                        corrupted_result = BrokenCharacters(column=column, fraction=corruption_fraction).transform(
                            pandas_df)
                else:
                    raise NotImplementedError("TODO")
        elif data_type == DataType.NUM:
            for column in input_df.columns:
                corrupted_result = Scaling(column=column, fraction=corruption_fraction).transform(input_df)
        else:
            raise NotImplementedError(f"TODO: Add support for datatype {DataType.value}!")
        corrupted_result._mlinspect_provenance = None
        return corrupted_result

    @staticmethod
    def apply_diff_filter(input_df, corrupted_index):
        # TODO
        if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
            input_df = input_df.reset_index(drop=True)
        if isinstance(input_df, (pandas.DataFrame, pandas.Series)):
            corrupted_diff = input_df.iloc[corrupted_index]
        else:
            corrupted_diff = input_df[corrupted_index]
        if isinstance(corrupted_diff, (pandas.Series, pandas.DataFrame)):
            corrupted_diff = corrupted_diff.reset_index(drop=True)
        corrupted_diff._mlinspect_provenance = None

        return corrupted_diff

    @staticmethod
    def update_prediction_diff(old_predictions, prediction_diff, prediction_index):
        updated_predictions = old_predictions.copy()
        updated_predictions[prediction_index] = prediction_diff
        return updated_predictions

    @staticmethod
    def fix_data(input_df, data_type):
        # TODO
        if data_type == DataType.TEXT:
            typo_fixer = get_typo_fixer()
            for column in input_df.columns:
                fixed_corrupted = typo_fixer.fit_transform(input_df[[column]])
        elif data_type == DataType.CAT:
            typo_fixer = get_typo_fixer()
            for column in input_df.columns:
                # TODO: There are also smarter ways to do this
                fixed_corrupted = typo_fixer.fit_transform(input_df[[column]])
        elif data_type == DataType.NUM:
            # FIXME: This doesn't work that well because the OutlierCleaner never sees clean rows this way
            #  ALso, this might not perform any changes. In these cases, the shadow pipeline shouldn't crash
            for column in input_df.columns:
                fixed_corrupted = OutlierCleaner.fit_transform_all(input_df, detection_strategy='IF',
                                                                   repair_strategy='mean', column=column)
            # fixed_corrupted = MinMaxScaler(feature_range=(0, 10)).fit_transform(input_df)
        else:
            raise NotImplementedError(f"TODO: Add support for datatype {DataType.value}!")

        fixed_corrupted = wrap_in_mlinspect_array_if_necessary(fixed_corrupted)
        fixed_corrupted._mlinspect_provenance = None

        return fixed_corrupted

    @staticmethod
    def corrupt_data_diff_detection(input_df, corrupted_result):
        if isinstance(input_df, pandas.Series):
            corrupt_diff_mask = (corrupted_result != input_df).to_numpy()
        else:
            corrupt_diff_mask = numpy.any(corrupted_result != input_df, axis=1)
        changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
        return changed_indices_corrupt

    @staticmethod
    def fix_data_diff_detection_mask_only(input_df, corrupted_result):
        if isinstance(input_df, pandas.Series):
            corrupt_diff_mask = (corrupted_result != input_df).to_numpy()
        else:
            corrupt_diff_mask = numpy.any(corrupted_result != input_df, axis=1)
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
    def label_flip_processing_func_llm(rag_join_result, encoded_train_labels, shapley_result, inputs):
        # TODO: Should we propagate provenance here? Might be important for explanations later
        mislabeled_indices = shapley_result['train_id']
        classes = set()
        class_search_index = 0
        label_key = None
        while len(classes) != 2 and class_search_index < len(encoded_train_labels):
            label_dict_items = list(encoded_train_labels[class_search_index].items())
            assert len(label_dict_items) == 1
            label_key, label_value = label_dict_items[0]
            classes.add(label_value)
            class_search_index += 1
        classes = list(classes)
        diff_encoded_train_labels = numpy.array(encoded_train_labels)[mislabeled_indices]
        for mislabeled_row in diff_encoded_train_labels:
            assert label_key is not None
            current_val = mislabeled_row[label_key]
            current_val_index = classes.index(current_val)
            mislabeled_row[label_key] = classes[1 - current_val_index]
        diff_encoded_train_labels = list(diff_encoded_train_labels)

        # Update the labels in the vectorstore
        vectorstore = rag_join_result[5]
        vectorstore_ids = list(map(str, mislabeled_indices))
        old_entries = vectorstore.get(ids=vectorstore_ids, include=["embeddings", "documents", "metadatas"])
        documents = old_entries['documents']
        embeddings = old_entries['embeddings']
        old_metadata = old_entries['metadatas']
        vectorstore._collection.update(vectorstore_ids, embeddings, diff_encoded_train_labels, documents)
        retrieval_index = rag_join_result[6]

        pandas_retrieval_index_df = pandas.DataFrame(retrieval_index,
                                                     columns=['train_retrieved_1', 'train_retrieved_2',
                                                              'train_retrieved_3', 'train_retrieved_4'])
        pandas_retrieval_index_df['prediction_id'] = list(range(len(rag_join_result[2])))
        changed_df = shapley_result[['train_id']]
        all_predictions_to_rerun = duckdb.query("""
                    SELECT DISTINCT prediction_id
                    FROM changed_df c JOIN pandas_retrieval_index_df p 
                    ON c.train_id = train_retrieved_1 
                    OR c.train_id = train_retrieved_2 
                    OR c.train_id = train_retrieved_3 
                    OR c.train_id = train_retrieved_4 
                """).fetchnumpy()['prediction_id']

        diff_inputs = list(numpy.array(inputs)[all_predictions_to_rerun])
        diff_rag_result, diff_retrieval_index = RunnableSequencePatching.execute_rag_join_diff(
            rag_join_result[7], diff_inputs, vectorstore)
        # Revert vectorstore changes again
        vectorstore._collection.update(vectorstore_ids, embeddings, old_metadata, documents)

        new_rag_join_text_result = numpy.array(rag_join_result[2])
        new_rag_join_text_result[all_predictions_to_rerun] = diff_rag_result
        new_rag_join_text_result = list(new_rag_join_text_result)

        new_retrieval_index = retrieval_index.copy()
        new_retrieval_index[all_predictions_to_rerun, :] = diff_retrieval_index

        new_rag_join_result = (rag_join_result[0], rag_join_result[1], new_rag_join_text_result, rag_join_result[3],
                               rag_join_result[4], rag_join_result[5], new_retrieval_index, rag_join_result[7])
        return new_rag_join_result

    @staticmethod
    def label_flip_processing_func_ml(encoded_train_labels, shapley_result):
        # TODO: Should we propagate provenance here? Might be important for explanations later
        unfair_indices = shapley_result['train_id'].to_numpy()
        if isinstance(encoded_train_labels, (pandas.Series, pandas.DataFrame)):
            modified_encoded_train_labels = encoded_train_labels.reset_index(drop=True, inplace=False)
        else:
            modified_encoded_train_labels = encoded_train_labels.copy()
        if isinstance(modified_encoded_train_labels, pandas.Series):
            is_bool = pandas.api.types.is_bool_dtype(modified_encoded_train_labels)
            modified_encoded_train_labels[unfair_indices] = 1 - modified_encoded_train_labels[unfair_indices]
            if is_bool:
                modified_encoded_train_labels = modified_encoded_train_labels.astype(bool)
        else:
            modified_encoded_train_labels[unfair_indices, :] = 1 - modified_encoded_train_labels[unfair_indices, :]
        return modified_encoded_train_labels

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
