from functools import partial

import networkx
import numpy
import pandas
from fairlearn.metrics import MetricFrame
from jenga.corruptions.generic import MissingValues
from jenga.corruptions.numerical import Scaling
from sklearn.impute import SimpleImputer

from mlidea import OperatorType, DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlidea.analysis._analysis_utils import find_nodes_by_type
from mlidea.analysis._cleaning_methods import detect_outlier_interquartile_range
from mlidea.execution._pipeline_executor import singleton
from mlidea.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline
from mlidea.shadow_pipelines._utils import get_intermediate_extraction_node, copy_node_with_new_id, \
    get_sorted_parent_nodes, get_typo_adder, duplicate_descendants, \
    get_typo_fixer, get_conditional_stop_node, DataType, get_transformer_parents_with_data_types, \
    get_relative_score_change, add_orig_score_extraction_nodes, fix_data_diff_detection_mask_only, \
    fix_data_mask_to_indices, rag_join_update, \
    get_diff_filter_node, get_changed_indices_node, merge_prediction_diff_with_old_predictions, \
    add_new_score_and_score_extraction_nodes, assert_standard_llm_shape, assert_standard_ml_shape, get_top_n_df_rows


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
        self._transformer_inputs_to_check = []
        self._corruption_significant_relative_threshold = corruption_significant_relative_threshold

    @property
    def shadow_pipeline_id(self):
        return self._shadow_pipeline_id

    @property
    def simple_name(self):
        return "data_errors"

    def generate_shadow_pipeline_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
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
        assert_standard_ml_shape(dag, "Data Errors")

        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        add_orig_score_extraction_nodes(singleton, new_dag, score_operators)
        self.score_operator_count = len(score_operators)

        data_parent_transformer_and_data_type = get_transformer_parents_with_data_types(dag)
        self._transformer_inputs_to_check = []

        for data_type_index, (data_parent, data_type) in enumerate(data_parent_transformer_and_data_type):
            corrupted_predictions, corruption_diff, corruption, new_score_nodes = self._add_corruption_computation_ml(
                dag, data_parent, data_type, data_type_index, new_dag, score_operators)

            conditional_corruption_significant_node = self._get_corruption_significant_conditional_node(data_type_index,
                                                                                                        new_dag,
                                                                                                        new_score_nodes,
                                                                                                        score_operators)
            self._add_fix_computation_ml(conditional_corruption_significant_node, corrupted_predictions,
                                         corruption_diff, corruption, dag, data_parent, data_type,
                                         data_type_index, new_dag, score_operators)
            # End evaluate
        return new_dag

    def get_llm_rag_dag(self, dag):
        new_dag = dag.copy()
        assert_standard_llm_shape(dag, "Data Errors")

        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)

        self.score_operator_count = len(score_operators)
        self._transformer_inputs_to_check = [DataType.TEXT.value]

        add_orig_score_extraction_nodes(singleton, new_dag, score_operators)

        corrupted_predictions, corruption_diff, corruption, new_score_nodes = self._add_corruption_computation_llm(
            new_dag, predict_operators, rag_join_operators, score_operators, test_data_operators)

        conditional_corruption_significant_node = self._get_corruption_significant_conditional_node(0,
                                                                                                    new_dag,
                                                                                                    new_score_nodes,
                                                                                                    score_operators)
        self._add_fix_computation_llm(conditional_corruption_significant_node,
                                      corrupted_predictions, corruption_diff,
                                      corruption, new_dag, predict_operators, rag_join_operators,
                                      score_operators)
        # End evaluate
        return new_dag

    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        orig_result = []
        for score_index in range(self.score_operator_count):
            orig_result.append(extracted_plan_results[f"orig-{score_index}"])
        report = ""
        corrupted_data_types = []
        data_types_w_repairs = []
        corruption_score_decreases = []
        fix_score_increases = []
        for transformer_index, data_type_name in enumerate(self._transformer_inputs_to_check):
            report += self._add_data_type_results_to_report(corrupted_data_types, corruption_score_decreases,
                                                            data_type_name, data_types_w_repairs, extracted_plan_results,
                                                            fix_score_increases, orig_result, transformer_index)
        if len(corrupted_data_types) > 0:
            report += (f"\n\nOverall, your pipeline does not seem very robust to the corruptions "
                       f"{corrupted_data_types} that were tried (performance drops as extreme as "
                       f"{min(corruption_score_decreases)})!")
            if len(data_types_w_repairs) > 0:
                report += (f" However, for the cases {data_types_w_repairs}, data errors already found "
                           f"a potential way to address them (that improve the performance on corrupted data by up to "
                           f"{max(fix_score_increases)})! (However, you might want to do more detailed experiments "
                           f"yourself, but the suggestions by Data Errors might be a good starting point).")
            else:
                report += (" While Data Errors was not able to find a promising way to make your pipeline more "
                           "robust, you might want to investigate this issue yourself.")
        return report

    def _add_data_type_results_to_report(self, corrupted_data_types, corruption_score_decreases, data_type_name,
                                         data_types_w_repairs, extracted_plan_results, fix_score_increases, orig_result,
                                         transformer_index):
        report = f"Issue {transformer_index}: {data_type_name}\n-\n"
        if extracted_plan_results[f"data-errors-corruption-made-changes-{transformer_index}"] is False:
            report += "The corruption function did not make any changes."
        else:
            corruption_diff_df = extracted_plan_results[
                f"data-errors-corruption-diff-{transformer_index}"]
            corruption_diff_df_sample = get_top_n_df_rows(corruption_diff_df, 20)
            corrupt_result = []
            for score_index in range(self.score_operator_count):
                corrupt_result.append(
                    extracted_plan_results[f"data-errors-corrupt-{transformer_index}-{score_index}"])
            max_score_decrease = get_relative_score_change(max_not_min=False, *orig_result, *corrupt_result)
            corruption_score_decreases.append(max_score_decrease)
            report += (
                f"The original result was {orig_result}. After corrupting {self._corruption_fraction} of rows, "
                f"the pipeline metric was {corrupt_result} (a relative change of {max_score_decrease} in the "
                f"most extreme scenario).\n")
            if max_score_decrease <= self._corruption_significant_relative_threshold:
                report += "This indicates robustness problems you might want to take a look at!\n"
                corrupted_data_types.append(data_type_name)
            else:
                report += ("This shows that your pipeline is relatively robust agaisnt the tested data quality "
                           "problems. However, this doesn't mean that it is robust against other data quality "
                           "problems!\n")
            report += f"A sample of the corrupted rows: \n{str(corruption_diff_df_sample)}\n"

            if extracted_plan_results[f"data-errors-corruption-significant-{transformer_index}"] is True:
                corruption_diff_fix_df = extracted_plan_results[
                    f"data-errors-corruption-diff-fix-{transformer_index}"]
                corruption_diff_fix_df_sample = get_top_n_df_rows(corruption_diff_fix_df, 20)
                score_after_fixing = []
                for score_index in range(self.score_operator_count):
                    score_after_fixing.append(
                        extracted_plan_results[f"data-errors-corrupt-fix-{transformer_index}-{score_index}"])
            else:
                report += (
                    "Fortunately, corruption function was not able to significantly affect the performance beyond "
                    "the configured acceptable threshold.\n")
            if (extracted_plan_results[f"data-errors-corruption-significant-{transformer_index}"] is True and
                    extracted_plan_results[
                        f"data-errors-corruption-diff-fix-not-empty-{transformer_index}"] is True):
                max_score_increase = get_relative_score_change(*corrupt_result, *score_after_fixing)
                report += (f"After adding a fix method, the pipeline metric was "
                           f"{score_after_fixing} (a relative change of {max_score_increase}).\n")
                if max_score_increase > 1.:
                    report += ("This shows that the repair strategy Data Errors tried could help with making your "
                               "pipeline more robust (although other repair strategies Data Errors did not try "
                               "might be even better).\n")
                    data_types_w_repairs.append(data_type_name)
                    fix_score_increases.append(max_score_increase)
                else:
                    report += ("This shows that the repair strategy Data Errors tried could not help with making "
                               "your pipeline more robust. However, you might still want to fix the robustness "
                               "problems Data Errors found.\n")
                report += f"A sample of the fixed rows:\n{str(corruption_diff_fix_df_sample)}\n"
            elif extracted_plan_results[f"data-errors-corruption-significant-{transformer_index}"] is True:
                report += "Unfortunately, the fix method was not able to automatically address the corrupted rows."
        report += "\n"
        return report

    def _add_fix_computation_ml(self, conditional_corruption_significant_node, corrupted_predictions_node,
                                corruption_diff_node, corruption_node, dag, data_parent, data_type, data_type_index,
                                new_dag, score_operators):
        # pylint: disable=too-many-arguments
        # This is important so the non-text-based fix functions can also see the clean data if necessary
        fix_input_node = corruption_node
        processing_func = partial(DataErrorRobustness.fix_data, data_type=data_type)
        new_fix_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation("Data Errors", None),
                               OperatorContext(OperatorType.ESTIMATOR, None),
                               DagNodeDetails(
                                   f"Fix {self._corruption_fraction} of {data_type.value} values", None),
                               None,
                               processing_func)
        new_dag.add_edge(fix_input_node, new_fix_node, arg_index=0)
        new_dag.add_edge(corruption_diff_node, new_fix_node, arg_index=1)
        new_dag.add_edge(conditional_corruption_significant_node, new_fix_node, arg_index=2)
        new_fix_with_corruption_change_filter_node = get_diff_filter_node(singleton, "Data Errors")
        new_dag.add_edge(new_fix_node, new_fix_with_corruption_change_filter_node, arg_index=0)
        new_dag.add_edge(corruption_diff_node, new_fix_with_corruption_change_filter_node, arg_index=1)
        new_dag.add_edge(conditional_corruption_significant_node, new_fix_with_corruption_change_filter_node,
                         arg_index=2)
        fix_node_to_extract = new_fix_with_corruption_change_filter_node
        extraction_node = get_intermediate_extraction_node(singleton, new_fix_node,
                                                           f"data-errors-corruption-diff-fix-{data_type_index}")
        new_dag.add_edge(fix_node_to_extract, extraction_node, arg_index=0)
        new_fix_diff_mask_node = DagNode(singleton.get_next_op_id(),
                                         BasicCodeLocation("Data Errors", None),
                                         OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                         DagNodeDetails(
                                             "Compute change mask from fixing", None),
                                         None,
                                         fix_data_diff_detection_mask_only)
        new_dag.add_edge(fix_input_node, new_fix_diff_mask_node, arg_index=0)
        new_dag.add_edge(new_fix_node, new_fix_diff_mask_node, arg_index=1)
        new_fix_diff_indices_node = DagNode(singleton.get_next_op_id(),
                                            BasicCodeLocation("Data Errors", None),
                                            OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                            DagNodeDetails(
                                                "Compute changed indices from fixing", None),
                                            None,
                                            fix_data_mask_to_indices)
        new_dag.add_edge(new_fix_diff_mask_node, new_fix_diff_indices_node, arg_index=0)
        condition_fix_function = lambda np_array: len(np_array) != 0
        conditional_fixes_changed_something_node = get_conditional_stop_node(
            singleton, condition_fix_function, f"data-errors-corruption-diff-fix-not-empty-{data_type_index}",
            "Check if fix function made changes", new_fix_diff_indices_node)
        new_dag.add_edge(new_fix_diff_indices_node, conditional_fixes_changed_something_node, arg_index=0)
        new_fix_diff_filter_node = get_diff_filter_node(singleton, "Data Errors")
        new_dag.add_edge(new_fix_node, new_fix_diff_filter_node, arg_index=0)
        new_dag.add_edge(new_fix_diff_indices_node, new_fix_diff_filter_node, arg_index=1)
        new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_diff_filter_node, arg_index=2)
        # Evaluate with fixed data
        _, new_nodes = duplicate_descendants(
            dag, new_dag, data_parent, new_fix_diff_filter_node, singleton)
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
                        new_concat_parent_filter_node = get_diff_filter_node(singleton, "Data Errors")
                        new_dag.add_edge(concat_parent, new_concat_parent_filter_node, arg_index=0)
                        new_dag.add_edge(new_fix_diff_indices_node, new_concat_parent_filter_node, arg_index=1)
                        new_dag.add_edge(new_concat_parent_filter_node, concat, **edge_data)
        test_predict = [node for node in new_nodes
                        if node.operator_info.operator == OperatorType.PREDICT][0]
        prediction_filter_index_node = new_fix_diff_indices_node
        new_fix_predict_diff_update_node = merge_prediction_diff_with_old_predictions(singleton, "Data Errors")
        new_dag.add_edge(corrupted_predictions_node, new_fix_predict_diff_update_node, arg_index=0)
        new_dag.add_edge(test_predict, new_fix_predict_diff_update_node, arg_index=1)
        new_dag.add_edge(prediction_filter_index_node, new_fix_predict_diff_update_node, arg_index=2)
        new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_predict_diff_update_node, arg_index=3)
        add_new_score_and_score_extraction_nodes(singleton, new_dag, new_fix_predict_diff_update_node,
                                                 score_operators, f"data-errors-corrupt-fix-{data_type_index}")

    def _get_corruption_significant_conditional_node(self, data_type_index, new_dag, new_score_nodes, score_operators):
        condition_processing_func = partial(DataErrorRobustness.condition_corruption_significant_function,
                                            self._corruption_significant_relative_threshold)
        conditional_corruption_significant_node = get_conditional_stop_node(
            singleton, condition_processing_func, f"data-errors-corruption-significant-{data_type_index}",
            "Check if fix function made changes", new_score_nodes[0])
        for score_index, score_operator in enumerate(score_operators):
            new_dag.add_edge(score_operator, conditional_corruption_significant_node,
                             arg_index=score_index)
        for score_index, score_operator in enumerate(new_score_nodes):
            new_dag.add_edge(score_operator, conditional_corruption_significant_node,
                             arg_index=score_index + self.score_operator_count)
        return conditional_corruption_significant_node

    def _add_corruption_computation_ml(self, dag, data_parent, data_type, data_type_index, new_dag, score_operators):
        self._transformer_inputs_to_check.append(data_type.value)
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
        new_corruption_diff_node = get_changed_indices_node(singleton, "Data Errors")
        new_dag.add_edge(data_parent, new_corruption_diff_node, arg_index=0)
        new_dag.add_edge(new_corruption_node, new_corruption_diff_node, arg_index=1)
        condition_corrupt_function = lambda np_array: len(np_array) != 0
        conditional_corruption_made_changes_node = get_conditional_stop_node(
            singleton, condition_corrupt_function, f"data-errors-corruption-made-changes-{data_type_index}",
            "Check if corrupt function made changes", new_corruption_diff_node)
        new_dag.add_edge(new_corruption_diff_node, conditional_corruption_made_changes_node, arg_index=1)
        new_corruption_diff_filter_node = get_diff_filter_node(singleton, "Data Errors")
        new_dag.add_edge(new_corruption_node, new_corruption_diff_filter_node, arg_index=0)
        new_dag.add_edge(new_corruption_diff_node, new_corruption_diff_filter_node, arg_index=1)
        new_dag.add_edge(conditional_corruption_made_changes_node, new_corruption_diff_filter_node, arg_index=2)
        extraction_node = get_intermediate_extraction_node(singleton, new_corruption_diff_filter_node,
                                                           f"data-errors-corruption-diff-{data_type_index}")
        new_dag.add_edge(new_corruption_diff_filter_node, extraction_node, arg_index=0)
        # Evaluate with corrupted data
        old_copied_nodes, new_nodes = duplicate_descendants(
            dag, new_dag, data_parent, new_corruption_diff_filter_node, singleton)
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
                        new_concat_parent_filter_node = get_diff_filter_node(singleton, "Data Errors")
                        new_dag.add_edge(concat_parent, new_concat_parent_filter_node, arg_index=0)
                        new_dag.add_edge(new_corruption_diff_node, new_concat_parent_filter_node, arg_index=1)
                        new_dag.add_edge(conditional_corruption_made_changes_node, new_concat_parent_filter_node,
                                         arg_index=2)
                        new_dag.add_edge(new_concat_parent_filter_node, concat, **edge_data)
        test_predict = [node for node in new_nodes
                        if node.operator_info.operator == OperatorType.PREDICT][0]
        old_predict = [node for node in old_copied_nodes
                       if node.operator_info.operator == OperatorType.PREDICT][0]
        new_corrupt_predict_diff_update_node = merge_prediction_diff_with_old_predictions(singleton, "Data Errors")
        new_dag.add_edge(old_predict, new_corrupt_predict_diff_update_node, arg_index=0)
        new_dag.add_edge(test_predict, new_corrupt_predict_diff_update_node, arg_index=1)
        new_dag.add_edge(new_corruption_diff_node, new_corrupt_predict_diff_update_node, arg_index=2)
        new_dag.add_edge(conditional_corruption_made_changes_node, new_corrupt_predict_diff_update_node,
                         arg_index=3)
        new_score_nodes = add_new_score_and_score_extraction_nodes(singleton, new_dag,
                                                                   new_corrupt_predict_diff_update_node,
                                                                   score_operators,
                                                                   f"data-errors-corrupt-{data_type_index}")
        return new_corrupt_predict_diff_update_node, new_corruption_diff_node, new_corruption_node, new_score_nodes

    def _add_fix_computation_llm(self,
                                 conditional_corruption_significant_node,
                                 corrupted_predictions_node, corruption_diff_node, corruption_node,
                                 new_dag, predict_operators, rag_join_operators, score_operators):
        fix_input_node = corruption_node
        data_type = DataType.TEXT
        processing_func = partial(DataErrorRobustness.fix_data, data_type=data_type)
        new_fix_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation("Data Errors", None),
                               OperatorContext(OperatorType.ESTIMATOR, None),
                               DagNodeDetails(
                                   f"Fix {self._corruption_fraction} of {data_type.value} values", None),
                               None,
                               processing_func)
        new_dag.add_edge(fix_input_node, new_fix_node, arg_index=0)
        new_dag.add_edge(corruption_diff_node, new_fix_node, arg_index=1)
        new_dag.add_edge(conditional_corruption_significant_node, new_fix_node, arg_index=2)
        new_fix_with_corruption_change_filter_node = get_diff_filter_node(singleton, "Data Errors")
        new_dag.add_edge(new_fix_node, new_fix_with_corruption_change_filter_node, arg_index=0)
        new_dag.add_edge(corruption_diff_node, new_fix_with_corruption_change_filter_node, arg_index=1)
        new_dag.add_edge(conditional_corruption_significant_node, new_fix_with_corruption_change_filter_node,
                         arg_index=2)
        fix_node_to_extract = new_fix_with_corruption_change_filter_node
        extraction_node = get_intermediate_extraction_node(singleton, new_fix_node,
                                                           "data-errors-corruption-diff-fix-0")
        new_dag.add_edge(fix_node_to_extract, extraction_node, arg_index=0)
        new_fix_diff_mask_node = DagNode(singleton.get_next_op_id(),
                                         BasicCodeLocation("Data Errors", None),
                                         OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                         DagNodeDetails(
                                             "Compute change mask from fixing", None),
                                         None,
                                         fix_data_diff_detection_mask_only)
        new_dag.add_edge(fix_input_node, new_fix_diff_mask_node, arg_index=0)
        new_dag.add_edge(new_fix_node, new_fix_diff_mask_node, arg_index=1)
        new_fix_diff_indices_node = DagNode(singleton.get_next_op_id(),
                                            BasicCodeLocation("Data Errors", None),
                                            OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                            DagNodeDetails(
                                                "Compute changed indices from fixing", None),
                                            None,
                                            fix_data_mask_to_indices)
        new_dag.add_edge(new_fix_diff_mask_node, new_fix_diff_indices_node, arg_index=0)
        condition_fix_function = lambda np_array: len(np_array) != 0
        conditional_fixes_changed_something_node = get_conditional_stop_node(
            singleton, condition_fix_function, "data-errors-corruption-diff-fix-not-empty-0",
            "Check if fix function made changes", new_fix_diff_indices_node)
        new_dag.add_edge(new_fix_diff_indices_node, conditional_fixes_changed_something_node, arg_index=0)
        new_fix_diff_filter_node = get_diff_filter_node(singleton, "Data Errors")
        new_dag.add_edge(new_fix_node, new_fix_diff_filter_node, arg_index=0)
        new_dag.add_edge(new_fix_diff_indices_node, new_fix_diff_filter_node, arg_index=1)
        new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_diff_filter_node, arg_index=2)
        # Evaluate with fixed data
        # Operator to get the rag join results
        new_rag_join_update_node = DagNode(singleton.get_next_op_id(),
                                           BasicCodeLocation("Data Errors", None),
                                           OperatorContext(OperatorType.RAG_JOIN, None),
                                           DagNodeDetails("RAG join for test set diff", None),
                                           None,
                                           rag_join_update)
        new_dag.add_edge(rag_join_operators[0], new_rag_join_update_node, arg_index=0)
        new_dag.add_edge(new_fix_diff_filter_node, new_rag_join_update_node, arg_index=1)
        # Duplicate predict operator and connect with rag join result update and prediction update
        test_predict = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_rag_join_update_node, test_predict, arg_index=0)
        prediction_filter_index_node = new_fix_diff_indices_node
        new_fix_predict_diff_update_node = merge_prediction_diff_with_old_predictions(singleton, "Data Errors")
        new_dag.add_edge(corrupted_predictions_node, new_fix_predict_diff_update_node, arg_index=0)
        new_dag.add_edge(test_predict, new_fix_predict_diff_update_node, arg_index=1)
        new_dag.add_edge(prediction_filter_index_node, new_fix_predict_diff_update_node, arg_index=2)
        new_dag.add_edge(conditional_fixes_changed_something_node, new_fix_predict_diff_update_node, arg_index=3)
        add_new_score_and_score_extraction_nodes(singleton, new_dag, new_fix_predict_diff_update_node,
                                                 score_operators, "data-errors-corrupt-fix-0")

    def _add_corruption_computation_llm(self, new_dag, predict_operators, rag_join_operators, score_operators,
                                        test_data_operators):
        data_parent = test_data_operators[0]
        data_type = DataType.TEXT
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
        new_corruption_diff_node = get_changed_indices_node(singleton, "Data Errors")
        new_dag.add_edge(data_parent, new_corruption_diff_node, arg_index=0)
        new_dag.add_edge(new_corruption_node, new_corruption_diff_node, arg_index=1)
        condition_corrupt_function = lambda np_array: len(np_array) != 0
        conditional_corruption_made_changes_node = get_conditional_stop_node(
            singleton, condition_corrupt_function, "data-errors-corruption-made-changes-0",
            "Check if corrupt function made changes", new_corruption_diff_node)
        new_dag.add_edge(new_corruption_diff_node, conditional_corruption_made_changes_node, arg_index=1)
        new_corruption_diff_filter_node = get_diff_filter_node(singleton, "Data Errors")
        new_dag.add_edge(new_corruption_node, new_corruption_diff_filter_node, arg_index=0)
        new_dag.add_edge(new_corruption_diff_node, new_corruption_diff_filter_node, arg_index=1)
        new_dag.add_edge(conditional_corruption_made_changes_node, new_corruption_diff_filter_node, arg_index=2)
        extraction_node = get_intermediate_extraction_node(singleton, new_corruption_diff_filter_node,
                                                           "data-errors-corruption-diff-0")
        new_dag.add_edge(new_corruption_diff_filter_node, extraction_node, arg_index=0)
        # Evaluate with corrupted data
        # Operator to get the rag join results
        new_rag_join_update_node = DagNode(singleton.get_next_op_id(),
                                           BasicCodeLocation("Data Errors", None),
                                           OperatorContext(OperatorType.RAG_JOIN, None),
                                           DagNodeDetails("RAG join for test set diff", None),
                                           None,
                                           rag_join_update)
        new_dag.add_edge(rag_join_operators[0], new_rag_join_update_node, arg_index=0)
        new_dag.add_edge(new_corruption_diff_filter_node, new_rag_join_update_node, arg_index=1)
        # Duplicate predict operator and connect with rag join result update and prediction update
        test_predict = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_rag_join_update_node, test_predict, arg_index=0)
        old_predict = predict_operators[0]
        new_corrupt_predict_diff_update_node = merge_prediction_diff_with_old_predictions(singleton, "Data Errors")
        new_dag.add_edge(old_predict, new_corrupt_predict_diff_update_node, arg_index=0)
        new_dag.add_edge(test_predict, new_corrupt_predict_diff_update_node, arg_index=1)
        new_dag.add_edge(new_corruption_diff_node, new_corrupt_predict_diff_update_node, arg_index=2)
        new_dag.add_edge(conditional_corruption_made_changes_node, new_corrupt_predict_diff_update_node,
                         arg_index=3)
        new_score_nodes = add_new_score_and_score_extraction_nodes(singleton, new_dag,
                                                                   new_corrupt_predict_diff_update_node,
                                                                   score_operators, "data-errors-corrupt-0")
        return new_corrupt_predict_diff_update_node, new_corruption_diff_node, new_corruption_node, new_score_nodes

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
    def fix_data(input_df, only_fix_indices=None, data_type=None):
        fixed_corrupted = input_df.copy()
        if data_type == DataType.TEXT:
            was_series = False
            was_numpy = False
            series_column_name = None
            if isinstance(fixed_corrupted, pandas.Series):
                series_column_name = fixed_corrupted.name
                if series_column_name is None:
                    series_column_name = "column"
                fixed_corrupted = pandas.DataFrame({series_column_name: fixed_corrupted})
                was_series = True
            elif isinstance(fixed_corrupted, (numpy.ndarray, list)):
                fixed_corrupted = pandas.DataFrame({"column": fixed_corrupted})
                was_numpy = True
            for column_index, column in enumerate(fixed_corrupted.columns):
                if fixed_corrupted[column].dtype == object:
                    typo_fixer = get_typo_fixer(column)
                    fixed_corrupted.iloc[only_fix_indices, [column_index]] = typo_fixer.fit_transform(
                        fixed_corrupted.iloc[only_fix_indices, [column_index]])
            if was_series is True:
                fixed_corrupted = fixed_corrupted[series_column_name]
            elif was_numpy is True:
                fixed_corrupted = fixed_corrupted["column"].to_numpy()
        elif data_type == DataType.CAT:
            fixed_corrupted = input_df.reset_index(drop=True)
            clean = fixed_corrupted.drop(only_fix_indices, axis=0)
            for column in fixed_corrupted.columns:
                imputer = SimpleImputer(strategy="most_frequent", copy=True, missing_values="0")
                imputer.fit(clean[[column]])
                fixed_corrupted.iloc[only_fix_indices, [column]] = imputer.transform(
                    fixed_corrupted.iloc[only_fix_indices, [column]])

        elif data_type == DataType.NUM:
            fixed_corrupted = input_df.reset_index(drop=True)
            clean = fixed_corrupted.drop(only_fix_indices, axis=0)
            for column_index, column in enumerate(fixed_corrupted.columns):
                is_int = fixed_corrupted[column].dtype == int
                # This is if we want to just apply fit_transform on all data instead of fixing only the corrupted data
                #  with a detection and cleaning method fitted on the clean data
                # fixed_corrupted = OutlierCleaner.fit_transform_all(fixed_corrupted, detection_strategy='IQR',
                #                                                    repair_strategy='mean', column=column)
                _, fitted_detector = detect_outlier_interquartile_range(clean[[column]])
                imputer = SimpleImputer(strategy='mean', copy=True)
                imputer.fit(clean[[column]])
                outlier_indicator, _ = detect_outlier_interquartile_range(
                    fixed_corrupted.iloc[only_fix_indices, [column_index]], fitted_detector=fitted_detector)
                detector_mask = fixed_corrupted.iloc[only_fix_indices, column_index].apply(outlier_indicator).to_numpy()
                if numpy.any(detector_mask):
                    fixed_corrupted.iloc[only_fix_indices[detector_mask], [column_index]] = numpy.nan
                    fixed_corrupted.iloc[only_fix_indices[detector_mask], [column_index]] = imputer.transform(
                        fixed_corrupted.iloc[only_fix_indices[detector_mask], [column_index]])
                if is_int:
                    fixed_corrupted[column] = fixed_corrupted[column].astype(int)
            # fixed_corrupted = MinMaxScaler(feature_range=(0, 10)).fit_transform(input_df)
        else:
            raise NotImplementedError(f"TODO: Add support for datatype {DataType.value}!")

        fixed_corrupted = wrap_in_mlinspect_array_if_necessary(fixed_corrupted)
        fixed_corrupted._mlinspect_provenance = None

        return fixed_corrupted

    @staticmethod
    def condition_corruption_significant_function(corruption_significant_relative_threshold, *scores):
        # This function compares all scores of the original pipeline and the corrupted pipeline
        # So the number of scores in both pipeline variants should be equal
        assert len(scores) % 2 == 0
        number_of_scores_each = int(len(scores) / 2)
        scores_different_enough = any(
            # TODO: More sophisticated handling of FairLearn MetricFrames
            (isinstance(scores[score_index], float) and
             scores[score_index] * corruption_significant_relative_threshold >=
             scores[score_index + number_of_scores_each]) or
            (isinstance(scores[score_index], MetricFrame) and
             scores[score_index].overall * corruption_significant_relative_threshold >=
             scores[score_index + number_of_scores_each].overall)
            for score_index in range(number_of_scores_each))
        return scores_different_enough
