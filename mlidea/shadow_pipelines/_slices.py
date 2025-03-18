import dataclasses
from collections import defaultdict
from enum import Enum
from functools import partial
from inspect import cleandoc

import networkx
import numpy
import pandas
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sliceline import Slicefinder

from mlidea import OperatorType, DagNode, OperatorContext, DagNodeDetails, FunctionInfo
from mlidea.analysis._analysis_utils import find_nodes_by_type
from mlidea.analysis._cleaning_methods import detect_outlier_interquartile_range
from mlidea.execution._pipeline_executor import singleton
from mlidea.instrumentation._operator_call_info import OperatorCallInfo
from mlidea.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline
from mlidea.shadow_pipelines._utils import get_intermediate_extraction_node, copy_node_with_new_id, \
    duplicate_descendants_and_filter_concat_inputs, get_typo_fixer, get_conditional_stop_node, \
    filter_estimator_transformer_edges, \
    get_transformer_parents_with_data_types, \
    DataType, get_translate_transformer, get_relative_score_change, add_orig_score_extraction_nodes, \
    get_diff_filter_node, get_changed_indices_node, merge_prediction_diff_with_old_predictions, \
    add_new_score_and_score_extraction_nodes, assert_standard_llm_shape, assert_standard_ml_shape, \
    prov_join_node_with_data_sources, df_or_array_non_empty, df_or_array_non_empty_func_info, add_parent_node_edges, \
    get_rag_join_update_node, get_basic_code_location_for_current_line


@dataclasses.dataclass
class PotentialSuggestion:
    improves_score: bool
    suggestion: str or None
    suggestion_metric_results: any
    suggestion_max_score_improvement: float or None
    suggestion_df_before: any or None
    suggestion_df_after: any or None
    source_code_to_integrate: str or None


@dataclasses.dataclass
class ScreenedIssue:
    description: str
    issue_found: bool
    problematic_slice: any or None
    suggestion_found: bool
    issue_suggestions: list[PotentialSuggestion]


@dataclasses.dataclass
class FairnessSlicesReport:
    """
    The class the PipelineExecutor returns when doing runtime estimation only
    """
    orig_metric_results: any
    screened_issues: list[ScreenedIssue]
    summary: str


class FixType(Enum):
    """
    The different data types that we base our error detection techniques on
    """
    NUM = "Num: IQR + Mean Impute"
    CAT = "Cat: Isolation Forest + Simple Impute"
    TEXT_TRANSLATE = "Text: Translate"
    TEXT_SPELLCHECK = "Text: Spellcheck"


DATA_TYPE_TO_FIX_STRATEGY = {
    DataType.TEXT: [FixType.TEXT_TRANSLATE, FixType.TEXT_SPELLCHECK],
    DataType.NUM: [FixType.NUM],
    DataType.CAT: [FixType.CAT]
}

FIX_STRATEGY_TO_CODE = {
    FixType.NUM.value: cleandoc("""
        def detect_outlier_interquartile_range(x, k=1.5, fitted_detector=None):
            if fitted_detector is None:
                q25, q75 = numpy.percentile(x, 25), numpy.percentile(x, 75)
                iqr = q75 - q25
                cut_off = iqr * k
                lower, upper = q25 - cut_off, q75 + cut_off
            else:
                lower, upper = fitted_detector
            return lambda y: (y > upper) | (y < lower), (lower, upper)
        
        is_int = df[column_to_clean].dtype == int
        
        _, fitted_detector = detect_outlier_interquartile_range(df[[column_to_clean]], k=0.25)

        imputer = SimpleImputer(strategy='mean', copy=True)
        imputer.fit(df[[column_to_clean]])
        
        outlier_indicator, _ = detect_outlier_interquartile_range(df[[column_to_clean]], fitted_detector=fitted_detector)
        detector_mask = df[[column_to_clean]].apply(outlier_indicator).to_numpy()
        if numpy.any(detector_mask):
            df.iloc[detector_mask, [column_to_clean]] = numpy.nan
            df.iloc[detector_mask, [column_to_clean]] = imputer.transform(df.iloc[detector_mask, [column_to_clean]])
        if is_int:
            df[column_to_clean] = df[column_to_clean].astype(int)
    """),
    FixType.CAT.value: cleandoc("""
        one_hot = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df)
        isolation_forest = IsolationForest(contamination=0.5, random_state=42)
        isolation_forest.fit(one_hot)
        outlier_indicator = isolation_forest.predict(one_hot) == -1
        
        df[outlier_indicator, :] = -1
        imputer = SimpleImputer(strategy="most_frequent", copy=True, missing_values=-1)
        df[columns_to_clean] = imputer.fit_transform(df[columns_to_clean).ravel()
    """),
    FixType.TEXT_TRANSLATE.value: cleandoc("""
        import nest_asyncio
        nest_asyncio.apply()
        from googletrans import Translator
        from functools import partial
        from sklearn.pipeline import Pipeline
        
        translator = Translator()

        def translate(series):
            if isinstance(series, pd.Series):
                series = [result.text for result in asyncio.run(translator.translate(series.to_list()))]
            else:
                series = [result.text for result in asyncio.run(translator.translate(series))]
            return series
    
        translate_transformer = FunctionTransformer(translate)
        # If the pipeline is a ML pipelien that uses a ColumnTransformer:
        # pipeline_transformer = Pipeline([
        #     ('translate', translate_transformer),
        #     ('...previous transformer...', ...previous transformer...)
        # ])
        # The previous transformer can now be replaced with the new pipeline_transformer
    """),
    FixType.TEXT_SPELLCHECK.value: cleandoc("""
        from autocorrect import Speller
        from functools import partial
        
        spell = Speller()

        def fix_typos(series):
            series = series.map(spell)
            return series
    
        typo_fixer = FunctionTransformer(spell)
        # If the pipeline is a ML pipeline that uses a ColumnTransformer:
        # pipeline_transformer = Pipeline([
        #     ('translate', typo_fixer),
        #     ('...previous transformer...', ...previous transformer...)
        # ])
        # The previous transformer can now be replaced with the new pipeline_transformer
    """)
}


class FairnessSlices(ShadowPipeline):
    """
    The Data Error Robustness Shadow Pipeline
    """

    def check_rebuilding_necessary(self, extracted_plan_results: dict[str, any]) -> any:
        return False

    def __init__(self, additional_column_names=None, database_path=".function_transformer_cache.db",
                 slice_finder_alpha=0.95):
        if additional_column_names is None:
            additional_column_names = []
        self._additional_column_names = additional_column_names
        self._shadow_pipeline_id = (tuple(additional_column_names), database_path, slice_finder_alpha)
        self.score_operator_count = 0
        self.sensitive_column_count = 0
        self.database_path = database_path
        self.slice_finder_alpha = slice_finder_alpha
        self.fix_strategy_names = []
        self.sensitive_columns = []

    @property
    def shadow_pipeline_id(self):
        return self._shadow_pipeline_id

    @property
    def simple_name(self):
        return "slices"

    @staticmethod
    def is_column_sensitive(column_name, additional_column_names):
        # TODO: There are many different ways to do this to explore in the future, e.g., using LLMs
        return column_name in {"race", "gender", "age", "lang", "country", "sex"}.union(additional_column_names)

    def generate_shadow_pipeline_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        data_sources_with_sensitive_columns = FairnessSlices.get_data_sources_to_sensitive_columns(
            dag, self._additional_column_names)
        self.sensitive_column_count = len(data_sources_with_sensitive_columns)
        self.fix_strategy_names = []
        self.sensitive_columns = []
        self.score_operator_count = 0

        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)

        for column_names in data_sources_with_sensitive_columns.values():
            self.sensitive_columns.extend(column_names)

        if len(rag_join_operators) == 0:
            new_dag = self.get_traditional_ml_dag(dag, data_sources_with_sensitive_columns)
        else:
            new_dag = self.get_llm_rag_dag(dag, data_sources_with_sensitive_columns)

        return new_dag

    def get_traditional_ml_dag(self, dag, data_sources_with_sensitive_columns):
        new_dag = dag.copy()
        assert_standard_ml_shape(dag, "Fairness Slices")

        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)
        add_orig_score_extraction_nodes(singleton, new_dag, score_operators)
        self.score_operator_count = len(score_operators)

        if len(data_sources_with_sensitive_columns) == 0:
            return new_dag

        new_slice_finder_node = self._add_slice_finder_computation(data_sources_with_sensitive_columns, new_dag,
                                                                   predict_operators, test_data_operators,
                                                                   test_labels_operators)

        conditional_slices_found_node = FairnessSlices._get_slice_found_conditional_node(new_dag, new_slice_finder_node)

        self._add_fix_computation_ml(conditional_slices_found_node, dag, new_dag, new_slice_finder_node,
                                     predict_operators, score_operators)

        return new_dag

    def get_llm_rag_dag(self, dag, data_sources_with_sensitive_columns):
        new_dag = dag.copy()
        assert_standard_llm_shape(dag, "Fairness Slices")

        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)
        add_orig_score_extraction_nodes(singleton, new_dag, score_operators)
        self.score_operator_count = len(score_operators)

        if len(data_sources_with_sensitive_columns) == 0:
            return new_dag

        new_slice_finder_node = self._add_slice_finder_computation(data_sources_with_sensitive_columns, new_dag,
                                                                   predict_operators, test_data_operators,
                                                                   test_labels_operators)

        conditional_slices_found_node = self._get_slice_found_conditional_node(new_dag, new_slice_finder_node)

        self._add_fix_computation_llm(conditional_slices_found_node, new_dag, new_slice_finder_node, predict_operators,
                                      rag_join_operators, score_operators, test_data_operators)

        return new_dag

    def _add_fix_computation_ml(self, conditional_slices_found_node, dag, new_dag, new_slice_finder_node,
                                predict_operators, score_operators):
        slice_finder_indices_node = FairnessSlices._get_slice_finder_indices_node(
            new_dag, [new_slice_finder_node, conditional_slices_found_node])
        data_parent_transformer_and_data_type = get_transformer_parents_with_data_types(dag)
        fix_strategy_index = 0
        for data_parent, data_type in data_parent_transformer_and_data_type:
            for fix_strategy in DATA_TYPE_TO_FIX_STRATEGY[data_type]:
                new_fix_diff_node, new_fix_node = self.fix_function_computation_node(data_parent, fix_strategy, new_dag,
                                                                                     slice_finder_indices_node)

                conditional_fix_function_made_changes_node = FairnessSlices.get_fix_made_changes_conditional_node(
                    fix_strategy_index, new_dag, new_fix_diff_node)

                FairnessSlices._add_fix_evaluation_computation_ml(conditional_fix_function_made_changes_node, dag,
                                                                  data_parent,
                                                                  fix_strategy_index, new_dag, new_fix_diff_node,
                                                                  new_fix_node, predict_operators, score_operators)
                fix_strategy_index += 1

    @staticmethod
    def _get_slice_finder_indices_node(new_dag, parents):
        operator_context = OperatorContext(OperatorType.GROUP_BY_AGG,
                                           FunctionInfo('mlidea.shadow_pipelines._slices.FairnessSlices',
                                                        'extract_slice_finder_result'),
                                           {})
        operator_call_info = OperatorCallInfo(operator_context, parents)
        slice_finder_indices_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                            get_basic_code_location_for_current_line(),
                                            operator_context,
                                            DagNodeDetails("Compute slice finder indexes", None),
                                            None,
                                            FairnessSlices.extract_slice_finder_result)
        add_parent_node_edges(singleton, new_dag, slice_finder_indices_node, parents)
        return slice_finder_indices_node

    @staticmethod
    def extract_slice_finder_result(slice_finder_result):
        return slice_finder_result[1]

    def _add_fix_computation_llm(self, conditional_slices_found_node, new_dag, new_slice_finder_node, predict_operators,
                                 rag_join_operators, score_operators, test_data_operators):
        slice_finder_indices_node = FairnessSlices._get_slice_finder_indices_node(
            new_dag, [new_slice_finder_node, conditional_slices_found_node])
        data_parent = test_data_operators[0]
        data_type = DataType.TEXT
        for fix_strategy_index, fix_strategy in enumerate(DATA_TYPE_TO_FIX_STRATEGY[data_type]):
            new_fix_diff_node, new_fix_node = self.fix_function_computation_node(data_parent, fix_strategy, new_dag,
                                                                                 slice_finder_indices_node)

            conditional_fix_function_made_changes_node = FairnessSlices.get_fix_made_changes_conditional_node(
                fix_strategy_index, new_dag, new_fix_diff_node)

            FairnessSlices._add_fix_evaluation_computation_llm(conditional_fix_function_made_changes_node, data_parent,
                                                               fix_strategy_index, new_dag, new_fix_diff_node,
                                                               new_fix_node,
                                                               predict_operators, rag_join_operators, score_operators)

    @staticmethod
    def _add_fix_evaluation_computation_llm(conditional_fix_function_made_changes_node, data_parent,
                                            fix_strategy_index, new_dag, new_fix_diff_node, new_fix_node,
                                            predict_operators,
                                            rag_join_operators, score_operators):
        new_unmodified_fix_filter_node = get_diff_filter_node(singleton, new_dag, [data_parent, new_fix_diff_node])
        _ = get_intermediate_extraction_node(singleton, new_dag, [new_unmodified_fix_filter_node],
                                             f"fairness-slices-data-to-fix-{fix_strategy_index}")
        new_fix_diff_filter_node = get_diff_filter_node(singleton, new_dag, [new_fix_node, new_fix_diff_node,
                                                                             conditional_fix_function_made_changes_node])
        _ = get_intermediate_extraction_node(singleton, new_dag, [new_fix_diff_filter_node],
                                             f"fairness-slice-fixing-diff-{fix_strategy_index}")
        # Evaluate with updated data
        # Operator to get the rag join results
        new_rag_join_update_node = get_rag_join_update_node(
            singleton, new_dag, [rag_join_operators[0], new_fix_diff_filter_node])
        # Duplicate predict operator and connect with rag join result update and prediction update
        test_predict = copy_node_with_new_id(singleton, new_dag, predict_operators[0], [new_rag_join_update_node])
        new_fix_predict_diff_update_node = merge_prediction_diff_with_old_predictions(singleton, new_dag,
                                                                                      [predict_operators[0],
                                                                                       test_predict, new_fix_diff_node,
                                                                                       conditional_fix_function_made_changes_node])
        add_new_score_and_score_extraction_nodes(singleton, new_dag, new_fix_predict_diff_update_node,
                                                 score_operators, f"fairness-slice-fixing-{fix_strategy_index}")

    @staticmethod
    def get_fix_made_changes_conditional_node(fix_strategy_index, new_dag, new_fix_diff_node):
        function_info = df_or_array_non_empty_func_info()
        conditional_fix_made_changes_node = get_conditional_stop_node(
            singleton, new_dag, df_or_array_non_empty, function_info,
            f"fairness-slices-fixing-made-changes-{fix_strategy_index}",
            "Check if fixing function made changes", [new_fix_diff_node])
        return conditional_fix_made_changes_node

    @staticmethod
    def _get_slice_found_conditional_node(new_dag, new_slice_finder_node):
        function_info = FunctionInfo("mlidea.shadow_pipelines._slices.FairnessSlices", 'problematic_slice_found_func')
        conditional_fix_made_changes_node = get_conditional_stop_node(
            singleton, new_dag, FairnessSlices.problematic_slice_found_func, function_info,
            "fairness-slices-slice-line-problematic-slice-found",
            "Check if problematic slice was found", [new_slice_finder_node])
        return conditional_fix_made_changes_node

    @staticmethod
    def problematic_slice_found_func(slice_finder_result):
        return slice_finder_result[0] is not None and slice_finder_result[1] is not None

    @staticmethod
    def get_data_sources_to_sensitive_columns(dag, additional_column_names):
        data_sources_to_columns = defaultdict(list)
        data_sources = find_nodes_by_type(dag, OperatorType.DATA_SOURCE)

        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        dag_to_consider = networkx.subgraph_view(dag, filter_edge=filter_estimator_transformer_edges)
        for data_source in data_sources:
            for column_name in data_source.details.columns:
                if (FairnessSlices.is_column_sensitive(column_name, additional_column_names) is True and
                        networkx.has_path(dag_to_consider, source=data_source, target=test_data_operators[0]) is True):
                    data_sources_to_columns[data_source].append(column_name)
        return data_sources_to_columns

    @staticmethod
    def _add_fix_evaluation_computation_ml(conditional_fix_function_made_changes_node, dag, data_parent,
                                           fix_strategy_index, new_dag, new_fix_diff_node, new_fix_node,
                                           predict_operators, score_operators):
        new_unmodified_fix_filter_node = get_diff_filter_node(singleton, new_dag, [data_parent, new_fix_diff_node])
        _ = get_intermediate_extraction_node(singleton, new_dag, [new_unmodified_fix_filter_node],
                                             f"fairness-slices-data-to-fix-{fix_strategy_index}")
        new_fix_diff_filter_node = get_diff_filter_node(singleton, new_dag, [new_fix_node, new_fix_diff_node,
                                                                             conditional_fix_function_made_changes_node])
        _ = get_intermediate_extraction_node(singleton, new_dag, [new_fix_diff_filter_node],
                                             f"fairness-slice-fixing-diff-{fix_strategy_index}")
        # Evaluate with updated data
        new_predict = duplicate_descendants_and_filter_concat_inputs(singleton, dag, new_dag, data_parent,
                                                                     new_fix_diff_filter_node, new_fix_diff_node,
                                                                     conditional_fix_function_made_changes_node)

        parents = [predict_operators[0], new_predict, new_fix_diff_node, conditional_fix_function_made_changes_node]
        new_fix_predict_diff_update_node = merge_prediction_diff_with_old_predictions(singleton, new_dag, parents)
        add_new_score_and_score_extraction_nodes(singleton, new_dag, new_fix_predict_diff_update_node,
                                                 score_operators,
                                                 f"fairness-slice-fixing-{fix_strategy_index}")

    def fix_function_computation_node(self, data_parent, fix_strategy, new_dag, slice_finder_indices_node):
        self.fix_strategy_names.append(fix_strategy.value)
        new_fix_node = self._get_fix_node(fix_strategy, new_dag, [data_parent, slice_finder_indices_node])
        new_fix_diff_node = get_changed_indices_node(singleton, new_dag, [data_parent, new_fix_node])
        return new_fix_diff_node, new_fix_node

    def _get_fix_node(self, fix_strategy, new_dag, parents):
        non_data_kwargs = {'database_path': self.database_path, 'fix_strategy': fix_strategy}
        processing_func = partial(FairnessSlices.fix_data, **non_data_kwargs)
        if fix_strategy in {FixType.TEXT_TRANSLATE, FixType.TEXT_SPELLCHECK}:
            # TODO: For slow text processing functions, we need to be able to have IVM, estimators are not an option
            operator_type = OperatorType.PROJECTION_MODIFY_SUBSET
        else:
            operator_type = OperatorType.TRANSFORMER_MODIFY_SUBSET
        operator_context = OperatorContext(operator_type,
                                           FunctionInfo('mlidea.shadow_pipelines._slices.FairnessSlices',
                                                        'fix_data'),
                                           non_data_kwargs)
        operator_call_info = OperatorCallInfo(operator_context, parents)
        new_fix_node = DagNode(singleton.get_next_op_id(operator_call_info),
                               get_basic_code_location_for_current_line(),
                               operator_context,
                               DagNodeDetails(f"Trying to fix slice: {fix_strategy.value}",
                                              parents[0].details.columns),
                               None,
                               processing_func)
        add_parent_node_edges(singleton, new_dag, new_fix_node, parents)
        return new_fix_node

    def _add_slice_finder_computation(self, data_sources_with_sensitive_columns, new_dag, predict_operators,
                                      test_data_operators, test_labels_operators):
        concat_node = prov_join_node_with_data_sources(singleton, data_sources_with_sensitive_columns, new_dag,
                                                       test_data_operators[0])
        new_slice_finder_node = self._get_slice_finder_node(
            new_dag, [concat_node, test_labels_operators[0], predict_operators[0]])
        _ = get_intermediate_extraction_node(singleton, new_dag, [new_slice_finder_node],
                                             "fairness-slices-slice-line-result")
        return new_slice_finder_node

    def _get_slice_finder_node(self, new_dag, parents):
        non_data_kwargs = {'alpha': self.slice_finder_alpha}
        slice_finder_process_func = partial(FairnessSlices.get_slice_finder_slice_and_indices, **non_data_kwargs)
        operator_context = OperatorContext(OperatorType.GROUP_BY_AGG,
                                           FunctionInfo('mlidea.shadow_pipelines._slices.FairnessSlices',
                                                        'get_slice_finder_slice_and_indices'),
                                           non_data_kwargs)
        operator_call_info = OperatorCallInfo(operator_context, parents)
        new_slice_finder_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                        get_basic_code_location_for_current_line(),
                                        operator_context,
                                        DagNodeDetails("Run Slice Finder", None),
                                        None,
                                        slice_finder_process_func)
        add_parent_node_edges(singleton, new_dag, new_slice_finder_node, parents)
        return new_slice_finder_node

    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        summary = ""
        orig_result = []
        for score_index in range(self.score_operator_count):
            orig_result.append(extracted_plan_results[f"orig-{score_index}"])
        summary += f"The original result was {orig_result}.\n"
        if self.sensitive_column_count == 0:
            summary += "Slice finding could not be applied since no sensitive column could be found!"
            report = FairnessSlicesReport(orig_result, [], summary)
        elif extracted_plan_results["fairness-slices-slice-line-problematic-slice-found"] is False:
            summary += ("No problematic slice could be found by Fairness Slices. However, this does not mean that "
                       "there are no fairness problems, Fairness Slices only could not find any with the given config.")
            report = FairnessSlicesReport(orig_result, [ScreenedIssue("Underperforming slices", False,
                                                                      None, False, [])], summary)
        else:
            slice_line_result = extracted_plan_results["fairness-slices-slice-line-result"]
            column_with_slice_value = []
            for sensitive_column, column_value in zip(self.sensitive_columns, list(slice_line_result[0])):
                column_with_slice_value.append(f"{sensitive_column}={column_value}")
            readable_slice_result = ", ".join(column_with_slice_value)
            readable_slice_result = f"[{readable_slice_result}]"
            summary += f"The problematic slice that was found is {readable_slice_result}.\n"

            promising_fix_strategies = []
            performance_increases = []
            suggestions = []
            for fix_strategy_index, fix_strategy_name in enumerate(self.fix_strategy_names):
                summary_strategy, suggestion = self.generate_report_for_fix_strategy(
                    extracted_plan_results, fix_strategy_index, fix_strategy_name, orig_result, performance_increases,
                    promising_fix_strategies)
                summary += summary_strategy
                suggestions.append(suggestion)
            fix_found = len(promising_fix_strategies) != 0
            if fix_found:
                summary += (f"\n\nFairness Slices found the problematic slice {column_with_slice_value}. "
                           f"It seems that the fix strategies {promising_fix_strategies} that Fairness Slices"
                           f" tried to improve the predictions for the problematic slice "
                           f"can lead to performance improvements by up to {max(performance_increases)}. "
                           f"You could take a look at these.")
            else:
                summary += (f"While the slice {column_with_slice_value} seems to be problematic, Fairness Slices"
                           f" cannot find any promising repair strategy automatically. However, you could try finding"
                           f" one on your own.")
            report = FairnessSlicesReport(orig_result, [ScreenedIssue(
                "Underperforming slices", True, readable_slice_result, fix_found, suggestions)], summary)

        return report

    def generate_report_for_fix_strategy(self, extracted_plan_results, fix_strategy_index, fix_strategy_name,
                                         orig_result, performance_increases, promising_fix_strategies):
        report = f"-\nRepair strategy {fix_strategy_index}: {fix_strategy_name}\n-\n"
        if extracted_plan_results[f"fairness-slices-fixing-made-changes-{fix_strategy_index}"] is False:
            report += "The fixing function did not make any changes.\n"
            suggestion = PotentialSuggestion(False, f"{fix_strategy_name}", orig_result, 1.0,
                                             None, None, None)
        else:
            fix_diff_df = extracted_plan_results[
                f"fairness-slice-fixing-diff-{fix_strategy_index}"]
            if isinstance(fix_diff_df, (pandas.DataFrame, pandas.Series)):
                fix_diff_df_sample = fix_diff_df.head(20)
            elif isinstance(fix_diff_df, numpy.ndarray) and fix_diff_df.ndim == 1:
                fix_diff_df_sample = fix_diff_df[:20]
            elif isinstance(fix_diff_df, numpy.ndarray) and fix_diff_df.ndim == 2:
                fix_diff_df_sample = fix_diff_df[:20, :]
            else:
                raise NotImplementedError("TODO")

            unmodified_diff = extracted_plan_results[
                f"fairness-slices-data-to-fix-{fix_strategy_index}"]
            if isinstance(unmodified_diff, (pandas.DataFrame, pandas.Series)):
                unmodified_diff_sample = unmodified_diff.head(20)
            elif isinstance(unmodified_diff, numpy.ndarray) and unmodified_diff.ndim == 1:
                unmodified_diff_sample = unmodified_diff[:20]
            elif isinstance(unmodified_diff, numpy.ndarray) and unmodified_diff.ndim == 2:
                unmodified_diff_sample = unmodified_diff[:20, :]
            else:
                raise NotImplementedError("TODO")

            fix_result = []
            for score_index in range(self.score_operator_count):
                fix_result.append(
                    extracted_plan_results[f"fairness-slice-fixing-{fix_strategy_index}-{score_index}"])

            max_score_improvement = get_relative_score_change(*orig_result, *fix_result)
            performance_increases.append(max_score_improvement)
            report += (
                f"After trying to automatically repair rows from this slice, "
                f"the pipeline metric was {fix_result} (A change of {max_score_improvement}). "
                f"A sample of the modified rows:\n{str(fix_diff_df_sample)}.\n\n"
                f"Before, these rows had the following values:\n{str(unmodified_diff_sample)}.\n")
            is_improvement = max_score_improvement > 1.
            if is_improvement:
                promising_fix_strategies.append(fix_strategy_name)
                report += (
                    f" It seems like changing the preprocessing of this datatype with a repair strategy like "
                    f"{fix_strategy_name} could help to improve the pipeline.\n")
            else:
                report += (
                    f" Repair strategy {fix_strategy_name} did not help to automatically improve the "
                    f"the pipeline performance. However, this does not mean that changing the preprocessing "
                    f"cannot help, it only means that Fairness Slices cannot find a promising "
                    f"repair strategy automatically.\n")
            source_code = FIX_STRATEGY_TO_CODE[fix_strategy_name]
            suggestion = PotentialSuggestion(is_improvement, fix_strategy_name, fix_result, max_score_improvement,
                                             unmodified_diff_sample, fix_diff_df_sample, source_code)
        return report, suggestion

    @staticmethod
    def fix_data(input_df, only_fix_indices=None, fix_strategy=None, database_path=None):
        # For now, this function is the same as in data_errors. Might want to consider different things here
        #  at some point
        fixed_corrupted = input_df.copy()
        if fix_strategy in {FixType.TEXT_TRANSLATE, FixType.TEXT_SPELLCHECK}:
            fixed_corrupted = FairnessSlices.fix_data_type_text(database_path, fix_strategy, fixed_corrupted,
                                                                only_fix_indices)
        elif fix_strategy == FixType.CAT:
            fixed_corrupted = FairnessSlices.fix_data_type_cat(fixed_corrupted, only_fix_indices)
        elif fix_strategy == FixType.NUM:
            fixed_corrupted = FairnessSlices.fix_data_type_num(fixed_corrupted, only_fix_indices)
        else:
            raise NotImplementedError(f"TODO: Add support for fix strategy {fix_strategy.value}!")

        fixed_corrupted = wrap_in_mlinspect_array_if_necessary(fixed_corrupted)
        fixed_corrupted._mlinspect_provenance = None

        return fixed_corrupted

    @staticmethod
    def fix_data_type_num(fixed_corrupted, only_fix_indices):
        fixed_corrupted = fixed_corrupted.reset_index(drop=True)
        clean = fixed_corrupted.drop(only_fix_indices, axis=0)
        for column_index, column in enumerate(fixed_corrupted.columns):
            is_int = fixed_corrupted[column].dtype == int
            # This is if we want to just apply fit_transform on all data instead of fixing only the corrupted data
            #  with a detection and cleaning method fitted on the clean data
            # fixed_corrupted = OutlierCleaner.fit_transform_all(fixed_corrupted, detection_strategy='IQR',
            #                                                    repair_strategy='mean', column=column)
            _, fitted_detector = detect_outlier_interquartile_range(clean[[column]], k=0.25)
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
        return fixed_corrupted

    @staticmethod
    def fix_data_type_cat(fixed_corrupted, only_fix_indices):
        is_dataframe = isinstance(fixed_corrupted, pandas.DataFrame)
        if is_dataframe:
            fixed_corrupted = fixed_corrupted.reset_index(drop=True)
            clean = fixed_corrupted.drop(only_fix_indices, axis=0)
        else:
            # For NumPy array, create a mask and remove rows
            mask = numpy.ones(fixed_corrupted.shape[0], dtype=bool)
            mask[only_fix_indices] = False
            clean = fixed_corrupted[mask]
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        one_hot_clean = one_hot_encoder.fit_transform(clean)
        isolation_forest = IsolationForest(contamination=0.5, random_state=42)
        isolation_forest.fit(one_hot_clean)
        if is_dataframe:
            one_hot_dirty = one_hot_encoder.transform(fixed_corrupted.iloc[only_fix_indices, :])
            outlier_indicator = isolation_forest.predict(one_hot_dirty) == -1
            fixed_corrupted[only_fix_indices[outlier_indicator], :] = -1
        else:
            one_hot_dirty = one_hot_encoder.transform(fixed_corrupted[only_fix_indices, :])
            outlier_indicator = isolation_forest.predict(one_hot_dirty) == -1
            fixed_corrupted[only_fix_indices[outlier_indicator], :] = -1
        # Iterate over columns
        num_columns = fixed_corrupted.shape[1]
        for col in range(num_columns):
            imputer = SimpleImputer(strategy="most_frequent", copy=True, missing_values=-1)

            if is_dataframe:
                # For DataFrame, fit on the clean column and transform specified rows
                imputer.fit(clean[[fixed_corrupted.columns[col]]])

                fixed_corrupted.iloc[only_fix_indices, col] = imputer.transform(
                    fixed_corrupted.iloc[only_fix_indices, [col]]
                ).ravel()
            else:
                # For NumPy array, fit on the clean column and transform specified rows
                imputer.fit(clean[:, col].reshape(-1, 1))
                fixed_corrupted[only_fix_indices, col] = imputer.transform(
                    fixed_corrupted[only_fix_indices, col].reshape(-1, 1)
                ).ravel()
        return fixed_corrupted

    @staticmethod
    def fix_data_type_text(database_path, fix_strategy, fixed_corrupted, only_fix_indices):
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
                if fix_strategy == FixType.TEXT_TRANSLATE:
                    translate_transformer = get_translate_transformer(column, database_path)
                    fixed_corrupted.iloc[only_fix_indices, [column_index]] = translate_transformer.fit_transform(
                        fixed_corrupted.iloc[only_fix_indices, [column_index]])
                elif fix_strategy == FixType.TEXT_SPELLCHECK:
                    typo_fixer = get_typo_fixer(column)
                    fixed_corrupted.iloc[only_fix_indices, [column_index]] = typo_fixer.fit_transform(
                        fixed_corrupted.iloc[only_fix_indices, [column_index]])
                else:
                    raise NotImplementedError("TODO")
        if was_series is True:
            fixed_corrupted = fixed_corrupted[series_column_name]
        elif was_numpy is True:
            fixed_corrupted = fixed_corrupted["column"].to_numpy()
        return fixed_corrupted

    @staticmethod
    def get_slice_finder_slice_and_indices(side_info_df, encoded_test_labels, predicted_test_labels, alpha):
        # TODO: In the documentation, it is only used on train and with known float loss
        sf = Slicefinder(
            alpha=alpha,
            k=1,
            max_l=2,
            min_sup=1,
            verbose=True,
        )
        if isinstance(encoded_test_labels, pandas.Series):
            encoded_test_labels = encoded_test_labels.to_numpy()
        if isinstance(predicted_test_labels, list):
            predicted_test_labels = numpy.array(predicted_test_labels)

        side_info_df = side_info_df.fillna("nan")
        sf_result = sf.fit(side_info_df, (encoded_test_labels.reshape(-1, ) != predicted_test_labels.reshape(-1, )))
        if len(sf_result.top_slices_) == 0:
            return None, None
        top_slice = sf_result.top_slices_[0]

        test_mask = numpy.ones(shape=(len(side_info_df)), dtype=bool)
        for column_index, column_value in enumerate(top_slice):
            if column_value is not None:
                test_mask = test_mask & (side_info_df.iloc[:, column_index] == column_value).to_numpy()
        test_indices = numpy.where(test_mask)[0]
        return top_slice, test_indices
