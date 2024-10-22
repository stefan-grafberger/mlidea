from functools import partial

import duckdb
import networkx
import numpy
import pandas
from numba import prange, njit
from scipy.sparse import csr_matrix

from mlidea.instrumentation._operator_call_info import OperatorCallInfo
from mlidea import OperatorType, DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlidea.analysis._analysis_utils import find_nodes_by_type
from mlidea.execution._pipeline_executor import singleton
from mlidea.monkeypatching._patch_langchain import RunnableSequencePatching
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline
from mlidea.shadow_pipelines._utils import get_intermediate_extraction_node, copy_node_with_new_id, \
    get_conditional_stop_node, get_relative_score_change, add_orig_score_extraction_nodes, \
    get_diff_filter_node, merge_prediction_diff_with_old_predictions, add_new_score_and_score_extraction_nodes, \
    assert_standard_llm_shape, assert_standard_ml_shape, get_proxy_model_node, df_or_array_non_empty, \
    df_or_array_non_empty_func_info


class LabelErrors(ShadowPipeline):
    """
    The Label Error Shadow Pipeline
    """

    def check_rebuilding_necessary(self, extracted_plan_results: dict[str, any]) -> any:
        return False

    def __init__(self, train_fraction_to_consider=1., test_fraction_to_consider=1., proxy_model=False,
                 cleaning_batch_size=20, only_consider_negative_shapley_values=False):
        # TODO: We should probably also implement the second proxy version from the workshop paper
        self._train_fraction_to_consider = train_fraction_to_consider
        self._test_fraction_to_consider = test_fraction_to_consider
        self._proxy_model = proxy_model
        self._cleaning_batch_size = cleaning_batch_size
        self._only_consider_negative_shapley_values = only_consider_negative_shapley_values
        self._shadow_pipeline_id = (train_fraction_to_consider, test_fraction_to_consider, proxy_model,
                                    cleaning_batch_size, only_consider_negative_shapley_values)
        self.score_operator_count = 0

    @property
    def shadow_pipeline_id(self):
        return self._shadow_pipeline_id

    @property
    def simple_name(self):
        return "label_errors"

    def generate_shadow_pipeline_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        # TODO: Maybe it would be better to delete all unrelated DAG nodes here that are not specifically mentioned
        #  below. But this only works once intermediate resutl caching is implemented

        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)

        if len(rag_join_operators) == 0:
            new_dag = self._get_traditional_ml_dag(dag)
        else:
            new_dag = self._get_llm_rag_dag(dag)

        return new_dag

    def _get_traditional_ml_dag(self, dag):
        new_dag = dag.copy()
        assert_standard_ml_shape(dag, "Label Errors")

        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        model_operators = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
        train_data_operators = find_nodes_by_type(dag, OperatorType.TRAIN_DATA)
        train_labels_operators = find_nodes_by_type(dag, OperatorType.TRAIN_LABELS)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)
        add_orig_score_extraction_nodes(singleton, new_dag, score_operators)
        self.score_operator_count = len(score_operators)

        new_shapley_node = self._add_shapley_value_computation_ml(new_dag, test_data_operators, test_labels_operators,
                                                                  train_data_operators, train_labels_operators)

        likely_mislabeled_rows_condition_node = LabelErrors._get_likely_mislabeled_rows_present_condition_node(
            new_dag, new_shapley_node)

        self._add_orig_proxy_score_computation_ml(likely_mislabeled_rows_condition_node, model_operators, new_dag,
                                                  predict_operators, score_operators, test_data_operators,
                                                  train_data_operators, train_labels_operators)

        self._add_label_flip_computation_ml(likely_mislabeled_rows_condition_node, model_operators, new_dag,
                                            new_shapley_node, predict_operators, score_operators, test_data_operators,
                                            train_data_operators, train_labels_operators)

        return new_dag

    def _get_llm_rag_dag(self, dag):
        if self._proxy_model is True:
            raise ValueError("Proxy model is not supported for LLM pipelines!")
        new_dag = dag.copy()
        assert_standard_llm_shape(dag, "Label Errors")

        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)
        train_labels_operators = find_nodes_by_type(dag, OperatorType.TRAIN_LABELS)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)
        label_encoder_operators = list(new_dag.predecessors(test_labels_operators[0]))
        if len(label_encoder_operators) != 1 or "label_binarize" not in label_encoder_operators[0].details.description:
            raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                      "pattern!")

        self.score_operator_count = len(score_operators)

        new_shapley_node = self._add_shapley_value_computation_llm(label_encoder_operators, new_dag, rag_join_operators,
                                                                   score_operators, test_data_operators,
                                                                   test_labels_operators, train_labels_operators)

        likely_mislabeled_rows_condition_node = LabelErrors._get_likely_mislabeled_rows_present_condition_node(
            new_dag, new_shapley_node)

        self._add_label_flip_computation_llm(likely_mislabeled_rows_condition_node, new_dag, new_shapley_node,
                                             predict_operators, rag_join_operators, score_operators, test_data_operators,
                                             train_labels_operators)
        return new_dag

    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        report = ""
        orig_result = []
        for score_index in range(self.score_operator_count):
            orig_result.append(extracted_plan_results[f"orig-{score_index}"])
        report += f"The original result was {orig_result}.\n"
        proxy_result = []
        if self._proxy_model is True:
            for score_index in range(self.score_operator_count):
                proxy_result.append(extracted_plan_results[f"label-errors-proxy-{score_index}"])
            report += f"The proxy result was {proxy_result}.\n"
        shapley_values = extracted_plan_results["label-errors-shapley-values"]
        if extracted_plan_results["label-errors-shapley-values-non-empty"] is False:
            report += "No likely mislabeled rows were found with the given label error config!\nNothing to do for now."
        else:
            flip_result = []
            for score_index in range(self.score_operator_count):
                flip_result.append(extracted_plan_results[f"label-errors-flip-retrain-{score_index}"])
            report += (f"After flipping the top {self._cleaning_batch_size} most "
                       f"likely incorrect row labels, the pipeline metric was {flip_result}")
            if self._proxy_model is True:
                report += " (with the proxy model)"
            report += (f".\nThe shapley values of the "
                       f"most likely mislabeled rows:\n{str(shapley_values)}")
            if self._proxy_model is True:
                max_score_improvement = get_relative_score_change(*proxy_result, *flip_result)
            else:
                max_score_improvement = get_relative_score_change(*orig_result, *flip_result)
            if max_score_improvement > 1.:
                report += (f"\n\nThe score increased by relabeling {self._cleaning_batch_size} rows by "
                           f"{max_score_improvement}. You probably want to take a look at "
                           f"the row labels again!")
            else:
                report += (f"\n\nWhile there are rows with potentially problematic shapley values that you could "
                           f"take a look at, automatically flipping the top {self._cleaning_batch_size} most likely "
                           f"incorrect labels did not lead to an improvement (the max relative score "
                           f"was {max_score_improvement}).")
            if self._proxy_model is True:
                report += (" (However, that relative score difference is only calculated using the proxy model, so "
                           "the score changes with the proxy model are not guaranteed to be similar to score changes "
                           "for your actual model.)")
        return report

    def _add_label_flip_computation_ml(self, likely_mislabeled_rows_condition_node, model_operators, new_dag,
                                       new_shapley_node, predict_operators, score_operators, test_data_operators,
                                       train_data_operators, train_labels_operators):
        # pylint: disable=too-many-arguments
        non_data_kwargs = {'cleaning_batch_size': self._cleaning_batch_size,
                           'func': LabelErrors._label_flip_processing_func_ml}
        operator_context = OperatorContext(OperatorType.PROJECTION, None, non_data_kwargs)
        parents = [train_labels_operators[0], new_shapley_node, likely_mislabeled_rows_condition_node]
        operator_call_info = OperatorCallInfo(operator_context, parents)
        new_label_flip_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                      BasicCodeLocation("Label Errors", None),
                                      operator_context,
                                      DagNodeDetails(
                                          f"Flip {self._cleaning_batch_size} most likely incorrect labels", None),
                                      None,
                                      LabelErrors._label_flip_processing_func_ml)
        new_dag.add_edge(train_labels_operators[0], new_label_flip_node, arg_index=0)
        new_dag.add_edge(new_shapley_node, new_label_flip_node, arg_index=1)
        new_dag.add_edge(likely_mislabeled_rows_condition_node, new_label_flip_node, arg_index=4)
        if self._proxy_model is False:
            new_model_node = copy_node_with_new_id(singleton, model_operators[0])
        else:
            parent_nodes = [train_data_operators[0], new_label_flip_node]
            new_model_node = get_proxy_model_node(singleton, model_operators[0], parent_nodes)
        new_dag.add_edge(train_data_operators[0], new_model_node, arg_index=0)
        new_dag.add_edge(new_label_flip_node, new_model_node, arg_index=1)
        new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_model_node, new_predict_node, arg_index=0)
        new_dag.add_edge(test_data_operators[0], new_predict_node, arg_index=1)
        add_new_score_and_score_extraction_nodes(singleton, new_dag, new_predict_node, score_operators,
                                                 "label-errors-flip-retrain")

    def _add_orig_proxy_score_computation_ml(self, likely_mislabeled_rows_condition_node, model_operators, new_dag,
                                             predict_operators, score_operators, test_data_operators,
                                             train_data_operators,
                                             train_labels_operators):
        if self._proxy_model is True:
            parent_nodes = [train_data_operators[0], train_labels_operators[0], likely_mislabeled_rows_condition_node]
            new_model_node = get_proxy_model_node(singleton, model_operators[0], parent_nodes)
            new_dag.add_edge(train_data_operators[0], new_model_node, arg_index=0)
            new_dag.add_edge(train_labels_operators[0], new_model_node, arg_index=1)
            new_dag.add_edge(likely_mislabeled_rows_condition_node, new_model_node, arg_index=2)

            new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
            new_dag.add_edge(new_model_node, new_predict_node, arg_index=0)
            new_dag.add_edge(test_data_operators[0], new_predict_node, arg_index=1)
            new_dag.add_edge(likely_mislabeled_rows_condition_node, new_predict_node, arg_index=2)
            add_new_score_and_score_extraction_nodes(singleton, new_dag, new_predict_node, score_operators,
                                                     "label-errors-proxy")

    def _add_shapley_value_computation_ml(self, new_dag, test_data_operators, test_labels_operators,
                                          train_data_operators,
                                          train_labels_operators):
        processing_func = partial(LabelErrors._shapley_top_k_func_ml,
                                  train_fraction_to_consider=self._train_fraction_to_consider,
                                  test_fraction_to_consider=self._test_fraction_to_consider,
                                  cleaning_batch_size=self._cleaning_batch_size,
                                  only_consider_negative_shapley_values=self._only_consider_negative_shapley_values)
        non_data_kwargs = {'cleaning_batch_size': self._cleaning_batch_size,
                           'func': LabelErrors._shapley_top_k_func_ml,
                           'train_fraction_to_consider': self._train_fraction_to_consider,
                           'test_fraction_to_consider': self._test_fraction_to_consider,
                           'only_consider_negative_shapley_values': self._only_consider_negative_shapley_values}
        operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, None, non_data_kwargs)
        operator_call_info = OperatorCallInfo(operator_context, [train_data_operators[0], train_labels_operators[0],
                                                                 test_data_operators[0], test_labels_operators[0]])
        new_shapley_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                   BasicCodeLocation("Label Errors", None),
                                   operator_context,
                                   DagNodeDetails(
                                       f"Top {self._cleaning_batch_size} Shapley values", None),
                                   None,
                                   processing_func)
        new_dag.add_edge(train_data_operators[0], new_shapley_node, arg_index=0)
        new_dag.add_edge(train_labels_operators[0], new_shapley_node, arg_index=1)
        new_dag.add_edge(test_data_operators[0], new_shapley_node, arg_index=2)
        new_dag.add_edge(test_labels_operators[0], new_shapley_node, arg_index=3)
        _ = get_intermediate_extraction_node(singleton, new_dag, new_shapley_node, "label-errors-shapley-values")
        return new_shapley_node

    def _add_label_flip_computation_llm(self, likely_mislabeled_rows_condition_node, new_dag, new_shapley_node,
                                        predict_operators, rag_join_operators, score_operators, test_data_operators,
                                        train_labels_operators):
        non_data_kwargs = {'cleaning_batch_size': self._cleaning_batch_size,
                           'func': LabelErrors._get_rows_to_flip_llm}
        operator_context = OperatorContext(OperatorType.PROJECTION, None, non_data_kwargs)
        parents = [rag_join_operators[0], new_shapley_node, likely_mislabeled_rows_condition_node]
        operator_call_info = OperatorCallInfo(operator_context, parents)
        new_label_flip_indices_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                              BasicCodeLocation("Label Errors", None),
                                              operator_context,
                                              DagNodeDetails(
                                                  f"Flip {self._cleaning_batch_size} most likely incorrect labels",
                                                  None),
                                              None,
                                              LabelErrors._get_rows_to_flip_llm)
        new_dag.add_edge(rag_join_operators[0], new_label_flip_indices_node, arg_index=0)
        new_dag.add_edge(new_shapley_node, new_label_flip_indices_node, arg_index=1)
        new_dag.add_edge(likely_mislabeled_rows_condition_node, new_label_flip_indices_node, arg_index=2)
        non_data_kwargs = {'cleaning_batch_size': self._cleaning_batch_size,
                           'func': LabelErrors._label_flip_processing_func_llm}
        operator_context = OperatorContext(OperatorType.PROJECTION, None, non_data_kwargs)
        parents = [rag_join_operators[0], train_labels_operators[0], new_shapley_node, new_label_flip_indices_node,
                   test_data_operators[0], likely_mislabeled_rows_condition_node]
        operator_call_info = OperatorCallInfo(operator_context, parents)
        new_label_flip_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                      BasicCodeLocation("Label Errors", None),
                                      operator_context,
                                      DagNodeDetails(
                                          f"Flip {self._cleaning_batch_size} most likely incorrect labels", None),
                                      None,
                                      LabelErrors._label_flip_processing_func_llm)
        new_dag.add_edge(rag_join_operators[0], new_label_flip_node, arg_index=0)
        new_dag.add_edge(train_labels_operators[0], new_label_flip_node, arg_index=1)
        new_dag.add_edge(new_shapley_node, new_label_flip_node, arg_index=2)
        new_dag.add_edge(new_label_flip_indices_node, new_label_flip_node, arg_index=3)
        new_dag.add_edge(test_data_operators[0], new_label_flip_node, arg_index=4)
        new_dag.add_edge(likely_mislabeled_rows_condition_node, new_label_flip_indices_node, arg_index=5)
        parents = [new_label_flip_node, new_label_flip_indices_node]
        new_fix_diff_filter_node = get_diff_filter_node(singleton, new_dag, "Data Errors", parents)
        new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_fix_diff_filter_node, new_predict_node, arg_index=0)
        parents = [predict_operators[0], new_predict_node, new_label_flip_indices_node]
        new_fix_predict_diff_update_node = merge_prediction_diff_with_old_predictions(singleton, new_dag,
                                                                                      "Label Errors", parents)
        add_new_score_and_score_extraction_nodes(singleton, new_dag, new_fix_predict_diff_update_node,
                                                 score_operators, "label-errors-flip-retrain")

    def _add_shapley_value_computation_llm(self, label_encoder_operators, new_dag, rag_join_operators, score_operators,
                                           test_data_operators, test_labels_operators, train_labels_operators):
        train_labels_dict_conversion = list(new_dag.predecessors(train_labels_operators[0]))[0]
        train_labels_before_dict = list(new_dag.predecessors(train_labels_dict_conversion))[0]
        add_orig_score_extraction_nodes(singleton, new_dag, score_operators)
        processing_func = partial(LabelErrors._shapley_top_k_func_llm,
                                  train_fraction_to_consider=self._train_fraction_to_consider,
                                  test_fraction_to_consider=self._test_fraction_to_consider,
                                  cleaning_batch_size=self._cleaning_batch_size,
                                  label_encoding_op=label_encoder_operators[0],
                                  only_consider_negative_shapley_values=self._only_consider_negative_shapley_values)
        non_data_kwargs = {'cleaning_batch_size': self._cleaning_batch_size,
                           'func': LabelErrors._shapley_top_k_func_llm,
                           'train_fraction_to_consider': self._train_fraction_to_consider,
                           'test_fraction_to_consider': self._test_fraction_to_consider,
                           'only_consider_negative_shapley_values': self._only_consider_negative_shapley_values,
                           'label_encoding_op': label_encoder_operators[0]}
        operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, None, non_data_kwargs)
        parents = [rag_join_operators[0], train_labels_before_dict, test_data_operators[0], test_labels_operators[0]]
        operator_call_info = OperatorCallInfo(operator_context, parents)
        new_shapley_node = DagNode(singleton.get_next_op_id(operator_call_info),
                                   BasicCodeLocation("Label Errors", None),
                                   operator_context,
                                   DagNodeDetails(
                                       f"Top {self._cleaning_batch_size} Shapley values", None),
                                   None,
                                   processing_func)
        new_dag.add_edge(rag_join_operators[0], new_shapley_node, arg_index=0)
        new_dag.add_edge(train_labels_before_dict, new_shapley_node, arg_index=1)
        new_dag.add_edge(test_data_operators[0], new_shapley_node, arg_index=2)
        new_dag.add_edge(test_labels_operators[0], new_shapley_node, arg_index=3)
        _ = get_intermediate_extraction_node(singleton, new_dag, new_shapley_node, "label-errors-shapley-values")
        return new_shapley_node

    @staticmethod
    def _get_likely_mislabeled_rows_present_condition_node(new_dag, new_shapley_node):
        function_info = df_or_array_non_empty_func_info()
        likely_mislabeled_rows_condition_node = get_conditional_stop_node(
            singleton, new_dag, df_or_array_non_empty, function_info, "label-errors-shapley-values-non-empty",
            "Check if there are likely mislabeled rows", [new_shapley_node])
        return likely_mislabeled_rows_condition_node

    @staticmethod
    @njit(fastmath=True, parallel=True, cache=True)
    def _compute_shapley_values(X_train, y_train, X_test, y_test, K=1):
        # pylint: disable=invalid-name
        """Compute approximate shapley values as presented in the DataScope paper. Here, we only do it for the
        estimator input data though and not for the input data of the surrounding pipeline.
        """
        N = len(X_train)
        M = len(X_test)
        result = numpy.zeros(N, dtype=numpy.float32)

        for j in prange(M):  # pylint: disable=not-an-iterable
            score = numpy.zeros(N, dtype=numpy.float32)
            dist = numpy.zeros(N, dtype=numpy.float32)
            div_range = numpy.arange(1.0, N)
            div_min = numpy.minimum(div_range, K)
            for i in range(N):
                dist[i] = numpy.sqrt(numpy.sum(numpy.square(X_train[i] - X_test[j])))
            indices = numpy.argsort(dist)
            y_sorted = y_train[indices]
            eq_check = (y_sorted == y_test[j]) * 1.0
            diff = - 1 / K * (eq_check[1:] - eq_check[:-1])
            diff /= div_range
            diff *= div_min
            score[indices[:-1]] = diff
            score[indices[-1]] = eq_check[-1] / N
            score[indices] += numpy.sum(score[indices]) - numpy.cumsum(score[indices])
            result += score / M

        return result

    @staticmethod
    def _shapley_top_k_func_llm(rag_join_result, train_labels_before_dict, encoded_test_data, encoded_test_labels,
                                train_fraction_to_consider, test_fraction_to_consider, cleaning_batch_size,
                                label_encoding_op, only_consider_negative_shapley_values):
        # TODO: Should we propagate provenance here? Might be important for explanations later
        test_indices_to_consider, train_indices_to_consider = LabelErrors._get_train_and_test_indices_to_consider(
            encoded_test_labels, test_fraction_to_consider, train_fraction_to_consider, train_labels_before_dict)

        x_train, y_train, x_test, y_test = LabelErrors._prepare_shapley_arguments(
            encoded_test_data, encoded_test_labels, label_encoding_op, test_indices_to_consider,
            train_indices_to_consider, train_labels_before_dict, rag_join_result[5])

        shapley_values = LabelErrors._compute_shapley_values(x_train, numpy.squeeze(y_train),
                                                             x_test, numpy.squeeze(y_test))
        df_with_id_and_shapley_value = pandas.DataFrame(
            {"train_id": train_indices_to_consider, "shapley_value": shapley_values})

        rows_to_fix = df_with_id_and_shapley_value.nsmallest(cleaning_batch_size, "shapley_value")
        if only_consider_negative_shapley_values:
            rows_to_fix = rows_to_fix[rows_to_fix["shapley_value"] <= 0.]
        return rows_to_fix

    @staticmethod
    def _prepare_shapley_arguments(encoded_test_data, encoded_test_labels, label_encoding_op, test_indices_to_consider,
                                   train_indices_to_consider, train_labels_before_dict, vectorstore):
        train_data_sample = numpy.array(vectorstore.get(
            ids=[str(index) for index in train_indices_to_consider], include=["embeddings"])['embeddings'])
        to_label_encode = train_labels_before_dict.iloc[train_indices_to_consider, 0]
        to_label_encode._mlinspect_provenance = None
        train_label_sample = label_encoding_op.processing_func(to_label_encode)
        test_data_sample = numpy.array(vectorstore.embeddings.embed_documents(
            numpy.array(encoded_test_data)[test_indices_to_consider]))
        test_label_sample = encoded_test_labels[test_indices_to_consider]
        return train_data_sample, train_label_sample, test_data_sample, test_label_sample

    @staticmethod
    def _get_train_and_test_indices_to_consider(encoded_test_labels, test_fraction_to_consider,
                                                train_fraction_to_consider, train_labels_before_dict):
        indices = numpy.arange(len(train_labels_before_dict))
        numpy.random.shuffle(indices)
        num_values_to_typo = int(len(train_labels_before_dict) * train_fraction_to_consider)
        train_indices_to_consider = indices[:num_values_to_typo]
        indices = numpy.arange(len(encoded_test_labels))
        num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
        test_indices_to_consider = indices[:num_values_to_typo]
        return test_indices_to_consider, train_indices_to_consider

    @staticmethod
    def _shapley_top_k_func_ml(encoded_train_data, encoded_train_labels, encoded_test_data, encoded_test_labels,
                               train_fraction_to_consider, test_fraction_to_consider, cleaning_batch_size,
                               only_consider_negative_shapley_values):
        # TODO: Should we propagate provenance here? Might be important for explanations later
        indices = numpy.arange(len(encoded_train_labels))
        numpy.random.shuffle(indices)

        num_values_to_typo = int(len(encoded_train_labels) * train_fraction_to_consider)
        train_indices_to_consider = indices[:num_values_to_typo]
        if isinstance(encoded_train_data, (pandas.DataFrame, pandas.Series)):
            encoded_train_data = encoded_train_data.reset_index(drop=True)
        if isinstance(encoded_train_labels, (pandas.DataFrame, pandas.Series)):
            encoded_train_labels = encoded_train_labels.reset_index(drop=True).to_numpy()
        train_data_sample = encoded_train_data[train_indices_to_consider]
        train_label_sample = encoded_train_labels[train_indices_to_consider]

        indices = numpy.arange(len(encoded_test_labels))
        num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
        test_indices_to_consider = indices[:num_values_to_typo]
        if isinstance(encoded_test_data, (pandas.DataFrame, pandas.Series)):
            encoded_test_data = encoded_test_data.reset_index(drop=True)
        if isinstance(encoded_test_labels, (pandas.DataFrame, pandas.Series)):
            encoded_test_labels = encoded_test_labels.reset_index(drop=True).to_numpy()
        test_data_sample = encoded_test_data[test_indices_to_consider]
        test_label_sample = encoded_test_labels[test_indices_to_consider]

        if isinstance(train_data_sample, csr_matrix):
            train_data_sample = train_data_sample.todense()
        if isinstance(test_data_sample, csr_matrix):
            test_data_sample = test_data_sample.todense()
        shapley_values = LabelErrors._compute_shapley_values(train_data_sample, numpy.squeeze(train_label_sample),
                                                             test_data_sample, numpy.squeeze(test_label_sample))
        df_with_id_and_shapley_value = pandas.DataFrame(
            {"train_id": train_indices_to_consider, "shapley_value": shapley_values})

        rows_to_fix = df_with_id_and_shapley_value.nsmallest(cleaning_batch_size, "shapley_value")
        if only_consider_negative_shapley_values:
            rows_to_fix = rows_to_fix[rows_to_fix["shapley_value"] <= 0.]
        return rows_to_fix

    @staticmethod
    def _get_rows_to_flip_llm(rag_join_result, shapley_result):
        retrieval_index = rag_join_result[6]
        changed_df = shapley_result[['train_id']]  # pylint: disable=unused-variable
        pandas_retrieval_index_df = pandas.DataFrame(retrieval_index,
                                                     columns=['train_retrieved_1', 'train_retrieved_2',
                                                              'train_retrieved_3', 'train_retrieved_4'])
        pandas_retrieval_index_df['prediction_id'] = list(range(len(rag_join_result[2])))
        all_predictions_to_rerun = duckdb.query("""
                    SELECT DISTINCT prediction_id
                    FROM changed_df c JOIN pandas_retrieval_index_df p 
                    ON c.train_id = train_retrieved_1 
                    OR c.train_id = train_retrieved_2 
                    OR c.train_id = train_retrieved_3 
                    OR c.train_id = train_retrieved_4 
                """).fetchnumpy()['prediction_id']

        return all_predictions_to_rerun

    @staticmethod
    def _label_flip_processing_func_llm(rag_join_result, encoded_train_labels, shapley_result, all_predictions_to_rerun,
                                        inputs):
        # TODO: Should we propagate provenance here? Might be important for explanations later
        # Flip the row labels that need flipping
        mislabeled_indices = shapley_result['train_id']
        diff_encoded_train_labels = LabelErrors._flip_specified_row_labels(encoded_train_labels,
                                                                           mislabeled_indices)

        # Update the labels in the vectorstore
        vectorstore = rag_join_result[5]
        vectorstore_ids = [str(index) for index in mislabeled_indices]
        old_entries = vectorstore.get(ids=vectorstore_ids, include=["embeddings", "documents", "metadatas"])
        vectorstore._collection.update(vectorstore_ids, old_entries['embeddings'], diff_encoded_train_labels,
                                       old_entries['documents'])

        # Rerun the RAG join on the diff
        diff_inputs = list(numpy.array(inputs)[all_predictions_to_rerun])
        diff_rag_result, diff_retrieval_index = RunnableSequencePatching.execute_rag_join_diff(
            rag_join_result[7], diff_inputs, vectorstore)

        # Revert vectorstore changes again
        vectorstore._collection.update(vectorstore_ids, old_entries['embeddings'], old_entries['metadatas'],
                                       old_entries['documents'])

        # Prepare the usual RAG join output
        new_rag_join_text_result = numpy.array(rag_join_result[2])
        new_rag_join_text_result[all_predictions_to_rerun] = diff_rag_result
        new_rag_join_text_result_list = list(new_rag_join_text_result)

        new_retrieval_index = rag_join_result[6].copy()
        new_retrieval_index[all_predictions_to_rerun, :] = diff_retrieval_index

        new_rag_join_result = (rag_join_result[0], rag_join_result[1], new_rag_join_text_result_list,
                               rag_join_result[3], None, rag_join_result[5], new_retrieval_index, rag_join_result[7])
        return new_rag_join_result

    @staticmethod
    def _flip_specified_row_labels(encoded_train_labels, mislabeled_indices):
        label_name, label_value_list = LabelErrors._get_label_name_and_label_value_list(encoded_train_labels)
        diff_encoded_train_labels = numpy.array(encoded_train_labels)[mislabeled_indices]
        for mislabeled_row in diff_encoded_train_labels:
            assert label_name is not None
            current_val = mislabeled_row[label_name]
            current_val_index = label_value_list.index(current_val)
            mislabeled_row[label_name] = label_value_list[1 - current_val_index]
        diff_encoded_train_labels = list(diff_encoded_train_labels)
        return diff_encoded_train_labels

    @staticmethod
    def _get_label_name_and_label_value_list(encoded_train_labels):
        classes = set()
        class_search_index = 0
        label_key = None
        while len(classes) != 2 and class_search_index < len(encoded_train_labels):
            label_dict_items = list(encoded_train_labels[class_search_index].items())
            assert len(label_dict_items) == 1
            label_key, label_value = label_dict_items[0]
            classes.add(label_value)
            class_search_index += 1
        classes_list = list(classes)
        return label_key, classes_list

    @staticmethod
    def _label_flip_processing_func_ml(encoded_train_labels, shapley_result):
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
        elif modified_encoded_train_labels.ndim == 2:
            modified_encoded_train_labels[unfair_indices, :] = 1 - modified_encoded_train_labels[unfair_indices, :]
        elif modified_encoded_train_labels.ndim == 1:
            modified_encoded_train_labels[unfair_indices] = 1 - modified_encoded_train_labels[unfair_indices]
        else:
            raise NotImplementedError("TODO")
        return modified_encoded_train_labels
