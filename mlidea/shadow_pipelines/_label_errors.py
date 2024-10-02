from copy import copy
from functools import partial

import duckdb
import networkx
import numpy
import pandas
from numba import prange, njit
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier

from mlidea.execution._pipeline_executor import singleton
from mlidea.analysis._analysis_utils import find_nodes_by_type
from mlidea import OperatorType, DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline
from mlidea.shadow_pipelines._utils import get_intermediate_extraction_node, copy_node_with_new_id, \
    get_sorted_parent_nodes, get_conditional_stop_node, get_relative_score_change, add_orig_score_extraction_nodes
from mlidea.monkeypatching._patch_langchain import RunnableSequencePatching
from mlidea.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


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
        add_orig_score_extraction_nodes(singleton, new_dag, score_operators)
        self.score_operator_count = len(score_operators)

        processing_func = partial(LabelErrors.shapley_top_k_func_ml,
                                  train_fraction_to_consider=self._train_fraction_to_consider,
                                  test_fraction_to_consider=self._test_fraction_to_consider,
                                  cleaning_batch_size=self._cleaning_batch_size,
                                  only_consider_negative_shapley_values=self._only_consider_negative_shapley_values)
        new_shapley_node = DagNode(singleton.get_next_op_id(),
                                   BasicCodeLocation("Label Errors", None),
                                   OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                   DagNodeDetails(
                                       f"Top {self._cleaning_batch_size} Shapley values", None),
                                   None,
                                   processing_func)
        new_dag.add_edge(train_data_operators[0], new_shapley_node, arg_index=0)
        new_dag.add_edge(train_labels_operators[0], new_shapley_node, arg_index=1)
        new_dag.add_edge(test_data_operators[0], new_shapley_node, arg_index=2)
        new_dag.add_edge(test_labels_operators[0], new_shapley_node, arg_index=3)
        extraction_node = get_intermediate_extraction_node(singleton, new_shapley_node, "label-errors-shapley-values")
        new_dag.add_edge(new_shapley_node, extraction_node, arg_index=0)

        likely_mislabeled_rows_not_empty_func = lambda shapley_df: len(shapley_df) != 0
        likely_mislabeled_rows_condition_node = get_conditional_stop_node(
            singleton, likely_mislabeled_rows_not_empty_func, "label-errors-shapley-values-non-empty",
            "Check if there are likely mislabeled rows", new_shapley_node)
        new_dag.add_edge(new_shapley_node, likely_mislabeled_rows_condition_node, arg_index=0)

        new_label_flip_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("Label Errors", None),
                                      OperatorContext(OperatorType.PROJECTION, None),
                                      DagNodeDetails(
                                          f"Flip {self._cleaning_batch_size} most likely incorrect labels", None),
                                      None,
                                      LabelErrors.label_flip_processing_func_ml)
        new_dag.add_edge(train_labels_operators[0], new_label_flip_node, arg_index=0)
        new_dag.add_edge(extraction_node, new_label_flip_node, arg_index=1)
        new_dag.add_edge(likely_mislabeled_rows_condition_node, new_label_flip_node, arg_index=4)

        if self._proxy_model is True:
            new_model_node = LabelErrors.get_proxy_model_node(singleton, model_operators[0])
            new_dag.add_edge(train_data_operators[0], new_model_node, arg_index=0)
            new_dag.add_edge(train_labels_operators[0], new_model_node, arg_index=1)
            new_dag.add_edge(likely_mislabeled_rows_condition_node, new_model_node, arg_index=2)
            new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
            new_dag.add_edge(new_model_node, new_predict_node, arg_index=0)
            new_dag.add_edge(test_data_operators[0], new_predict_node, arg_index=1)
            new_dag.add_edge(likely_mislabeled_rows_condition_node, new_predict_node, arg_index=2)
            LabelErrors.add_new_score_and_score_extraction_nodes(new_dag, new_predict_node, score_operators,
                                                                 test_labels_operators, "label-errors-proxy")

        if self._proxy_model is False:
            new_model_node = copy_node_with_new_id(singleton, model_operators[0])
        else:
            new_model_node = LabelErrors.get_proxy_model_node(singleton, model_operators[0])
        new_dag.add_edge(train_data_operators[0], new_model_node, arg_index=0)
        new_dag.add_edge(new_label_flip_node, new_model_node, arg_index=1)
        new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_model_node, new_predict_node, arg_index=0)
        new_dag.add_edge(test_data_operators[0], new_predict_node, arg_index=1)
        LabelErrors.add_new_score_and_score_extraction_nodes(new_dag, new_predict_node, score_operators,
                                                             test_labels_operators, "label-errors-flip-retrain")

        return new_dag

    def get_llm_rag_dag(self, dag):
        if self._proxy_model is True:
            raise ValueError("Proxy model is not supported for LLM pipelines!")
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

        add_orig_score_extraction_nodes(singleton, new_dag, score_operators)

        processing_func = partial(LabelErrors.shapley_top_k_func_llm,
                                  train_fraction_to_consider=self._train_fraction_to_consider,
                                  test_fraction_to_consider=self._test_fraction_to_consider,
                                  cleaning_batch_size=self._cleaning_batch_size,
                                  label_encoding_op=label_encoder_operators[0],
                                  only_consider_negative_shapley_values=self._only_consider_negative_shapley_values)
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

        likely_mislabeled_rows_not_empty_func = lambda shapley_df: len(shapley_df) != 0
        likely_mislabeled_rows_condition_node = get_conditional_stop_node(
            singleton, likely_mislabeled_rows_not_empty_func, "label-errors-shapley-values-non-empty",
            "Check if there are likely mislabeled rows", new_shapley_node)
        new_dag.add_edge(new_shapley_node, likely_mislabeled_rows_condition_node, arg_index=0)

        new_label_flip_indices_node = DagNode(singleton.get_next_op_id(),
                                              BasicCodeLocation("Label Errors", None),
                                              OperatorContext(OperatorType.PROJECTION, None),
                                              DagNodeDetails(
                                                  f"Flip {self._cleaning_batch_size} most likely incorrect labels",
                                                  None),
                                              None,
                                              LabelErrors.get_rows_to_flip_llm)
        new_dag.add_edge(rag_join_operators[0], new_label_flip_indices_node, arg_index=0)
        new_dag.add_edge(new_shapley_node, new_label_flip_indices_node, arg_index=1)
        new_dag.add_edge(likely_mislabeled_rows_condition_node, new_label_flip_indices_node, arg_index=2)

        new_label_flip_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("Label Errors", None),
                                      OperatorContext(OperatorType.PROJECTION, None),
                                      DagNodeDetails(
                                          f"Flip {self._cleaning_batch_size} most likely incorrect labels", None),
                                      None,
                                      LabelErrors.label_flip_processing_func_llm)
        new_dag.add_edge(rag_join_operators[0], new_label_flip_node, arg_index=0)
        new_dag.add_edge(train_labels_operators[0], new_label_flip_node, arg_index=1)
        new_dag.add_edge(new_shapley_node, new_label_flip_node, arg_index=2)
        new_dag.add_edge(new_label_flip_indices_node, new_label_flip_node, arg_index=3)
        new_dag.add_edge(test_data_operators[0], new_label_flip_node, arg_index=4)
        new_dag.add_edge(likely_mislabeled_rows_condition_node, new_label_flip_indices_node, arg_index=5)

        new_fix_diff_filter_node = DagNode(singleton.get_next_op_id(),
                                           BasicCodeLocation("Data Errors", None),
                                           OperatorContext(OperatorType.SELECTION, None),
                                           DagNodeDetails(
                                               "Filter for diff only",
                                               None),
                                           None,
                                           LabelErrors.apply_diff_filter)
        new_dag.add_edge(new_label_flip_node, new_fix_diff_filter_node, arg_index=0)
        new_dag.add_edge(new_label_flip_indices_node, new_fix_diff_filter_node, arg_index=1)

        new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_fix_diff_filter_node, new_predict_node, arg_index=0)

        new_fix_predict_diff_update_node = DagNode(singleton.get_next_op_id(),
                                                   BasicCodeLocation("Label Errors", None),
                                                   OperatorContext(OperatorType.SELECTION, None),
                                                   DagNodeDetails(
                                                       "Merge fixing diff with old predictions",
                                                       None),
                                                   None,
                                                   LabelErrors.update_prediction_diff)
        new_dag.add_edge(predict_operators[0], new_fix_predict_diff_update_node, arg_index=0)
        new_dag.add_edge(new_predict_node, new_fix_predict_diff_update_node, arg_index=1)
        new_dag.add_edge(new_label_flip_indices_node, new_fix_predict_diff_update_node, arg_index=2)

        LabelErrors.add_new_score_and_score_extraction_nodes(new_dag, new_fix_predict_diff_update_node, score_operators,
                                                             test_labels_operators, "label-errors-flip-retrain")
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

    @staticmethod
    @njit(fastmath=True, parallel=True, cache=True)
    def _compute_shapley_values(X_train, y_train, X_test, y_test, K=1):
        # pylint: disable=invalid-name,too-many-locals
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
    def shapley_top_k_func_llm(rag_join_result, train_labels_before_dict, encoded_test_data, encoded_test_labels,
                               train_fraction_to_consider, test_fraction_to_consider, cleaning_batch_size,
                               label_encoding_op, only_consider_negative_shapley_values):
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

        shapley_values = LabelErrors._compute_shapley_values(train_data_sample, numpy.squeeze(train_label_sample),
                                                             test_data_sample, numpy.squeeze(test_label_sample))
        df_with_id_and_shapley_value = pandas.DataFrame(
            {"train_id": train_indices_to_consider, "shapley_value": shapley_values})

        rows_to_fix = df_with_id_and_shapley_value.nsmallest(cleaning_batch_size, "shapley_value")
        if only_consider_negative_shapley_values:
            rows_to_fix = rows_to_fix[rows_to_fix["shapley_value"] <= 0.]
        return rows_to_fix

    @staticmethod
    def shapley_top_k_func_ml(encoded_train_data, encoded_train_labels, encoded_test_data, encoded_test_labels,
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
    def get_rows_to_flip_llm(rag_join_result, shapley_result):
        retrieval_index = rag_join_result[6]
        changed_df = shapley_result[['train_id']]
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
    def label_flip_processing_func_llm(rag_join_result, encoded_train_labels, shapley_result, all_predictions_to_rerun,
                                       inputs):
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

        # TODO: Should we propagate provenance here? Might be important for explanations later
        new_rag_join_result = (rag_join_result[0], rag_join_result[1], new_rag_join_text_result, rag_join_result[3],
                               None, rag_join_result[5], new_retrieval_index, rag_join_result[7])
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
                                                 test_labels_operators, label_prefix):
        for score_index, score_operator in enumerate(score_operators):
            new_score_node = copy_node_with_new_id(singleton, score_operator)
            new_dag.add_edge(new_predict_node, new_score_node, arg_index=0)
            new_dag.add_edge(test_labels_operators[0], new_score_node, arg_index=1)
            parents = get_sorted_parent_nodes(new_dag, score_operator)[2:]
            for parent_index, parent in enumerate(parents):
                # TODO: There might be shadow pipeline edge cases where this does not work without further work
                new_dag.add_edge(parent, new_score_node, arg_index=parent_index + 2)

            retrain_extraction_node = get_intermediate_extraction_node(singleton, new_score_node,
                                                                       f"{label_prefix}-{score_index}")
            new_dag.add_edge(new_score_node, retrain_extraction_node, arg_index=0)

    @staticmethod
    def get_proxy_model_node(singleton, old_estimator_node):
        model_function = partial(SGDClassifier, loss='log_loss', max_iter=30, n_jobs=1)
        new_processing_func = partial(LabelErrors.fit_model_variant, make_classifier_func=model_function)
        new_description = "Fast proxy model"
        new_estimator_node = DagNode(singleton.get_next_op_id(),
                                     old_estimator_node.code_location,
                                     old_estimator_node.operator_info,
                                     DagNodeDetails(new_description, old_estimator_node.details.columns,
                                                    old_estimator_node.details.optimizer_info),
                                     old_estimator_node.optional_code_info,
                                     new_processing_func)
        return new_estimator_node

    @staticmethod
    def fit_model_variant(train_data, train_labels, make_classifier_func):
        """Create the classifier and fit it"""
        estimator = make_classifier_func()
        estimator.fit(train_data, train_labels)
        return estimator

    @staticmethod
    def update_prediction_diff(old_predictions, prediction_diff, prediction_index):
        updated_predictions = numpy.array(old_predictions.copy())
        updated_predictions[prediction_index] = prediction_diff
        return updated_predictions

    @staticmethod
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
