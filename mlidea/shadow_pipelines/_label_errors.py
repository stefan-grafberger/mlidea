# 1. Mislabel: in mlwhatif, two approaches, shapley and cleanlab. for mlidea workshop paper we only used shapley.
# in general, for LLM+RAG, we need the embeddings, that we don't have specifically in the DAG right now.
# Do we need to update the DAG? Or use some hack like letting the RAG join output the embeddings next to the text?
# but might have a big of added performance overhead. then, conditional operator depending on how many mislabels
# found. but maybe not that problematic here. but maybe for this we do want to use the provenance since the labeling
# might not be the final step in the data preprocessing and there might be filte
from functools import partial

import duckdb
import networkx
import numpy
import pandas
from numba import prange, njit

from mlidea.execution._pipeline_executor import singleton
from mlidea.analysis._analysis_utils import find_nodes_by_type
from mlidea import OperatorType, DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline
from mlidea.shadow_pipelines._utils import get_intermediate_extraction_node, copy_node_with_new_id
from monkeypatching._patch_langchain import call_info_singleton, RunnableSequencePatching


class LabelErrors(ShadowPipeline):
    """
    The Label Error Shadow Pipeline
    """

    def check_rebuilding_necessary(self, extracted_plan_results: dict[str, any]) -> any:
        return False

    def __init__(self, train_fraction_to_consider=1., test_fraction_to_consider=1., proxy_model=False,
                 cleaning_batch_size=20):
        # TODO: We should probably also implement the second proxy version from the workshop paper
        self._train_fraction_to_consider = train_fraction_to_consider
        self._test_fraction_to_consider = test_fraction_to_consider
        self._proxy_model = proxy_model
        self._cleaning_batch_size = cleaning_batch_size
        if proxy_model is True:
            raise NotImplementedError("TODO")
        self._shadow_pipeline_id = (
            train_fraction_to_consider, test_fraction_to_consider, proxy_model, cleaning_batch_size)

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
        if len(predict_operators) != 1 or len(score_operators) != 1 or len(model_operators) != 1 \
                or len(train_data_operators) != 1 or len(train_labels_operators) != 1 \
                or len(test_data_operators) != 1 or len(test_labels_operators) != 1:
            raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                      "pattern!")
        orig_extraction_node = get_intermediate_extraction_node(singleton, score_operators[0], "label-errors-orig")
        new_dag.add_edge(score_operators[0], orig_extraction_node, arg_index=0)

        def shapley_top_k_func(encoded_train_data, encoded_train_labels, encoded_test_data, encoded_test_labels,
                               train_fraction_to_consider, test_fraction_to_consider, cleaning_batch_size):
            indices = numpy.arange(len(encoded_train_labels))
            numpy.random.shuffle(indices)

            num_values_to_typo = int(len(encoded_train_labels) * train_fraction_to_consider)
            train_indices_to_consider = indices[:num_values_to_typo]
            train_data_sample = encoded_train_data[train_indices_to_consider]
            train_label_sample = encoded_train_labels[train_indices_to_consider]

            indices = numpy.arange(len(encoded_test_labels))
            num_values_to_typo = int(len(encoded_test_labels) * test_fraction_to_consider)
            test_indices_to_consider = indices[:num_values_to_typo]
            test_data_sample = encoded_test_data[test_indices_to_consider]
            test_label_sample = encoded_test_labels[test_indices_to_consider]

            shapley_values = LabelErrors._compute_shapley_values(train_data_sample, numpy.squeeze(train_label_sample),
                                                                 test_data_sample, numpy.squeeze(test_label_sample))
            df_with_id_and_shapley_value = pandas.DataFrame(
                {"train_id": train_indices_to_consider, "shapley_value": shapley_values})

            rows_to_fix = df_with_id_and_shapley_value.nsmallest(cleaning_batch_size, "shapley_value")
            return rows_to_fix

        processing_func = partial(shapley_top_k_func, train_fraction_to_consider=self._train_fraction_to_consider,
                                  test_fraction_to_consider=self._test_fraction_to_consider,
                                  cleaning_batch_size=self._cleaning_batch_size)
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

        def label_flip_processing_func(encoded_train_labels, shapley_result):
            unfair_indices = shapley_result['train_id']
            modified_encoded_train_labels = encoded_train_labels.copy()
            modified_encoded_train_labels[unfair_indices, :] = 1 - modified_encoded_train_labels[unfair_indices, :]
            return modified_encoded_train_labels

        new_label_flip_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("Label Errors", None),
                                      OperatorContext(OperatorType.PROJECTION, None),
                                      DagNodeDetails(
                                          f"Flip {self._cleaning_batch_size} most likely incorrect labels", None),
                                      None,
                                      label_flip_processing_func)
        new_dag.add_edge(train_labels_operators[0], new_label_flip_node, arg_index=0)
        new_dag.add_edge(extraction_node, new_label_flip_node, arg_index=1)
        new_model_node = copy_node_with_new_id(singleton, model_operators[0])
        new_dag.add_edge(train_data_operators[0], new_model_node, arg_index=0)
        new_dag.add_edge(new_label_flip_node, new_model_node, arg_index=1)
        new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_model_node, new_predict_node, arg_index=0)
        new_dag.add_edge(test_data_operators[0], new_predict_node, arg_index=1)
        new_score_node = copy_node_with_new_id(singleton, score_operators[0])
        new_dag.add_edge(new_predict_node, new_score_node, arg_index=0)
        new_dag.add_edge(test_labels_operators[0], new_score_node, arg_index=1)
        retrain_extraction_node = get_intermediate_extraction_node(singleton, new_shapley_node,
                                                                   "label-errors-flip-retrain")
        new_dag.add_edge(new_score_node, retrain_extraction_node, arg_index=0)
        return new_dag

    def get_llm_rag_dag(self, dag):
        # FIXME: This won't work yet
        new_dag = dag.copy()

        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        rag_join_operators = find_nodes_by_type(dag, OperatorType.RAG_JOIN)
        train_data_operators = find_nodes_by_type(dag, OperatorType.TRAIN_DATA)
        train_labels_operators = find_nodes_by_type(dag, OperatorType.TRAIN_LABELS)
        test_data_operators = find_nodes_by_type(dag, OperatorType.TEST_DATA)
        test_labels_operators = find_nodes_by_type(dag, OperatorType.TEST_LABELS)

        if len(predict_operators) != 1 or len(score_operators) != 1 or len(rag_join_operators) != 1 \
                or len(train_data_operators) != 1 or len(train_labels_operators) != 1 \
                or len(test_data_operators) != 1 or len(test_labels_operators) != 1:
            raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                      "pattern!")
        label_encoder_operators = list(new_dag.predecessors(test_labels_operators[0]))
        if len(label_encoder_operators) != 1 or "label_binarize" not in label_encoder_operators[0].details.description:
            raise NotImplementedError("Currently, Label Errors only supports pipelines following a very specific "
                                      "pattern!")
        train_labels_dict_conversion = list(new_dag.predecessors(train_labels_operators[0]))[0]
        train_labels_before_dict = list(new_dag.predecessors(train_labels_dict_conversion))[0]

        orig_extraction_node = get_intermediate_extraction_node(singleton, score_operators[0], "label-errors-orig")
        new_dag.add_edge(score_operators[0], orig_extraction_node, arg_index=0)

        def shapley_top_k_func(rag_join_result, train_labels_before_dict, encoded_test_data, encoded_test_labels,
                               train_fraction_to_consider, test_fraction_to_consider, cleaning_batch_size):
            indices = numpy.arange(len(train_labels_before_dict))
            numpy.random.shuffle(indices)

            num_values_to_typo = int(len(train_labels_before_dict) * train_fraction_to_consider)
            train_indices_to_consider = indices[:num_values_to_typo]

            vectorstore = rag_join_result[5]
            train_data_sample = numpy.array(vectorstore.get(
                ids=list(map(str, train_indices_to_consider)), include=["embeddings"])['embeddings'])
            to_label_encode = train_labels_before_dict.iloc[train_indices_to_consider, 0]
            # FIXME: What should we do provenance-wise in shadow pipelines?
            to_label_encode._mlinspect_provenance = None
            train_label_sample = label_encoder_operators[0].processing_func(to_label_encode)

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
            return rows_to_fix

        processing_func = partial(shapley_top_k_func, train_fraction_to_consider=self._train_fraction_to_consider,
                                  test_fraction_to_consider=self._test_fraction_to_consider,
                                  cleaning_batch_size=self._cleaning_batch_size)
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

        def label_flip_processing_func(rag_join_result, encoded_train_labels, shapley_result, inputs):
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

        new_label_flip_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("Label Errors", None),
                                      OperatorContext(OperatorType.PROJECTION, None),
                                      DagNodeDetails(
                                          f"Flip {self._cleaning_batch_size} most likely incorrect labels", None),
                                      None,
                                      label_flip_processing_func)
        new_dag.add_edge(rag_join_operators[0], new_label_flip_node, arg_index=0)
        new_dag.add_edge(train_labels_operators[0], new_label_flip_node, arg_index=1)
        new_dag.add_edge(extraction_node, new_label_flip_node, arg_index=2)
        new_dag.add_edge(test_data_operators[0], new_label_flip_node, arg_index=3)

        new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
        new_dag.add_edge(new_label_flip_node, new_predict_node, arg_index=0)
        new_score_node = copy_node_with_new_id(singleton, score_operators[0])
        new_dag.add_edge(new_predict_node, new_score_node, arg_index=0)
        new_dag.add_edge(test_labels_operators[0], new_score_node, arg_index=1)
        retrain_extraction_node = get_intermediate_extraction_node(singleton, new_shapley_node,
                                                                   "label-errors-flip-retrain")
        new_dag.add_edge(new_score_node, retrain_extraction_node, arg_index=0)
        return new_dag

    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        # result_df = pandas.DataFrame({'todo': []})
        orig_result = extracted_plan_results["label-errors-orig"]
        shapley_values = extracted_plan_results["label-errors-shapley-values"]
        flip_result = extracted_plan_results["label-errors-flip-retrain"]
        return (f"The original result was {orig_result}. After flipping the top {self._cleaning_batch_size} most "
                f"likely incorrect row labels, the pipeline metric was {flip_result}. The shapley values of the "
                f"most likely mislabeled rows: {str(shapley_values)}.")

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
