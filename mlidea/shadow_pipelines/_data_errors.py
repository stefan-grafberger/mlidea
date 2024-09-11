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
from numba import prange, njit
from scipy.sparse import csr_matrix

from mlidea.execution._pipeline_executor import singleton
from mlidea.analysis._analysis_utils import find_nodes_by_type
from mlidea import OperatorType, DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline
from mlidea.shadow_pipelines._utils import get_intermediate_extraction_node, copy_node_with_new_id, \
    get_sorted_parent_nodes, find_train_or_test_pipeline_part_end, get_typo_adder
from mlidea.monkeypatching._patch_langchain import RunnableSequencePatching


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

    def __init__(self, corruption_fraction=.1):
        self._corruption_fraction = corruption_fraction
        self._shadow_pipeline_id = (corruption_fraction,)
        self.score_operator_count = 0

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

            extraction_node = get_intermediate_extraction_node(singleton, new_corruption_node,
                                                               "data-errors-corruption")
            new_dag.add_edge(new_corruption_node, extraction_node, arg_index=0)

            new_corruption_diff_node = DagNode(singleton.get_next_op_id(),
                                               BasicCodeLocation("Data Errors", None),
                                               OperatorContext(OperatorType.GROUP_BY_AGG, None),
                                               DagNodeDetails(
                                                   f"Detect changed indices", None),
                                               None,
                                               DataErrorRobustness.corrupt_data_diff_detection)
            new_dag.add_edge(data_parent, new_corruption_diff_node, arg_index=0)
            new_dag.add_edge(new_corruption_node, new_corruption_diff_node, arg_index=1)

            extraction_node = get_intermediate_extraction_node(singleton, new_corruption_diff_node,
                                                               "data-errors-corruption-diff")
            new_dag.add_edge(new_corruption_diff_node, extraction_node, arg_index=0)

            #
            # new_dag.add_edge(train_labels_operators[0], new_shapley_node, arg_index=1)
            # new_dag.add_edge(test_data_operators[0], new_shapley_node, arg_index=2)
            # new_dag.add_edge(test_labels_operators[0], new_shapley_node, arg_index=3)
            # extraction_node = get_intermediate_extraction_node(singleton, new_shapley_node,
            #                                                    "label-errors-shapley-values")
            # new_dag.add_edge(new_shapley_node, extraction_node, arg_index=0)
            #
            # new_label_flip_node = DagNode(singleton.get_next_op_id(),
            #                               BasicCodeLocation("Label Errors", None),
            #                               OperatorContext(OperatorType.PROJECTION, None),
            #                               DagNodeDetails(
            #                                   f"Flip {self._cleaning_batch_size} most likely incorrect labels", None),
            #                               None,
            #                               DataErrorRobustness.label_flip_processing_func_ml)
            # new_dag.add_edge(train_labels_operators[0], new_label_flip_node, arg_index=0)
            # new_dag.add_edge(extraction_node, new_label_flip_node, arg_index=1)
            # new_model_node = copy_node_with_new_id(singleton, model_operators[0])
            # new_dag.add_edge(train_data_operators[0], new_model_node, arg_index=0)
            # new_dag.add_edge(new_label_flip_node, new_model_node, arg_index=1)
            # new_predict_node = copy_node_with_new_id(singleton, predict_operators[0])
            # new_dag.add_edge(new_model_node, new_predict_node, arg_index=0)
            # new_dag.add_edge(test_data_operators[0], new_predict_node, arg_index=1)
            #
            # DataErrorRobustness.add_new_score_and_score_extraction_nodes(new_dag, new_predict_node, score_operators,
            #                                                              test_labels_operators)
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
        corrupted_df = extracted_plan_results["data-errors-corruption"]
        corruption_index = extracted_plan_results["data-errors-corruption-diff"]
        corrupted_sample = corrupted_df.reset_index(drop=True).iloc[corruption_index, :].head(20)
        score_after_corruption = "todo"
        score_after_fixing = "todo"
        fixed_sample = None
        # flip_result = []
        # for score_index in range(self.score_operator_count):
        #     flip_result.append(extracted_plan_results[f"label-errors-flip-retrain-{score_index}"])
        return (f"The original result was {orig_result}. After corrupting {self._corruption_fraction} of rows, "
                f"the pipeline metric was {score_after_corruption}, indicating robustness problems. A sample of the corrupted "
                f"rows: {str(corrupted_sample)}. After adding a fix method, the pipeline metric was "
                f"{score_after_fixing}. A sample of the fixed rows: {str(fixed_sample)}")

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
            for column in input_df.columns:
                corrupted_result = get_typo_adder(column).fit_transform(input_df)
        elif data_type == DataType.CAT:
            for column in input_df.columns:
                # TODO: Broken Characters is pretty slow, maybe do not use it
                """Corrupt broken characters that may be in a pandas df, but may also be in a different format"""
                if isinstance(input_df, pandas.DataFrame):
                    corrupted_result = BrokenCharacters(column=column, fraction=corruption_fraction).transform(input_df)
                elif isinstance(input_df, list):
                    pandas_df = pandas.DataFrame({column: input_df})
                    corrupted_result = BrokenCharacters(column=column, fraction=corruption_fraction).transform(
                        pandas_df)
                else:
                    pandas_df = pandas.DataFrame(input_df)
                    corrupted_result = BrokenCharacters(column=column, fraction=corruption_fraction).transform(
                        pandas_df)
        elif data_type == DataType.NUM:
            for column in input_df.columns:
                corrupted_result = Scaling(column=column, fraction=corruption_fraction).transform(input_df)
        else:
            raise NotImplementedError(f"TODO: Add support for datatype {DataType.value}!")
        return corrupted_result

    @staticmethod
    def corrupt_data_diff_detection(input_df, corrupted_result):
        corrupt_diff_mask = corrupted_result != input_df
        changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
        return changed_indices_corrupt

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
        return data_parent_and_data_type
