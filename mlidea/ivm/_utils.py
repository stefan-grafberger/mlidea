from copy import copy

import duckdb
import numpy
import pandas
from langchain_core.runnables import RunnableSequence

from mlidea.monkeypatching._mlinspect_ndarray import MlinspectNdarray, MlinspectList, MlinspectDict, MlinspectTuple


def update_prediction_diff(old_predictions, prediction_diff, prediction_index):
    updated_predictions = numpy.array(old_predictions.copy())
    updated_predictions[prediction_index] = prediction_diff
    return updated_predictions


def _get_rag_join_results_to_rerun(rag_join_result, change_indices):
    retrieval_index = rag_join_result[6]
    changed_df = pandas.DataFrame({'train_id': change_indices})  # pylint: disable=unused-variable
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


def rag_join_update(rag_join_result, inputs, vectorstore, all_predictions_to_rerun):
    # TODO: Should we propagate provenance here? Might be important for explanations later

    # Rerun the RAG join on the diff
    diff_inputs = list(numpy.array(inputs)[all_predictions_to_rerun])
    diff_rag_result, diff_retrieval_index = RunnableSequence.execute_rag_join_diff(
        rag_join_result[7], diff_inputs, vectorstore)

    # Prepare the usual RAG join output
    new_rag_join_text_result = numpy.array(rag_join_result[2])
    new_rag_join_text_result[all_predictions_to_rerun] = diff_rag_result
    new_rag_join_text_result_list = list(new_rag_join_text_result)

    new_retrieval_index = rag_join_result[6].copy()
    new_retrieval_index[all_predictions_to_rerun, :] = diff_retrieval_index

    new_rag_join_result = (rag_join_result[0], rag_join_result[1], new_rag_join_text_result_list,
                           rag_join_result[3], None, rag_join_result[5], new_retrieval_index, rag_join_result[7])
    return new_rag_join_result



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
        if input_df.shape == corrupted_result.shape:
            corrupt_diff_mask = numpy.any(corrupted_result != input_df, axis=1)
        else:
            # FIXME: This case can happen if the shape mismatches. Ideally, in the case of filters, we know that
            #  a filter happened and we immediately find out which rows are added/removed and do not need to run this
            #  function at all. Otherwise, we would have to try to understand the changes here. But maybe that
            #  is also okay, e.g., with a DuckDB join. But then we should really only run this function if the
            #  runtime of the operation we do not want to fully execute is sufficiently high.
            corrupt_diff_mask = numpy.ones(corrupted_result.shape, dtype=bool)
    else:
        corrupt_diff_mask = corrupted_result != input_df
    return corrupt_diff_mask


def fix_data_mask_to_indices(corrupt_diff_mask):
    changed_indices_corrupt = numpy.where(corrupt_diff_mask)[0]
    return changed_indices_corrupt


def wrap_in_mlinspect_array_if_necessary(df_object):
    """
    Makes sure annotations can be stored in a df_object. For example, numpy arrays need a wrapper for this.
    """
    prov = None
    if hasattr(df_object, "_mlinspect_provenance"):
        # Not really sure yet why this is necessary, we should clean this up in the future
        prov = df_object._mlinspect_provenance
    if isinstance(df_object, numpy.ndarray) and not isinstance(df_object, MlinspectNdarray):
        df_object = MlinspectNdarray(df_object)
    elif isinstance(df_object, list):
        df_object = MlinspectList(df_object)
    elif isinstance(df_object, dict):
        df_object = MlinspectDict(df_object)
    elif isinstance(df_object, tuple):
        df_object = MlinspectTuple(df_object)
    if prov is not None:
        df_object._mlinspect_provenance = prov
    return df_object


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


def changed_data_diff_detection(input_df, corrupted_result):
    corrupt_diff_mask = fix_data_diff_detection_mask_only(input_df, corrupted_result)
    changed_indices_corrupt = fix_data_mask_to_indices(corrupt_diff_mask)
    return changed_indices_corrupt
