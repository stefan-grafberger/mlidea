"""
Tests whether the monkey patching works for all patched sklearn methods
"""
import os
from functools import partial
from types import FunctionType

import networkx
import numpy
from testfixtures import compare, Comparison, RangeComparison

from mlidea.analysis._operator_impact import OperatorImpact
from mlidea.analysis._data_cleaning import DataCleaning, ErrorType
from mlidea.analysis._permutation_feature_importance import PermutationFeatureImportance
from mlidea import OperatorType, OperatorContext, FunctionInfo, PipelineAnalyzer
from mlidea.analysis._data_corruption import DataCorruption, CorruptionType
from mlidea.execution import _pipeline_executor
from mlidea.execution._dag_executor import DagExecutor
from mlidea.execution._pipeline_executor import singleton
from mlidea.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo
from mlidea.monkeypatching._mlinspect_ndarray import MlinspectList
from mlidea.testing._testing_helper_utils import get_llm_rag_mini_example_code


def test_binary_rag_classification(tmpdir):
    """
    Tests whether the monkey patching of langchain pipelines works
    """
    # pylint: disable=too-many-locals,too-many-statements
    test_code = get_llm_rag_mini_example_code()

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_0 = DagNode(0, BasicCodeLocation('<string-source>', 12),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame'),
                                         Comparison(dict)),
                         DagNodeDetails(None, ['text', 'label'],
                                        OptimizerInfo(RangeComparison(0, 10000), (4, 2), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(12, 5, 13, 56),
                                          'pd.DataFrame({\'text\': ["positive", "positive", "negative", "negative"], \n'
                                          '                   \'label\': [\'no\', \'no\', \'yes\', \'yes\']})'),
                         Comparison(partial))
    expected_1 = DagNode(1, BasicCodeLocation('<string-source>', 15),
                         OperatorContext(OperatorType.PROJECTION, FunctionInfo('pandas.core.frame', '__getitem__'),
                                         Comparison(dict)),
                         DagNodeDetails("to ['text']", ['text'],
                                        OptimizerInfo(RangeComparison(0, 10000), (4, 1), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(15, 38, 15, 48), "df['text']"), Comparison(partial))
    expected_dag.add_edge(expected_0, expected_1, arg_index=0)
    expected_2 = DagNode(2, BasicCodeLocation('<string-source>', 15),
                         OperatorContext(OperatorType.PROJECTION, FunctionInfo('pandas.core.series.Series', 'to_list'),
                                         Comparison(dict)),
                         DagNodeDetails('list conversion', ['text'],
                                        OptimizerInfo(RangeComparison(0, 10000), (4, 1), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(15, 38, 15, 58), "df['text'].to_list()"),
                         Comparison(partial))
    expected_dag.add_edge(expected_1, expected_2, arg_index=0)
    expected_3 = DagNode(3, BasicCodeLocation('<string-source>', 15),
                         OperatorContext(OperatorType.PROJECTION, FunctionInfo('pandas.core.frame', '__getitem__'),
                                         Comparison(dict)),
                         DagNodeDetails("to ['label']", ['label'],
                                        OptimizerInfo(RangeComparison(0, 10000), (4, 1), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(15, 70, 15, 83), "df[['label']]"), Comparison(partial))
    expected_dag.add_edge(expected_0, expected_3, arg_index=0)
    expected_4 = DagNode(4, BasicCodeLocation('<string-source>', 15),
                         OperatorContext(OperatorType.PROJECTION, FunctionInfo('pandas.core.frame', 'to_dict'),
                                         Comparison(dict)),
                         DagNodeDetails('dict conversion', ['label'],
                                        OptimizerInfo(RangeComparison(0, 10000), (4, 1), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(15, 70, 15, 102), "df[['label']].to_dict('records')"),
                         Comparison(partial))
    expected_dag.add_edge(expected_3, expected_4, arg_index=0)
    expected_5 = DagNode(5, BasicCodeLocation('<string-source>', 15),
                         OperatorContext(OperatorType.TRAIN_DATA,
                                         FunctionInfo('langchain_community.vectorstores.Chroma', 'from_texts'),
                                         Comparison(dict)),
                         DagNodeDetails(None, ['text'],
                                        OptimizerInfo(RangeComparison(0, 10000), (4, 1), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(15, 14, 16, 101),
                                          "Chroma.from_texts(texts=df['text'].to_list(), metadatas=df[['label']]."
                                          "to_dict('records'),\n"
                                          "                embedding=HuggingFaceEmbeddings(model_name="
                                          "'sentence-transformers/all-MiniLM-L6-v2'))"),
                         Comparison(FunctionType))
    expected_dag.add_edge(expected_2, expected_5, arg_index=0)
    expected_6 = DagNode(6, BasicCodeLocation('<string-source>', 15),
                         OperatorContext(OperatorType.TRAIN_LABELS,
                                         FunctionInfo('langchain_community.vectorstores.Chroma', 'from_texts'),
                                         Comparison(dict)),
                         DagNodeDetails(None, ['label'],
                                        OptimizerInfo(RangeComparison(0, 10000), (4, 1), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(15, 14, 16, 101),
                                          "Chroma.from_texts(texts=df['text'].to_list(), metadatas=df[['label']]."
                                          "to_dict('records'),\n"
                                          "                embedding=HuggingFaceEmbeddings(model_name="
                                          "'sentence-transformers/all-MiniLM-L6-v2'))"),
                         Comparison(FunctionType))
    expected_dag.add_edge(expected_4, expected_6, arg_index=0)
    expected_7 = DagNode(7, BasicCodeLocation('<string-source>', 15),
                         OperatorContext(OperatorType.CONCATENATION,
                                         FunctionInfo('langchain_community.vectorstores.Chroma', 'from_texts'),
                                         Comparison(dict)),
                         DagNodeDetails(None, ['text', 'label'],
                                        OptimizerInfo(RangeComparison(0, 10000), (4, 2), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(15, 14, 16, 101),
                                          "Chroma.from_texts(texts=df['text'].to_list(), metadatas=df[['label']]."
                                          "to_dict('records'),\n"
                                          "                embedding=HuggingFaceEmbeddings(model_name="
                                          "'sentence-transformers/all-MiniLM-L6-v2'))"),
                         Comparison(FunctionType))
    expected_dag.add_edge(expected_5, expected_7, arg_index=0)
    expected_dag.add_edge(expected_6, expected_7, arg_index=1)
    expected_8 = DagNode(8, BasicCodeLocation('<string-source>', 20),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame'),
                                         Comparison(dict)),
                         DagNodeDetails(None, ['text', 'label'],
                                        OptimizerInfo(RangeComparison(0, 10000), (2, 2), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(20, 7, 20, 70),
                                          'pd.DataFrame({\'text\': ["pos", "neg."], \'label\': [\'no\', \'yes\']})'),
                         Comparison(partial))
    expected_9 = DagNode(9, BasicCodeLocation('<string-source>', 21),
                         OperatorContext(OperatorType.PROJECTION, FunctionInfo('pandas.core.frame', '__getitem__'),
                                         Comparison(dict)),
                         DagNodeDetails("to ['text']", ['text'],
                                        OptimizerInfo(RangeComparison(0, 10000), (2, 1), RangeComparison(0, 10000))),
                         OptionalCodeInfo(CodeReference(21, 53, 21, 65), "test['text']"), Comparison(partial))
    expected_dag.add_edge(expected_8, expected_9, arg_index=0)
    expected_10 = DagNode(10, BasicCodeLocation('<string-source>', 21),
                          OperatorContext(OperatorType.PROJECTION,
                                          FunctionInfo('pandas.core.series.Series', 'to_list'), Comparison(dict)),
                          DagNodeDetails('list conversion', ['text'],
                                         OptimizerInfo(RangeComparison(0, 10000), (2, 1), RangeComparison(0, 10000))),
                          OptionalCodeInfo(CodeReference(21, 53, 21, 75), "test['text'].to_list()"),
                          Comparison(partial))
    expected_dag.add_edge(expected_9, expected_10, arg_index=0)
    expected_11 = DagNode(11, BasicCodeLocation('<string-source>', 21), OperatorContext(OperatorType.TEST_DATA,
                                                                                        FunctionInfo(
                                                                                            'langchain_community.vectorstores.Chroma',
                                                                                            'from_texts'),
                                                                                        Comparison(dict)),
                          DagNodeDetails(None, ['text'],
                                         OptimizerInfo(RangeComparison(0, 10000), (2, 1), RangeComparison(0, 10000))),
                          OptionalCodeInfo(CodeReference(21, 14, 21, 83),
                                           "wait_llm_call(partial(rag_chain.batch, test['text'].to_list()), test)"),
                          Comparison(FunctionType), make_classifier_func=None)
    expected_dag.add_edge(expected_10, expected_11, arg_index=0)
    expected_12 = DagNode(12, BasicCodeLocation('<string-source>', 15), OperatorContext(OperatorType.RAG_JOIN,
                                                                                        FunctionInfo(
                                                                                            'langchain_community.vectorstores.Chroma',
                                                                                            'from_texts'),
                                                                                        Comparison(dict)),
                          DagNodeDetails('Embedding similarity join', ['text', 'label'],
                                         OptimizerInfo(RangeComparison(0, 10000), None, RangeComparison(0, 10000))),
                          OptionalCodeInfo(CodeReference(15, 14, 16, 101),
                                           "Chroma.from_texts(texts=df['text'].to_list(), "
                                           "metadatas=df[['label']].to_dict('records'),\n"
                                           "                embedding=HuggingFaceEmbeddings(model_name="
                                           "'sentence-transformers/all-MiniLM-L6-v2'))"),
                          Comparison(partial), make_classifier_func=None)
    expected_dag.add_edge(expected_7, expected_12, arg_index=0)
    expected_dag.add_edge(expected_11, expected_12, arg_index=1)
    expected_13 = DagNode(13, BasicCodeLocation('<string-source>', 21),
                          OperatorContext(OperatorType.PREDICT, FunctionInfo('langchain_core.runnables.base', 'batch'),
                                          Comparison(dict)),
                          DagNodeDetails('LLM', [],
                                         OptimizerInfo(RangeComparison(0, 10000), (2, 1), RangeComparison(0, 10000))),
                          OptionalCodeInfo(CodeReference(21, 14, 21, 83),
                                           "wait_llm_call(partial(rag_chain.batch, test['text'].to_list()), test)"),
                          Comparison(partial))
    expected_dag.add_edge(expected_12, expected_13, arg_index=0)
    expected_14 = DagNode(14, BasicCodeLocation('<string-source>', 22),
                          OperatorContext(OperatorType.PROJECTION, FunctionInfo('pandas.core.frame', '__getitem__'),
                                          Comparison(dict)),
                          DagNodeDetails("to ['label']", ['label'],
                                         OptimizerInfo(RangeComparison(0, 10000), (2, 1), RangeComparison(0, 10000))),
                          OptionalCodeInfo(CodeReference(22, 34, 22, 47), "test['label']"), Comparison(partial))
    expected_dag.add_edge(expected_8, expected_14, arg_index=0)
    expected_15 = DagNode(15, BasicCodeLocation('<string-source>', 22),
                          OperatorContext(OperatorType.PROJECTION_MODIFY, FunctionInfo('sklearn.preprocessing._label',
                                                                                       'label_binarize'),
                                          Comparison(dict)),
                          DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array'],
                                         OptimizerInfo(RangeComparison(0, 10000), (2, 1), RangeComparison(0, 10000))),
                          OptionalCodeInfo(CodeReference(22, 19, 22, 71),
                                           "label_binarize(test['label'], classes=['no', 'yes'])"),
                          Comparison(partial))
    expected_dag.add_edge(expected_14, expected_15, arg_index=0)
    expected_16 = DagNode(16, BasicCodeLocation('<string-source>', 23),
                          OperatorContext(OperatorType.TEST_LABELS, FunctionInfo('sklearn.metrics._classification',
                                                                                 'accuracy_score'), Comparison(dict)),
                          DagNodeDetails(None, ['array'],
                                         OptimizerInfo(RangeComparison(0, 10000), (2, 1), RangeComparison(0, 10000))),
                          OptionalCodeInfo(CodeReference(23, 11, 23, 56),
                                           'accuracy_score(y_test_binarized, y_predicted)'), Comparison(FunctionType))
    expected_dag.add_edge(expected_15, expected_16, arg_index=0)
    expected_17 = DagNode(17, BasicCodeLocation('<string-source>', 23),
                          OperatorContext(OperatorType.SCORE, FunctionInfo('sklearn.metrics._classification',
                                                                           'accuracy_score'), Comparison(dict)),
                          DagNodeDetails('accuracy_score', [],
                                         OptimizerInfo(RangeComparison(0, 10000), (1, 1), RangeComparison(0, 10000))),
                          OptionalCodeInfo(CodeReference(23, 11, 23, 56),
                                           'accuracy_score(y_test_binarized, y_predicted)'), Comparison(FunctionType))
    expected_dag.add_edge(expected_13, expected_17, arg_index=0)
    expected_dag.add_edge(expected_16, expected_17, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    vectorstore_creation_node = list(inspector_result.original_dag.nodes)[7]
    vectorstore_join_node = list(inspector_result.original_dag.nodes)[12]
    llm_node = list(inspector_result.original_dag.nodes)[13]

    vectorstore_texts = MlinspectList(["positive", "positive", "negative", "negative"])
    vectorstore_texts._mlinspect_provenance = {"11_0": numpy.array(range(4))}
    # TODO: Also track label provenance here? But for now not necessary
    vectorstore_labels = [{"label": "yes"}, {"label": "yes"}, {"label": "no"}, {"label": "no"}]
    concat_result = vectorstore_creation_node.processing_func(vectorstore_texts, vectorstore_labels)
    test_data = MlinspectList(["pos.", "pos."])
    test_data._mlinspect_provenance = {"13_0": numpy.array(range(2))}
    rag_result = vectorstore_join_node.processing_func(concat_result, test_data)
    assert len(rag_result[4].items()) == 5
    llm_result = llm_node.processing_func(rag_result)
    assert len(llm_result._mlinspect_provenance.items()) == 5

    expected = numpy.array([1, 1]).reshape(-1, 1)
    assert numpy.allclose(llm_result, expected, atol=1)

    # Also test if the DAG is fully re-executable
    DagExecutor(singleton).execute(inspector_result.original_dag)

    # Test if the what-if analyses work for this simple LLM+RAG pipeline
    data_corruption = DataCorruption([('text', CorruptionType.BROKEN_CHARACTERS)],
                                     also_corrupt_train=True)
    data_cleaning = DataCleaning({'text': ErrorType.CAT_MISSING_VALUES})

    analysis_result = PipelineAnalyzer \
        .on_previously_extracted_pipeline(inspector_result.dag_extraction_info) \
        .add_what_if_analysis(data_corruption) \
        .add_what_if_analysis(data_cleaning) \
        .add_what_if_analysis(PermutationFeatureImportance()) \
        .add_what_if_analysis(OperatorImpact(True, True)) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (4, 4)

    report = analysis_result.analysis_to_result_reports[PermutationFeatureImportance()]
    assert report.shape == (2, 2)

    report = analysis_result.analysis_to_result_reports[OperatorImpact(True, True)]
    assert report.shape == (1, 5)

    report = analysis_result.analysis_to_result_reports[data_cleaning]
    assert report.shape == (4, 4)

    analysis_result.save_original_dag_to_path(os.path.join(str(tmpdir), "orig-dag"))
    analysis_result.save_what_if_dags_to_path(os.path.join(str(tmpdir), "whatif-dags"))
    analysis_result.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir), "opt-dag"))
