from inspect import cleandoc

from example_pipelines import HEALTHCARE_PY, ANHEDONIA_ML_PY, ANHEDONIA_LLM_PY, ADULT_COMPLEX_PY, \
    COMPAS_PY
from example_pipelines.healthcare import custom_monkeypatching
from mlidea import PipelineAnalyzer
from mlidea.testing._testing_helper_utils import visualize_dags_shadow_pipelines, get_llm_rag_mini_example_code
from mlidea.shadow_pipelines._data_errors import DataErrorRobustness


def test_label_errors_mini_example_with_transformer_processing_multiple_columns(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    test_code = cleandoc("""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np

        df = pd.DataFrame({'A': [0, 0, 0, 0], 'B': [0, 1, 3, 4], 'target': ['no', 'no', 'yes', 'yes']})

        standard_scaler = StandardScaler()
        train = standard_scaler.fit_transform(df[['A', 'B']])
        target = label_binarize(df['target'], classes=['no', 'yes'])

        clf = DecisionTreeClassifier()
        clf = clf.fit(train, target)

        test_df = pd.DataFrame({'A': [0, 0, 0, 0], 'B':  [4, 3, 4, 3], 
            'sensitive': ["cat_a", "cat_b", "cat_a", "cat_b"], 'target': ['yes', 'yes', 'yes', 'yes']})
        test_data = standard_scaler.transform(test_df[['A', 'B']])
        test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
        test_score = clf.score(test_data, test_labels)
        assert test_score == 1.0
        """)

    data_errors = DataErrorRobustness(corruption_fraction=0.5)
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_shadow_pipeline(data_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[data_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)


def test_label_errors_mini_example_llm_rag(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    test_code = get_llm_rag_mini_example_code()

    data_errors = DataErrorRobustness()
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_shadow_pipeline(data_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[data_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)


def test_label_errors_compas(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    data_errors = DataErrorRobustness()
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_shadow_pipeline(data_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[data_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)


def test_label_errors_anhedonia_ml(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    data_errors = DataErrorRobustness()
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ANHEDONIA_ML_PY) \
        .add_shadow_pipeline(data_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[data_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)


def test_label_errors_anhedonia_llm(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    data_errors = DataErrorRobustness()
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ANHEDONIA_LLM_PY) \
        .add_shadow_pipeline(data_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[data_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)


def test_label_errors_adult_complex(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    data_errors = DataErrorRobustness()
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ADULT_COMPLEX_PY) \
        .add_shadow_pipeline(data_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[data_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)


def test_label_errors_healthcare(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    data_errors = DataErrorRobustness()
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_shadow_pipeline(data_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[data_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report
    # TODO: Does not work yet because of multiple score functions

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)


def test_label_errors_healthcare_fraction(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    data_errors = DataErrorRobustness(corruption_fraction=0.3)
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_shadow_pipeline(data_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[data_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report
    # TODO: Does not work yet because of multiple score functions

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)
