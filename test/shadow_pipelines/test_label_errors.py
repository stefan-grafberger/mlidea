from inspect import cleandoc

from mlidea import PipelineAnalyzer
from shadow_pipelines._label_errors import LabelErrors
from testing._testing_helper_utils import visualize_dags_shadow_pipelines


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

    label_errors = LabelErrors()
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_shadow_pipeline(label_errors) \
        .execute()

    report = analysis_result.shadow_pipelines_to_result_reports[label_errors]
    # assert report.shape == (4, 2)
    assert "the pipeline metric was" in report

    visualize_dags_shadow_pipelines(analysis_result, tmpdir)
