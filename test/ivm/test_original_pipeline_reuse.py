"""
Tests whether the fluent API works
"""
import os
from inspect import cleandoc

from testfixtures import compare

from mlidea import PipelineAnalyzer
from mlidea.utils import get_project_root

DATABASE_PATH_FUNC_TRANSFORMER = f"{str(get_project_root())}/test/offline/.function_transformer_cache.db"

# TODO: Differentiate more between cases: operator replacement, operator deletion, operator addition, transformer
#  change or not. However, for the end-to-end pipelines we already do this, but we need to do this in more detail
#  when working on the IVM

def test_changed_pipeline_code_regex_change_or(tmpdir):
    """
    Tests whether the Data Cleaning analysis works for a very simple pipeline with a DecisionTree score
    """

    test_code_before = cleandoc("""
            import pandas as pd

            pd_series = pd.Series(['aa', 'b', 'ccc', ''], name='A')
            regex = r"^(a|c)*$"
            mask1 = pd_series.str.contains(regex, regex=True)
            mask2 = pd.Series([True, False, False, True], name='B')
            mask3 = mask1 & mask2
            """)
    test_code_after = cleandoc("""
            import pandas as pd

            pd_series = pd.Series(['aa', 'b', 'ccc', ''], name='A')
            regex = r"^(b|c)*$"
            mask1 = pd_series.str.contains(regex, regex=True)
            mask2 = pd.Series([True, False, False, True], name='B')
            mask3 = mask1 & mask2
            """)

    before, after = run_code_before_and_after(test_code_after, test_code_before, tmpdir)
    all_nodes_before = {node.node_id for node in list(before.original_dag.nodes)}
    all_nodes_after = {node.node_id for node in list(after.original_dag.nodes)}

    compare(expected={0, 1, 2, 3}, actual=all_nodes_before)
    compare(expected={0, 2, 4, 5}, actual=all_nodes_after)

    reuse_info = after.dag_extraction_info.reuse_info
    assert len(reuse_info.operator_reexecuted) == 1
    assert len(reuse_info.operator_replacement) == 1


def run_code_before_and_after(test_code_after, test_code_before, tmpdir):
    analysis_result_before = PipelineAnalyzer \
        .on_pipeline_from_string(test_code_before) \
        .execute()
    analysis_result_after = PipelineAnalyzer \
        .on_changed_pipeline_from_string(analysis_result_before.dag_extraction_info, test_code_after) \
        .execute()
    analysis_result_before.save_original_dag_to_path(os.path.join(str(tmpdir), "orig-old"))
    analysis_result_after.save_original_dag_to_path(os.path.join(str(tmpdir), "orig-new"))
    return analysis_result_before, analysis_result_after
