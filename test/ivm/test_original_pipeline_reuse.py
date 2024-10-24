"""
Tests whether the fluent API works
"""
import os

from example_pipelines import ANHEDONIA_LLM_MODIFIED_PY, \
    ANHEDONIA_LLM_PY, ADULT_COMPLEX_PY, ADULT_COMPLEX_MODIFIED_PY
from mlidea import PipelineAnalyzer
from mlidea.utils import get_project_root

DATABASE_PATH_FUNC_TRANSFORMER = f"{str(get_project_root())}/test/offline/.function_transformer_cache.db"

# TODO: Start with Anhedonia and projections that don't affect relevant columns
#  Maybe we have something like that in Adult Complex

def test_changed_pipeline_code_anhedonia(tmpdir):
    """
    Tests whether the Data Cleaning analysis works for a very simple pipeline with a DecisionTree score
    """
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ANHEDONIA_LLM_PY) \
        .execute()

    analysis_result.save_original_dag_to_path(os.path.join(str(tmpdir), "orig-old"))

    analysis_result = PipelineAnalyzer \
        .on_changed_pipeline_from_py_file(analysis_result.dag_extraction_info, ANHEDONIA_LLM_MODIFIED_PY) \
        .execute()
    analysis_result.save_original_dag_to_path(os.path.join(str(tmpdir), "orig-new"))

    # Somehow assert that the output only contains the DAG nodes it is supposed to contain
