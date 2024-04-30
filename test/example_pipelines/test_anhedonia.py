"""
Tests whether the healthcare demo works
"""
import ast

from example_pipelines.healthcare import custom_monkeypatching
from example_pipelines import ANHEDONIA_ML_PY, ANHEDONIA_ML_PNG, ANHEDONIA_LLM_PY, ANHEDONIA_LLM_PNG
from mlidea.testing._testing_helper_utils import run_and_assert_all_op_outputs_inspected


def test_ml_py_pipeline_runs():
    """
    Tests whether the pipeline works without instrumentation
    """
    with open(ANHEDONIA_ML_PY, encoding="utf-8") as file:
        healthcare_code = file.read()
        parsed_ast = ast.parse(healthcare_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"), {})


def test_instrumented_ml_py_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    dag = run_and_assert_all_op_outputs_inspected(ANHEDONIA_ML_PY, None, ANHEDONIA_ML_PNG,
                                                  [custom_monkeypatching])
    assert len(dag) == 36


def test_llm_py_pipeline_runs():
    """
    Tests whether the pipeline works without instrumentation
    """
    with open(ANHEDONIA_LLM_PY, encoding="utf-8") as file:
        healthcare_code = file.read()
        parsed_ast = ast.parse(healthcare_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"), {})


def test_instrumented_llm_py_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    dag = run_and_assert_all_op_outputs_inspected(ANHEDONIA_LLM_PY, None, ANHEDONIA_LLM_PNG,
                                                  [custom_monkeypatching])
    assert len(dag) == 37
