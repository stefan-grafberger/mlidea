"""
Some useful utils for the project
"""
import os

from mlidea.utils import get_project_root

ADULT_SIMPLE_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_simple", "adult_simple.py")
ADULT_SIMPLE_IPYNB = os.path.join(str(get_project_root()), "example_pipelines", "adult_simple", "adult_simple.ipynb")
ADULT_SIMPLE_PNG = os.path.join(str(get_project_root()), "example_pipelines", "adult_simple", "adult_simple.png")

ADULT_COMPLEX_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_complex.py")
ADULT_COMPLEX_PNG = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_complex.png")
ADULT_COMPLEX_MODIFIED_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex",
                                         "adult_complex_modified.py")

COMPAS_PY = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.py")
COMPAS_PNG = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.png")

HEALTHCARE_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.py")
HEALTHCARE_PNG = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.png")
HEALTHCARE_MODIFIED_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                      "healthcare_replaced.py")
HEALTHCARE_DELETED_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                      "healthcare_deleted.py")
HEALTHCARE_ADDED_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                      "healthcare_added.py")

ANHEDONIA_ML_PY = os.path.join(str(get_project_root()), "example_pipelines", "anhedonia_ml", "anhedonia_ml.py")
ANHEDONIA_ML_PNG = os.path.join(str(get_project_root()), "example_pipelines", "anhedonia_ml", "anhedonia_ml.png")
ANHEDONIA_ML_MODIFIED_PY = os.path.join(str(get_project_root()), "example_pipelines", "anhedonia_ml",
                                        "anhedonia_ml_modified.py")

ANHEDONIA_LLM_PY = os.path.join(str(get_project_root()), "example_pipelines", "anhedonia_llm", "anhedonia_llm.py")
ANHEDONIA_LLM_PNG = os.path.join(str(get_project_root()), "example_pipelines", "anhedonia_llm", "anhedonia_llm.png")
ANHEDONIA_LLM_MODIFIED_PY = os.path.join(str(get_project_root()), "example_pipelines", "anhedonia_llm",
                                         "anhedonia_llm_modified.py")
ANHEDONIA_LLM_CONCAT_DATA_LOADING_PY = os.path.join(str(get_project_root()), "example_pipelines", "anhedonia_llm",
                                                    "anhedonia_llm_concat_data_loading.py")

