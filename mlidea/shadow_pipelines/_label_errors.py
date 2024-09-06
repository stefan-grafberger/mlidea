# 1. Mislabel: in mlwhatif, two approaches, shapley and cleanlab. for mlidea workshop paper we only used shapley.
# in general, for LLM+RAG, we need the embeddings, that we don't have specifically in the DAG right now.
# Do we need to update the DAG? Or use some hack like letting the RAG join output the embeddings next to the text?
# but might have a big of added performance overhead. then, conditional operator depending on how many mislabels
# found. but maybe not that problematic here. but maybe for this we do want to use the provenance since the labeling
# might not be the final step in the data preprocessing and there might be filte

import networkx
import pandas

from shadow_pipelines._shadow_pipeline import ShadowPipeline


class LabelErrors(ShadowPipeline):
    """
    The Label Error Shadow Pipeline
    """

    def check_rebuilding_necessary(self, extracted_plan_results: dict[str, any]) -> any:
        return False

    def __init__(self):
        self._shadow_pipeline_id = None

    @property
    def shadow_pipeline_id(self):
        return self._shadow_pipeline_id

    def generate_shadow_pipeline_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        # pylint: disable=too-many-locals,too-many-statements
        return dag

    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        result_df = pandas.DataFrame({'todo': []})
        return result_df
