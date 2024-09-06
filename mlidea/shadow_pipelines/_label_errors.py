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

    def __init__(self, train_fraction_to_consider=1., test_fraction_to_consider=1., proxy_model=False):
        # TODO: We should probably also implement the second proxy version from the workshop paper
        self._train_fraction_to_consider = train_fraction_to_consider
        self._test_fraction_to_consider = test_fraction_to_consider
        self._proxy_model = proxy_model
        if proxy_model is True:
            raise NotImplementedError("TODO")
        self._shadow_pipeline_id = (train_fraction_to_consider, test_fraction_to_consider, proxy_model)

    @property
    def shadow_pipeline_id(self):
        return self._shadow_pipeline_id

    def generate_shadow_pipeline_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        # pylint: disable=too-many-locals,too-many-statements
        # 1. first search for predict labels

        # 2. then get other info required for shapley:
        # train_data_sample, train_label_sample, test_data_sample, test_label_sample,
        # Also, have a configurable threshold how much of the train and test data gets used for the shapley stuff
        #  because this is calculated using sampling, we also need to keep track of their indices (or at least prov ids)
        # train_indices_to_consider
        # output: the train indices to flip

        # 3. then, look into label flipping:
        # copy the original train labels, flip them. rerun the model fitting, run predict, and rerun the eval

        return dag

    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        result_df = pandas.DataFrame({'todo': []})
        return result_df
