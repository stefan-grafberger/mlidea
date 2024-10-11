"""
The Interface for the Shadow Pipelines
"""
import abc

import networkx


class ShadowPipeline(metaclass=abc.ABCMeta):
    """
    The Interface for the Shadow Pipelines
    """
    # Maybe they should start by building an empty DAG that has a few nodes from the original DAG
    #  That way, prioritisation etc will be easier later

    @property
    def shadow_pipeline_id(self):
        """The Interface for the Shadow Pipelines"""
        return None

    @property
    def simple_name(self):
        """The Simple String name for the Shadow Pipeline"""
        return None

    @abc.abstractmethod
    def generate_shadow_pipeline_dag(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        """Generate the shadow pipeline to run"""
        raise NotImplementedError

    @abc.abstractmethod
    def check_rebuilding_necessary(self, extracted_plan_results: dict[str, any]) -> bool:
        """Get the final report after trying out the different pipeline variants"""
        # TODO: Not sure if we really need something like this or not
        raise NotImplementedError

    # TODO: Maybe we want more methods in the future like checking if there is some warning or suggestion etc

    @abc.abstractmethod
    def generate_final_report(self, extracted_plan_results: dict[str, any]) -> any:
        """Get the final report after trying out the different pipeline variants"""
        raise NotImplementedError

    def __eq__(self, other):
        """Shadow Pipelines must implement equals"""
        return (isinstance(other, self.__class__) and
                self.shadow_pipeline_id == other.shadow_pipeline_id)

    def __hash__(self):
        """Shadow Pipelines must be hashable"""
        return hash((self.__class__.__name__, self.shadow_pipeline_id))

    def __repr__(self):
        """Shadow Pipelines must have a str representation"""
        return f"{self.__class__.__name__}({self.shadow_pipeline_id})"
