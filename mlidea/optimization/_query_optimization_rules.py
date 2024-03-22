"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
import abc

import networkx

from mlidea.execution._patches import PipelinePatch


class QueryOptimizationRule(metaclass=abc.ABCMeta):
    """
    The Interface for Query Optimization Rules
    """
    # pylint: disable=unused-argument

    def optimize_dag(self, dag: networkx.DiGraph, patches: list[list[PipelinePatch]]) -> \
            tuple[networkx.DiGraph, list[list[PipelinePatch]]]:
        """Transform the original DAG into something that is better for optimizations without changing the
        final result"""
        return dag, patches

    def optimize_patches(self, dag: networkx.DiGraph, patches: list[list[PipelinePatch]]) -> list[list[PipelinePatch]]:
        """Transform the patches into more efficient ones"""
        return patches
