"""
Instrument and executes the pipeline
"""
# TODO: At some point, this should be split into two files, one for mere orchestration, one for instrumentation
import ast
import copy
import linecache
import logging
import sys
import time
from contextlib import redirect_stdout
from io import StringIO

import gorilla
import nbformat
import networkx
from astmonkey.transformers import ParentChildNodeTransformer
from nbconvert import PythonExporter

from mlidea.instrumentation._call_capture_transformer import CallCaptureTransformer
from mlidea import monkeypatching
from mlidea.instrumentation._operator_types import OperatorType
from mlidea._analysis_results import AnalysisResults, RuntimeInfo, DagExtractionInfo, ReuseInfo
from mlidea.analysis._what_if_analysis import WhatIfAnalysis
from mlidea.execution._dag_executor import DagExecutor
from mlidea.optimization._multi_query_optimizer import MultiQueryOptimizer
from mlidea.optimization._query_optimization_rules import QueryOptimizationRule
from mlidea.shadow_pipelines._shadow_pipeline import ShadowPipeline

logging.basicConfig(format='%(asctime)s %(levelname)-5s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
for _ in ("gensim", "tensorflow", "h5py"):
    logging.getLogger(_).setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """
    # pylint: disable=too-many-instance-attributes

    source_code_path = None
    source_code = None
    script_scope = {}
    lineno_next_call_or_subscript = -1
    col_offset_next_call_or_subscript = -1
    end_lineno_next_call_or_subscript = -1
    end_col_offset_next_call_or_subscript = -1
    next_op_id = 0
    next_patch_id = 0
    next_missing_op_id = -1
    track_code_references = True
    analyses = []
    shadow_pipelines = []
    custom_monkey_patching = []
    # TODO: Do we want to add the analysis to the key next to label to isolate analyses and avoid name clashes?
    original_pipeline_labels_to_extracted_plan_results = {}
    labels_to_extracted_plan_results = {}
    analysis_results = AnalysisResults({}, {}, networkx.DiGraph(), [], {}, networkx.DiGraph(),
                                       RuntimeInfo(0, 0, 0, 0, None, None, 0, 0, 0, 0, 0, 0, 0),
                                       DagExtractionInfo(networkx.DiGraph(), [], {}, 0, 0,
                                                         ReuseInfo({}, {}, {}, set(), set(), set(), set(), set(), set(),
                                                                   {}, {}, set())), None)
    monkey_patch_duration = 0
    skip_optimizer = False
    force_optimization_rules = None
    estimate_only = False
    operators_to_runtime_during_analysis = {}
    use_dfs_exec_strategy = False
    disable_monkey_patching = False
    prov_enabled = True
    old_dag = None
    old_shadow_pipelines = None
    enable_caching = True
    enable_cache_reuse = True
    global_old_dag = None
    global_new_dag = networkx.DiGraph()
    # Put this into a new data class
    reuse_info = ReuseInfo({}, {}, {}, set(), set(), set(), set(), set(), set(), {}, {}, set())

    def run(self, *,
            notebook_path: str or None = None,
            python_path: str or None = None,
            python_code: str or None = None,
            extraction_info: DagExtractionInfo or None = None,
            analyses: list[WhatIfAnalysis] or None = None,
            shadow_pipelines: list[ShadowPipeline] or None = None,
            reset_state: bool = True,
            track_code_references: bool = True,
            custom_monkey_patching: list[any] = None,
            skip_optimizer=False,
            force_optimization_rules: list[QueryOptimizationRule] or None = None,
            use_dfs_exec_strategy: bool = False,
            estimate_only=False,
            prov_enabled=True,
            caching_enabled=True
            ) -> AnalysisResults:
        """
        Instrument and execute the pipeline and evaluate all checks
        """
        self.analysis_results.pipeline_executor = self
        if reset_state:
            # reset_state=False should only be used internally for performance experiments etc!
            # It does not ensure the same inspections are still used as args etc.
            self.reset()

        if custom_monkey_patching is None:
            custom_monkey_patching = []
        if analyses is None:
            analyses = []
        if shadow_pipelines is None:
            shadow_pipelines = []

        self.track_code_references = track_code_references
        self.custom_monkey_patching = custom_monkey_patching
        self.analyses = analyses
        self.shadow_pipelines = shadow_pipelines
        self.skip_optimizer = skip_optimizer
        self.force_optimization_rules = force_optimization_rules
        self.estimate_only = estimate_only
        self.use_dfs_exec_strategy = use_dfs_exec_strategy
        self.prov_enabled = prov_enabled
        self.enable_caching = caching_enabled
        self.enable_cache_reuse = caching_enabled

        if extraction_info is not None:
            logger.info('Reusing DAG extraction results results from previously instrumented pipeline...')
            self.analysis_results.runtime_info.original_pipeline_without_importing_and_monkeypatching = None
            self.next_op_id = extraction_info.next_op_id
            self.next_patch_id = 0
            self.next_missing_op_id = extraction_info.next_missing_op_id
            self.reuse_info.cached_intermediates = extraction_info.reuse_info.cached_intermediates
            self.reuse_info.operator_call_info_to_dag_node = extraction_info.reuse_info.operator_call_info_to_dag_node.copy()
            self.reuse_info.op_id_to_dag_node = extraction_info.reuse_info.op_id_to_dag_node.copy()
            self.old_dag = extraction_info.original_dag.copy()
            self.old_shadow_pipelines = copy.deepcopy(extraction_info.shadow_pipelines)
            self.global_old_dag = networkx.compose_all([self.old_dag, *(self.old_shadow_pipelines or [])])

        if notebook_path is None and python_code is None and python_path is None:
            self.analysis_results.original_dag = extraction_info.original_dag.copy()
            self.original_pipeline_labels_to_extracted_plan_results = \
                extraction_info.original_pipeline_labels_to_extracted_plan_results.copy()
        else:
            logger.info('Running instrumented original pipeline...')
            orig_instrumented_exec_start = time.time()
            sys.stdout.flush()
            stdout_output = StringIO()
            with redirect_stdout(stdout_output):
                self.run_instrumented_pipeline(notebook_path, python_code, python_path)
            # TODO: Do we ever need the captured output from the original pipeline version?
            #  Maybe this gets relevant once we add the DAG as input to mlwhat in case there are multiple executions
            # captured_output = stdout_output.getvalue()
            self.prepare_runtime_info(orig_instrumented_exec_start)
            # FIXME: Training Data Matrix shape
            pipeline_exec_time = self.analysis_results.runtime_info.original_pipeline_without_importing_and_monkeypatching
            logger.info(f'---RUNTIME: Original pipeline execution took {pipeline_exec_time} ms '
                        f'(excluding imports and monkey-patching)')

        logger.info(f'Starting execution of {len(self.analyses)} what-if analyses...')
        self.run_what_if_analyses()
        self.gen_and_exec_shadow_pipelines()

        self.analysis_results.dag_extraction_info = DagExtractionInfo(
            self.analysis_results.original_dag.copy(),
            copy.deepcopy(list(self.analysis_results.shadow_pipeline_to_dags.values())),
            self.original_pipeline_labels_to_extracted_plan_results.copy(),
            self.next_op_id, self.next_missing_op_id, self.reuse_info)

        logger.info('Done!')
        return self.analysis_results

    def prepare_runtime_info(self, orig_instrumented_exec_start):
        orig_instrumented_exec_duration = (time.time() - orig_instrumented_exec_start -
                                           singleton.monkey_patch_duration)
        self.analysis_results.runtime_info.original_pipeline_without_importing_and_monkeypatching = \
            orig_instrumented_exec_duration * 1000
        original_estimator_runtime = [node.details.optimizer_info.runtime
                                      for node in self.analysis_results.original_dag.nodes
                                      if node.operator_info.operator == OperatorType.ESTIMATOR]
        self.analysis_results.runtime_info.original_model_training = sum(original_estimator_runtime)
        train_data_nodes = [node for node in self.analysis_results.original_dag.nodes
                            if node.operator_info.operator == OperatorType.TRAIN_DATA]
        if len(train_data_nodes) != 0:
            train_data_node = train_data_nodes[0]
            self.analysis_results.runtime_info.original_pipeline_train_data_shape = \
                train_data_node.details.optimizer_info.shape
        test_data_nodes = [node for node in self.analysis_results.original_dag.nodes
                           if node.operator_info.operator == OperatorType.TEST_DATA]
        if len(test_data_nodes) != 0:
            test_data_node = test_data_nodes[0]
            self.analysis_results.runtime_info.original_pipeline_test_data_shape = \
                test_data_node.details.optimizer_info.shape

    def gen_and_exec_shadow_pipelines(self):
        # Required for the execution engine for the IVM to detect changes
        for shadow_pipeline in self.shadow_pipelines:
            original_dag_copy = copy.deepcopy(self.analysis_results.original_dag)
            shadow_dag = shadow_pipeline.generate_shadow_pipeline_dag(original_dag_copy)
            self.global_new_dag = networkx.compose_all([self.global_new_dag, shadow_dag])
            DagExecutor(self).execute(shadow_dag, self.use_dfs_exec_strategy)
            filtered_shadow_dag = filter_shadow_dag(original_dag_copy, shadow_dag)

            # Update the runtime info
            for node in filtered_shadow_dag.nodes:
                if node in self.operators_to_runtime_during_analysis:
                    node.details.optimizer_info = self.operators_to_runtime_during_analysis[node]
                else:
                    print(node)

            self.analysis_results.shadow_pipeline_to_dags[shadow_pipeline] = filtered_shadow_dag
        for shadow_pipeline in self.shadow_pipelines:
            report = shadow_pipeline.generate_final_report(self.labels_to_extracted_plan_results)
            self.analysis_results.shadow_pipelines_to_result_reports[shadow_pipeline] = report

    def run_what_if_analyses(self):
        """
        Execute the specified what-if analyses
        """
        caching_status = self.enable_caching
        cache_reuse_status = self.enable_cache_reuse
        self.enable_caching = False  # We do not want to cache the large what-if intermediates
        # TODO: The QueryOptimizationRule.optimize_dag function cannot deal with reuse yet
        self.enable_cache_reuse = False
        for analysis in self.analyses:
            logger.info(f'Start plan generation for analysis {type(analysis).__name__}...')
            plan_generation_start = time.time()
            for patches in analysis.generate_plans_to_try(self.analysis_results.original_dag):
                self.analysis_results.what_if_dags.append((patches, networkx.DiGraph()))
            plan_generation_duration = time.time() - plan_generation_start
            logger.info(f'---RUNTIME: Plan generation took {plan_generation_duration * 1000} ms')
            self.analysis_results.runtime_info.what_if_plan_generation = plan_generation_duration * 1000

        # TODO: Add try catch statements so we can see intermediate DAGs even if something goes wrong for debugging
        MultiQueryOptimizer(self, self.force_optimization_rules) \
            .create_optimized_plan(self.analysis_results, self.skip_optimizer)

        if self.estimate_only is False:
            logger.info("Executing generated plans")
            execution_start = time.time()
            if self.skip_optimizer is False:
                DagExecutor(self).execute(self.analysis_results.combined_optimized_dag, self.use_dfs_exec_strategy)
            else:
                for _, what_if_dag in self.analysis_results.what_if_dags:
                    DagExecutor(self).execute(what_if_dag, self.use_dfs_exec_strategy)
            execution_duration = time.time() - execution_start
            logger.info(f'---RUNTIME: Execution took {execution_duration * 1000} ms')
            self.analysis_results.runtime_info.what_if_execution = execution_duration * 1000

            analysis_estimator_runtimes = [optimizer_info.runtime
                                           for node, optimizer_info in self.operators_to_runtime_during_analysis.items()
                                           if node.operator_info.operator == OperatorType.ESTIMATOR]
            self.analysis_results.runtime_info.what_if_execution_combined_model_training = sum(
                analysis_estimator_runtimes)

            # TODO: self.analysis_results.combined_optimized_dag currently only contains estimates.
            #  However, ideally, we have both estimates and the true numbers. This is how we can compute the real
            #  numbers. However, we have some tests that use the estimated numbers in combined_optimized_dag
            #  currently to check if optimizations work. So, we would need to duplicate the combined_optimized_dag
            #  to have a version with estimates and one with the actual numbers. However, this is not a priority for now
            # if self.skip_optimizer is False:
            #     for node in self.analysis_results.combined_optimized_dag.nodes:
            #         if node in self.operators_to_runtime_during_analysis:
            #             node.details.optimizer_info = self.operators_to_runtime_during_analysis[node]

            # Some debugging code to look at actual executon time of different operators in optimized plan
            # ops_with_runtimes = [(operator, optimizer_info.runtime) for operator, optimizer_info
            #                      in self.operators_to_runtime_during_analysis]
            # ops_with_runtimes.sort(key=lambda tuple: tuple[1], reverse=True)
            # ops_with_runtimes = [(op.details.description, runtime) for op, runtime in ops_with_runtimes]
            self.labels_to_extracted_plan_results.update(self.original_pipeline_labels_to_extracted_plan_results)
            for analysis in self.analyses:
                report = analysis.generate_final_report(self.labels_to_extracted_plan_results)
                self.analysis_results.analysis_to_result_reports[analysis] = report
        self.enable_caching = caching_status
        self.enable_cache_reuse = cache_reuse_status

    def run_instrumented_pipeline(self, notebook_path, python_code, python_path):
        """
        Instrument and execute the pipeline
        """
        self.source_code, self.source_code_path = self.load_source_code(notebook_path, python_path, python_code)
        parsed_ast = ast.parse(self.source_code)
        parsed_modified_ast = self.instrument_pipeline(parsed_ast, self.track_code_references)

        # Cache the source code in linecache under the fake filename
        linecache.cache[self.source_code_path] = (
            len(self.source_code),  # Size of code in bytes (not really needed here)
            None,  # Last modification time (unused)
            self.source_code.splitlines(True),  # Lines of the code
            self.source_code_path  # Filename
        )
        exec(compile(parsed_modified_ast, filename=self.source_code_path, mode="exec"), self.script_scope)

    def get_next_op_id(self, operator_call_info):
        """
        Each operator in the DAG gets a consecutive unique id
        """
        if operator_call_info in self.reuse_info.operator_call_info_to_dag_node:
            result = self.reuse_info.operator_call_info_to_dag_node[operator_call_info].node_id
        else:
            result = self.next_op_id
            self.next_op_id += 1
        return result

    def get_next_patch_id(self):
        """
        Each operator in the DAG gets a consecutive unique id
        """
        current_patch_id = self.next_patch_id
        self.next_patch_id += 1
        return current_patch_id

    def get_next_missing_op_id(self):
        """
        Each unknown operator in the DAG gets a consecutive unique negative id
        """
        current_missing_op_id = self.next_missing_op_id
        self.next_missing_op_id -= 1
        return current_missing_op_id

    def get_dag_node_for_id(self, dag_node_id: int):
        """
        Get a DAG node by id
        """
        return self.reuse_info.op_id_to_dag_node[dag_node_id]

    def reset(self):
        """
        Reset all attributes in the singleton object. This can be used when there are multiple repeated calls to mlidea
        """
        self.source_code_path = None
        self.source_code = None
        self.script_scope = {}
        self.lineno_next_call_or_subscript = -1
        self.col_offset_next_call_or_subscript = -1
        self.end_lineno_next_call_or_subscript = -1
        self.end_col_offset_next_call_or_subscript = -1
        self.next_op_id = 0
        self.next_patch_id = 0
        self.next_missing_op_id = -1
        self.track_code_references = True
        self.analysis_results = AnalysisResults({}, {}, networkx.DiGraph(), [], {}, networkx.DiGraph(),
                                                RuntimeInfo(0, 0, 0, 0, None, None, 0, 0, 0, 0, 0, 0, 0),
                                                DagExtractionInfo(networkx.DiGraph(), [], {}, 0, 0,
                                                                  ReuseInfo({}, {}, {}, set(), set(), set(), set(), set(),
                                                                            set(),{}, {}, set())), None)
        self.analyses = []
        self.shadow_pipelines = []
        self.original_pipeline_labels_to_extracted_plan_results = {}
        self.labels_to_extracted_plan_results = {}
        self.custom_monkey_patching = []
        self.monkey_patch_duration = 0
        self.skip_optimizer = False
        self.force_optimization_rules = None
        self.estimate_only = False
        self.operators_to_runtime_during_analysis = {}
        self.use_dfs_exec_strategy = False
        self.disable_monkey_patching = False
        self.prov_enabled = True
        self.old_dag = None
        self.old_shadow_pipelines = None
        self.enable_caching = True
        self.enable_cache_reuse = True
        self.global_old_dag = None
        self.global_new_dag = networkx.DiGraph()
        self.reuse_info = ReuseInfo({}, {}, {}, set(), set(), set(), set(), set(), set(), {}, {}, set())

    @staticmethod
    def instrument_pipeline(parsed_ast, track_code_references):
        """
        Instrument the pipeline AST to instrument function calls
        """
        # insert set_code_reference calls
        if track_code_references:
            # Needed to get the parent assign node for subscript assigns.
            #  Without this, "pandas_df['baz'] = baz + 1" would only be "pandas_df['baz']"
            parent_child_transformer = ParentChildNodeTransformer()
            parsed_ast = parent_child_transformer.visit(parsed_ast)
            call_capture_transformer = CallCaptureTransformer()
            parsed_ast = call_capture_transformer.visit(parsed_ast)
            parsed_ast = ast.fix_missing_locations(parsed_ast)

        # from mlinspect2._pipeline_executor import set_code_reference, monkey_patch
        func_import_node = ast.ImportFrom(module='mlidea.execution._pipeline_executor',
                                          names=[ast.alias(name='set_code_reference_call', asname=None),
                                                 ast.alias(name='set_code_reference_subscript', asname=None),
                                                 ast.alias(name='monkey_patch', asname=None),
                                                 ast.alias(name='undo_monkey_patch', asname=None)],
                                          level=0)
        parsed_ast.body.insert(0, func_import_node)

        # monkey_patch()
        inspect_import_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='monkey_patch', ctx=ast.Load()), args=[], keywords=[]))
        parsed_ast.body.insert(1, inspect_import_node)
        # undo_monkey_patch()
        inspect_import_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='undo_monkey_patch', ctx=ast.Load()), args=[], keywords=[]))
        parsed_ast.body.append(inspect_import_node)

        parsed_ast = ast.fix_missing_locations(parsed_ast)

        return parsed_ast

    @staticmethod
    def load_source_code(notebook_path, python_path, python_code):
        """
        Load the pipeline source code from the specified source
        """
        sources = [notebook_path, python_path, python_code]
        assert sum(source is not None for source in sources) == 1
        if python_path is not None:
            with open(python_path, encoding="utf-8") as file:
                source_code = file.read()
            source_code_path = python_path
        elif notebook_path is not None:
            with open(notebook_path, encoding="utf-8") as file:
                notebook = nbformat.reads(file.read(), nbformat.NO_CONVERT)
                exporter = PythonExporter()
                source_code, _ = exporter.from_notebook_node(notebook)
            source_code_path = notebook_path
        elif python_code is not None:
            source_code = python_code
            source_code_path = "<string-source>"
        else:
            assert False
        return source_code, source_code_path


# How we instrument the calls

# This instance works as our singleton: we avoid to pass the class instance to the instrumented
# pipeline. This keeps the DAG nodes to be inserted very simple.
singleton = PipelineExecutor()


def set_code_reference_call(lineno, col_offset, end_lineno, end_col_offset, **kwargs):
    """
    Method that gets injected into the pipeline code
    """
    singleton.lineno_next_call_or_subscript = lineno
    singleton.col_offset_next_call_or_subscript = col_offset
    singleton.end_lineno_next_call_or_subscript = end_lineno
    singleton.end_col_offset_next_call_or_subscript = end_col_offset
    return kwargs


def set_code_reference_subscript(lineno, col_offset, end_lineno, end_col_offset, arg):
    """
    Method that gets injected into the pipeline code
    """
    singleton.lineno_next_call_or_subscript = lineno
    singleton.col_offset_next_call_or_subscript = col_offset
    singleton.end_lineno_next_call_or_subscript = end_lineno
    singleton.end_col_offset_next_call_or_subscript = end_col_offset
    return arg


def monkey_patch():
    """
    Function that does the actual monkey patching
    """
    # The first time this is called, this can take a bit because all of the libraries need to be
    #  loaded by Python, but this cost is present anyway if those libraries are used.
    #  Because of this, we need to be careful how we fair benchmarking.
    logger.info("Importing libraries and monkey-patching them... (Imports are slow if not in sys.modules cache yet!)")
    monkey_patch_start = time.time()
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.apply(patch)
    singleton.monkey_patch_duration = time.time() - monkey_patch_start
    logger.info(f'---RUNTIME: Importing and monkey-patching took {singleton.monkey_patch_duration * 1000} ms')
    singleton.analysis_results.runtime_info.original_pipeline_importing_and_monkeypatching = singleton.monkey_patch_duration * 1000


def undo_monkey_patch():
    """
    Function that does the actual monkey patching
    """
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.revert(patch)


def get_monkey_patching_patch_sources():
    """
    Get monkey patches provided by mlidea and custom patches provided by the user
    """
    patch_sources = [monkeypatching]
    patch_sources.extend(singleton.custom_monkey_patching)
    return patch_sources


def filter_shadow_dag(orig_dag, shadow_dag):
    # Step 1: Find nodes that are only in G2 (not in G1)
    nodes_only_in_shadow = set(shadow_dag.nodes) - set(orig_dag.nodes)

    # Step 2: Find nodes directly connected to nodes only in G2
    predecessors_of_shadow_only = set()
    for node in nodes_only_in_shadow:
        predecessors_of_shadow_only.update(shadow_dag.predecessors(node))

    # Step 3: Determine all nodes to keep (those only in G2 + their neighbors)
    nodes_to_keep = nodes_only_in_shadow | predecessors_of_shadow_only

    # Step 4: Remove nodes not in the set of nodes to keep from G2
    nodes_to_remove = set(shadow_dag.nodes) - nodes_to_keep
    shadow_dag.remove_nodes_from(nodes_to_remove)

    return shadow_dag
