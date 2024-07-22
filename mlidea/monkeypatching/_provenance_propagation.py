from functools import partial

import numpy

from mlidea.execution._pipeline_executor import singleton
from mlidea.execution._stat_tracking import get_df_shape
from monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


def generate_and_add_provenance_data_source(df_obj):
    df_len = get_df_shape(df_obj)[0]
    provenance = numpy.array(range(df_len))
    if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
        df_obj._mlinspect_provenance = {}
    current_data_source = singleton.next_op_id - 1
    df_obj._mlinspect_provenance[f"{current_data_source}_0"] = provenance

def wrap_data_source_func(source_func):
    def edit_data_source_result(source_func):
        df_obj = source_func()
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)
        generate_and_add_provenance_data_source(df_obj)
        return df_obj

    return partial(edit_data_source_result, source_func)
