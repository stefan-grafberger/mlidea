from functools import partial

import numpy
import pandas

from mlidea.execution._pipeline_executor import singleton
from mlidea.execution._stat_tracking import get_df_shape
from monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


def generate_and_add_provenance_data_source(df_obj, op_id):
    df_len = get_df_shape(df_obj)[0]
    provenance = numpy.array(range(df_len))
    if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
        df_obj._mlinspect_provenance = {}
    df_obj._mlinspect_provenance[f"{op_id}_0"] = provenance

def wrap_data_source_func(source_func, op_id):
    def edit_data_source_result(source_func):
        df_obj = source_func()
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)
        generate_and_add_provenance_data_source(df_obj, op_id)
        return df_obj

    return partial(edit_data_source_result, source_func)

def get_input_provenance(df_obj):
    return df_obj._mlinspect_provenance

def set_output_provenance(df_obj, new_provenance):
    if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
        df_obj._mlinspect_provenance = {}
    df_obj._mlinspect_provenance = new_provenance
    return df_obj


def wrap_projection_func(source_func):
    def propagate_provenance(source_func, *inputs):
        provenance = inputs[0]._mlinspect_provenance
        df_obj = source_func(*inputs)
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)
        if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
            df_obj._mlinspect_provenance = {}
        df_obj._mlinspect_provenance = provenance
        return df_obj

    return partial(propagate_provenance, source_func)

def wrap_filter_func(source_func):
    def propagate_provenance(source_func, *inputs):
        provenance = inputs[0]._mlinspect_provenance
        for prov_key, prov_value in provenance.items():
            assert isinstance(inputs[0], pandas.DataFrame)
            inputs[0][prov_key] = prov_value
        df_obj = source_func(*inputs)
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)
        if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
            df_obj._mlinspect_provenance = {}

        new_provenance = {}
        for prov_key in provenance.keys():
            assert isinstance(df_obj, pandas.DataFrame)
            new_provenance[prov_key] = df_obj[prov_key].to_numpy()
            df_obj.drop([prov_key],  axis=1, inplace=True)
        df_obj._mlinspect_provenance = new_provenance
        return df_obj

    return partial(propagate_provenance, source_func)

def wrap_join_func(source_func):
    def propagate_provenance(source_func, *inputs):
        provenance_a = inputs[0]._mlinspect_provenance
        provenance_b = inputs[1]._mlinspect_provenance

        prov_names_a = set(provenance_a.keys())
        prov_names_b = set(provenance_b.keys())
        column_clashes = prov_names_a.intersection(prov_names_b)
        all_prov_columns = prov_names_a.union(prov_names_b)
        for column_clash in column_clashes:
            data_source, duplicate_index = column_clash.rsplit('_', 1)
            num_occurrences_in_data_columns = len([column for column in prov_names_a
                                                   if column.startswith(data_source)])
            new_duplicate_index = int(duplicate_index) + num_occurrences_in_data_columns
            new_col_name = f"{data_source}_{new_duplicate_index}"
            provenance_b[new_col_name] = provenance_b.pop(column_clash)
            all_prov_columns.add(new_col_name)

        for prov_key, prov_value in provenance_a.items():
            assert isinstance(inputs[0], pandas.DataFrame)
            inputs[0][prov_key] = prov_value
        for prov_key, prov_value in provenance_b.items():
            assert isinstance(inputs[1], pandas.DataFrame)
            inputs[1][prov_key] = prov_value

        df_obj = source_func(*inputs)
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)
        if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
            df_obj._mlinspect_provenance = {}

        new_provenance = {}
        for prov_key in all_prov_columns:
            assert isinstance(df_obj, pandas.DataFrame)
            new_provenance[prov_key] = df_obj[prov_key].to_numpy()
            df_obj.drop([prov_key],  axis=1, inplace=True)
        df_obj._mlinspect_provenance = new_provenance
        return df_obj

    return partial(propagate_provenance, source_func)