from functools import partial

import numpy
import pandas

from mlidea.execution._pipeline_executor import singleton
from mlidea.execution._stat_tracking import get_df_shape
from mlidea.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary
from mlidea.monkeypatching._mlinspect_ndarray import MlinspectList

class ProvTrackingInfo:
    """ Contains info if the current calls originate from provenance tracking only """
    # pylint: disable=too-few-public-methods
    prov_tracking_operations_active: bool = False


prov_info_singleton = ProvTrackingInfo()


def generate_and_add_provenance_data_source(df_obj, op_id):
    if singleton.prov_enabled is True:
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
    if singleton.prov_enabled is True:
        if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
            df_obj._mlinspect_provenance = {}
        df_obj._mlinspect_provenance = new_provenance
    return df_obj


def wrap_projection_func(source_func):
    def propagate_provenance(source_func, *inputs):
        if singleton.prov_enabled is True:
            if isinstance(inputs[0], list) and not isinstance(inputs[0], MlinspectList):
                # This is special handling for the sklearn ColumnTransformer hstack
                provenance = inputs[0][0]._mlinspect_provenance
            else:
                provenance = inputs[0]._mlinspect_provenance
        df_obj = source_func(*inputs)
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)
        if singleton.prov_enabled is True:
            if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
                df_obj._mlinspect_provenance = {}
            df_obj._mlinspect_provenance = provenance
        return df_obj

    return partial(propagate_provenance, source_func)

def wrap_predict_func(source_func):
    def propagate_provenance(source_func, *inputs):
        if singleton.prov_enabled is True:
            provenance = inputs[1]._mlinspect_provenance
        df_obj = source_func(*inputs)
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)
        if singleton.prov_enabled is True:
            if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
                df_obj._mlinspect_provenance = {}
            df_obj._mlinspect_provenance = provenance
        return df_obj

    return partial(propagate_provenance, source_func)

def wrap_filter_func(source_func):
    def propagate_provenance(source_func, *inputs):
        prov_info_singleton.prov_tracking_operations_active = True
        df_input = inputs[0]
        if singleton.prov_enabled is True:
            provenance = df_input._mlinspect_provenance
            if isinstance(df_input, pandas.Series):
                df_input = pandas.DataFrame(df_input)
                was_series = True
            else:
                was_series = False
            for prov_key, prov_value in provenance.items():
                assert isinstance(df_input, pandas.DataFrame)
                df_input[prov_key] = prov_value
        prov_info_singleton.prov_tracking_operations_active = False
        df_obj = source_func(df_input, *inputs[1:])
        prov_info_singleton.prov_tracking_operations_active = True
        if singleton.prov_enabled is True:
            if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
                df_obj._mlinspect_provenance = {}
            new_provenance = {}
            for prov_key in provenance.keys():
                assert isinstance(df_obj, pandas.DataFrame)
                new_provenance[prov_key] = df_obj[prov_key].to_numpy()
                df_obj.drop([prov_key], axis=1, inplace=True)
                df_input.drop([prov_key], axis=1, inplace=True)

            if was_series is True:
                df_obj = df_obj.iloc[:, 0]
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)

        if singleton.prov_enabled is True:
            df_obj._mlinspect_provenance = new_provenance
        prov_info_singleton.prov_tracking_operations_active = False
        return df_obj

    return partial(propagate_provenance, source_func)

def wrap_join_func(source_func):
    def propagate_provenance(source_func, *inputs):
        prov_info_singleton.prov_tracking_operations_active = True
        if singleton.prov_enabled is True:
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
        prov_info_singleton.prov_tracking_operations_active = False

        df_obj = source_func(*inputs)
        df_obj = wrap_in_mlinspect_array_if_necessary(df_obj)

        prov_info_singleton.prov_tracking_operations_active = True
        if singleton.prov_enabled is True:
            if not hasattr(df_obj, "_mlinspect_provenance") or df_obj._mlinspect_provenance is None:
                df_obj._mlinspect_provenance = {}

            new_provenance = {}
            for prov_key in all_prov_columns:
                assert isinstance(df_obj, pandas.DataFrame)
                new_provenance[prov_key] = df_obj[prov_key].to_numpy()
                df_obj.drop([prov_key],  axis=1, inplace=True)
            df_obj._mlinspect_provenance = new_provenance
            for prov_key, prov_value in provenance_a.items():
                assert isinstance(inputs[0], pandas.DataFrame)
                inputs[0].drop([prov_key],  axis=1, inplace=True)
            for prov_key, prov_value in provenance_b.items():
                assert isinstance(inputs[1], pandas.DataFrame)
                inputs[1].drop([prov_key],  axis=1, inplace=True)
        prov_info_singleton.prov_tracking_operations_active = False
        return df_obj

    return partial(propagate_provenance, source_func)

def wrap_train_test_split_func(source_func):
    def propagate_provenance(source_func, *inputs):
        prov_info_singleton.prov_tracking_operations_active = True
        if singleton.prov_enabled is True:
            provenance = inputs[0]._mlinspect_provenance
            for prov_key, prov_value in provenance.items():
                assert isinstance(inputs[0], pandas.DataFrame)
                inputs[0][prov_key] = prov_value
        prov_info_singleton.prov_tracking_operations_active = False

        df_objs = source_func(*inputs)
        assert isinstance(df_objs, list)

        prov_info_singleton.prov_tracking_operations_active = True
        if singleton.prov_enabled is True:
            for split_result in df_objs:
                split_result = wrap_in_mlinspect_array_if_necessary(split_result)
                if not hasattr(split_result, "_mlinspect_provenance") or split_result._mlinspect_provenance is None:
                    split_result._mlinspect_provenance = {}

                new_provenance = {}
                for prov_key in provenance.keys():
                    assert isinstance(split_result, pandas.DataFrame)
                    new_provenance[prov_key] = split_result[prov_key].to_numpy()
                    split_result.drop([prov_key],  axis=1, inplace=True)
                split_result._mlinspect_provenance = new_provenance
        prov_info_singleton.prov_tracking_operations_active = False
        return df_objs

    return partial(propagate_provenance, source_func)


def wrap_rag_join_func_func(source_func):
    def propagate_provenance(source_func, *inputs):
        prov_info_singleton.prov_tracking_operations_active = True
        if singleton.prov_enabled is True:
            provenance = inputs[0]._mlinspect_provenance
            for prov_key, prov_value in provenance.items():
                assert isinstance(inputs[0], pandas.DataFrame)
                inputs[0][prov_key] = prov_value
        prov_info_singleton.prov_tracking_operations_active = False
        df_objs = source_func(*inputs)
        assert isinstance(df_objs, list)
        prov_info_singleton.prov_tracking_operations_active = True
        if singleton.prov_enabled is True:
            for split_result in df_objs:
                split_result = wrap_in_mlinspect_array_if_necessary(split_result)
                if not hasattr(split_result, "_mlinspect_provenance") or split_result._mlinspect_provenance is None:
                    split_result._mlinspect_provenance = {}

                new_provenance = {}
                for prov_key in provenance.keys():
                    assert isinstance(split_result, pandas.DataFrame)
                    new_provenance[prov_key] = split_result[prov_key].to_numpy()
                    split_result.drop([prov_key],  axis=1, inplace=True)
                split_result._mlinspect_provenance = new_provenance
        prov_info_singleton.prov_tracking_operations_active = False
        return df_objs

    return partial(propagate_provenance, source_func)
