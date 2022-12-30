# -*- coding: utf-8 -*-

import time
from collections import namedtuple
from typing import NamedTuple, List
from itertools import product
from roost.optiflow import *
from roost.constants import hour
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#: :obj:`tuple` :
#: TimeSelector = namedtuple("TimeSelector", ['time', 'step'])
TimeSelector = namedtuple("TimeSelector", ['time', 'step'])


def add_times_to_steps_sources_targets(steps_sources_targets, dataset):
    """Given a dict mapping (source step, source step) -> (target step), return a dict replacing step keys and values
    with analogous TimeSelector values that also include time information.

        :param steps_sources_targets:
        :type steps_sources_targets:

        :param dataset:
        :type dataset:

        :returns: None

    """
    new_dict = {}
    times = dataset.coords['time'].values
    for t in times:
        for sources, target in steps_sources_targets.items():
            source_0 = TimeSelector(time=t, step=sources[0])
            source_1 = TimeSelector(time=t, step=sources[1])
            target = TimeSelector(time=t, step=target)
            new_dict[(source_0, source_1)] = target
    return new_dict


def add_steps_to_times_sources_targets(times_sources_targets, steps_list=(0*hour,)):
    """Given a dict mapping (source step, source step) -> (target step), return a dict replacing step keys and values
    with analogous TimeSelector values that also include time information.

        :param times_sources_targets:
        :type times_sources_targets:

        :param steps_list: default 0
        :type steps_list:

        :returns: None

    """
    new_dict = {}
    for s in steps_list:
        for sources, target in times_sources_targets.items():
            source_0 = TimeSelector(time=sources[0], step=s)
            source_1 = TimeSelector(time=sources[1], step=s)
            target = TimeSelector(time=target, step=s)
            new_dict[(source_0, source_1)] = target
    return new_dict


def filter_dims(d, dims):
    """Returns d filtered so that only key-value pairs where the key is in dims are kept

        :param d:
        :type d:

        :param dims:
        :type dims:

        :returns: None
    """
    return {k: v for k, v in d.items() if k in dims}


def evaluate_flow_on_dataset(flow_calc, dataset, variables, sources_and_targets,
                             array_norm=None, aggregation_norm=None, cut_evaluation_array=None,
                             aggregate=True):
    """

        :param flow_calc:
        :type flow_calc:

        :param dataset:
        :type dataset: xarray.Dataset

        :param variables:
        :type variables:

        :param sources_and_targets:
        :type sources_and_targets:

        :param array_norms:
        :type array_norms:

        :param aggregation_norm:
        :type aggregation_norm:

        :param cut_evaluation_array:
        :type cut_evaluation_array:

        :param aggregate:
        :type aggregate:

        :returns: None

    """
    if array_norm is None:
        def array_norm(x): return np.linalg.norm(x, ord=2)
    if aggregation_norm is None:
        def aggregation_norm(x): return np.linalg.norm(x, ord=2)
    non_ll_coords = {k: v for k, v in dataset.coords.items()
                     if k not in ('latitude', 'longitude', 'step', 'valid_time', 'time')
                     and v.values.size > 1}
    coord_names = non_ll_coords.keys()
    coord_axes = [non_ll_coords[c_name].values for c_name in coord_names]
    var_norms = {}
    for var in variables:
        norms_dict = {}
        norms = []
        for sources, target in sources_and_targets.items():
            sources: List[NamedTuple]
            source_selector_0 = filter_dims(sources[0]._asdict(), dataset.dims)
            source_selector_1 = filter_dims(sources[1]._asdict(), dataset.dims)
            target_selector = filter_dims(target._asdict(), dataset.dims)
            coord_values = product(*coord_axes)
            for cv in coord_values:
                selector = dict(zip(coord_names, cv))
                ds_subset = dataset[var].sel(**selector)
                source_start = ds_subset.loc[source_selector_0].values
                source_end = ds_subset.loc[source_selector_1].values
                target_frame = ds_subset.loc[target_selector].values
                interpolant = TwoFrameInterpolant(source_start, source_end, flow_calc)
                prediction_at_half_point = interpolant(.5)
                delta_array = prediction_at_half_point - target_frame
                if cut_evaluation_array is None:
                    norms.append(array_norm(delta_array))
                else:
                    cea = cut_evaluation_array
                    if aggregate:
                        norms.append(array_norm(delta_array[cea:-cea, cea:-cea]))
                    else:
                        norms_dict[(sources, target)] = array_norm(delta_array[cea:-cea, cea:-cea])
        if aggregate:
            var_norms[var] = aggregation_norm(norms)
        else:
            var_norms[var] = norms_dict
    return var_norms


def evaluate_flow_on_dataset_norms(flow_calc, dataset, variables, sources_and_targets,
                             array_norms=None, aggregation_norm=None, cut_evaluation_array=None,
                             aggregate=False):
    """

        :param flow_calc:
        :type flow_calc:

        :param dataset:
        :type dataset: xarray.Dataset

        :param variables:
        :type variables:

        :param sources_and_targets:
        :type sources_and_targets:

        :param array_norms:
        :type array_norms:

        :param aggregation_norm:
        :type aggregation_norm:

        :param cut_evaluation_array:
        :type cut_evaluation_array:

        :param aggregate:
        :type aggregate:

        :returns: None

    """
    if array_norms is None:
        def arr_norm(x): return np.linalg.norm(x, ord=2)
        array_norms = [arr_norm]
    if aggregation_norm is None:
        def aggregation_norm(x): return np.linalg.norm(x, ord=2)
    non_ll_coords = {k: v for k, v in dataset.coords.items()
                     if k not in ('latitude', 'longitude', 'step', 'valid_time', 'time')
                     and v.values.size > 1}
    coord_names = non_ll_coords.keys()
    coord_axes = [non_ll_coords[c_name].values for c_name in coord_names]
    var_norms = {}
    for var in variables:
        norms_dict = {arr_norm: {} for arr_norm in array_norms}
        norms = []
        for sources, target in sources_and_targets.items():
            sources: List[NamedTuple]
            source_selector_0 = filter_dims(sources[0]._asdict(), dataset.dims)
            source_selector_1 = filter_dims(sources[1]._asdict(), dataset.dims)
            target_selector = filter_dims(target._asdict(), dataset.dims)
            coord_values = product(*coord_axes)
            for cv in coord_values:
                selector = dict(zip(coord_names, cv))
                ds_subset = dataset[var].sel(**selector)
                source_start = ds_subset.loc[source_selector_0].values
                source_end = ds_subset.loc[source_selector_1].values
                target_frame = ds_subset.loc[target_selector].values
                interpolant = TwoFrameInterpolant(source_start, source_end, flow_calc)
                prediction_at_half_point = interpolant(.5)
                delta_array = prediction_at_half_point - target_frame
                if cut_evaluation_array is None:
                    # fix for disaggregated norms
                    norms.append(array_norm(delta_array))
                else:
                    cea = cut_evaluation_array
                    if aggregate:
                        for array_norm in array_norms:
                            norms_dict[array_norm][(sources, target)] = array_norm(delta_array[cea:-cea, cea:-cea])
                    else:
                        pass  # integrate later
        if aggregate:
            var_norms[var] = aggregation_norm(norms)
        else:
            var_norms[var] = norms_dict
    return var_norms


def optimize_flow(parameter_space, flow_generator, dataset, sources_and_targets,
                  var_weights=None, max_evals=100, cut_evaluation_array=None):
    """Optimizes the parameters of the algorithm for performance on the dataset

        :param parameter_space:
        :type parameter_space:

        :param flow_generator:
        :type flow_generator:

        :param dataset:
        :type dataset:

        :param sources_and_targets:
        :type sources_and_targets:

        :param var_weights:
        :type var_weights:

        :param max_evals:
        :type max_evals:

        :param cut_evaluation_array:
        :type cut_evaluation_array:

        :return: None


    """
    if var_weights is None:
        var_weights = {'u': 1, 'v': 1, 't': 1}
    variables = var_weights.keys()

    def objective(args):
        flow = flow_generator(args)
        t0 = time.time()
        losses = evaluate_flow_on_dataset(flow, dataset, variables, sources_and_targets,
                                          cut_evaluation_array=cut_evaluation_array)
        delta_t = time.time() - t0
        loss = sum(w * losses[var] for var, w in var_weights.items())
        print(loss, delta_t)
        print(args)
        print("===")
        return {'loss': loss,
                'status': STATUS_OK,
                'var_losses': losses,
                'time': delta_t}

    trials = Trials()
    best = fmin(objective,
                space=parameter_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    return best, trials


if __name__ == "__main__":
    farneback_space = {
        'pyr_scale': hp.uniform('pyr_scale', 0.1, 0.9),
        'levels': hp.qloguniform('levels', 0, np.log(11), q=1),
        'winsize': hp.qlognormal('winsize', np.log(100), 1/6*np.log(100), 1),
        'iterations': hp.quniform('iterations', 2, 6, 1),
        'poly_n': hp.quniform('poly_n', 2, 14, 1),
        'poly_sigma': hp.uniform('poly_sigma', 0.1, 2.0),
        'scale': hp.qloguniform('scale', np.log(0.1), np.log(1e7), 1),
    }

    def flow_calculator(args):
        return FarnebackFlow(**args)

    snt_steps = {(0*hour, 2*hour): hour, (0*hour, 4*hour): 2*hour, (0*hour, 6*hour): 3*hour}


