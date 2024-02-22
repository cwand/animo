from typing import OrderedDict, Any

import numpy as np

import animo


def image_series_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: LOAD IMAGE SERIES")
    print(task)

    data_path = task['path']
    meta_tags = None
    if 'meta' in task:
        meta_tags = task['meta'].split(',')
    result_name = task['result_name']
    img = animo.load_image_series_from_file(data_path, meta_tags)
    if 'decay' in task:
        decay_task = task['decay']
        ref: animo.ImageData = named_obj[decay_task['ref']]
        t12 = float(decay_task['t12'])
        img.decay_correction(ref, t12)
    named_obj[result_name] = img
    print("")


def image_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: LOAD IMAGE")
    print(task)

    data_path = task['path']
    result_name = task['result_name']
    named_obj[result_name] = animo.load_image_from_file(data_path)
    print("")


def tac_from_labelmap(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: CALCULATE TAC")
    print(task)

    image_series = task['image_series']
    roi = task['roi']
    t_name = task['frame_start_name']
    d_name = task['frame_duration_name']
    tac_name = task['tac_name']
    named_obj[t_name], named_obj[d_name], named_obj[tac_name] = animo.extract_tac_from_01labelmap(
        named_obj[image_series], named_obj[roi])
    print("")


def tac_integrate(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: INTEGRATE TAC")
    print(task)

    frame_start = task['frame_start']
    tac = task['tac']
    result_name = task['result_name']
    named_obj[result_name] = animo.integrate_tac(named_obj[frame_start], named_obj[tac])
    print("")


def mean_var_calc(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: CALCULATE MEAN AND VARIANCE")
    print(task)

    data = task['data']
    mean_name = task['mean_name']
    var_name = task['var_name']

    d = named_obj[data]

    named_obj[mean_name] = np.mean(d)
    named_obj[var_name] = np.var(d, ddof=1)

    print("")


def writer(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: WRITE DATA TO FILE")
    print(task)

    write_path: str = task['path']

    headers = [col['header'] for col in task['dataset']['col']]
    data_arrays = [named_obj[col['data']] for col in task['dataset']['col']]

    animo.write_data(tuple(data_arrays), tuple(headers), write_path)

    print("")
