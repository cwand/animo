from typing import OrderedDict, Any
import numpy as np
import numpy.typing as npt
import animo
import math


def image_series_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: LOAD IMAGE SERIES")
    print(task)

    data_path = task['path']
    meta_tags = None
    if 'meta' in task:
        meta_tags = task['meta'].split(',')
    result_name = task['result_name']
    named_obj[result_name] = animo.load_image_series_from_file(data_path, meta_tags)
    print("")


def image_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: LOAD IMAGE")
    print(task)

    data_path = task['path']
    result_name = task['result_name']
    named_obj[result_name] = animo.load_image_from_file(data_path)
    print("")


def image_decay_correction(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: DECAY CORRECTION")
    print(task)

    img: animo.ImageData = named_obj[task['image']]
    ref: animo.ImageData = named_obj[task['ref']]
    t12 = float(task['t12'])
    img.decay_correction(ref, t12)
    print("")


def tac_from_labelmap(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: CALCULATE TAC")
    print(task)

    image_series = task['image_series']
    roi = task['roi']
    t_name = task['t_name']
    tac_name = task['tac_name']
    named_obj[t_name], named_obj[tac_name] = animo.extract_tac_from_01labelmap(
        named_obj[image_series], named_obj[roi])
    print("")


def integrate(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: INTEGRATE XY-DATA")
    print(task)
    x: npt.NDArray[np.float64] = named_obj[task['x']]
    y: npt.NDArray[np.float64] = named_obj[task['y']]
    result_name = task['result_name']
    named_obj[result_name] = np.trapz(y, x)
    print("")


def eval_expr(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: EVALUATE EXPRESSION")
    print(task)

    res = eval(task['expression'], {'math': math}, named_obj)

    result_name = task['result_name']
    named_obj[result_name] = res

    print("")
