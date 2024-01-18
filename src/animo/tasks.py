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
    named_obj[result_name] = animo.load_image_series_from_file(data_path, meta_tags)
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
    result_name = task['result_name']
    named_obj[result_name] = animo.extract_tac_from_01labelmap(
        named_obj[image_series], named_obj[roi])
    print("")


def int_xy(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: INTEGRATE XY-DATA")
    print(task)
    xydata : animo.XYData = named_obj[task['xydata']]
    result_name = task['result_name']
    named_obj[result_name] = np.trapz(xydata.y, xydata.x)
    print("")


def avg_xy(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: AVERAGE XY-DATA")
    print(task)
    xydata: animo.XYData = named_obj[task['xydata']]
    result_name = task['result_name']
    named_obj[result_name] = np.trapz(xydata.y, xydata.x)/(xydata.x[-1]-xydata.x[0])
    print("")


def multiply(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: CALCULATE PRODUCT")
    print(task)

    product = 1.0
    factors: OrderedDict[str, Any] = task['factors']

    for factor in factors['factor']:
        if factor in named_obj:
            product = product * named_obj[factor]
        else:
            product = product * float(factor)

    result_name = task['result_name']
    named_obj[result_name] = product

    print("")


def xyplotter(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: PLOT XY-DATA")
    print(task)

    plot_list = []
    dataset: OrderedDict[str, Any] = task['dataset']

    for data in dataset['data']:
        dataname = data['name']
        xydata = named_obj[dataname]
        linestyle = data['style']
        label = data['label']
        plot_list.append(animo.XYDataPlotWrapper(xydata, linestyle, label))

    xlabel = task['xlabel']
    ylabel = task['ylabel']
    ylim = None
    if 'ylim_low' in task:
        ylim = float(task['ylim_low'])

    out_file = None
    if 'out_file' in task:
        out_file = task['out_file']
    animo.plot_xy(plot_list, out_file=out_file, xlabel=xlabel, ylabel=ylabel, ylim_low=ylim)
    print("")
