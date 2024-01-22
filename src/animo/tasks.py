from typing import OrderedDict, Any
import numpy as np
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
    result_name = task['result_name']
    named_obj[result_name] = animo.extract_tac_from_01labelmap(
        named_obj[image_series], named_obj[roi])
    print("")


def int_xy(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: INTEGRATE XY-DATA")
    print(task)
    xydata: animo.XYData = named_obj[task['xydata']]
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


def eval_expr(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: EVALUATE EXPRESSION")
    print(task)

    res = eval(task['expression'], {'math': math}, named_obj)

    result_name = task['result_name']
    named_obj[result_name] = res

    print("")


def to_xydata(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None:
    print("TASK: APPEND TO DATA ARRAY")
    print(task)

    array_name = task['array']

    xdata = task['x'].split(',')
    ydata = task['y'].split(',')

    if len(xdata) != len(ydata):
        raise ValueError("ANIMO: Unequal length of x- and y-data in TO_XYDATA")

    app_vals_x = []
    app_vals_y = []

    for x, y in zip(xdata, ydata):
        if x in named_obj:
            x_val = named_obj[x]
        else:
            x_val = float(x)
        app_vals_x.append(x_val)
        if y in named_obj:
            y_val = named_obj[y]
        else:
            y_val = float(y)
        app_vals_y.append(y_val)

    if array_name not in named_obj:
        named_obj[array_name] = animo.XYData(np.array(app_vals_x), np.array(app_vals_y))
    else:
        old_array: animo.XYData = named_obj[array_name]
        named_obj[array_name] = animo.XYData(
            np.append(old_array.x, app_vals_x), np.append(old_array.y, app_vals_y))
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
    ylim_low = None
    if 'ylim_low' in task:
        ylim_low = float(task['ylim_low'])
    ylim_high = None
    if 'ylim_high' in task:
        ylim_high = float(task['ylim_high'])

    out_file = None
    if 'out_file' in task:
        out_file = task['out_file']
    animo.plot_xy(plot_list, out_file=out_file, xlabel=xlabel, ylabel=ylabel,
                  ylim_low=ylim_low, ylim_high=ylim_high)
    print("")
