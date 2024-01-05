from typing import OrderedDict, Any
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
    out_file = None
    if 'out_file' in task:
        out_file = task['out_file']
    animo.plot_xy(plot_list, out_file=out_file, xlabel=xlabel, ylabel=ylabel)
    print("")
