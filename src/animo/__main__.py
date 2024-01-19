import sys
import animo
from typing import Any
import xmltodict


def main(argv: list[str]):

    print("ANIMO STARTED")

    task_types = {
        'ImageSeriesLoad': animo.image_series_loader,
        'ImageLoad': animo.image_loader,
        'TAC': animo.tac_from_labelmap,
        'IntXY': animo.int_xy,
        'AvgXY': animo.avg_xy,
        'Eval': animo.eval_expr,
        'ToXYData': animo.to_xydata,
        'PlotXY': animo.xyplotter,
    }

    named_obj: dict[str, Any] = {}

    # Parse XML input file
    if len(argv) != 1:
        exit("Missing command line argument: path to an XML file.")
    xml_file = open(argv[0], "r")
    task_tree = xmltodict.parse(xml_file.read(), force_list=('task', 'data'))
    root = task_tree['animo']

    for task in root['task']:
        task_types[task['@type']](task, named_obj)

    print("ANIMO ENDED")


if __name__ == "__main__":
    main(sys.argv[1:])
