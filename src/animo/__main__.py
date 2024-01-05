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
        'PlotXY': animo.xyplotter
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

    '''
    a = animo.load_image_series_from_file(os.path.join('test', 'data', '8_3V'),
                                   tags=['0008|0022', '0008|0032'])

    y = animo.load_image_from_file(os.path.join('test', 'data', 'segs', 'Cyl101.nrrd'))

    xy = animo.extract_tac_from_01labelmap(a, y)
    xy_wrap = animo.XYDataPlotWrapper(xy, 'kx-', '8.3V')
    animo.plot_xy([xy_wrap], out_file=None, xlabel='Time [s]', ylabel='Activity') '''


if __name__ == "__main__":
    main(sys.argv[1:])
