import sys
import animo
from typing import Any
import xmltodict


def main(argv: list[str]):

    print("ANIMO STARTED")

    # Definition of available task types and the functions in tasks.py they will be calling
    task_types = {
        # Load an image series from a directory
        'ImageSeriesLoad': animo.image_series_loader,
        # Load an image from a single file
        'ImageLoad': animo.image_loader,
        # Apply decay correction to an image
        'DecayCorrection': animo.image_decay_correction,
        # Calculate a time-activity curve from an image and a ROI
        'TAC': animo.tac_from_labelmap,
        # Calculate integral of XYData
        'IntXY': animo.integrate,
        # Evaluate an arithmetic expression
        'Eval': animo.eval_expr,
    }

    # This object will collect all the results from running the tasks.
    # E.g.: when an image is loaded in the image_loader function, the image is
    # put in this dictionary with the key given in the input xml-file.
    # The named object structure is passed to all tasks, so all tasks have access
    # to all data from tasks before them.
    named_obj: dict[str, Any] = {}

    # Parse XML input file
    # The path to the input xml-file must be supplied
    if len(argv) != 1:
        exit("Missing command line argument: path to an XML file.")
    xml_file = open(argv[0], "r")

    # Create the task tree.
    # Some elements of the xml-file are expected to be lists, but
    # might only contain one element. These are forced into lists.
    task_tree = xmltodict.parse(xml_file.read(), force_list=('task', 'data'))

    # The <animo> tag defines the root of the task tree.
    root = task_tree['animo']

    # Run the tasks sequentially
    for task in root['task']:
        task_types[task['@type']](task, named_obj)

    print("ANIMO ENDED")


if __name__ == "__main__":
    main(sys.argv[1:])
