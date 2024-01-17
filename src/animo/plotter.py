import matplotlib.pyplot as plt
from animo import XYData
from typing import Union, Optional


class XYDataPlotWrapper:

    data: XYData
    linestyle: Union[str, None]
    label: Union[str, None]

    def __init__(self, data: XYData, linestyle: str, label: str):
        self.data = data
        self.linestyle = linestyle
        self.label = label


def plot_xy(xydata: list[XYDataPlotWrapper], out_file: Optional[str] = None,
            xlabel: Optional[str] = None, ylabel: Optional[str] = None,
            ylim_low: Optional[float] = None) -> None:
    fig, ax = plt.subplots()
    for xy in xydata:
        ax.plot(xy.data.x, xy.data.y, xy.linestyle, label=xy.label)
    if ylim_low is not None:
        ax.set_ylim(bottom=ylim_low)
    plt.grid()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
        plt.close()
