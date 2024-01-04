import matplotlib.pyplot as plt
from animo import XYData


class XYDataPlotWrapper:

    data: XYData
    linestyle: str
    label: str

    def __init__(self, data: XYData, linestyle: str, label: str):
        self.data = data
        self.linestyle = linestyle
        self.label = label


def plot_xy(xydata: list[XYDataPlotWrapper], out_file: str | None = None,
            xlabel: str | None = None, ylabel: str | None = None) -> None:
    fig, ax = plt.subplots()
    for xy in xydata:
        ax.plot(xy.data.x, xy.data.y, xy.linestyle, label=xy.label)
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
