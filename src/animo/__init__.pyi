import numpy as np
import numpy.typing as npt

# From common.py

class XYData:

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __init__(self, x_data: npt.NDArray[np.float64], y_data: npt.NDArray[np.float64]): ...


# From plotter.py

class XYDataPlotWrapper:

    data: XYData
    linestyle: str | None
    label: str | None

    def __init__(self, data: XYData, linestyle: str, label: str): ...

def plot_xy(xydata: list[XYDataPlotWrapper], out_file: str | None = ..., xlabel: str | None = ..., ylabel: str | None = ...) -> None: ...
