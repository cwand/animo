import numpy as np
import numpy.typing as npt
from typing import Any
import animo

# From common.py

class XYData:

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __init__(self, x_data: npt.NDArray[np.float64], y_data: npt.NDArray[np.float64]): ...


# From image.py

class ImageData:

    voxel_data: npt.NDArray[np.float64]
    meta_data: dict[str, list[str]]

    def __init__(self, voxel_data: npt.NDArray[np.float64], meta_data: dict[str, list[str]]): ...

def load_image_from_file(fp: str, tags: list[str] | None = ...) -> animo.ImageData : ...


# From plotter.py

class XYDataPlotWrapper:

    data: XYData
    linestyle: str | None
    label: str | None

    def __init__(self, data: XYData, linestyle: str, label: str): ...

def plot_xy(xydata: list[XYDataPlotWrapper], out_file: str | None = ..., xlabel: str | None = ..., ylabel: str | None = ...) -> None: ...
