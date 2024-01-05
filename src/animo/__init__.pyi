import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import OrderedDict, Any

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

def load_image_series_from_file(fp: str, tags: list[str] | None = ...) -> ImageData : ...

def load_image_from_file(fp: str) -> ImageData : ...

def get_acq_datetime(image: ImageData) -> list[datetime] : ...


# From plotter.py

class XYDataPlotWrapper:

    data: XYData
    linestyle: str | None
    label: str | None

    def __init__(self, data: XYData, linestyle: str, label: str): ...

def plot_xy(xydata: list[XYDataPlotWrapper], out_file: str | None = ..., xlabel: str | None = ..., ylabel: str | None = ...) -> None: ...


# From tac.py

def extract_tac_from_01labelmap(image_series: ImageData, roi: ImageData) -> XYData: ...


# From tasks.py

def image_series_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def image_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def tac_from_labelmap(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def xyplotter(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...
