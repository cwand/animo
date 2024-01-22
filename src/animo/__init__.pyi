import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import OrderedDict, Any, Optional, Union

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

    def get_no_frames(self) -> int : ...

    def get_matrix_size(selfself) -> tuple[int, int, int] : ...

    def get_acq_datetime(self) -> list[datetime]: ...

    def decay_correction(self, ref: ImageData, t12_sec: float) -> None : ...

def load_image_series_from_file(fp: str, tags: Optional[list[str]] = ...) -> ImageData : ...

def load_image_from_file(fp: str) -> ImageData : ...




# From plotter.py

class XYDataPlotWrapper:

    data: XYData
    linestyle: Union[str, None]
    label: Union[str, None]

    def __init__(self, data: XYData, linestyle: str, label: str): ...

def plot_xy(xydata: list[XYDataPlotWrapper], out_file: Optional[str] = ...,
            xlabel: Optional[str] = ..., ylabel: Optional[str] = ...,
			ylim_low: Optional[float] = ..., ylim_high: Optional[float] = ...) -> None: ...


# From tac.py

def extract_tac_from_01labelmap(image_series: ImageData, roi: ImageData) -> XYData: ...


# From tasks.py

def image_series_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def image_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def image_decay_correction(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def tac_from_labelmap(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def int_xy(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def avg_xy(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def eval_expr(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def to_xydata(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def xyplotter(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...
