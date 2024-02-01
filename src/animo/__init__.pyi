import numpy as np
import numpy.typing as npt
from datetime import datetime
from typing import OrderedDict, Any, Optional, Union

# From common.py

def write_data(data: tuple[npt.NDArray[np.float64], ...], col_hdr: tuple[str, ...], fp: str) -> None : ...


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


# From tac.py

def extract_tac_from_01labelmap(image_series: ImageData, roi: ImageData) \
        -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


# From tasks.py

def image_series_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def image_loader(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def image_decay_correction(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def tac_from_labelmap(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...

def writer(task: OrderedDict[str, Any], named_obj: dict[str, Any]) -> None: ...
