import numpy as np
import numpy.typing as npt


class XYData:

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __init__(self, x_data: npt.NDArray[np.float64], y_data: npt.NDArray[np.float64]):
        if x_data.size != y_data.size:
            raise ValueError("XYDATA: X and Y data must have the same length.")
        self.x = x_data
        self.y = y_data
