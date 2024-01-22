import numpy as np
import numpy.typing as npt


class XYData:
    """
    A contianer for a set of x- and y-data.

    Attributes
    ----------
    x:
        The x-data
    y:
        The y-data

    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __init__(self, x_data: npt.NDArray[np.float64], y_data: npt.NDArray[np.float64]):
        """
        Initialise a XYData object with a set of x- and y- data objects.
        The length of the two objects must be the same, and both must be one-dimensional.
        The input data-arrays are copied into this container. Subsequent modifications to
        the input parameters will not be reflected in the data.

        Parameters
        ----------
        x_data:
                The x-data
        y_data:
                The y-data

        Examples
        --------
        >>> xy = animo.XYData(np.array([1, 2, 3]), np.array([1.0, 4.0, -3.7]))
        """
        if x_data.size != y_data.size:
            raise ValueError("XYDATA: X and Y data must have the same length.")
        if x_data.ndim != 1 or y_data.ndim != 1:
            raise ValueError("XYDATA: X and Y data must be one-dimensional.")
        self.x = x_data.copy()
        self.y = y_data.copy()
