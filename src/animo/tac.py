import numpy as np
import numpy.typing as npt
import animo


def extract_tac_from_01labelmap(image_series: animo.ImageData, roi: animo.ImageData) \
        -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    nims = image_series.voxel_data.shape[0]

    # Get acquisition times and difference between times and start
    acq_times = image_series.get_acq_datetime()
    t = [(acq_time - acq_times[0]).total_seconds() for acq_time in acq_times]

    # Get acquisition durations
    fd = image_series.get_acq_duration()

    # Calculate roi sum
    x = [np.sum(np.multiply(image_series.voxel_data[i], roi.voxel_data)) for i in range(nims)]

    return np.array(t), np.array(fd), np.array(x)
