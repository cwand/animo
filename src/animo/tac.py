import numpy as np
import animo


def extract_tac_from_01labelmap(image_series: animo.ImageData,
                                roi: animo.ImageData) -> animo.XYData:

    nims = image_series.voxel_data.shape[0]

    # Get acquisition times and difference between times and start
    acq_times = image_series.get_acq_datetime()
    t = [(acq_time - acq_times[0]).total_seconds() for acq_time in acq_times]

    # Calculate roi sum
    x = [np.sum(np.multiply(image_series.voxel_data[i], roi.voxel_data)) for i in range(nims)]

    return animo.XYData(np.array(t), np.array(x))
