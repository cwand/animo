from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from datetime import datetime
from typing import Optional
if TYPE_CHECKING:
    import animo


class ImageData:

    voxel_data: npt.NDArray[np.float64]
    meta_data: dict[str, list[str]]

    def __init__(self, voxel_data: npt.NDArray[np.float64], meta_data: dict[str, list[str]]):
        self.voxel_data = voxel_data
        self.meta_data = meta_data

    def get_no_frames(self) -> int:
        if len(self.voxel_data.shape) == 4:
            return self.voxel_data.shape[0]
        else:
            return 1

    def get_matrix_size(self) -> tuple[int, ...]:
        if len(self.voxel_data.shape) == 4:
            print(self.voxel_data.shape)
            return self.voxel_data.shape[1:]
        if len(self.voxel_data.shape) == 3:
            return self.voxel_data.shape
        else:
            return self.voxel_data.shape[0], self.voxel_data.shape[1]

    def get_acq_datetime(self) -> list[datetime]:
        if '0008|0032' not in self.meta_data:
            raise KeyError("ANIMO: Missing key 0008|0032 in GET_ACQ_DATETIME")
        if '0008|0022' not in self.meta_data:
            raise KeyError("ANIMO: Missing key 0008|0022 in GET_ACQ_DATETIME")
        acq_datetimes = []
        for i in range(self.get_no_frames()):
            acq_time_tag = self.meta_data['0008|0032'][i]
            acq_date_tag = self.meta_data['0008|0022'][i]
            sd = acq_date_tag[:4] + "-" + acq_date_tag[4:6] + "-" + acq_date_tag[6:]
            sd = sd + " " + acq_time_tag[:2] + ":" + acq_time_tag[2:4] + ":" + acq_time_tag[4:6]
            sd = sd + "." + acq_time_tag[-1] + "00"
            acq_datetimes.append(datetime.fromisoformat(sd))
        return acq_datetimes

    def get_acq_duration(self) -> list[float]:
        if '0018|1242' not in self.meta_data:
            raise KeyError("ANIMO: Missing key 0018|1242 in GET_ACQ_DURATION")
        acq_durations = []
        for i in range(self.get_no_frames()):
            acq_durations.append(float(self.meta_data['0018|1242'][i])/1000)
        return acq_durations

    def decay_correction(self, ref: "animo.ImageData", t12_sec: float) -> None:
        my_time = self.get_acq_datetime()
        ref_time = ref.get_acq_datetime()
        tdiff = (ref_time[0] - my_time[0]).total_seconds()
        corr_factor = 0.5**(tdiff/t12_sec)
        self.voxel_data = self.voxel_data * corr_factor


def load_image_series_from_file(fp: str, tags: Optional[list[str]] = None) -> ImageData:

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(fp)
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()

    a: sitk.Image = reader.Execute()
    adata = sitk.GetArrayFromImage(a)
    n_slices = adata.shape[0]
    if tags is None:
        mdata = {}
    else:
        try:
            mdata = {tag: [reader.GetMetaData(i, tag) for i in range(n_slices)] for tag in tags}
        except RuntimeError:
            raise ValueError("ANIMO: Invalid meta-data tag requested in LOAD_IMAGE_FROM_FILE")

    return ImageData(adata, mdata)


def load_image_from_file(fp: str) -> ImageData:

    adata = sitk.GetArrayFromImage(sitk.ReadImage(fp))
    return ImageData(adata, {})
