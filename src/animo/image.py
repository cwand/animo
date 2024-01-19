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

    def decay_correction(self, ref: "animo.ImageData", t12_sec: float) -> None:
        return


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


def get_acq_datetime(image: ImageData) -> list[datetime]:
    if '0008|0032' not in image.meta_data:
        raise KeyError("ANIMO: Missing key 0008|0032 in GET_ACQ_DATETIME")
    if '0008|0022' not in image.meta_data:
        raise KeyError("ANIMO: Missing key 0008|0022 in GET_ACQ_DATETIME")
    acq_datetimes = []
    for i in range(image.voxel_data.shape[0]):
        acq_time_tag = image.meta_data['0008|0032'][i]
        acq_date_tag = image.meta_data['0008|0022'][i]
        sd = acq_date_tag[:4] + "-" + acq_date_tag[4:6] + "-" + acq_date_tag[6:]
        sd = sd + " " + acq_time_tag[:2] + ":" + acq_time_tag[2:4] + ":" + acq_time_tag[4:6]
        sd = sd + "." + acq_time_tag[-1] + "00"
        acq_datetimes.append(datetime.fromisoformat(sd))
    return acq_datetimes
