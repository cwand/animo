import numpy as np
import numpy.typing as npt
import SimpleITK as sitk


class ImageData:

    voxel_data: npt.NDArray[np.float64]
    meta_data: dict[str, list[str]]

    def __init__(self, voxel_data: npt.NDArray[np.float64], meta_data: dict[str, list[str]]):
        self.voxel_data = voxel_data
        self.meta_data = meta_data


def load_image_series_from_file(fp: str, tags: list[str] | None = None) -> ImageData:

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
