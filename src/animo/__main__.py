import animo
import SimpleITK as SITK
import numpy as np
import os
from datetime import datetime


def main():

    reader = SITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(os.path.join('test', 'data', '8_3V'))
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()

    a: SITK.Image = reader.Execute()
    adata = SITK.GetArrayFromImage(a)

    y = SITK.GetArrayFromImage(SITK.ReadImage(os.path.join('test', 'data', 'segs', 'Cyl101.nrrd')))
    nvox = y.sum()

    nims = a.GetSize()[3]

    t = np.zeros(nims)
    x = np.zeros(nims)

    for i in range(nims):

        acq_time_tag = reader.GetMetaData(i, '0008|0032')
        acq_date_tag = reader.GetMetaData(i, '0008|0022')
        sd = (acq_date_tag[:4] + "-" + acq_date_tag[4:6] + "-" + acq_date_tag[6:] + " " +
              acq_time_tag[:2] + ":" + acq_time_tag[2:4] + ":" + acq_time_tag[4:6] + "." +
              acq_time_tag[-1] + "00")
        acqtime = datetime.fromisoformat(sd)
        if i == 0:
            early_time = acqtime
            t[0] = 0.0
        else:
            t[i] = (acqtime-early_time).total_seconds()

        masked = np.multiply(adata[i], y)
        x[i] = np.sum(masked)/nvox

    print(np.trapz(x, t))

    xy = animo.XYData(t, x)
    xy_wrap = animo.XYDataPlotWrapper(xy, 'kx-', '8.3V')
    animo.plot_xy([xy_wrap], out_file=None, xlabel='Time [s]', ylabel='Activity')


if __name__ == "__main__":
    main()
