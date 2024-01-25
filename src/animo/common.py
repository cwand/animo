import numpy as np
import numpy.typing as npt


def write_data(data: tuple[npt.NDArray[np.float64], ...], col_hdr: tuple[str, ...], fp: str) -> \
        None:
    if len(col_hdr) != len(data):
        raise ValueError("ANIMO: Non-equal number of columns in header and data in WRITE_DATA")
    hdr_string = ""
    for hdr in col_hdr:
        hdr_string = hdr_string + hdr + "\t"
    np.savetxt(fp, np.c_[data], delimiter=',', header=hdr_string)
