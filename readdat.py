import numpy as np

def readdat(filename):
    with open(filename, 'rb') as fid:
                    D = np.fromfile(fid, dtype=np.float32)

    fs   = D[10]
    dt   = 1 / fs
    dx   = D[13]
    nx   = int(D[16])
    nt   = int(fs * D[17])
    data = D[64:].reshape((nx, nt), order='F').T  # 使用Fortran顺序进行数据的reshape

    return data, dx, dt, nt, nx