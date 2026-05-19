import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    def _gpu_dtype(vec_len: int):
        if vec_len < 2**15:
            return cp.int16
        elif vec_len < 2**31:
            return cp.int32
        else:
            return cp.int64

    def block_dot(df1: cp.ndarray, df2: cp.ndarray, block_size: int):
            M, K = df1.shape
            _, N = df2.shape
            result = np.zeros((M, N), dtype=np.int32)
            for i in range(0, M, block_size):
                for j in range(0, N, block_size):
                    block_C = cp.zeros((block_size, block_size), dtype=cp.int32)
                    for k in range(0, K, block_size):
                        block_C += cp.dot(
                            cp.asarray(df1[i:i+block_size, k:k+block_size], dtype=cp.int32),
                            cp.asarray(df2[k:k+block_size, j:j+block_size], dtype=cp.int32)
                        )
                    result[i:i+block_size, j:j+block_size] += cp.asnumpy(block_C)

            return result
except ImportError:
    cp = None
    block_dot = None
    CUPY_AVAILABLE = False
    

