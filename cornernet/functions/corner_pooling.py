import numpy 


class CornerPoolingLeft(object):
    def forward_gpu(self, x):
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        n, c, h, w = x[0].shape

        y = cuda.cupy.zeros((n, c, h, w), dtype=x[0].dtype)
        z = cuda.cupy.zeros((n, c, h, w), dtype=x[0].dtype)

        self.indexes = cuda.cupy.empty((n, c, h, w), dtype=numpy.int32)
        print(x[0].reshape(-1, w))

        print(x[0])
        cuda.elementwise(
            'raw T in, int32 w',
            'raw T out, raw S indexes, S z',
            '''
            int ind[] = {i, 0};
            out[ind] = in[ind];
            indexes[ind] = i * w;
            for (int j = 1; j < w; j++) {
                int ind2[] = {i, j};
                int prev[] = {i, j-1};
                if (out[prev] < in[ind2]) {
                    out[ind2] = in[ind2];
                    indexes[ind2] = i*w+j;
                }
                else {
                    out[ind2] = out[prev];
                    indexes[ind2] = indexes[prev];
                }
            }
            //z = in[i];
            ''', 'corner_pool_left_fwd')(x[0].reshape(-1, w), w, y.reshape(-1, w), self.indexes.reshape(-1, w), cuda.cupy.arange(n * c * h, dtype=numpy.int32))
        print(x[0])
        print(self.indexes)
        #print(z)
        #print(x[0])
        return y

if __name__ == '__main__':
    from chainer import cuda
    import cupy as xp
    import numpy as np
    
    cp = CornerPoolingLeft()
    a = xp.arange(24*2)[::-1].reshape((2, 2, 4, 3)).astype(np.float32)
    a[0, 0, 1, 1] = 100
    print(a.reduced_view())
    print(a)
    print(cp.forward_gpu([a]))

