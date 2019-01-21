import numpy 


class CornerPoolingLeft(object):
    def forward_gpu(self, x):
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        n, c, h, w = x[0].shape

        y = cuda.cupy.zeros((n, c, h, w), dtype=x[0].dtype)
        z = cuda.cupy.zeros((n, c, h, w), dtype=x[0].dtype)

        self.indexes = cuda.cupy.empty((n, c, h, w), dtype=numpy.int32)

        cuda.elementwise(
            'raw T in, int32 h, int32 w',
            'raw T out, S indexes, T z',
            '''
               int c0    = i / (h * w);
               int y = i / w % h;
               int x = i % w;
               int prev_x = x - 1;
               if (x == 0) {
                   out[i] = in[i];
                   indexes = i;
                   return;
               }
               int prev_index = ((c0 * h + y) * w) + prev_x;
               if (out[prev_index] < in[i]){
                   out[i] = in[i];
                   indexes = i;
               }
               else {
                    out[i] = out[prev_index];
                    indexes = prev_index;
               }
               z = out[prev_index];
            ''', 'corner_pool_left_fwd')(x[0].reduced_view(), h, w, y.reduced_view(), self.indexes, z)
        print(self.indexes)
        print(z)
        return y

if __name__ == '__main__':
    from chainer import cuda
    import cupy as xp
    import numpy as np
    
    cp = CornerPoolingLeft()
    a = xp.arange(9)[::-1].reshape((1, 1, 3, 3)).astype(np.float32)
    print(a)
    print(cp.forward_gpu([a]))

