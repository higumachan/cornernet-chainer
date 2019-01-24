import numpy 
from chainer.function_node import FunctionNode
import chainer.functions as F
from chainer.utils import type_check


class CornerPoolingLeft(FunctionNode):
    def check_type_forward(self, in_types):
            type_check.expect(
                in_types.size() == 1,
                in_types[0].dtype.kind == 'f',
                in_types[0].ndim == 4
            )

    def forward_cpu(self, x):
        pass

    def forward_gpu(self, x):
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        n, c, h, w = x[0].shape

        y = cuda.cupy.zeros((n, c, h, w), dtype=x[0].dtype)
        z = cuda.cupy.zeros((n, c, h, w), dtype=x[0].dtype)

        self.indexes = cuda.cupy.empty((n, c, h, w), dtype=numpy.int32)
        cuda.elementwise(
            'raw T in, int32 w',
            'raw T out, raw S indexes, S z',
            '''
            int ind[] = {i, w-1};
            out[ind] = in[ind];
            indexes[ind] = i * w + (w-1);
            for (int j = w-1; j >= 0; j--) {
                int ind2[] = {i, j};
                int prev[] = {i, j+1};
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
        return y,

    def backward_gpu(self, gout):
        gy, = gout
        n, c, h, w = gy.shape

        gx = cuda.cupy.empty((n, c, h, w), dtype=gy.dtype)
        
        cuda.elementwise(
                'raw T in, raw S indexes', 'T out',
                '''
                int index = indexes[i];
                out = in[index];
                ''', 'max_pool_grad_fwd')(
                    gy.reduced_view(), self.indexes.reduced_view(), gx.reduced_view())
        return gx,


def corner_pooling_left(x):
    func = CornerPoolingLeft()

    return func.apply((x,))[0]


def corner_pooling_top(x):
    h = F.transpose(x, (0, 1, 3, 2))
    func = CornerPoolingLeft()
    h = func.apply((h,))[0]
    return F.transpose(h, (0, 1, 3, 2))


if __name__ == '__main__':
    from chainer import cuda
    import chainer
    import cupy as xp
    import numpy as np
    
    cp = CornerPoolingLeft()
    a = xp.arange(24*2).reshape((2, 2, 4, 3)).astype(np.float32)
    b = xp.arange(24*2).reshape((2, 2, 4, 3)).astype(np.float32)
    a[0, 0, 1, 1] = 100
    print(a.reduced_view())
    print(a)
    print(cp.forward_gpu([a]))
    print(cp.backward_gpu([b]))

    va = chainer.Variable(a)   
    print(va)
    print(corner_pooling_left(va))
    print(corner_pooling_left(va).shape)

    va = chainer.Variable(a)
    print(va)
    print(corner_pooling_top(va))
    print(corner_pooling_top(va).shape)

