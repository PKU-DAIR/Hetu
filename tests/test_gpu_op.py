import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op


def test_array_set():
    ctx = ht.gpu(0)
    shape = (500, 200)
    # oneslike
    arr_x = ht.empty(shape, ctx=ctx)
    gpu_op.array_set(arr_x, 1.)
    x = arr_x.asnumpy()
    np.testing.assert_allclose(np.ones(shape), x)
    # zeroslike
    gpu_op.array_set(arr_x, 0.)
    x = arr_x.asnumpy()
    np.testing.assert_allclose(np.zeros(shape), x)


def test_broadcast_to():
    ctx = ht.gpu(0)
    shape = (200, 300)
    to_shape = (130, 200, 300)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(to_shape, ctx=ctx)
    gpu_op.broadcast_to(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.broadcast_to(x, to_shape), y)


def test_reduce_sum_axis_zero():
    ctx = ht.gpu(0)
    shape = (20, 1, 1)
    temp_shape = list(shape)
    temp_shape[0] = (temp_shape[0] + 1) // 2
    temp_shape = tuple(temp_shape)
    to_shape = (1, 1)
    x = np.random.uniform(0, 20, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(to_shape, ctx=ctx)
    arr_workspace = ht.empty(shape=temp_shape, ctx=ctx)
    gpu_op.reduce_sum_axis_zero(arr_x, arr_y, arr_workspace)
    y = arr_y.asnumpy()
    y_ = np.sum(x, axis=0)
    for index, _ in np.ndenumerate(y):
        v = y[index]
        v_ = y_[index]
        if abs((v - v_) / v_) > 1e-4:
            print(index, v, v_)
    np.testing.assert_allclose(np.sum(x, axis=0), y, rtol=1e-5)


def test_matrix_elementwise_add():
    ctx = ht.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_add(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x + y, z, rtol=1e-5)


def test_matrix_elementwise_add_by_const():
    shape = (2000, 3000)
    ctx = ht.gpu(0)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    val = np.random.uniform(-5, 5)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_add_by_const(arr_x, val, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x + val, y, rtol=1e-5)


def test_matrix_elementwise_multiply():
    ctx = ht.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_multiply(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x * y, z, rtol=1e-5)


def test_matrix_elementwise_multiply_by_const():
    shape = (2000, 3000)
    ctx = ht.gpu(0)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    val = np.random.uniform(-5, 5)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_multiply_by_const(arr_x, val, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x * val, y, rtol=1e-5)


def test_matrix_elementwise_divide():
    ctx = ht.gpu(0)
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    y = np.random.uniform(1, 10, size=shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_divide(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x / y, z, rtol=1e-5)


def test_matrix_elementwise_divide_const():
    shape = (2000, 3000)
    ctx = ht.gpu(0)
    val = np.random.uniform(-5, 5)
    x = np.random.uniform(1, 10, size=shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_elementwise_divide_const(val, arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(val / x, y, rtol=1e-5)


def test_matrix_opposite():
    shape = (2000, 2500)
    ctx = ht.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_opposite(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(-x, y)


def test_matrix_multiply():
    ctx = ht.gpu(0)
    x = np.random.uniform(0, 10, size=(500, 700)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(700, 1000)).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty((500, 1000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, False, arr_y, False, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, y), z, rtol=1e-5)

    x = np.random.uniform(0, 10, size=(1000, 500)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty((1000, 2000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, False, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(x, np.transpose(y)), z, rtol=1e-5)

    x = np.random.uniform(0, 10, size=(500, 1000)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty((1000, 2000), ctx=ctx)
    gpu_op.matrix_multiply(arr_x, True, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.dot(np.transpose(x), np.transpose(y)), z,
                               rtol=1e-5)


def test_matrix_sqrt():
    shape = (2000, 2500)
    ctx = ht.gpu(0)
    x = np.random.uniform(0, 10, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_sqrt(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.sqrt(x), y, rtol=1e-5)


def test_matrix_rsqrt():
    shape = (2000, 2500)
    ctx = ht.gpu(0)
    x = np.random.uniform(0, 10, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.matrix_rsqrt(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(1 / np.sqrt(x), y, rtol=1e-5)


def test_relu():
    shape = (2000, 2500)
    ctx = ht.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.relu(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.maximum(x, 0).astype(np.float32), y)


def test_leaky_relu():
    shape = (2000, 2500)
    ctx = ht.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    alpha = 10
    gpu_op.leaky_relu(arr_x, float(alpha), arr_y)
    y = arr_y.asnumpy()


def test_relu_gradient():
    shape = (2000, 2500)
    ctx = ht.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    grad_x = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_grad_x = ht.array(grad_x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.relu_gradient(arr_x, arr_grad_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(((x > 0) * grad_x).astype(np.float32), y)


def test_leaky_relu_gradient():
    shape = (2000, 2500)
    ctx = ht.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    grad_x = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_grad_x = ht.array(grad_x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    alpha = 10
    gpu_op.leaky_relu_gradient(arr_x, arr_grad_x, alpha, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(
        np.where(np.greater(x, 0), grad_x, alpha * grad_x).astype(np.float32), y)


def test_softmax():
    def softmax_func(y):
        """Numerically stable softmax."""
        b = y - np.max(y, axis=1, keepdims=True)
        expb = np.exp(b)
        softmax = expb / np.sum(expb, axis=1, keepdims=True)
        return softmax
    ctx = ht.gpu(0)
    shape = (400, 1000)
    x = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shape, ctx=ctx)
    gpu_op.softmax(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(softmax_func(x), y, rtol=1e-5)


def test_softmax_cross_entropy():
    def softmax_func(y):
        """Numerically stable softmax."""
        b = y - np.max(y, axis=1, keepdims=True)
        expb = np.exp(b)
        softmax = expb / np.sum(expb, axis=1, keepdims=True)
        return softmax
    def convert_to_one_hot(vals, max_val=0, ignored_index = -1):
        """Helper method to convert label array to one-hot array."""
        if max_val == 0:
            max_val = vals.max() + 1
        one_hot_vals = np.zeros((vals.size, max_val))
        for i in range(vals.size):
            if vals[i] == ignored_index:
                continue
            one_hot_vals[i, vals[i]] = 1
        return one_hot_vals
    ctx = ht.gpu(0)
    shape = (400, 1000)
    #shape = (4,5)
    y = np.random.uniform(-5, 5, shape).astype(np.float32)
    # print('y value:')
    # print(y)
    y_sparse = np.random.randint(-1,shape[1],(shape[0],))
    y_ = convert_to_one_hot(y_sparse, shape[1])
    # print("y_ value:")
    # print(y_)
    # print(y_sparse)

    # numpy calculation
    cross_entropy = -np.sum(y_ * np.log(softmax_func(y)), axis=1)
    # print("Numpy result:")
    # print(cross_entropy)

    # test manual kernel
    arr_y = ht.array(y, ctx=ctx)
    arr_y_ = ht.array(y_, ctx=ctx)
    arr_out = ht.empty((shape[0],), ctx=ctx)
    gpu_op.softmax_cross_entropy(arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    # print("Manual fused kernel result:")
    # print(out)
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-4, atol=1e-8)

    # test cudnn
    gpu_op.CuDNN_softmax_cross_entropy(arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    # print("CuDNN fused kernel result:")
    # print(out)
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-4, atol=1e-8)

    # test softmax_op + crossentropy_op split version
    arr_softmax = ht.empty(shape, ctx=ctx)
    gpu_op.CuDNN_softmax(arr_y, arr_softmax)
    gpu_op.cross_entropy(arr_softmax, arr_y_, arr_out)
    out = arr_out.asnumpy()
    # print("softmax_op + crossentropy_op split version result:")
    # print(out)
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-4, atol=1e-8)

    # test softmax_op + crossentropy_sparse_op
    ignored_index = -1
    arr_y_sparse = ht.array(y_sparse, ctx=ctx)
    gpu_op.CuDNN_softmax(arr_y, arr_softmax)
    gpu_op.cross_entropy_sparse(arr_softmax, arr_y_sparse, ignored_index, arr_out)
    out = arr_out.asnumpy()
    # print("softmax_op + crossentropy_sparse_op split version result:")
    # print(out)
    np.testing.assert_allclose(cross_entropy, out, rtol=1e-4, atol=1e-8)


def test_softmax_cross_entropy_gradient():
    def softmax_func(y):
        """Numerically stable softmax."""
        b = y - np.max(y, axis=1, keepdims=True)
        expb = np.exp(b)
        softmax = expb / np.sum(expb, axis=1, keepdims=True)
        return softmax
    def convert_to_one_hot(vals, max_val=0, ignored_index = -1):
        """Helper method to convert label array to one-hot array."""
        if max_val == 0:
            max_val = vals.max() + 1
        one_hot_vals = np.zeros((vals.size, max_val))
        for i in range(vals.size):
            if vals[i] == ignored_index:
                continue
            one_hot_vals[i, vals[i]] = 1
        return one_hot_vals

    ctx = ht.gpu(0)
    shape = (400, 1000)
    #shape = (4,5)
    y = np.random.uniform(-5, 5, shape).astype(np.float32)
    y_sparse = np.random.randint(0,shape[1],(shape[0],))
    y_ = convert_to_one_hot(y_sparse, shape[1])
    grad = np.random.uniform(-5, 5, (shape[0],)).astype(np.float32)
    arr_y = ht.array(y, ctx=ctx)
    arr_y_ = ht.array(y_, ctx=ctx)
    arr_grad = ht.array(grad, ctx=ctx)
    arr_out = ht.empty(shape, ctx=ctx)

    # numpy calculation
    np_grad = (softmax_func(y) + -1 * y_) * np.expand_dims(grad, -1)
    
    # test manual kernel 
    gpu_op.softmax_cross_entropy_gradient(arr_y, arr_y_, arr_grad, arr_out)
    out = arr_out.asnumpy()
    np.testing.assert_allclose(np_grad, out, rtol=1e-4, atol=1e-8)

    # test cudnn
    gpu_op.CuDNN_softmax_cross_entropy_gradient(
        arr_grad, arr_y, arr_y_, arr_out)
    out = arr_out.asnumpy()
    np.testing.assert_allclose(np_grad, out, rtol=1e-4, atol=1e-8)

    # test softmax_op + crossentropy_op split version
    arr_softmax = ht.empty(shape, ctx=ctx)
    arr_softmax_grad = ht.empty(shape, ctx=ctx)
    arr_crossentropy_out = ht.empty((shape[0],), ctx=ctx)
    gpu_op.CuDNN_softmax(arr_y, arr_softmax)
    gpu_op.cross_entropy(arr_softmax, arr_y_, arr_crossentropy_out)

    # print("Hetu crossentropy output:")
    # print(arr_crossentropy_out.asnumpy())

    gpu_op.cross_entropy_gradient(arr_grad, arr_softmax, arr_y_, arr_softmax_grad)
    gpu_op.CuDNN_softmax_gradient(arr_softmax, arr_softmax_grad, arr_out)
    out = arr_out.asnumpy()

    # print("Hetu crossentropy gradient:")
    # print(arr_softmax_grad.asnumpy())
    # print("Hetu softmax gradient:")
    # print(out)

    np.testing.assert_allclose(np_grad, out, rtol=1e-4, atol=1e-8)

    # test softmax_op + crossentropy_sparse_op
    y_sparse = np.random.randint(-1,shape[1],(shape[0],))
    y_ = convert_to_one_hot(y_sparse, shape[1])
    np_grad = (softmax_func(y) + -1 * y_) * np.expand_dims(grad, -1)
    for i in range(y_sparse.size):
        if y_sparse[i] == -1:
            np_grad[i] = np.zeros(shape = (1,shape[1]))
    ignored_index = -1
    arr_y_sparse = ht.array(y_sparse, ctx=ctx)
    gpu_op.CuDNN_softmax(arr_y, arr_softmax)
    gpu_op.cross_entropy_sparse(arr_softmax, arr_y_sparse, ignored_index, arr_crossentropy_out)
    gpu_op.cross_entropy_sparse_gradient(arr_grad, arr_softmax, arr_y_sparse, ignored_index, arr_softmax_grad)
    gpu_op.CuDNN_softmax_gradient(arr_softmax, arr_softmax_grad, arr_out)
    out = arr_out.asnumpy()
    np.testing.assert_allclose(np_grad, out, rtol=1e-4, atol=1e-8)

    # pytorch autograd test
    import torch
    import torch.autograd as autograd
    from torch import Tensor
    import torch.nn.functional as F
    from torch.autograd import Variable
    tc_y = Tensor(y)
    tc_y = Variable(tc_y, requires_grad = True)
    tc_y_ = Tensor(y_)
    tc_grad = Tensor(grad)

    def torch_cross_entropy(pred, label):
        temp = - torch.log(pred) * label
        return torch.sum(temp, dim = -1)

    tc_softmax = F.softmax(tc_y, dim = -1)
    tc_softmax_1 = Variable(tc_softmax.data, requires_grad = True)
    tc_crossentropy = torch_cross_entropy(tc_softmax_1, tc_y_)
    # print("Torch crossentropy output:")
    # print(tc_crossentropy)
    autograd.backward(tc_crossentropy, tc_grad)
    # print("Torch crossentropy gradient:")
    # print(tc_softmax_1.grad)
    autograd.backward(tc_softmax, tc_softmax_1.grad)
    # print("Torch softmax gradient:")
    # print(tc_y.grad)

    # print("Numpy softmax gradient:")
    # print(np_grad)
    np.testing.assert_allclose(tc_y.grad, out, rtol=1e-4, atol=1e-8)

def test_softmax_gradient():
    def softmax_func(y):
        """Numerically stable softmax."""
        b = y - np.max(y, axis=1, keepdims=True)
        expb = np.exp(b)
        softmax = expb / np.sum(expb, axis=1, keepdims=True)
        return softmax
    ctx = ht.gpu(0)
    # shape = (400, 1000)
    shape = (4,5)
    y = np.random.uniform(-5, 5, shape).astype(np.float32)
    grad = np.random.uniform(-5, 5, shape).astype(np.float32)
    arr_y = ht.array(y, ctx=ctx)
    arr_grad = ht.array(grad, ctx=ctx)
    arr_out = ht.empty(shape, ctx=ctx)
    arr_grad_out = ht.empty(shape, ctx=ctx)

    np_softmax = softmax_func(y)

    gpu_op.CuDNN_softmax(arr_y, arr_out)
    print(np_softmax)
    print(arr_out.asnumpy())

    gpu_op.CuDNN_softmax_gradient_recompute(arr_y, arr_grad, arr_grad_out)
    print(arr_grad_out.asnumpy())

    gpu_op.CuDNN_softmax_gradient(arr_out, arr_grad, arr_grad_out)
    print(arr_grad_out.asnumpy())


    # out = arr_out.asnumpy()
    # # numpy calculation
    # np_grad = (softmax_func(y) + -1 * y_) * np.expand_dims(grad, -1)
    # np.testing.assert_allclose(np_grad, out, rtol=1e-4, atol=1e-8)

    # # test cudnn
    # gpu_op.CuDNN_softmax_cross_entropy_gradient(
    #     arr_grad, arr_y, arr_y_, arr_out)
    # out = arr_out.asnumpy()
    # np.testing.assert_allclose(np_grad, out, rtol=1e-4, atol=1e-8)

def test_conv2d():
    ctx = ht.gpu(0)
    # im2col and np_conv2d are helper functions

    def im2col(X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        print("shape", im2col_matrix.shape)
        print("shape", filter_matrix.shape)
        print("shape", np.matmul(filter_matrix, im2col_matrix).shape)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)
        # return im2col_matrix

    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    shapeW = (100, 3*5*5, 24*24)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)
    y = np.zeros(shapeY).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_f = ht.array(f, ctx=ctx)
    arr_y = ht.empty(shapeY, ctx=ctx)
    arr_workspace = ht.empty(shapeW, ctx=ctx)

    gpu_op.conv2d(arr_x, arr_f, arr_y, arr_workspace)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np_conv2d(x, f), y, rtol=1e-5)


def test_conv2d_Gradient():
    ctx = ht.gpu(0)

    def im2col(X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)

    def im2col_transpose(X, filter_H, filter_W, Y, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1
        _, y_row_size, y_col_size = Y.shape

        der_X_shape = (N, C, H, W)
        der_X = np.zeros(der_X_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                der_X[batch_index, c, y,
                                      x] += Y[batch_index, row_idx, col_index]
                            row_idx += 1
        return der_X

    def np_conv2d_transpose(X, Filter, Y, padding=0, stride=1):
        """Implement a conv2d_transpose as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        X_N, X_C, X_H, X_W = X.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        # XX = X.reshape((X_N, X_C, X_W * X_H))   # transformed to im2col X
        F_filter = Filter.reshape((filter_outChannel, -1))
        gradient_im2col_XX = np.matmul(F_filter.T, YY)

        gradient_X = im2col_transpose(
            X, filter_H, filter_W, gradient_im2col_XX, padding, stride)    # gradient of x
        im2col_XX = im2col(X, filter_H, filter_W, padding, stride)
        gradient_filter = np.zeros(shape=F_filter.shape, dtype=X.dtype)

        for i in range(X_N):
            gradient_filter += np.matmul(YY[i], im2col_XX[i].T)
        gradient_filter = gradient_filter.reshape(Filter.shape)

        return gradient_X, gradient_filter

    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    shapeW = (100, 3*5*5, 24*24)
    shapeFF = (100, 10, 3, 5, 5)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)

    der_y = np.ones(shape=shapeY)
    gradient_x, gradient_f = np_conv2d_transpose(x, f, der_y)

    arr_x = ht.array(x, ctx=ctx)
    arr_f = ht.array(f, ctx=ctx)
    gradient_y = ht.array(der_y, ctx=ctx)
    gradient_xx = ht.array(x, ctx=ctx)
    gradient_ff = ht.array(f, ctx=ctx)

    arr_workspace_im2col = ht.empty(shapeW, ctx=ctx)
    arr_workspace_batch_filter = ht.empty(shapeFF, ctx=ctx)
    gpu_op.conv2d_gradient_of_filter(
        arr_x, gradient_y, gradient_ff, arr_workspace_im2col, arr_workspace_batch_filter)
    gpu_op.conv2d_gradient_of_data(
        arr_f, gradient_y, gradient_xx, arr_workspace_im2col)

    np.testing.assert_allclose(gradient_x, gradient_xx.asnumpy(), rtol=1e-5)
    # test ok
    np.testing.assert_allclose(gradient_f, gradient_ff.asnumpy(), rtol=1e-5)


def test_average_pooling():
    ctx = ht.gpu(0)

    def np_average_pooling(input, kernel_H, kernel_W, padding=0, stride=1):
        N, C, H, W = input.shape
        assert((H + 2 * padding - kernel_H) % stride == 0)
        assert((W + 2 * padding - kernel_W) % stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) // stride + 1
        pooled_W = (W + 2 * padding - kernel_W) // stride + 1
        pooled_layer = np.zeros(
            shape=(N, C, pooled_H, pooled_W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                pooled_layer[n][c][h][w] += input[n][c][i][j]
                        pooled_layer[n][c][h][w] /= pooling_size
        return pooled_layer

    def np_average_pooling_gradient(gradient_y, kernel_H, kernel_W, padding=0, stride=1):
        N, C, pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding

        gradient_x = np.zeros(shape=(N, C, H, W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                gradient_x[n][c][i][j] += gradient_y[n][c][h][w] / \
                                    pooling_size

        return gradient_x

    shapeX = (100, 3, 28, 28)
    # (1,1,5,5)
    shapeY = (100, 3, 24, 24)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    gradient_y = np.random.uniform(0, 10, size=shapeY).astype(np.float32)

    arr_x = ht.array(x, ctx=ctx)
    arr_gradient_y = ht.array(gradient_y, ctx=ctx)
    arr_pool_layer = ht.empty(shapeY, ctx=ctx)
    arr_gradient_x = ht.empty(shapeX, ctx=ctx)

    gpu_op.average_pooling2d(arr_x, 5, 5, arr_pool_layer)
    gpu_op.average_pooling2d_gradient(arr_gradient_y, 5, 5, arr_gradient_x)

    np_pool_layer = np_average_pooling(x, 5, 5)
    np_gradient_x = np_average_pooling_gradient(gradient_y, 5, 5)

    np.testing.assert_allclose(
        np_pool_layer, arr_pool_layer.asnumpy(), rtol=1e-5)

    np.testing.assert_allclose(
        np_gradient_x, arr_gradient_x.asnumpy(), rtol=1e-5)


def test_reshape():
    ctx = ht.gpu(0)

    def np_reshape(X, output_shape):
        return X.reshape(output_shape)

    shapeX = (10, 5, 28, 28)
    shapeY = (50, 28, 28)

    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    y = np_reshape(x, shapeY)

    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shapeY, ctx=ctx)
    gpu_op.array_reshape(arr_x, arr_y)
    np.testing.assert_allclose(y, arr_y.asnumpy(), rtol=1e-5)


def test_conv2d_broadcast_to():
    ctx = ht.gpu(0)
    shapeX = (32)
    shapeY = (100, 32, 28, 28)
    shapeW = (100, 28, 28, 32)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    np_y = np.broadcast_to(x, shapeW)
    np_y = np_y.swapaxes(1, 3)

    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shapeY, ctx=ctx)
    gpu_op.conv2d_broadcast_to(arr_x, arr_y)

    np.testing.assert_allclose(np_y, arr_y.asnumpy(), rtol=1e-5)


def test_conv2d_reduce_sum():
    ctx = ht.gpu(0)
    shapeX = (32,)
    shapeY = (100, 32, 28, 28)
    shapeW = (100, 28, 28, 32)
    x = np.random.uniform(0, 10, size=shapeY).astype(np.float32)
    np_y = np.sum(x, axis=(0, 2, 3))

    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shapeX, ctx=ctx)

    gpu_op.conv2d_reduce_sum(arr_x, arr_y)
    np.testing.assert_allclose(np_y, arr_y.asnumpy(), rtol=1e-5)


def test_cudnn_conv2d():
    ctx = ht.gpu(0)
    # im2col and np_conv2d are helper functions

    def im2col(X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        print("shape", im2col_matrix.shape)
        print("shape", filter_matrix.shape)
        print("shape", np.matmul(filter_matrix, im2col_matrix).shape)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)
        # return im2col_matrix

    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    shapeW = (100, 3*5*5, 24*24)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)
    y = np.zeros(shapeY).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_f = ht.array(f, ctx=ctx)
    arr_y = ht.empty(shapeY, ctx=ctx)
    arr_workspace = ht.empty(shapeW, ctx=ctx)

    gpu_op.CuDNN_conv2d(arr_x, arr_f, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np_conv2d(x, f), y, rtol=1e-5)


def test_cudnn_conv2d_Gradient():
    ctx = ht.gpu(0)

    def im2col(X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)

    def im2col_transpose(X, filter_H, filter_W, Y, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1
        _, y_row_size, y_col_size = Y.shape

        der_X_shape = (N, C, H, W)
        der_X = np.zeros(der_X_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                der_X[batch_index, c, y,
                                      x] += Y[batch_index, row_idx, col_index]
                            row_idx += 1
        return der_X

    def np_conv2d_transpose(X, Filter, Y, padding=0, stride=1):
        """Implement a conv2d_transpose as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        X_N, X_C, X_H, X_W = X.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        # XX = X.reshape((X_N, X_C, X_W * X_H))   # transformed to im2col X
        F_filter = Filter.reshape((filter_outChannel, -1))
        gradient_im2col_XX = np.matmul(F_filter.T, YY)

        gradient_X = im2col_transpose(
            X, filter_H, filter_W, gradient_im2col_XX, padding, stride)    # gradient of x
        im2col_XX = im2col(X, filter_H, filter_W, padding, stride)
        gradient_filter = np.zeros(shape=F_filter.shape, dtype=X.dtype)

        for i in range(X_N):
            gradient_filter += np.matmul(YY[i], im2col_XX[i].T)
        gradient_filter = gradient_filter.reshape(Filter.shape)

        return gradient_X, gradient_filter

    shapeX = (100, 3, 28, 28)
    shapeF = (10, 3, 5, 5)
    shapeY = (100, 10, 24, 24)
    shapeW = (100, 3*5*5, 24*24)
    shapeFF = (100, 10, 3, 5, 5)
    #  input : x , filter : f , output: y
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    f = np.random.uniform(0, 10, size=shapeF).astype(np.float32)

    der_y = np.ones(shape=shapeY)
    gradient_x, gradient_f = np_conv2d_transpose(x, f, der_y)

    arr_x = ht.array(x, ctx=ctx)
    arr_f = ht.array(f, ctx=ctx)
    gradient_y = ht.array(der_y, ctx=ctx)
    gradient_xx = ht.array(x, ctx=ctx)
    gradient_ff = ht.array(f, ctx=ctx)

    arr_workspace_im2col = ht.empty(shapeW, ctx=ctx)
    arr_workspace_batch_filter = ht.empty(shapeFF, ctx=ctx)
    gpu_op.CuDNN_conv2d_gradient_of_filter(arr_x, gradient_y, gradient_ff)
    gpu_op.CuDNN_conv2d_gradient_of_data(arr_f, gradient_y, gradient_xx)

    np.testing.assert_allclose(gradient_x, gradient_xx.asnumpy(), rtol=1e-5)
    # test ok
    np.testing.assert_allclose(gradient_f, gradient_ff.asnumpy(), rtol=1e-5)


def test_average_pooling():
    ctx = ht.gpu(0)

    def np_average_pooling(input, kernel_H, kernel_W, padding=0, stride=1):
        N, C, H, W = input.shape
        assert((H + 2 * padding - kernel_H) % stride == 0)
        assert((W + 2 * padding - kernel_W) % stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) // stride + 1
        pooled_W = (W + 2 * padding - kernel_W) // stride + 1
        pooled_layer = np.zeros(
            shape=(N, C, pooled_H, pooled_W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                pooled_layer[n][c][h][w] += input[n][c][i][j]
                        pooled_layer[n][c][h][w] /= pooling_size
        return pooled_layer

    def np_average_pooling_gradient(gradient_y, kernel_H, kernel_W, padding=0, stride=1):
        N, C, pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding

        gradient_x = np.zeros(shape=(N, C, H, W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                gradient_x[n][c][i][j] += gradient_y[n][c][h][w] / \
                                    pooling_size

        return gradient_x

    shapeX = (100, 3, 28, 28)
    shapeY = (100, 3, 24, 24)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    gradient_y = np.random.uniform(0, 10, size=shapeY).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_gradient_y = ht.array(gradient_y, ctx=ctx)
    arr_pool_layer = ht.empty(shapeY, ctx=ctx)
    arr_gradient_x = ht.empty(shapeX, ctx=ctx)

    gpu_op.CuDNN_average_pooling2d(arr_x, 5, 5, arr_pool_layer)
    gpu_op.CuDNN_average_pooling2d_gradient(
        arr_pool_layer, arr_gradient_y, arr_x, 5, 5, arr_gradient_x)
    np_pool_layer = np_average_pooling(x, 5, 5)
    np_gradient_x = np_average_pooling_gradient(gradient_y, 5, 5)

    np.testing.assert_allclose(
        np_pool_layer, arr_pool_layer.asnumpy(), rtol=1e-5)

    np.testing.assert_allclose(
        np_gradient_x, arr_gradient_x.asnumpy(), rtol=1e-5)


def test_CuDNN_max_pooling():
    ctx = ht.gpu(0)

    def np_max_pooling_gradient(input, gradient_y, kernel_H, kernel_W, padding=0, stride=1):
        N, C, pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding
        gradient_x = np.zeros(shape=(N, C, H, W), dtype=np.float32)
        pooling_size = kernel_H * kernel_W

        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)

                        hargmax = hs
                        wargmax = ws
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                if input[n][c][i][j] > input[n][c][hargmax][wargmax]:
                                    hargmax = i
                                    wargmax = j
                        gradient_x[n][c][hargmax][wargmax] += gradient_y[n][c][h][w]

        return gradient_x

    shapeX = (100, 3, 28, 28)
    # # (1,1,5,5)
    shapeY = (100, 3, 24, 24)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    gradient_y = np.random.uniform(0, 10, size=shapeY).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_gradient_y = ht.array(gradient_y, ctx=ctx)
    arr_pool_layer = ht.empty(shapeY, ctx=ctx)
    arr_gradient_x = ht.empty(shapeX, ctx=ctx)

    arr_pool_layer1 = ht.empty(shapeY, ctx=ctx)
    arr_gradient_x1 = ht.empty(shapeX, ctx=ctx)

    gpu_op.CuDNN_max_pooling2d(arr_x, 2, 2, arr_pool_layer)
    gpu_op.CuDNN_max_pooling2d_gradient(
        arr_pool_layer, arr_gradient_y, arr_x, 2, 2, arr_gradient_x)

    gpu_op.max_pooling2d(arr_x, 2, 2, arr_pool_layer1)
    gpu_op.max_pooling2d_gradient(arr_x, arr_gradient_y, 2, 2, arr_gradient_x1)

    np.testing.assert_allclose(
        arr_pool_layer.asnumpy(), arr_pool_layer1.asnumpy(), rtol=1e-5)

    np.testing.assert_allclose(
        arr_gradient_x.asnumpy(), arr_gradient_x1.asnumpy(), rtol=1e-5)


def test_CuDNN_dropout_op():
    import ctypes
    ctx = ht.gpu(0)
    shapeX = (3, 10)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shapeX, ctx=ctx)
    shapeK = (1)
    keep_prob = 0.5
    reserve_size = ctypes.c_int(0)
    reserve_space = ctypes.c_void_p(0)
    gpu_op.CuDNN_Dropout(arr_x, keep_prob, arr_y,
                         reserve_size, reserve_space, 1)

    gradient_y = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    gradient_y = ht.array(gradient_y, ctx=ctx)
    gradient_x = ht.empty(shapeX, ctx=ctx)
    gpu_op.CuDNN_Dropout_gradient(
        gradient_y, keep_prob, gradient_x, reserve_size, reserve_space)
    print(arr_y.asnumpy())
    print(gradient_x.asnumpy())


def test_pad():
    ctx = ht.gpu(0)
    shape = (1, 1, 1, 3)
    paddings = [[1, 1], [1, 1]]
    to_shape = (1, 1, 3, 5)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(to_shape, ctx=ctx)
    gpu_op.pad(arr_x, arr_y, paddings)
    print(arr_x.asnumpy())
    print(arr_y.asnumpy())
    gradient_y = np.random.uniform(0, 10, size=to_shape).astype(np.float32)
    arr_gradient_y = ht.array(gradient_y, ctx=ctx)
    arr_gradient_x = ht.empty(shape, ctx=ctx)
    gpu_op.pad_gradient(arr_gradient_y, arr_gradient_x, paddings)
    print(arr_gradient_y.asnumpy())
    print(arr_gradient_x.asnumpy())


def test_concat():
    def unit_test(shape1, shape2, axis):
        ctx = ht.gpu(0)
        x1 = np.random.random(shape1).astype(np.float32)
        x2 = np.random.random(shape2).astype(np.float32)
        arr_x1 = ht.array(x1, ctx=ctx)
        arr_x2 = ht.array(x2, ctx=ctx)
        np_res = np.concatenate([x1, x2], axis)
        arr_res = ht.empty(np_res.shape, ctx=ctx)
        gpu_op.concat(arr_x1, arr_x2, arr_res, axis)
        np.testing.assert_allclose(arr_res.asnumpy(), np_res)

        grad_x1 = ht.empty(shape1, ctx=ctx)
        grad_x2 = ht.empty(shape2, ctx=ctx)
        gpu_op.concat_gradient(arr_res, grad_x1, axis=axis, idx=0)
        gpu_op.concat_gradient(arr_res, grad_x2, axis=axis, idx=1)
        np.testing.assert_allclose(x1, grad_x1.asnumpy())
        np.testing.assert_allclose(x2, grad_x2.asnumpy())

        print("Pass test with ", shape1, shape2, axis)

    unit_test((1, 2), (1, 2), 0)
    unit_test((12, 34, 56), (12, 43, 56), 1)


def test_matrix_transpose():
    shape = (4321, 1234)
    ctx = ht.gpu(0)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty((shape[1], shape[0]), ctx=ctx)
    gpu_op.matrix_transpose(arr_x, arr_y, perm=[1, 0])
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.transpose(x), y)

    shape = (21, 43, 65, 11)
    x = np.random.uniform(-1, 1, shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty((65, 11, 43, 21), ctx=ctx)
    gpu_op.matrix_transpose(arr_x, arr_y, perm=[2, 3, 1, 0])
    y = arr_y.asnumpy()
    np.testing.assert_allclose(np.transpose(x, [2, 3, 1, 0]), y)


def test_slice():
    i_shape = (123, 234, 13, 7)
    o_shape = (67, 209, 3, 5)
    begin_pos = (31, 11, 3, 1)

    ctx = ht.gpu(0)
    x = np.random.normal(size=i_shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(o_shape, ctx=ctx)

    gpu_op.matrix_slice(arr_x, arr_y, begin_pos)
    y = arr_y.asnumpy()
    index = tuple([slice(i, i+j) for i, j in zip(begin_pos, o_shape)])
    np.testing.assert_allclose(x[index], y)
    print("Slice op no bug.")

    begin_pos = (29, 3, 5, 0)
    gpu_op.matrix_slice_gradient(arr_y, arr_x, begin_pos)
    x = arr_x.asnumpy()
    index = tuple([slice(i, i+j) for i, j in zip(begin_pos, o_shape)])
    x_ = np.zeros(i_shape, dtype=np.float32)
    x_[index] = y
    np.testing.assert_allclose(x, x_)
    print("Slice gradient op no bug.")


def test_where():
    cond = np.random.randint(2, size=(5, 5)).astype(np.float32)
    x = np.random.rand(5, 5).astype(np.float32)
    y = np.random.rand(5, 5).astype(np.float32)
    ctx = ht.gpu(0)
    cond_gpu = ht.array(cond, ctx=ctx)
    x_gpu = ht.array(x, ctx=ctx)
    y_gpu = ht.array(y, ctx=ctx)
    output_gpu = ht.empty((5, 5), ctx=ctx)

    gpu_op.where(cond_gpu, x_gpu, y_gpu, output_gpu)
    output = np.where(cond, x, y)

    np.testing.assert_allclose(output, output_gpu.asnumpy())


def test_batch_matrix_multiply():
    ctx = ht.gpu(0)
    x = np.random.uniform(0, 10, size=(2, 3, 500, 700)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(2, 3, 700, 1000)).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty((2, 3, 500, 1000), ctx=ctx)
    gpu_op.batch_matrix_multiply(arr_x, False, arr_y, False, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.matmul(x, y), z, rtol=1e-5)
    print('Test 1 passed.')

    x = np.random.uniform(0, 10, size=(7, 11, 1000, 500)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(7, 11, 2000, 500)).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty((7, 11, 1000, 2000), ctx=ctx)
    gpu_op.batch_matrix_multiply(arr_x, False, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(
        np.matmul(x, np.transpose(y, [0, 1, 3, 2])), z, rtol=1e-5)
    print('Test 2 passed.')

    x = np.random.uniform(0, 10, size=(3, 2, 5, 500, 1000)).astype(np.float32)
    y = np.random.uniform(0, 10, size=(3, 2, 5, 2000, 500)).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.array(y, ctx=ctx)
    arr_z = ht.empty((3, 2, 5, 1000, 2000), ctx=ctx)
    gpu_op.batch_matrix_multiply(arr_x, True, arr_y, True, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(np.matmul(np.transpose(x, [0, 1, 2, 4, 3]), np.transpose(y, [0, 1, 2, 4, 3])), z,
                               rtol=1e-5)
    print('Test 3 passed.')


def test_broadcast_shape():
    def unit_test(shape1, shape2):
        ctx = ht.gpu(0)
        x = np.random.random(shape1).astype(np.float32)
        arr_x = ht.array(x, ctx=ctx)
        arr_y = ht.empty(shape2, ctx=ctx)
        gpu_op.broadcast_shape(arr_x, arr_y)
        np.testing.assert_allclose(arr_y.asnumpy(), np.broadcast_to(x, shape2))
        print('Passed test with input shape %s and output shape %s.' %
              (str(shape1), str(shape2)))

    unit_test((3, 1), (2, 3, 4))
    unit_test((1,), (2, 3, 4, 5))
    unit_test((1, 1, 3, 1), (9, 8, 3, 7))


def test_reduce_sum():
    def unit_test(shape, axes):
        ctx = ht.gpu(0)
        x = np.random.random(shape).astype(np.float32)
        arr_x = ht.array(x, ctx=ctx)
        o_shape = list(shape)
        for ax in axes:
            o_shape[ax] = 0
        o_shape = [i for i in o_shape if i > 0]
        arr_y = ht.empty(o_shape, ctx=ctx)
        gpu_op.reduce_sum(arr_x, arr_y, axes)
        np.testing.assert_allclose(arr_y.asnumpy(), np.sum(
            x, tuple(axes), keepdims=False), rtol=1e-6)
        print('Passed test with input shape %s and reduce axes %s.' %
              (str(shape), str(axes)))

    unit_test((2, 3, 4), [2])
    unit_test((2, 3, 4), [2, 1])
    unit_test((2, 3, 4), [2, 1, 0])
    unit_test((2, 3, 1, 5, 6), [1, 2, 4])


def test_reduce_mean():
    def unit_test(shape, axes):
        ctx = ht.gpu(0)
        x = np.random.random(shape).astype(np.float32)
        arr_x = ht.array(x, ctx=ctx)
        o_shape = list(shape)
        for ax in axes:
            o_shape[ax] = 0
        o_shape = [i for i in o_shape if i > 0]
        arr_y = ht.empty(o_shape, ctx=ctx)
        gpu_op.reduce_mean(arr_x, arr_y, axes)
        np.testing.assert_allclose(arr_y.asnumpy(), np.mean(
            x, tuple(axes), keepdims=False), rtol=1e-6)
        print('Passed test with input shape %s and reduce axes %s.' %
              (str(shape), str(axes)))

    unit_test((2, 3, 4), [2])
    unit_test((2, 3, 4), [2, 1])
    unit_test((2, 3, 4), [2, 1, 0])
    unit_test((2, 3, 1, 5, 6), [1, 2, 4])


def test_dropout_recompute():
    ctx = ht.gpu(0)
    # shapeX = (16, 8, 99, 64)
    shapeX = (10,)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shapeX, ctx=ctx)
    dropout_rate = 0.6
    import ctypes
    seed = ctypes.c_ulonglong(0)
    print(x)
    gpu_op.dropout(arr_x, dropout_rate, arr_y, seed)
    print(arr_y.asnumpy())
    print(seed)
    gpu_op.dropout_gradient_recompute(arr_x, dropout_rate, arr_y, seed)
    print(arr_y.asnumpy())
    print(seed)


def test_dropout():
    ctx = ht.gpu(0)
    shapeX = (200,500)
    #shapeX = (10,)
    x = np.random.uniform(-1, 1, size=shapeX).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shapeX, ctx=ctx)
    arr_y2 = ht.empty(shapeX, ctx=ctx)
    dropout_rate = 0.6
    import ctypes
    seed = ctypes.c_ulonglong(0)
    print("Forward Input:")
    print(x)
    gpu_op.dropout(arr_x, dropout_rate, arr_y, seed)
    print("Forward output:")
    y_out = arr_y.asnumpy()
    print(y_out)
    print(seed)
    print("Backward Input:")
    print(arr_x.asnumpy())
    gpu_op.dropout_gradient(arr_x, arr_y, dropout_rate, arr_y2)
    print("Backward Output:")
    y2_out = arr_y2.asnumpy()
    print(y2_out)

    np.testing.assert_allclose(y_out, y2_out, rtol=1e-6)
def test_softmaxdropout():
    def softmax_func(y):
        """Numerically stable softmax."""
        b = y - np.max(y, axis=-1, keepdims=True)
        expb = np.exp(b)
        softmax = expb / np.sum(expb, axis=-1, keepdims=True)
        return softmax
    def softmax_gradient_func(y, dy):
        dx = y * (dy - (dy * y).sum(axis=-1, keepdims=True))
        return dx
    ctx = ht.gpu(0)
    shapeX = (16, 512, 768)
    #shapeX = (3,4,5)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    grad = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shapeX, ctx=ctx)
    arr_grad = ht.array(grad, ctx=ctx)
    dropout_rate = 0
    import ctypes
    seed = ctypes.c_ulonglong(1)
    print(x)
    gpu_op.softmaxdropout(arr_x, dropout_rate, arr_y, seed)
    print(arr_y.asnumpy())
    np_y = softmax_func(x)

    np.testing.assert_allclose(arr_y.asnumpy(), np_y, atol = 1e-5)

    print(seed)
    gpu_op.softmaxdropout_gradient(arr_grad, arr_x, dropout_rate, arr_y, seed)
    print(arr_y.asnumpy())
    print(seed)

    np_dx = softmax_gradient_func(np_y, grad)
    print(np_dx)

    np.testing.assert_allclose(arr_y.asnumpy(), np_dx, atol = 1e-5)




def test_dropout2d():
    ctx = ht.gpu(0)
    shapeX = (2, 2, 2, 4)
    x = np.random.uniform(0, 10, size=shapeX).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(shapeX, ctx=ctx)
    dropout_rate = 0.6
    import ctypes
    seed = ctypes.c_ulonglong(0)
    print(x)
    gpu_op.dropout2d(arr_x, dropout_rate, arr_y, seed)
    print(arr_y.asnumpy())
    print(seed)
    gpu_op.dropout2d_gradient(arr_x, dropout_rate, arr_y, seed)
    print(arr_y.asnumpy())
    print(seed)


def test_instance_norm2d():
    from hetu import ndarray
    ctx = ht.gpu(0)
    shapeX = (2, 2, 2, 4)
    eps = 0.00000001
    x = ht.array(np.arange(32).reshape(shapeX), ctx=ctx)
    mid_shape = (2, 2, 1, 1)
    mean = ndarray.empty(mid_shape, ctx=ndarray.gpu(0))
    var = ndarray.empty(mid_shape, ctx=ndarray.gpu(0))
    out = ht.empty(shapeX, ctx=ctx)
    gpu_op.instance_normalization2d(x, mean, var, out, eps)

    def correct_res(arr_in, eps):
        mu = np.mean(arr_in, axis=(2, 3))
        sigma = np.var(arr_in, axis=(2, 3))

        arr_out = np.ones_like(arr_in)
        for i in range(arr_in.shape[0]):
            for j in range(arr_in.shape[1]):
                for k in range(arr_in.shape[2]):
                    for h in range(arr_in.shape[3]):
                        arr_out[i][j][k][h] = (
                            arr_in[i][j][k][h] - mu[i][j]) / np.sqrt(sigma[i][j] + eps)
        return arr_out

    np.testing.assert_allclose(
        out.asnumpy(), correct_res(x.asnumpy(), eps), rtol=1e-5)


def test_instance_norm2d_gradient():
    from hetu import ndarray
    ctx = ht.gpu(0)
    shapeX = (2, 2, 2, 4)
    eps = 0.00000001
    x = ht.array(np.arange(32).reshape(shapeX), ctx=ctx)
    mid_shape = (2, 2, 1, 1)
    mean = ndarray.empty(mid_shape, ctx=ndarray.gpu(0))
    var = ndarray.empty(mid_shape, ctx=ndarray.gpu(0))
    norm_out = ht.empty(shapeX, ctx=ctx)
    gpu_op.instance_normalization2d(x, mean, var, norm_out, eps)

    x_grad = ht.array(np.ones(shapeX), ctx=ctx)
    norm_grad_out = ht.empty(shapeX, ctx=ctx)
    gpu_op.instance_normalization2d_gradient(
        x_grad, x, norm_grad_out, mean, var, eps)

    def correct_res(arr_in, grad_in, eps):
        mu = np.mean(arr_in, axis=(2, 3))
        sigma = np.var(arr_in, axis=(2, 3))

        grad_out = np.ones_like(grad_in)
        for i in range(arr_in.shape[0]):
            for j in range(arr_in.shape[1]):
                for k in range(arr_in.shape[2]):
                    for h in range(arr_in.shape[3]):
                        y = (arr_in[i][j][k][h] - mu[i][j]) / \
                            np.sqrt(sigma[i][j] + eps)
                        grad_out[i][j][k][h] = grad_in[i][j][k][h] * \
                            (1 - 1/(shapeX[2]*shapeX[3]) - y **
                             2) / np.sqrt(sigma[i][j] + eps)
        return grad_out

    np.testing.assert_allclose(norm_grad_out.asnumpy(), correct_res(
        x.asnumpy(), x_grad.asnumpy(), eps), rtol=1e-5)


def test_onehot():
    ctx = ht.gpu(0)
    shapein = (2, 3)
    num_classes = 7
    x = np.random.randint(7, size=shapein)
    arr_x = ht.array(x, ctx=ctx)
    arr_y = ht.empty(list(shapein) + [num_classes], ctx=ctx)
    gpu_op.one_hot(arr_x, arr_y)
    print(x)
    print(arr_y.asnumpy())

test_softmaxdropout()
exit(0)

test_array_set()
test_broadcast_to()
test_reduce_sum_axis_zero()
test_matrix_elementwise_add()
test_matrix_elementwise_add_by_const()
test_matrix_elementwise_multiply()
test_matrix_elementwise_multiply_by_const()
test_matrix_elementwise_divide()
test_matrix_elementwise_divide_const()
test_matrix_opposite()
test_matrix_multiply()
test_matrix_sqrt()
test_matrix_rsqrt()
test_relu()
test_leaky_relu()
test_relu_gradient()
test_leaky_relu_gradient()
test_softmax()
test_softmax_cross_entropy()
test_softmax_cross_entropy_gradient()
test_conv2d()
test_conv2d_Gradient()
test_average_pooling()
test_reshape()
test_conv2d_broadcast_to()
test_conv2d_reduce_sum()
test_cudnn_conv2d()
test_cudnn_conv2d_Gradient()
test_CuDNN_max_pooling()
test_CuDNN_dropout_op()
test_pad()
test_concat()
test_matrix_transpose()
test_slice()
test_where()
test_batch_matrix_multiply()
test_broadcast_shape()
test_reduce_sum()
test_reduce_mean()
test_dropout()
test_dropout_recompute()
test_dropout2d()
test_instance_norm2d()
test_instance_norm2d_gradient()
test_onehot()
