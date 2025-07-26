import hetu
import hetu.nn as nn
import torch.optim as optim
import numpy as np
import torch
import unittest
from test_utils import allclose
import os
import sys
import time
from apex.normalization.fused_layer_norm import *

# Warning: Remember to set rtol = 1e-05, atol = 3e-05 in `test_utils.py`

GRAD_TEST = True
WARM_STEP = 10
TEST_STEP = 10
TORCH_TEST = True
HETU_TEST = True

def timepoint():
    torch.cuda.synchronize("cuda:0")
    return time.time() * 1000.0

class TestArithmeticOps(unittest.TestCase):

    _test_elementwise_shapes = [
        (8192, 2560), 
    ]

    _test_broadcast_shapes = [
        ((8192, 2560), (2560,)), 
    ]

    _test_pow_exponents = [
        0.0,
        -1.0,
        -2.0,
        4.0
    ]

    def test_elementwise_add(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            y = hetu.from_numpy(y_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            y_t = torch.from_numpy(y_np).to("cuda:0").to(torch.bfloat16)
            c = np.random.randn()


            # tensor + tensor
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = x_t + y_t
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = x_t + y_t
                    end = timepoint()
                    times += (end - start)
                print("Torch Elewise Add with shape ", shape, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.add(x, y)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.add(x, y)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'AddElewiseOp':
                                times += item[2]
                print("Hetu Elewise Add with shape ", shape, ":", times / TEST_STEP)
                
            # tensor + constant & constant + tensor
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = x_t + c
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = x_t + c
                    end = timepoint()
                    times += (end - start)
                print("Torch Const Add with shape ", shape, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.add(x, c)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.add(x, c)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'AddByConstOp':
                                times += item[2]
                print("Hetu Const Add with shape ", shape, ":", times / TEST_STEP)

    def test_broadcast_add(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            y = hetu.from_numpy(y_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            y_t = torch.from_numpy(y_np).to("cuda:0").to(torch.bfloat16)
            c = np.random.randn()


            # tensor + tensor
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = x_t + y_t
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = x_t + y_t
                    end = timepoint()
                    times += (end - start)
                print("Torch Elewise Add with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.add(x, y)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.add(x, y)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'AddElewiseOp':
                                times += item[2]
                print("Hetu Elewise Add with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)
            
            # tensor + constant & constant + tensor
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = x_t + c
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = x_t + c
                    end = timepoint()
                    times += (end - start)
                print("Torch Const Add with shape ", shape_x, ",", shape_y, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.add(x, c)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.add(x, c)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'AddByConstOp':
                                times += item[2]
                print("Hetu Const Add with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)

    def test_elementwise_mul(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            y = hetu.from_numpy(y_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            y_t = torch.from_numpy(y_np).to("cuda:0").to(torch.bfloat16)
            c = np.random.randn()


            # tensor + tensor
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = x_t * y_t
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = x_t * y_t
                    end = timepoint()
                    times += (end - start)
                print("Torch Elewise Mul with shape ", shape, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.mul(x, y)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.mul(x, y)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'MulElewiseOp':
                                times += item[2]
                print("Hetu Elewise Mul with shape ", shape, ":", times / TEST_STEP)
            
            # tensor + constant & constant + tensor
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = x_t * c
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = x_t * c
                    end = timepoint()
                    times += (end - start)
                print("Torch Const Mul with shape ", shape, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.mul(x, c)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.mul(x, c)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'MulByConstOp':
                                times += item[2]
                print("Hetu Const Mul with shape ", shape, ":", times / TEST_STEP)

    def test_broadcast_mul(self):
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            y = hetu.from_numpy(y_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            y_t = torch.from_numpy(y_np).to("cuda:0").to(torch.bfloat16)
            c = np.random.randn()


            # tensor + tensor
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = x_t * y_t
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = x_t * y_t
                    end = timepoint()
                    times += (end - start)
                print("Torch Elewise Mul with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.mul(x, y)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.mul(x, y)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'MulElewiseOp':
                                times += item[2]
                print("Hetu Elewise Mul with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)
            
            # tensor + constant & constant + tensor
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = x_t * c
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = x_t * c
                    end = timepoint()
                    times += (end - start)
                print("Torch Const Mul with shape ", shape_x, ",", shape_y, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.mul(x, c)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.mul(x, c)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'MulByConstOp':
                                times += item[2]
                print("Hetu Const Mul with shape ", shape_x, ",", shape_y, ":", times / TEST_STEP)


    def test_sqrt(self):
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = torch.sqrt(x_t) 
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.sqrt(x_t) 
                    end = timepoint()
                    times += (end - start)
                print("Torch Sqrt with shape ", shape, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.sqrt(x) 
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.sqrt(x) 
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'SqrtOp':
                                times += item[2]
                print("Hetu Sqrt with shape ", shape, ":", times / TEST_STEP)

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                torch_optimizer = optim.SGD([torch_in], lr = 1e-5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True, dtype=hetu.bfloat16)
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 1e-5)
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        torch_out = torch.sqrt(torch_in)
                        torch_loss = torch_out.sum()
                        torch_loss.backward()
                        torch_optimizer.step()
                    for i in range(TEST_STEP):
                        torch_out = torch.sqrt(torch_in)
                        torch_loss = torch_out.sum()
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch SqrtGradient with shape ", shape, ":", times / TEST_STEP)
                hetu_out = hetu.sqrt(hetu_in)
                hetu_loss = hetu_out.sum()
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(WARM_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'ReciprocalSqrtOp':
                                    times += item[2]
                    print("Hetu SqrtGradient with shape ", shape, ":", times / TEST_STEP)
    

class TestMatMulOps(unittest.TestCase):

    _test_shapes = [
        # 1D x 1D
        # ((64,), (64,)),
        # # 2D x 1D
        # ((64, 128), (128,)),
        # # 1D x 2D
        # ((128,), (128, 64)),
        # # 2D x 2D
        # ((64, 128), (128, 512)),
        # # ND x 1D
        # ((8, 64, 128), (128,)),
        # # 1D x ND
        # ((128,), (8, 128, 64)),
        # # ND x 2D
        # ((2, 64, 128), (128, 512)),
        # # 2D x ND
        # ((512, 128), (2, 128, 64)),
        # # ND x ND
        # ((8, 64, 256), (8, 256, 8)),
        # ((8, 64, 256), (8, 8, 256, 64)),
        # ((8, 16, 8, 64), (8, 16, 64, 256)),
        ((128, 2, 5120), (5120, 2560)),
    ]
    
    def test_matmul_op(self):
        for shape_x, shape_y in TestMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np).to(hetu.bfloat16)
            y = hetu.from_numpy(y_np).to(hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            y_t = torch.from_numpy(y_np).to("cuda:0").to(torch.bfloat16)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = torch.matmul(x_t, y_t) 
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.matmul(x_t, y_t)
                    end = timepoint()
                    times += (end - start)
                print("Torch Matmul with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.matmul(x, y)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.matmul(x, y)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'MatMulOp':
                                times += item[2]
                print("Hetu Matmul with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)
            if GRAD_TEST:
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        gt = torch.matmul(x_t, y_t) 
                    times = 0
                    for i in range(TEST_STEP):
                        torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                        y_in = torch.tensor(y_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                        torch_out = torch.matmul(torch_in, y_in)
                        torch_loss = torch_out.sum()
                        torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch MatmulGradient with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)
                    times = 0
                    for i in range(TEST_STEP):
                        torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                        y_in = torch.tensor(y_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                        torch_out = torch.matmul(torch_in, y_in)
                        torch_loss = torch_out.sum()
                        torch_optimizer = optim.SGD([y_in], lr = 0.5)
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch MatmulGradient2 with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)
                if HETU_TEST:
                    times = 0
                    for i in range(WARM_STEP):
                        hetu_in = hetu.Tensor(x_np, requires_grad=True, dtype=hetu.bfloat16)
                        hetu_yin = hetu.Tensor(y_np, requires_grad=False, dtype=hetu.bfloat16)
                        hetu_out = hetu.matmul(hetu_in, hetu_yin)
                        hetu_loss = hetu_out.sum()
                        hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_in = hetu.Tensor(x_np, requires_grad=True, dtype=hetu.bfloat16)
                            hetu_yin = hetu.Tensor(y_np, requires_grad=False, dtype=hetu.bfloat16)
                            hetu_out = hetu.matmul(hetu_in, hetu_yin)
                            hetu_loss = hetu_out.sum()
                            hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'MatMulGradientOp':
                                    times += item[2]
                    print("Hetu MatmulGradient with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)
                    
                    times = 0
                    for i in range(WARM_STEP):
                        hetu_in = hetu.Tensor(x_np, requires_grad=False, dtype=hetu.bfloat16)
                        hetu_yin = hetu.Tensor(y_np, requires_grad=True, dtype=hetu.bfloat16)
                        hetu_out = hetu.matmul(hetu_in, hetu_yin)
                        hetu_loss = hetu_out.sum()
                        hetu_optimizer = hetu.SGDOptimizer([hetu_yin], lr = 0.5)
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_in = hetu.Tensor(x_np, requires_grad=False, dtype=hetu.bfloat16)
                            hetu_yin = hetu.Tensor(y_np, requires_grad=True, dtype=hetu.bfloat16)
                            hetu_out = hetu.matmul(hetu_in, hetu_yin)
                            hetu_loss = hetu_out.sum()
                            hetu_optimizer = hetu.SGDOptimizer([hetu_yin], lr = 0.5)
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'MatMulGradientOp':
                                    times += item[2]
                    print("Hetu MatmulGradient2 with shape ", shape_x, "," , shape_y, ":", times / TEST_STEP)

class TestBatchMatMulOps(unittest.TestCase):

    _test_shapes = [
        # ((16, 128, 256), (16, 256, 512))
        ((32, 5120, 2), (32, 2, 2560))
    ]
    
    def test_batch_matmul_op(self):
        for shape_x, shape_y in TestBatchMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            gt = torch.bmm(torch.from_numpy(x_np), torch.from_numpy(y_np)).numpy()
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            y_t = torch.from_numpy(y_np).to("cuda:0").to(torch.bfloat16)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = torch.bmm(x_t, y_t) 
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.bmm(x_t, y_t)
                    end = timepoint()
                    times += (end - start)
                print("Torch BMM with shape ", shape_x, ",", shape_y, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.bmm(x, y)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.bmm(x, y)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'BatchMatMulOp':
                                times += item[2]
                print("Hetu BMM with shape ", shape_x, ",", shape_y, ":", times / TEST_STEP)
    

class TestActivationOps(unittest.TestCase):

    _test_shapes = [
        (8192, 2560)
    ]

    _test_softmax_shapes = [
        ((64, 64, 64, 64), 0),
        ((64, 64, 64, 64), 1),
        ((64, 64, 64, 64), 2),
        ((64, 64, 64, 64), 3)
    ]

    def test_sigmoid_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = torch.sigmoid(x_t) 
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.sigmoid(x_t) 
                    end = timepoint()
                    times += (end - start)
                print("Torch Sigmoid with shape ", shape, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.sigmoid(x)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.sigmoid(x)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'SigmoidOp':
                                times += item[2]
                print("Hetu Sigmoid with shape ", shape, ":", times / TEST_STEP)
    
    def test_relu_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32) - 0.5
            gt = x_np * (x_np > 0).astype(x_np.dtype)
            x = hetu.from_numpy(x_np)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = torch.relu(x_t) 
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.relu(x_t) 
                    end = timepoint()
                    times += (end - start)
                print("Torch Relu with shape ", shape, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.relu(x)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.relu(x)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'ReluOp':
                                times += item[2]
                print("Hetu Relu with shape ", shape, ":", times / TEST_STEP)

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                torch_optimizer = optim.SGD([torch_in], lr = 1e-5)
                times = 0
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        torch_out = torch.relu(torch_in)
                        torch_loss = torch_out.sum()
                        torch_loss.backward()
                        torch_optimizer.step()
                    for i in range(TEST_STEP):
                        torch_out = torch.relu(torch_in)
                        torch_loss = torch_out.sum()
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch ReluGradient with shape ", shape, ":", times / TEST_STEP)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.relu(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 1e-5)
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'ReluGradientOp':
                                    times += item[2]
                    print("Hetu ReluGradient with shape ", shape, ":", times / TEST_STEP)
            
    def test_tanh_op(self):
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32) - 0.5
            gt = x_np * (x_np > 0).astype(x_np.dtype)
            x = hetu.from_numpy(x_np)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = torch.tanh(x_t) 
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.tanh(x_t) 
                    end = timepoint()
                    times += (end - start)
                print("Torch Tanh with shape ", shape, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.tanh(x)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.tanh(x)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'TanhOp':
                                times += item[2]
                print("Hetu Tanh with shape ", shape, ":", times / TEST_STEP)

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                torch_optimizer = optim.SGD([torch_in], lr = 1e-5)
                times = 0
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        torch_out = torch.tanh(torch_in)
                        torch_loss = torch_out.sum()
                        torch_loss.backward()
                        torch_optimizer.step()
                    for i in range(TEST_STEP):
                        torch_out = torch.tanh(torch_in)
                        torch_loss = torch_out.sum()
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch TanhGradient with shape ", shape, ":", times / TEST_STEP)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.tanh(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 1e-5)
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'TanhGradientOp':
                                    times += item[2]
                    print("Hetu TanhGradient with shape ", shape, ":", times / TEST_STEP)

    def test_softmax_op(self):
        for shape, dim in TestActivationOps._test_softmax_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = torch.softmax(torch.from_numpy(x_np), dim).numpy()
            x = hetu.from_numpy(x_np)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = torch.softmax(x_t, dim) 
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.softmax(x_t, dim) 
                    end = timepoint()
                    times += (end - start)
                print("Torch Softmax with shape ", shape, ", dim ", dim, ":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.softmax(x, dim)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.softmax(x, dim)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'SoftmaxOp':
                                times += item[2]
                print("Hetu Softmax with shape ", shape, ", dim ", dim, ":", times / TEST_STEP)

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                torch_optimizer = optim.SGD([torch_in], lr = 1e-5)
                times = 0
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        torch_out = torch.softmax(torch_in, dim)
                        torch_loss = torch_out.sum()
                        torch_loss.backward()
                        torch_optimizer.step()
                    for i in range(TEST_STEP):
                        torch_out = torch.softmax(torch_in, dim)
                        torch_loss = torch_out.sum()
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch SoftmaxGradient with shape ", shape, ", dim ", dim, ":", times / TEST_STEP)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.softmax(hetu_in, dim)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 1e-5)
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'SoftmaxGradientOp':
                                    times += item[2]
                    print("Hetu SoftmaxGradient with shape ", shape, ", dim ", dim, ":", times / TEST_STEP)

class TestConv2dOps(unittest.TestCase):

    _data_shapes = [
        (16, 3, 128, 128),        
    ]

    _filter_shapes = [
        (3, 3, 2, 2),
        (4, 3, 4, 4)        
    ]

    def test_conv2d_op(self):
        for shape in TestConv2dOps._data_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            for f_shape in TestConv2dOps._filter_shapes:
                f_np = np.random.randn(*f_shape).astype(np.float32)
                f = hetu.from_numpy(f_np).to(dtype = hetu.bfloat16)
                f_t = torch.from_numpy(f_np).to("cuda:0").to(torch.bfloat16)
                bias_shape = [f_shape[0]]
                bias_np = np.random.randn(*bias_shape).astype(np.float32)
                bias = hetu.from_numpy(bias_np).to(dtype = hetu.bfloat16)
                bias_t = torch.from_numpy(bias_np).to("cuda:0").to(torch.bfloat16)
                times = 0
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        gt = torch.conv2d(x_t, f_t, bias_t, stride = 1, padding = 0)
                    for i in range(TEST_STEP):
                        start = timepoint()
                        gt = torch.conv2d(x_t, f_t, bias_t, stride = 1, padding = 0)
                        end = timepoint()
                        times += (end - start)
                    print("Torch Conv2d with shape ", shape, ", fshape ", f_shape, ":", times / TEST_STEP)
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        gt = hetu.conv2d(x, f, bias, 0, 1)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            gt = hetu.conv2d(x, f, bias, 0, 1)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'Conv2dAddBiasOp':
                                    times += item[2]
                    print("Hetu Conv2d with shape ", shape, ", fshape ", f_shape, ":", times / TEST_STEP)

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                    filter_in = torch.tensor(f_np, requires_grad=True, device="cuda:0")
                    bias_in = torch.tensor(bias_np, requires_grad=True, device="cuda:0")
                    torch_optimizer = optim.SGD([torch_in, ], lr = 1e-5)
                    times = 0
                    if TORCH_TEST:
                        for i in range(WARM_STEP):
                            torch_out = torch.conv2d(torch_in, filter_in, bias_in, stride = 1, padding = 0)
                            torch_loss = torch_out.sum()
                        for i in range(TEST_STEP):
                            torch_out = torch.conv2d(torch_in, filter_in, bias_in, stride = 1, padding = 0)
                            torch_loss = torch_out.sum()
                            start = timepoint()
                            torch_loss.backward()
                            end = timepoint()
                            torch_optimizer.step()
                            times += (end - start)
                        print("Torch Conv2dGradient with shape ", shape, ", fshape ", f_shape, ":", times / TEST_STEP)
                    times = 0
                    hetu_in = hetu.Tensor(x_np, requires_grad=True, dtype=hetu.bfloat16)
                    hetu_out = hetu.conv2d(hetu_in, f, bias, 0, 1)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    if HETU_TEST:
                        for i in range(WARM_STEP):
                            hetu_optimizer.minimize(hetu_loss)
                        for i in range(TEST_STEP):
                            with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                                hetu_optimizer.minimize(hetu_loss)
                                for item in profiler.summary()['optype_with_inputs_view']:
                                    if item[0] == 'Conv2dGradientofDataOp':
                                        times += item[2]
                        print("Hetu Conv2dGradient with shape ", shape, ", fshape ", f_shape, ":", times / TEST_STEP)



class TestPoolOps(unittest.TestCase):

    _test_shapes = [
        (16, 3, 128, 128),      
    ]


    def test_maxpool_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            maxpool2d = torch.nn.MaxPool2d(2, 1, 0)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = maxpool2d(x_t)
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = maxpool2d(x_t)
                    end = timepoint()
                    times += (end - start)
                print("Torch Maxpool with shape ", shape, ":", times / TEST_STEP)
            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.maxpool(x, 2, 2, 0, 1)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.maxpool(x, 2, 2, 0, 1)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'MaxPoolOp':
                                times += item[2]
                print("Hetu Maxpool with shape ", shape, ":", times / TEST_STEP)
            
            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                torch_optimizer = optim.SGD([torch_in], lr = 1e-5)
                times = 0
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        torch_out = maxpool2d(torch_in)
                        torch_loss = torch_out.sum()
                        torch_loss.backward()
                        torch_optimizer.step()
                    for i in range(TEST_STEP):
                        torch_out = maxpool2d(torch_in)
                        torch_loss = torch_out.sum()
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch MaxpoolGradient with shape ", shape, ":", times / TEST_STEP)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.maxpool(hetu_in, 2, 2, 0, 1)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 1e-5)
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'MaxPoolGradientOp':
                                    times += item[2]
                    print("Hetu MaxpoolGradient with shape ", shape, ":", times / TEST_STEP)

    def test_avgpool_op(self):
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            avgpool2d = torch.nn.AvgPool2d(2, 1, 0)
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = avgpool2d(x_t)
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = avgpool2d(x_t)
                    end = timepoint()
                    times += (end - start)
                print("Torch Avgpool with shape ", shape, ":", times / TEST_STEP)
            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.avgpool(x, 2, 2, 0, 1)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.avgpool(x, 2, 2, 0, 1)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'AvgPoolOp':
                                times += item[2]
                print("Hetu Avgpool with shape ", shape, ":", times / TEST_STEP)
            
            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                torch_optimizer = optim.SGD([torch_in], lr = 1e-5)
                times = 0
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        torch_out = avgpool2d(torch_in)
                        torch_loss = torch_out.sum()
                        torch_loss.backward()
                        torch_optimizer.step()
                    for i in range(TEST_STEP):
                        torch_out = avgpool2d(torch_in)
                        torch_loss = torch_out.sum()
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch AvgpoolGradient with shape ", shape, ":", times / TEST_STEP)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.avgpool(hetu_in, 2, 2, 0, 1)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 1e-5)
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'AvgPoolGradientOp':
                                    times += item[2]
                    print("Hetu AvgpoolGradient with shape ", shape, ":", times / TEST_STEP)

class TestTransformOps(unittest.TestCase):
    _test_shapes = [
        (8192, 256),
    ]
    
    _test_dims = [
        1,
    ]
    
    def test_concat_op(self):
        for shape in TestTransformOps._test_shapes:
            for dim in TestTransformOps._test_dims:
                x_np = np.random.randn(*shape).astype(np.float32)
                y_np = np.random.randn(*shape).astype(np.float32)
                x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
                y = hetu.from_numpy(y_np).to(dtype = hetu.bfloat16)
                x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
                y_t = torch.from_numpy(y_np).to("cuda:0").to(torch.bfloat16)
                times = 0
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        gt = torch.cat([x_t, y_t], dim=dim)
                    for i in range(TEST_STEP):
                        start = timepoint()
                        gt = torch.cat([x_t, y_t], dim=dim)
                        end = timepoint()
                        times += (end - start)
                    print("Torch Concat with shape ", shape, ", dim ", dim, ":", times / TEST_STEP)
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        gt = hetu.concat([x, y], axis=dim)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            gt = hetu.concat([x, y], axis=dim)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'ConcatOp':
                                    times += item[2]
                    print("Hetu Concat with shape ", shape, ", dim ", dim, ":", times / TEST_STEP)

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                    torch_yin = torch.tensor(y_np, requires_grad=True, device="cuda:0")
                    torch_optimizer = optim.SGD([torch_in, torch_yin], lr = 1e-5)
                    times = 0
                    if TORCH_TEST:
                        for i in range(WARM_STEP):
                            torch_out = torch.cat([torch_in, torch_yin], dim)
                            torch_loss = torch_out.sum()
                            torch_loss.backward()
                            torch_optimizer.step()
                        for i in range(TEST_STEP):
                            torch_out = torch.cat([torch_in, torch_yin], dim)
                            torch_loss = torch_out.sum()
                            start = timepoint()
                            torch_loss.backward()
                            end = timepoint()
                            torch_optimizer.step()
                            times += (end - start)
                        print("Torch ConcatGradient with shape ", shape, ", dim ", dim, ":", times / TEST_STEP)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True, dtype=hetu.bfloat16)
                    hetu_yin = hetu.Tensor(y_np, requires_grad=True, dtype=hetu.bfloat16)
                    hetu_out = hetu.concat([hetu_in, hetu_yin], dim)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in, hetu_yin], lr = 1e-5)
                    times = 0
                    if HETU_TEST:
                        for i in range(WARM_STEP):
                            hetu_optimizer.minimize(hetu_loss)
                        for i in range(TEST_STEP):
                            with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                                hetu_optimizer.minimize(hetu_loss)
                                for item in profiler.summary()['optype_with_inputs_view']:
                                    if item[0] == 'ConcatGradientOp':
                                        times += item[2]
                        print("Hetu ConcatGradient with shape ", shape, ", dim ", dim, ":", times / TEST_STEP)

class TestDropoutOps(unittest.TestCase):
    _test_shapes = [
        (64, 32),
        (8, 16, 64),
        (16, 3, 128, 128),
    ]
    
    def test_dropout_op(self):
        for shape in TestDropoutOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
            
            p = 0.5
            times = 0
            if TORCH_TEST:
                for i in range(WARM_STEP):
                    gt = torch.nn.functional.dropout(x_t, p=p)
                for i in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.nn.functional.dropout(x_t, p=p)
                    end = timepoint()
                    times += (end - start)
                print("Torch Dropout with shape ", shape, ":", times / TEST_STEP)
            times = 0
            if HETU_TEST:
                for i in range(WARM_STEP):
                    gt = hetu.dropout(x, p)
                for i in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.dropout(x, p)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'DropoutOp':
                                times += item[2]
                print("Hetu Dropout with shape ", shape, ":", times / TEST_STEP)
            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                times = 0
                if TORCH_TEST:
                    for i in range(WARM_STEP):
                        torch_out = torch.nn.functional.dropout(torch_in, p=p)
                        torch_loss = torch_out.sum()
                        torch_loss.backward()
                        torch_optimizer.step()
                    for i in range(TEST_STEP):
                        torch_out = torch.nn.functional.dropout(torch_in, p=p)
                        torch_loss = torch_out.sum()
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch DropoutGradient with shape ", shape, ":", times / TEST_STEP)
                hetu_in = hetu.Tensor(x_np, requires_grad=True, dtype=hetu.bfloat16)
                hetu_loss = hetu.dropout(hetu_in, p)
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                times = 0
                if HETU_TEST:
                    for i in range(WARM_STEP):
                        hetu_optimizer.minimize(hetu_loss)
                    for i in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'DropoutGradientOp':
                                    times += item[2]
                    print("Hetu DropoutGradient with shape ", shape, ":", times / TEST_STEP)

class TestNormOps(unittest.TestCase):

    _test_shapes = [
        (2, 2048, 2560),      
    ]

    def test_layernorm_op(self):
        for shape in TestNormOps._test_shapes:
            for i in range(1, 3):
                norm_shape = shape[i:]
                x_np = np.random.randn(*shape).astype(np.float32)
                x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
                x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
                scale_np = np.ones(norm_shape).astype(np.float32)
                scale = hetu.from_numpy(scale_np).to(dtype = hetu.bfloat16)
                bias_np = np.zeros(norm_shape).astype(np.float32)
                bias = hetu.from_numpy(bias_np).to(dtype = hetu.bfloat16)
                scale_t = torch.from_numpy(scale_np).to("cuda:0").to(torch.bfloat16)
                bias_t = torch.from_numpy(bias_np).to("cuda:0").to(torch.bfloat16)
                times = 0
                if TORCH_TEST:
                    for j in range(WARM_STEP):
                        gt = torch.layer_norm(x_t, normalized_shape=tuple(norm_shape), weight = scale_t, bias = bias_t, eps=1e-5)
                    for j in range(TEST_STEP):
                        start = timepoint()
                        gt = torch.layer_norm(x_t, normalized_shape=tuple(norm_shape), weight = scale_t, bias = bias_t, eps=1e-5)
                        end = timepoint()
                        times += (end - start)
                    print("Torch LayerNorm with shape ", shape, ",position:", i ,":", times / TEST_STEP)
                    
                    times = 0
                    for j in range(WARM_STEP):
                        gt = fused_layer_norm_affine(x_t, normalized_shape=tuple(norm_shape), weight = scale_t, bias = bias_t, eps=1e-5)
                    for j in range(TEST_STEP):
                        start = timepoint()
                        gt = fused_layer_norm_affine(x_t, normalized_shape=tuple(norm_shape), weight = scale_t, bias = bias_t, eps=1e-5)
                        end = timepoint()
                        times += (end - start)
                    print("Torch FusedLayerNorm with shape ", shape, ",position:", i ,":", times / TEST_STEP)

                times = 0
                if HETU_TEST:
                    for j in range(WARM_STEP):
                        gt = hetu.fused_layernorm(x, scale, bias, list(norm_shape), 1e-5)[0]
                    for j in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            gt = hetu.fused_layernorm(x, scale, bias, list(norm_shape), 1e-5)[0]
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'FusedLayerNormOp':
                                    times += item[2]
                    print("Hetu FusedLayerNorm with shape ", shape, ",position:", i ,":", times / TEST_STEP)
                    
                    for j in range(WARM_STEP):
                        gt = hetu.layer_norm(x, scale, bias, list(norm_shape), 1e-5)[0]
                    for j in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            gt = hetu.layer_norm(x, scale, bias, list(norm_shape), 1e-5)[0]
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'LayerNormOp':
                                    times += item[2]
                    print("Hetu LayerNorm with shape ", shape, ",position:", i ,":", times / TEST_STEP)

                if GRAD_TEST:
                    hetu_in = hetu.Tensor(x_np, requires_grad=True, dtype=hetu.bfloat16)
                    hetu_out = hetu.layer_norm(hetu_in, scale, bias, list(norm_shape), 1e-5)[0]
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    times = 0
                    if TORCH_TEST:
                        for j in range(WARM_STEP):
                            torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                            torch_out = torch.layer_norm(torch_in, normalized_shape=tuple(norm_shape),
                                                        eps=1e-5)
                            torch_loss = torch_out.sum()
                            torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                            torch_loss.backward()
                            torch_optimizer.step()
                        # for j in range(TEST_STEP):
                        #     torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0")
                        #     torch_out = torch.layer_norm(torch_in, normalized_shape=tuple(norm_shape),
                        #                                  eps=1e-5)
                        #     torch_loss = torch_out.sum()
                        #     torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                        #     torch.cuda.synchronize("cuda:0")
                        #     start = time.time()
                        #     torch_loss.backward()
                        #     torch.cuda.synchronize("cuda:0")
                        #     end = time.time()
                        #     torch_optimizer.step()
                        #     times += (end - start)
                        # print("Torch LayerNormGradient with shape ", shape, ",position:", i ,":", times / TEST_STEP)
                        for j in range(TEST_STEP):
                            torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                            torch_weight = torch.tensor(scale_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                            torch_bias = torch.tensor(bias_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                            torch_out = torch.layer_norm(torch_in, normalized_shape=tuple(norm_shape), weight = torch_weight,
                                                        bias = torch_bias, eps=1e-5)
                            torch_loss = torch_out.sum()
                            torch_optimizer = optim.SGD([torch_in, torch_weight, torch_bias], lr = 0.5)
                            start = timepoint()
                            torch_loss.backward()
                            end = timepoint()
                            torch_optimizer.step()
                            times += (end - start)
                        print("Torch LayerNormGradient with shape ", shape, ",position:", i ,":", times / TEST_STEP)
                        times = 0
                        for j in range(TEST_STEP):
                            torch_in = torch.tensor(x_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                            torch_weight = torch.tensor(scale_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                            torch_bias = torch.tensor(bias_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                            torch_out = fused_layer_norm_affine(torch_in, normalized_shape=tuple(norm_shape), weight = torch_weight,
                                                                bias = torch_bias, eps=1e-5)
                            torch_loss = torch_out.sum()
                            torch_optimizer = optim.SGD([torch_in, torch_weight, torch_bias], lr = 0.5)
                            start = timepoint()
                            torch_loss.backward()
                            end = timepoint()
                            torch_optimizer.step()
                            times += (end - start)
                        print("Torch FusedLayerNormGradient with shape ", shape, ",position:", i ,":", times / TEST_STEP)

                    times = 0
                    if HETU_TEST:
                        for j in range(WARM_STEP):
                            hetu_optimizer.minimize(hetu_loss)
                        for j in range(TEST_STEP):
                            with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                                hetu_optimizer.minimize(hetu_loss)
                                for item in profiler.summary()['optype_with_inputs_view']:
                                    if item[0] == 'LayerNormGradientOp':
                                        times += item[2]
                        print("Hetu LayerNormGradient with shape ", shape, ",position:", i ,":", times / TEST_STEP)
                        
                        times = 0
                        hetu_in = hetu.Tensor(x_np, requires_grad=True, dtype=hetu.bfloat16)
                        hetu_out = hetu.fused_layernorm(hetu_in, scale, bias, list(norm_shape), 1e-5)[0]
                        hetu_loss = hetu_out.sum()
                        hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                        for j in range(WARM_STEP):
                            hetu_optimizer.minimize(hetu_loss)
                        for j in range(TEST_STEP):
                            with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                                hetu_optimizer.minimize(hetu_loss)
                                for item in profiler.summary()['optype_with_inputs_view']:
                                    if item[0] == 'FusedLayerNormGradientOp':
                                        times += item[2]
                        print("Hetu FusedLayerNormGradient with shape ", shape, ",position:", i ,":", times / TEST_STEP)

class TestReduceOps(unittest.TestCase):

    _test_shapes = [
        # (256, 128, 16, 16),
        (32, 5120, 2560),
    ]
    
    def test_reduce_sum_op(self):
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
            for i in range(1, pow(2, len(shape_x))):
                tmp = i
                ins = 0
                axes = []
                while tmp > 0:
                    if (tmp % 2 == 1):
                        axes.append(ins)
                    tmp //= 2
                    ins += 1
                x = hetu.from_numpy(x_np).to(dtype = hetu.bfloat16)
                x_t = torch.from_numpy(x_np).to("cuda:0").to(torch.bfloat16)
                times = 0
                if TORCH_TEST:
                    for j in range(WARM_STEP):
                        gt = torch.sum(x_t, tuple(axes))
                    for j in range(TEST_STEP):
                        start = timepoint()
                        gt = torch.sum(x_t, tuple(axes))
                        end = timepoint()
                        times += (end - start)
                    print("Torch Reduce with shape ", shape_x, ",axes:", axes ,":", times / TEST_STEP)

                times = 0
                if HETU_TEST:
                    for j in range(WARM_STEP):
                        gt = hetu.sum(x, axes)
                    for j in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            gt = hetu.sum(x, axes)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'ReduceOp':
                                    times += item[2]
                    print("Hetu Reduce with shape ", shape_x, ",axes:", axes ,":", times / TEST_STEP)

class TestLossOps(unittest.TestCase):

    _test_cross_entropy_label_shapes = [
        (4096, 256)
    ]


    def test_softmax_cross_entropy_sparse_op(self):
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_cross_entropy_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            #labels_np = np.random.uniform(0.25, 0.5, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(shape[1]), size=(shape[0],)).astype(np.int64)
            # labels_onehot = torch.nn.functional.one_hot(torch.from_numpy(labels_np), 16).numpy().astype(np.float32)
            probs = hetu.from_numpy(probs_np).to(dtype = hetu.bfloat16)
            labels = hetu.from_numpy(labels_np)
            probs_t = torch.from_numpy(probs_np).to("cuda:0").to(torch.bfloat16)
            labels_t = torch.from_numpy(labels_np).to("cuda:0")
            times = 0
            if TORCH_TEST:
                for j in range(WARM_STEP):
                    gt = torch.nn.functional.cross_entropy(probs_t, labels_t)
                for j in range(TEST_STEP):
                    start = timepoint()
                    gt = torch.nn.functional.cross_entropy(probs_t, labels_t)
                    end = timepoint()
                    times += (end - start)
                print("Torch CrossEntropy with shape ", shape ,":", times / TEST_STEP)

            times = 0
            if HETU_TEST:
                for j in range(WARM_STEP):
                    gt = hetu.softmax_cross_entropy_sparse(probs, labels)
                for j in range(TEST_STEP):
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        gt = hetu.softmax_cross_entropy_sparse(probs, labels)
                        for item in profiler.summary()['optype_with_inputs_view']:
                            if item[0] == 'SoftmaxCrossEntropySparseOp':
                                times += item[2]
                print("Hetu CrossEntropy with shape ", shape ,":", times / TEST_STEP)

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True, device="cuda:0", dtype=torch.bfloat16)
                torch_optimizer = optim.SGD([torch_in], lr = 1e-5)
                times = 0
                if TORCH_TEST:
                    for j in range(WARM_STEP):
                        torch_out = torch.nn.functional.cross_entropy(torch_in, labels_t)
                        torch_loss = torch_out.sum()
                        torch_loss.backward()
                        torch_optimizer.step()
                    for j in range(TEST_STEP):
                        torch_out = torch.nn.functional.cross_entropy(torch_in, labels_t)
                        torch_loss = torch_out.sum()
                        start = timepoint()
                        torch_loss.backward()
                        end = timepoint()
                        torch_optimizer.step()
                        times += (end - start)
                    print("Torch CrossEntropy Gradient with shape ", shape ,":", times / TEST_STEP)

                hetu_in = hetu.Tensor(probs_np, requires_grad=True, dtype=hetu.bfloat16)
                hetu_out = hetu.softmax_cross_entropy_sparse(hetu_in, labels)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 1e-5)
                times = 0
                if HETU_TEST:
                    for j in range(WARM_STEP):
                        hetu_optimizer.minimize(hetu_loss)
                    for j in range(TEST_STEP):
                        with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                            hetu_optimizer.minimize(hetu_loss)
                            for item in profiler.summary()['optype_with_inputs_view']:
                                if item[0] == 'SoftmaxCrossEntropySparseGradientOp':
                                    times += item[2]
                    print("Hetu CrossEntropy Gradient with shape ", shape ,":", times / TEST_STEP)
                

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    with hetu.graph("eager"):
        with hetu.context(eager_device="cuda:0"):
            unittest.main()
