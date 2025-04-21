import hetu
import hetu.nn as nn
import torch.optim as optim
import numpy as np
import torch
import unittest
from test_utils import allclose
import os
import sys

# Warning: Remember to set rtol = 1e-05, atol = 3e-05 in `test_utils.py`

GRAD_TEST = True

class TestArithmeticOps(unittest.TestCase):

    _test_elementwise_shapes = [
        (1024,), 
        (64, 256), 
        (64, 32, 16), 
    ]

    _test_broadcast_shapes = [
        ((1024,), (1,)), 
        ((1024,), (1024,)), 
        ((64, 256), (64, 1)), 
        ((64, 256), (1, 256)), 
        ((64, 256), (256,)), 
    ]

    _test_pow_exponents = [
        0.0,
        -1.0,
        -2.0,
        4.0
    ]

    def test_elementwise_add(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            c = np.random.randn()


            # tensor + tensor
            gt = x_np + y_np
            self.assertTrue(allclose(x + y, gt))
            self.assertTrue(allclose(x.add(y), gt))
            self.assertTrue(allclose(hetu.add(x, y), gt))
            # tensor + constant & constant + tensor
            gt = x_np + c
            self.assertTrue(allclose(x + c, gt))
            self.assertTrue(allclose(c + x, gt))
            self.assertTrue(allclose(x.add(c), gt))
            self.assertTrue(allclose(hetu.add(x, c), gt))
            self.assertTrue(allclose(hetu.add(c, x), gt))
        print(sys._getframe().f_code.co_name)
    
    def test_broadcast_add(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            gt = x_np + y_np
            self.assertTrue(allclose(x + y, gt))
            self.assertTrue(allclose(y + x, gt))
            self.assertTrue(allclose(x.add(y), gt))
            self.assertTrue(allclose(y.add(x), gt))
            self.assertTrue(allclose(hetu.add(x, y), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(y_np, requires_grad=True)
              torch_out = torch.add(torch_in, torch.from_numpy(x_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(y_np, requires_grad=True)
              hetu_out = hetu.add(hetu_in, x)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_elementwise_sub(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            c = np.random.randn()
            # tensor - tensor
            gt = x_np - y_np
            self.assertTrue(allclose(x - y, gt))
            self.assertTrue(allclose(x.sub(y), gt))
            self.assertTrue(allclose(hetu.sub(x, y), gt))
            gt = y_np - x_np
            self.assertTrue(allclose(y - x, gt))
            self.assertTrue(allclose(y.sub(x), gt))
            self.assertTrue(allclose(hetu.sub(y, x), gt))
            # tensor - constant
            gt = x_np - c
            self.assertTrue(allclose(x - c, gt))
            self.assertTrue(allclose(x.sub(c), gt))
            self.assertTrue(allclose(hetu.sub(x, c), gt))
            # constant - tensor
            gt = c - x_np
            self.assertTrue(allclose(c - x, gt))
            self.assertTrue(allclose(hetu.sub(c, x), gt))
        print(sys._getframe().f_code.co_name)
    
    def test_broadcast_sub(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            gt = x_np - y_np
            self.assertTrue(allclose(x - y, gt))
            self.assertTrue(allclose(x.sub(y), gt))
            self.assertTrue(allclose(hetu.sub(x, y), gt))
            gt = y_np - x_np
            self.assertTrue(allclose(y - x, gt))
            self.assertTrue(allclose(y.sub(x), gt))
            self.assertTrue(allclose(hetu.sub(y, x), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(y_np, requires_grad=True)
              torch_out = torch.sub(torch_in, torch.from_numpy(x_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(y_np, requires_grad=True)
              hetu_out = hetu.sub(hetu_in, x)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_neg(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            gt = np.negative(x_np)
            self.assertTrue(allclose(x.neg(), gt))
            self.assertTrue(allclose(hetu.neg(x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.neg(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.neg(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_elementwise_mul(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            c = np.random.randn()
            # tensor * tensor
            gt = x_np * y_np
            self.assertTrue(allclose(x * y, gt))
            self.assertTrue(allclose(y * x, gt))
            self.assertTrue(allclose(x.mul(y), gt))
            self.assertTrue(allclose(y.mul(x), gt))
            self.assertTrue(allclose(hetu.mul(x, y), gt))
            # tensor * constant & constant * tensor
            gt = x_np * c
            self.assertTrue(allclose(x * c, gt))
            self.assertTrue(allclose(c * x, gt))
            self.assertTrue(allclose(x.mul(c), gt))
            self.assertTrue(allclose(hetu.mul(x, c), gt))
            self.assertTrue(allclose(hetu.mul(c, x), gt))
        print(sys._getframe().f_code.co_name)
    
    def test_broadcast_mul(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            gt = x_np * y_np
            self.assertTrue(allclose(x * y, gt))
            self.assertTrue(allclose(y * x, gt))
            self.assertTrue(allclose(x.mul(y), gt))
            self.assertTrue(allclose(y.mul(x), gt))
            self.assertTrue(allclose(hetu.mul(x, y), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(y_np, requires_grad=True)
              torch_out = torch.mul(torch_in, torch.from_numpy(x_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(y_np, requires_grad=True)
              hetu_out = hetu.mul(hetu_in, x)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_elementwise_div(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            c = np.random.randn()
            # tensor / tensor
            gt = x_np / y_np
            self.assertTrue(allclose(x / y, gt))
            self.assertTrue(allclose(x.div(y), gt))
            self.assertTrue(allclose(hetu.div(x, y), gt))
            gt = y_np / x_np
            self.assertTrue(allclose(y / x, gt))
            self.assertTrue(allclose(y.div(x), gt))
            self.assertTrue(allclose(hetu.div(y, x), gt))
            # tensor - constant
            gt = x_np / c
            self.assertTrue(allclose(x / c, gt))
            self.assertTrue(allclose(x.div(c), gt))
            self.assertTrue(allclose(hetu.div(x, c), gt))
            # constant - tensor
            gt = c / x_np
            self.assertTrue(allclose(c / x, gt))
            self.assertTrue(allclose(hetu.div(c, x), gt))
        print(sys._getframe().f_code.co_name)
    
    def test_broadcast_div(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestArithmeticOps._test_broadcast_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32) + 10
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            gt = x_np / y_np
            self.assertTrue(allclose(x / y, gt))
            self.assertTrue(allclose(x.div(y), gt))
            self.assertTrue(allclose(hetu.div(x, y), gt))
            gt = y_np / x_np
            self.assertTrue(allclose(y / x, gt))
            self.assertTrue(allclose(y.div(x), gt))
            self.assertTrue(allclose(hetu.div(y, x), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(y_np, requires_grad=True)
              torch_out = torch.div(torch_in, torch.from_numpy(x_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(y_np, requires_grad=True)
              hetu_out = hetu.div(hetu_in, x)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

              torch_in = torch.tensor(y_np, requires_grad=True)
              torch_out = torch.div(torch.from_numpy(x_np), torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(y_np, requires_grad=True)
              hetu_x = hetu.from_numpy(x_np)
              hetu_out = hetu.div(hetu_x, hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_reciprocal(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            gt = np.reciprocal(x_np)
            self.assertTrue(allclose(x.reciprocal(), gt))
            self.assertTrue(allclose(hetu.reciprocal(x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.reciprocal(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.reciprocal(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_sqrt(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float32)
            x = hetu.from_numpy(x_np)
            gt = np.sqrt(x_np)
            self.assertTrue(allclose(x.sqrt(), gt))
            self.assertTrue(allclose(hetu.sqrt(x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.sqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.sqrt(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_rsqrt(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.abs(np.random.randn(*shape)).astype(np.float32)
            gt = np.reciprocal(np.sqrt(x_np))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(x.rsqrt(), gt))
            self.assertTrue(allclose(hetu.rsqrt(x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.rsqrt(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.rsqrt(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_pow(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float64)
            for exponent in TestArithmeticOps._test_pow_exponents:
                gt = np.power(x_np, exponent)
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(x.pow(exponent), gt))
                self.assertTrue(allclose(hetu.pow(x, exponent), gt))

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    torch_out = torch.pow(torch_in, exponent)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    hetu_out = hetu.pow(hetu_in, exponent)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss)
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_ceil(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            gt = np.ceil(x_np)
            self.assertTrue(allclose(x.ceil(), gt))
            self.assertTrue(allclose(hetu.ceil(x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.ceil(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.ceil(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_floor(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            gt = np.floor(x_np)
            self.assertTrue(allclose(x.floor(), gt))
            self.assertTrue(allclose(hetu.floor(x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.floor(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.floor(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_round(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            gt = np.round(x_np)
            self.assertTrue(allclose(x.round(), gt))
            self.assertTrue(allclose(hetu.round(x), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.round(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.round(hetu_in)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    _test_sum_shapes = [
        ((1024,), (1,), (1024, 16)), 
        ((1024,), (1024,), (1024, 16)), 
        ((64, 256), (64, 1), (64, 256, 4)), 
        ((64, 256), (1, 256), (64, 256, 4)), 
        ((64, 256), (256,), (64, 256, 4)), 
    ]

    def test_sum(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestArithmeticOps._test_elementwise_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            y_np = np.random.randn(*shape).astype(np.float32)
            z_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            z = hetu.from_numpy(z_np)
            gt = x_np + y_np + z_np
            self.assertTrue(allclose(hetu.add([x,y,z]), gt))
        print(sys._getframe().f_code.co_name)

class TestMatMulOps(unittest.TestCase):

    _test_shapes = [
        # 1D x 1D
        ((64,), (64,)),
        # 2D x 1D
        ((64, 128), (128,)),
        # 1D x 2D
        ((128,), (128, 64)),
        # 2D x 2D
        ((64, 128), (128, 512)),
        # ND x 1D
        ((8, 64, 128), (128,)),
        # 1D x ND
        # ((128,), (8, 128, 64)),
        # ND x 2D
        ((2, 64, 128), (128, 512)),
        # 2D x ND
        # ((512, 128), (2, 128, 64)),
        # ND x ND
        ((8, 64, 256), (8, 256, 8)),
        ((8, 64, 256), (8, 8, 256, 64)),
        ((8, 16, 8, 64), (8, 16, 64, 256)),
    ]
    
    def test_matmul_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            gt = np.matmul(x_np, y_np)
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            self.assertTrue(allclose(hetu.matmul(x, y), gt))
            self.assertTrue(allclose(x.matmul(y), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.matmul(torch_in, torch.from_numpy(y_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.matmul(hetu_in, y)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    # def test_linear_op(self):
    #     for shape_x, shape_y in TestMatMulOps._test_shapes:
    #         x_np = np.random.randn(*shape_x).astype(np.float32)
    #         w_np = np.random.randn(*shape_y).astype(np.float32)
    #         bias_np = np.random.randn(shape_y[-1]).astype(np.float32)
    #         gt = np.matmul(x_np, w_np) + bias_np
    #         x = hetu.from_numpy(x_np)
    #         w = hetu.from_numpy(w_np)
    #         bias = hetu.from_numpy(bias_np)
    #         torch_test = torch.addmm(torch.from_numpy(bias_np), torch.from_numpy(x_np), 
    #                     torch.from_numpy(w_np)).numpy()
    #         self.assertTrue(allclose(hetu.linear(x, w, bias), gt))
    #         self.assertTrue(allclose(torch_test, gt))

    #         torch_in = torch.tensor(x_np, requires_grad=True)
    #         torch_out = torch.matmul(torch_in, torch.from_numpy(w_np)) + torch.from_numpy(bias_np)
    #         torch_out.sum().backward()
    #         hetu_in = hetu.Tensor(x_np, trainable=True)
    #         hetu_out = hetu.linear(hetu_in, w, bias)
    #         hetu_out.sum().backward()
    #         self.assertTrue(allclose(hetu_in.grad, torch_in.grad.numpy()))

class TestBatchMatMulOps(unittest.TestCase):

    _test_shapes = [
        ((1, 64, 128), (1, 128, 32)),
        ((16, 64, 256), (16, 256, 128))
    ]
    
    def test_batch_matmul_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y in TestBatchMatMulOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)
            gt = torch.bmm(torch.from_numpy(x_np), torch.from_numpy(y_np)).numpy()
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            self.assertTrue(allclose(hetu.bmm(x, y), gt))
            self.assertTrue(allclose(x.bmm(y), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.bmm(torch_in, torch.from_numpy(y_np))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.bmm(hetu_in, y)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

# class TestMatDotOps(unittest.TestCase):

#     _test_shapes = [
#         ((128, 64), (128, 64)),
#         ((256, 64), (256, 16))
#     ]
    
#     def test_batch_matmul_op(self):
#         for shape_x, shape_y in TestMatDotOps._test_shapes:
#             x_np = np.random.randn(*shape_x)
#             y_np = np.random.randn(*shape_y)
#             x = hetu.from_numpy(x_np)
#             y = hetu.from_numpy(y_np)
#             gt = np.dot(x_np,y_np)
#             self.assertTrue(allclose(hetu.dot(x, y), gt))
#             self.assertTrue(allclose(x.dot(y), gt))
    

class TestActivationOps(unittest.TestCase):

    _test_shapes = [
        (2, 2),
        (64, 256),
        (1024, 16)
    ]

    _test_softmax_shapes = [
        ((2, 3, 4, 5), 0),
        ((17, 8, 25, 7), 0),
        ((17, 8, 25, 7), 1),
        ((17, 8, 25, 7), 2),
        ((17, 8, 25, 7), 3)
    ]

    def test_sigmoid_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = 1 / (1 + np.exp(-x_np))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.sigmoid(x), gt))
            self.assertTrue(allclose(x.sigmoid(), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.sigmoid(torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.sigmoid(hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_sin_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.sin(x_np)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.sin(x), gt))
            self.assertTrue(allclose(x.sin(), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.sin(torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.sin(hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_relu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32) - 0.5
            gt = x_np * (x_np > 0).astype(x_np.dtype)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.relu(x), gt))
            self.assertTrue(allclose(x.relu(), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.relu(torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.relu(hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
            
    
    def test_leaky_relu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            alphas = [0.1, 0.2, 0.5]
            for alpha in alphas:
                gt = np.where(x_np > 0, x_np, alpha * x_np)
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.leakyrelu(x, alpha), gt))
                self.assertTrue(allclose(x.leakyrelu(alpha), gt))

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    torch_out = torch.nn.functional.leaky_relu(torch_in, alpha)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    hetu_out = hetu.leakyrelu(hetu_in, alpha)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss)
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_tanh_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = np.tanh(x_np)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.tanh(x), gt))
            self.assertTrue(allclose(x.tanh(), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.tanh(torch_in)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.tanh(hetu_in)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_triu_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = torch.triu(torch.from_numpy(x_np), 0).numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.triu(x, False, 0), gt))
            self.assertTrue(allclose(x.triu(False, 0), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.triu(torch_in, 0)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.triu(hetu_in, False, 0)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_tril_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestActivationOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = torch.tril(torch.from_numpy(x_np), 0).numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.triu(x, True, 0), gt))
            self.assertTrue(allclose(x.triu(True, 0), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.tril(torch_in, 0)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.triu(hetu_in, True, 0)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
            
    
    def test_softmax_op(self):
        print(sys._getframe().f_code.co_name)
        for shape, dim in TestActivationOps._test_softmax_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            gt = torch.softmax(torch.from_numpy(x_np), dim).numpy()
            x = hetu.from_numpy(x_np)
            # print(int(dim))

            self.assertTrue(allclose(hetu.softmax(x, dim), gt))
            self.assertTrue(allclose(x.softmax(dim), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.softmax(torch_in, 0)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.softmax(hetu_in, 0)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)


class TestTransformOps(unittest.TestCase):

    _test_shapes = [
        (64, 256),
        (128, 128),
        (64, 64, 256)
    ]

    _pad_shapes = [
        (8, 4, 32, 32),
        (16, 4, 16, 16)
    ]

    _transpose_shapes = [
        (16, 4, 16),
        (4, 8, 16, 32)
    ]

    def test_reshape_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            shape_to = list(shape)
            shape_to[0] = int(shape_to[0] / 2)
            shape_to[1] *= 2
            gt = np.reshape(x_np, tuple(shape_to))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.reshape(x, shape_to), gt))
            self.assertTrue(allclose(x.reshape(shape_to), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.reshape(torch_in, tuple(shape_to)).contiguous()
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.reshape(hetu_in, shape_to)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_broadcast_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            shape_to = list(shape)
            shape_to = [16] + shape_to
            gt = np.broadcast_to(x_np, tuple(shape_to))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.broadcast(x, shape_to, []), gt))
            self.assertTrue(allclose(x.broadcast(shape_to, []), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.broadcast_to(torch_in, tuple(shape_to))
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.broadcast(hetu_in, shape_to, [])
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_concat_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            dims = range(len(shape))
            for dim in dims:
                shape_1 = list(shape)
                shape_1[dim] = shape_1[dim] // 2
                shape_2 = list(shape)
                shape_2[dim] = shape_2[dim] * 2
                x_np = np.random.randn(*shape_1).astype(np.float32)
                y_np = np.random.randn(*shape_2).astype(np.float32)
                z_np = np.random.randn(*shape).astype(np.float32)
                gt = np.concatenate((x_np, y_np), dim)
                x = hetu.from_numpy(x_np)
                y = hetu.from_numpy(y_np)
                z = hetu.from_numpy(z_np)
                self.assertTrue(allclose(hetu.concat([x, y], axis=dim), gt))
                gt = np.concatenate((x_np, y_np, z_np), dim)
                self.assertTrue(allclose(hetu.concat([x, y, z], axis=dim), gt))
                gt = np.concatenate((x_np,), dim)
                self.assertTrue(allclose(hetu.concat([x], axis=dim), gt))
                
                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    torch_out = torch.cat([torch_in, torch.from_numpy(y_np), torch.from_numpy(z_np)], dim)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    hetu_out = hetu.concat([hetu_in, y, z], dim)
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss)
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_pad_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._pad_shapes:
            x_np = np.random.randn(*shape)
            x = hetu.from_numpy(x_np)
            gt = np.pad(x_np, ((0,0),(0,0),(1,1),(2,2)), "constant", constant_values = 0.1)
            self.assertTrue(allclose(hetu.pad(x, [1,1,2,2], "constant", 0.1), gt))
            self.assertTrue(allclose(x.pad([1,1,2,2], "constant", 0.1), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.nn.functional.pad(torch_in, (0,0,0,0,1,1,2,2), "constant", 0.1)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.pad(hetu_in, [1,1,2,2], "constant", 0.1)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_slice_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            begin_pos = list(np.random.randint(0, 16 ,size = [2]))
            out_size = list(np.random.randint(16, 32 ,size = [2]))
            gt = x_np[begin_pos[0]:begin_pos[0]+out_size[0], begin_pos[1]:begin_pos[1]+out_size[1]]
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.slice(x, begin_pos, out_size), gt))
            self.assertTrue(allclose(x.slice(begin_pos, out_size), gt))
        print(sys._getframe().f_code.co_name)

    def test_split_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            idx = list(np.random.randint(0, 8 ,size = [1]))
            gt = np.split(x_np, 8, 0)[idx[0]]
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.split(x, 8, 0)[idx[0]], gt))
            self.assertTrue(allclose(x.split(8, 0)[idx[0]], gt))
        print(sys._getframe().f_code.co_name)   
    
    def test_transpose_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestTransformOps._test_shapes:
            x_np = np.random.randn(*shape)
            gt = np.transpose(x_np, (1, 0))
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.transpose(x, [1, 0]), gt))
            self.assertTrue(allclose(x.transpose([1, 0]), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch.transpose(torch_in, 1, 0)
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.transpose(hetu_in, [1, 0])
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))

        for shape in TestTransformOps._transpose_shapes:
            x_np = np.random.randn(*shape)
            perm = np.arange(x_np.ndim)
            np.random.shuffle(perm)
            perm = list(perm)
            gt = np.transpose(x_np, perm)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.transpose(x, perm), gt))
            self.assertTrue(allclose(x.transpose(perm), gt))

            if GRAD_TEST:
              torch_in = torch.tensor(x_np, requires_grad=True)
              torch_out = torch_in.permute(perm).contiguous()
              torch_loss = torch_out.sum()
              torch_optimizer = optim.SGD([torch_in], lr = 0.5)
              hetu_in = hetu.Tensor(x_np, requires_grad=True)
              hetu_out = hetu.transpose(hetu_in, perm)
              hetu_loss = hetu_out.sum()
              hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
              torch_loss.backward()
              torch_optimizer.step()
              hetu_optimizer.minimize(hetu_loss)
              self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
'''
class TestConv2dOps(unittest.TestCase):

    _data_shapes = [
        (4, 3, 16, 16),        
    ]

    _filter_shapes = [
        (3, 3, 2, 2),
        (4, 3, 4, 4)        
    ]

    def test_conv2d_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestConv2dOps._data_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            for f_shape in TestConv2dOps._filter_shapes:
                f_np = np.random.randn(*f_shape).astype(np.float32)
                f = hetu.from_numpy(f_np)
                bias_np = np.random.randn()
                gt = torch.conv2d(torch.from_numpy(x_np), torch.from_numpy(f_np), stride = 1, padding = 0).numpy()
                bias_shape = [f_shape[0]]
                self.assertTrue(allclose(hetu.conv2d(x, f, 0, 1), gt))
                self.assertTrue(allclose(x.conv2d(f, 0, 1), gt))
                # test conv2d add bias
                bias_np = np.random.randn(*bias_shape).astype(np.float32)
                bias = hetu.from_numpy(bias_np)
                gt = torch.conv2d(torch.from_numpy(x_np), torch.from_numpy(f_np), torch.from_numpy(bias_np), stride = 1, padding = 0).numpy()
                self.assertTrue(allclose(hetu.conv2d(x, f, bias, 0, 1), gt))
                self.assertTrue(allclose(x.conv2d(f, bias, 0, 1), gt))

                if GRAD_TEST:
                    with hetu.profiler(enabled = True, record_shapes = True) as profiler:
                        print(0)
                        torch_in = torch.tensor(x_np, requires_grad=True)
                        torch_out = torch.conv2d(torch_in, torch.from_numpy(f_np), stride = 1, padding = 0)
                        torch_loss = torch_out.sum()
                        print(1)
                        torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                        hetu_in = hetu.Tensor(x_np, requires_grad=True)
                        hetu_out = hetu.conv2d(hetu_in, f, bias, 0, 1)
                        hetu_loss = hetu_out.sum()
                        print(2)
                        hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                        torch_loss.backward()
                        torch_optimizer.step()
                        print(3)
                        hetu_optimizer.minimize(hetu_loss)
                        self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
                        print(profiler.summary());
        print(sys._getframe().f_code.co_name)
'''


class TestPoolOps(unittest.TestCase):

    _test_shapes = [
        (4, 3, 16, 16),      
        (5, 8, 16, 16)  
    ]


    def test_maxpool_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            maxpool2d = torch.nn.MaxPool2d(2, 1, 0)
            gt = maxpool2d(torch.from_numpy(x_np)).numpy()
            self.assertTrue(allclose(hetu.maxpool(x, 2, 2, 0, 1), gt))
            self.assertTrue(allclose(x.maxpool(2, 2, 0, 1), gt))
            
            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = maxpool2d(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.maxpool(hetu_in, 2, 2, 0, 1)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_avgpool_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            avgpool2d = torch.nn.AvgPool2d(2, 1, 0)
            gt = avgpool2d(torch.from_numpy(x_np)).numpy()
            self.assertTrue(allclose(hetu.avgpool(x, 2, 2, 0, 1), gt))
            self.assertTrue(allclose(x.avgpool(2, 2, 0, 1), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = avgpool2d(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.avgpool(hetu_in, 2, 2, 0, 1)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestNormOps(unittest.TestCase):

    _test_shapes = [
        (1,2,2,2),      
        # (5, 8, 16, 16)  
    ]


    def test_batchnorm_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestPoolOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            scale_np = np.ones(shape[1]).astype(np.float32)
            scale = hetu.from_numpy(scale_np)
            bias_np = np.zeros(shape[1]).astype(np.float32)
            bias = hetu.from_numpy(bias_np)
            running_mean_np = np.empty(shape[1]).astype(np.float32)
            running_mean = hetu.from_numpy(running_mean_np)
            running_var_np = np.empty(shape[1]).astype(np.float32)
            running_var = hetu.from_numpy(running_var_np)
            save_mean_np = np.empty(shape[1]).astype(np.float32)
            save_mean = hetu.from_numpy(save_mean_np)
            save_var_np = np.empty(shape[1]).astype(np.float32)
            save_var = hetu.from_numpy(save_var_np)
            gt = torch.batch_norm(torch.from_numpy(x_np), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                 running_mean=None, running_var=None, training=True, momentum=0.1, eps=1e-5, cudnn_enabled=True).numpy()
            self.assertTrue(allclose(hetu.batch_norm(x, scale, bias, running_mean, running_var, 0.1 ,1e-5)[0], gt))
            self.assertTrue(allclose(x.batch_norm(scale, bias, running_mean, running_var, 0.1 ,1e-5)[0], gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.batch_norm(torch_in, weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                            running_mean=None, running_var=None, training=True, momentum=0.1, eps=1e-5, cudnn_enabled=True)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.batch_norm(hetu_in, scale, bias, running_mean, running_var, 0.1 ,1e-5)[0]
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_layernorm_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestPoolOps._test_shapes:
            for i in range(1, 4):
                norm_shape = shape[i:]
                x_np = np.random.randn(*shape).astype(np.float32)
                x = hetu.from_numpy(x_np)
                scale_np = np.ones(norm_shape).astype(np.float32)
                scale = hetu.from_numpy(scale_np)
                bias_np = np.zeros(norm_shape).astype(np.float32)
                bias = hetu.from_numpy(bias_np)
                layernorm = torch.nn.LayerNorm(norm_shape, 1e-5)
                # gt = layernorm(torch.from_numpy(x_np)).detach().numpy()
                gt = torch.layer_norm(torch.from_numpy(x_np), normalized_shape=tuple(norm_shape), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                    eps=1e-5).numpy()
                self.assertTrue(allclose(hetu.layer_norm(x, scale, bias, list(norm_shape), 1e-5)[0], gt))
                self.assertTrue(allclose(x.layer_norm(scale, bias, list(norm_shape), 1e-5)[0], gt))

                if GRAD_TEST:
                    torch_in = torch.tensor(x_np, requires_grad=True)
                    torch_out = torch.layer_norm(torch_in, normalized_shape=tuple(norm_shape), weight = torch.from_numpy(scale_np), bias = torch.from_numpy(bias_np),
                                                 eps=1e-5)
                    torch_loss = torch_out.sum()
                    torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                    hetu_in = hetu.Tensor(x_np, requires_grad=True)
                    hetu_out = hetu.layer_norm(hetu_in, scale, bias, list(norm_shape), 1e-5)[0]
                    hetu_loss = hetu_out.sum()
                    hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                    torch_loss.backward()
                    torch_optimizer.step()
                    hetu_optimizer.minimize(hetu_loss)
                    self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
            print(sys._getframe().f_code.co_name)
    
    def test_instancenorm_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestNormOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            temp_shape = list(shape)
            # temp_shape[-1] = 1
            # temp_shape[-2] = 1
            temp_shape = [temp_shape[1]]
            temp_shape = tuple(temp_shape)
            save_mean_np = np.empty(temp_shape).astype(np.float32)
            save_mean = hetu.from_numpy(save_mean_np)
            save_var_np = np.empty(temp_shape).astype(np.float32)
            save_var = hetu.from_numpy(save_var_np)
            instancenorm = torch.nn.InstanceNorm2d(num_features=shape[1], eps=1e-5)
            gt = instancenorm(torch.from_numpy(x_np)).detach().numpy()
            self.assertTrue(allclose(hetu.instance_norm(x, 1e-5)[0], gt))
            self.assertTrue(allclose(x.instance_norm(1e-5)[0], gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = instancenorm(torch_in)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.instance_norm(hetu_in, 1e-5)[0]
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

class TestReduceOps(unittest.TestCase):

    _test_shapes = [
        (17, 8, 25, 7),
        (16, 4, 16, 16),
        (1, 8, 32, 32),
        (1,),
    ]
    
    def test_reduce_sum_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            gt = np.sum(x_np)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.reduce(x, "sum"), gt))
            self.assertTrue(allclose(x.reduce("sum"), gt))
            self.assertTrue(allclose(hetu.sum(x), gt))
            self.assertTrue(allclose(x.sum(), gt))
            for i in range(1, pow(2, len(shape_x))):
                tmp = i
                ins = 0
                axes = []
                while tmp > 0:
                    if (tmp % 2 == 1):
                        axes.append(ins)
                    tmp //= 2
                    ins += 1
                gt = np.sum(x_np, tuple(axes))
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.sum(x, axes), gt))
                self.assertTrue(allclose(x.sum(axes), gt))
                #keepdim test
                gt = np.sum(x_np, tuple(axes), keepdims=True)
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.sum(x, axes, [True]), gt))
        print(sys._getframe().f_code.co_name)
                

    def test_reduce_mean_op(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestReduceOps._test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            gt = np.average(x_np)
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.reduce(x, "mean"), gt))
            self.assertTrue(allclose(x.reduce("mean"), gt))
            self.assertTrue(allclose(hetu.mean(x), gt))
            self.assertTrue(allclose(x.mean(), gt))
            for i in range(1, pow(2, len(shape_x))):
                tmp = i
                ins = 0
                axes = []
                while tmp > 0:
                    if (tmp % 2 == 1):
                        axes.append(ins)
                    tmp //= 2
                    ins += 1
                gt = np.average(x_np, tuple(axes))
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.mean(x, axes), gt))
                self.assertTrue(allclose(x.mean(axes), gt))
                #keepdim test
                gt = np.mean(x_np, tuple(axes), keepdims=True)
                x = hetu.from_numpy(x_np)
                self.assertTrue(allclose(hetu.mean(x, axes, [True]), gt))
        print(sys._getframe().f_code.co_name)

class TestLossOps(unittest.TestCase):
    _test_binary_label_shapes = [
        (64, 1)
    ]

    _test_nllloss_label_shapes = [
        ((64, 16), (64, ))
    ]

    _test_cross_entropy_label_shapes = [
        (64, 16)
    ]

    def test_bce_op(self):
        print(sys._getframe().f_code.co_name)
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            bce = torch.nn.BCELoss(reduction="mean")
            # gt = torch.nn.functional.binary_cross_entropy(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            gt =bce(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.binary_cross_entropy(probs, labels)
            # print(loss.shape, " ", gt.shape)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                torch_out = bce(torch_in, torch.from_numpy(labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                hetu_out = hetu.binary_cross_entropy(hetu_in, labels)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_nllloss_op(self):
        print(sys._getframe().f_code.co_name)
        for shape, lshape in TestLossOps._test_nllloss_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=lshape).astype(np.int64)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.nll_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            #gt = torch.nn.functional.nll_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.nll_loss(probs, labels)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                torch_out = torch.nn.functional.nll_loss(torch_in, torch.from_numpy(labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                hetu_out = hetu.nll_loss(hetu_in, labels)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_kldivloss_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.kl_div(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.kl_div(probs, labels)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                torch_out = torch.nn.functional.kl_div(torch_in, torch.from_numpy(labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                hetu_out = hetu.kl_div(hetu_in, labels)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_mseloss_op(self):
        print(sys._getframe().f_code.co_name)
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_binary_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            labels_np = np.random.choice([0, 1], size=shape).astype(np.float32)
            # t1_np = np.maximum(np.log(probs_np), MIN_VALUE)
            # t2_np = np.maximum(np.log(1 - probs_np), MIN_VALUE)
            # gt = -(labels_np * t1_np + (1 - labels_np) * t2_np)
            gt = torch.nn.functional.mse_loss(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.mse_loss(probs, labels)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                torch_out = torch.nn.functional.mse_loss(torch_in, torch.from_numpy(labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                hetu_out = hetu.mse_loss(hetu_in, labels)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_softmax_cross_entropy_op(self):
        print(sys._getframe().f_code.co_name)
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_cross_entropy_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            #labels_np = np.random.uniform(0.25, 0.5, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=(64,)).astype(np.int64)
            labels_onehot = torch.nn.functional.one_hot(torch.from_numpy(labels_np), 16).numpy().astype(np.float32)
            gt = torch.nn.functional.cross_entropy(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_onehot)
            loss = hetu.softmax_cross_entropy(probs, labels)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                torch_out = torch.nn.functional.cross_entropy(torch_in, torch.from_numpy(labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                hetu_out = hetu.softmax_cross_entropy(hetu_in, labels)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)


    def test_softmax_cross_entropy_sparse_op(self):
        print(sys._getframe().f_code.co_name)
        MIN_VALUE = -100.0
        for shape in TestLossOps._test_cross_entropy_label_shapes:
            probs_np = np.random.uniform(1e-10, 1, size=shape).astype(np.float32)
            #labels_np = np.random.uniform(0.25, 0.5, size=shape).astype(np.float32)
            labels_np = np.random.choice(range(16), size=(64,)).astype(np.int64)
            # labels_onehot = torch.nn.functional.one_hot(torch.from_numpy(labels_np), 16).numpy().astype(np.float32)
            gt = torch.nn.functional.cross_entropy(torch.from_numpy(probs_np), torch.from_numpy(labels_np)).numpy()
            probs = hetu.from_numpy(probs_np)
            labels = hetu.from_numpy(labels_np)
            loss = hetu.softmax_cross_entropy_sparse(probs, labels)
            self.assertTrue(allclose(loss, gt))

            if GRAD_TEST:
                torch_in = torch.tensor(probs_np, requires_grad=True)
                torch_out = torch.nn.functional.cross_entropy(torch_in, torch.from_numpy(labels_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(probs_np, requires_grad=True)
                hetu_out = hetu.softmax_cross_entropy_sparse(hetu_in, labels)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

# class TestEinsumOps(unittest.TestCase):

#     _test_args = [
#         ("ij->ji",((64, 32),)),
#         ("ij,ij->ij", ((64, 32), (64, 32))),
#         ("ii->i",((64, 64),)),
#         ("...ij->...ji",((64, 32, 4, 2, 4),)),
#         ("ij->",((64, 32),)),
#         ("ij->j",((64, 32),)),
#         ("ik,k",((64, 32),(32,))),
#         ("ik,kj",((64, 32),(32, 16))),
#         ("i,i",((2,),(2,))),
#         ("ij,ij",((64, 32),(64, 32))),
#         ("i,j",((64, ),(32, ))),
#         ("ijk,ikl->ijl",((64, 32, 16), (64, 16, 24))),
#         ("pqrs,tuqvr->pstuv", ((4, 5, 6, 8), (9, 7, 5, 13, 6))),
#         ("ik,jkl,il->ij",((64, 32), (16, 32, 48), (64, 48))),
#         ("ijk",((64, 32, 16),)),
#         ("b n h w, n d -> b d h w",((64, 32, 8, 4), (32, 16))),
#         ("n d, n d -> n",((64, 32), (64, 32))),
#         ("i d, j d -> i j",((64, 32), (48, 32))),
#         ("b h i d, b h j d -> b h i j",((64, 32, 4, 8), (64, 32, 6, 8))),
#         ("b h i j, b h j d -> b h i d",((64, 32, 4, 8), (64, 32, 8, 6))),
#         ("b i d, b i j d -> b i j",((64, 32, 4), (64, 32, 8, 4))),
#         ("b x i d, b j d -> b x i j",((64, 32, 4, 8), (64, 5, 8))),
#         ("b x i j, b j d -> b x i d",((64, 32, 4, 5), (64, 5, 8))),
#         ("hij, ijc->ihc",((64, 32, 16), (32, 16, 8))),
#         ("rac,rab->rbc",((64, 32, 4), (64, 32, 7))),
#         ("ra,rab->rb",((64, 32), (64, 32, 8))),
#         ("qhc,khc->qkh",((64, 32, 4), (48, 32, 4))),
#         ("nm, mrc->nrc",((64, 32), (32, 8, 6))),
#         ("abc,adc->bdc",((64, 32, 15), (64, 13, 15))),
#         ("dceb,cef->dbf",((64, 32, 4, 8), (32, 4, 13))),
#         ("acb,ade->dceb",((64, 32, 7), (64, 15, 9))),
#         ("qkc,ch->hqk",((64, 32, 4), (4, 13))),
#         ("bhqk,bkhc->bqhc",((64, 32, 4, 8), (64, 8, 32, 7))),
#         ("bqa,ahc->bqhc",((64, 32, 8), (8, 15, 9))),
#         ("...lc, ...c -> ...l",((64, 32, 7), (64, 7))),
#         ("...lc, ...lc -> ...l",((64, 32, 7), (64, 32, 7))),
#         ("...id,...jd->...ij",((64, 32, 4, 8), (64, 32, 5, 8))),
#         ("...klm,kmn->...kln",((64, 32, 4, 8), (32, 8, 11))),
#         ("...ikl, ...jk -> ...ijl",((64, 32, 4, 8), (64, 15, 4))),
#         ("...l,...l->...",((64, 32, 17), (64, 32, 17))),
#         ("ijk,ijk...->ij...",((64, 32, 4), (64, 32, 4, 9))),
#         ("bxi,oij,byj->boxy",((64, 32, 5), (17, 5, 13), (64, 9, 13))),
#         ("ijac,ijkp->ijakcp",((64, 32, 4, 8), (64, 32, 5, 7))),
#         ("cdij,cbi->cdbj",((64, 32, 4, 8), (64, 19, 4))),
#         ("bsid,bsjd->bijd",((64, 32, 4, 8), (64, 32, 17, 8))),
#         ("bsid,bsje->bijde",((64, 32, 4, 8), (64, 32, 17, 9))),
#         ("...bac,...dae->...bdce",((64, 32, 4, 8), (64, 19, 4, 5))),
#         ("...abc,...adc->...bdc",((64, 32, 4, 8), (64, 32, 7, 8))),
#         ("...qhd,...khd->...hqk",((64, 32, 4, 8), (64, 23, 4, 8))),
#         ("...vhf,...qhv->...qhf",((64, 32, 4, 8), (64, 19, 4, 32))),
#         ("...ij,jk->ik",((64, 32, 4, 8), (8, 13))),
#     ]
    
#     def test_einsum_op_simple(self):
#         print(sys._getframe().f_code.co_name)
#         for equation, nshapes in TestEinsumOps._test_args:
#             inputs_np = []
#             inputs_hetu = []
#             for shape in nshapes:
#                 input_np = np.random.randn(*shape).astype(np.float32)
#                 input_hetu = hetu.from_numpy(input_np)
#                 inputs_np.append(torch.from_numpy(input_np))
#                 inputs_hetu.append(input_hetu)
#             gt = torch.einsum(equation, *inputs_np).numpy()
#             # self.assertTrue(allclose(hetu.einsum(equation, inputs_hetu), gt))
#             print(equation, " ok")

#             if GRAD_TEST:
#                 inputs_tensor = []
#                 inputs_hetu = []
#                 for shape in nshapes:
#                     input_np = np.random.randn(*shape).astype(np.float32)
#                     inputs_tensor.append(torch.tensor(input_np, requires_grad=True))
#                     inputs_hetu.append(hetu.Tensor(input_np, requires_grad=True))
#                 torch_out = torch.einsum(equation, *inputs_tensor)
#                 torch_loss = torch_out.sum()
#                 torch_optimizer = optim.SGD(inputs_tensor, lr = 0.5)
#                 torch_loss.backward()
#                 torch_optimizer.step()


#                 hetu_out = hetu.einsum(equation, inputs_hetu)
#                 hetu_loss = hetu_out.sum()
#                 hetu_optimizer = hetu.SGDOptimizer(inputs_hetu, lr = 0.5)
#                 hetu_optimizer.minimize(hetu_loss)
#                 # for i in range(len(inputs_tensor)):
#                 #     self.assertTrue(allclose(inputs_hetu[i], inputs_tensor[i].detach().numpy()))
#         print(sys._getframe().f_code.co_name)

class TestDropoutOps(unittest.TestCase):

    _test_shapes = [
        (64, 32),
        (16, 8, 32, 32),
        (8, 16, 64)
    ]

    def test_dropout_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestDropoutOps._test_shapes:
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)

            p = 0.5
            hetu_out = hetu.dropout(x, p)

            hetu_out_numpy = hetu_out.numpy(force=True)
            nonzero_scale = np.mean(np.abs(hetu_out_numpy[hetu_out_numpy != 0] / x_np[hetu_out_numpy != 0]))
            self.assertTrue(abs(nonzero_scale - 1/(1-p)) < 0.1)
        print(sys._getframe().f_code.co_name)

    def test_dropout2d_op(self):
        print(sys._getframe().f_code.co_name)
        for shape in TestDropoutOps._test_shapes:
            if len(shape) != 4:
                continue
            x_np = np.random.randn(*shape).astype(np.float32)
            x = hetu.from_numpy(x_np)
            
            p = 0.5
            hetu_out = hetu.dropout2d(x, p)

            hetu_out_numpy = hetu_out.numpy(force=True)
            nonzero_channels = hetu_out_numpy[~np.all(hetu_out_numpy == 0, axis=(2,3))]
            if nonzero_channels.size > 0:
                scale = np.mean(np.abs(nonzero_channels / x_np[~np.all(hetu_out_numpy == 0, axis=(2,3))]))
                self.assertTrue(abs(scale - 1/(1-p)) < 0.1)
        print(sys._getframe().f_code.co_name)

class TestOtherOps(unittest.TestCase):

    _asstrided_test_shapes = [
        ((8, 8), (4, 4), (1, 2)),
        ((6, 4, 6, 8), (2, 3, 4, 5), (1, 2, 1, 1))
    ]

    _embedding_test_shapes = [
        ((4, 4), (5)),
        ((16, 32), (16))
    ]

    _interpolate_test_shapes = [
        ((1, 1, 2, 2), (4, 4)),
        ((3, 4, 5, 5), (20, 20))
    ]

    _maskedfill_test_shapes = [
        ((3, 4, 5, 6),),
        ((1, 9, 1, 10),)
    ]

    _norm_test_shapes = [
        ((4, 5, 2, 3), 2, 2),
        ((3, 4, 5, 5), 0, 1)
    ]

    _repeat_test_shapes = [
        ((3, 5, 7), (2, 2, 3, 4)),
        ((2, 4, 6, 8), (2, 3, 4, 5) )
    ]

    _roll_test_shapes = [
        ((2, 2), (1,), (0,)),
        ((3, 6, 7, 9), (2, 4, 6), (0, 1, 3)),
        ((2, 4, 6, 8), (1, 7), (2, 3) )
    ]

    _gather_test_shapes = [
        ((2, 2), (2, 1), 1),
        ((5, 16, 32), (1, 16, 32), 0)
    ]

    _onehot_test_shapes = [
        (32, 4),
        (64,)
    ]

    def test_arangeop(self):
        print(sys._getframe().f_code.co_name)
        gt = torch.arange(0, 100, 4).numpy()
        self.assertTrue(allclose(hetu.arange(0, 100, 4), gt))
        print(sys._getframe().f_code.co_name)
    
    def test_asstridedop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_y, stride in TestOtherOps._asstrided_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.as_strided(torch.from_numpy(x_np), shape_y, stride).numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.as_strided(x, list(shape_y), list(stride)), gt))
            self.assertTrue(allclose(x.as_strided(list(shape_y), list(stride)), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.as_strided(torch_in, shape_y, stride)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.as_strided(hetu_in, list(shape_y), list(stride))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_gatherop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_id, dim in TestOtherOps._gather_test_shapes:
            x_np = np.random.randn(*shape_x)
            id_np = np.random.randint(0, shape_x[dim], size=shape_id)
            gt = torch.gather(torch.from_numpy(x_np), dim, torch.from_numpy(id_np)).numpy()
            x = hetu.from_numpy(x_np)
            id = hetu.from_numpy(id_np)
            self.assertTrue(allclose(hetu.gather(x, dim, id), gt))
            self.assertTrue(allclose(x.gather(dim, id), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.gather(torch_in, dim, torch.from_numpy(id_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.gather(hetu_in, dim, id)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_interpolateop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_o in TestOtherOps._interpolate_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.nn.functional.interpolate(torch.from_numpy(x_np), shape_o, mode='bicubic').numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.interpolate(x, list(shape_o)), gt))
            self.assertTrue(allclose(x.interpolate(list(shape_o)), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.nn.functional.interpolate(torch_in, shape_o, mode='bicubic')
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.interpolate(hetu_in, list(shape_o))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_maskedfillop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestOtherOps._maskedfill_test_shapes:
            shape_x = shape_x[0]
            x_np = np.random.randn(*shape_x)
            mask_np = np.random.choice([0, 1], size=shape_x).astype(np.bool_)
            val = np.random.random()
            gt = torch.masked_fill(torch.from_numpy(x_np), torch.from_numpy(mask_np), val).numpy()
            x = hetu.from_numpy(x_np)
            mask = hetu.from_numpy(mask_np.astype(np.int64))
            self.assertTrue(allclose(hetu.masked_fill(x, mask, val), gt))
            self.assertTrue(allclose(x.masked_fill(mask, val), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.masked_fill(torch_in, torch.from_numpy(mask_np), val)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.masked_fill(hetu_in, mask)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_normop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, dim0, p0 in TestOtherOps._norm_test_shapes:
            x_np = np.random.randn(*shape_x).astype(np.float32)
            gt = torch.norm(torch.from_numpy(x_np), p=p0, dim=dim0).numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.norm(x, p0, dim0), gt))
            self.assertTrue(allclose(x.norm(p0, dim0), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.norm(torch_in, p=p0, dim=dim0)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.norm(hetu_in, p0, dim0)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_repeatop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, repeats in TestOtherOps._repeat_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.from_numpy(x_np).repeat(*repeats).numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.repeat(x, list(repeats)), gt))
            self.assertTrue(allclose(x.repeat(list(repeats)), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch_in.repeat(*repeats)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.repeat(hetu_in, list(repeats))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_rollop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shifts, dims in TestOtherOps._roll_test_shapes:
            x_np = np.random.randn(*shape_x)
            gt = torch.roll(torch.from_numpy(x_np), shifts=shifts, dims=dims).numpy()
            x = hetu.from_numpy(x_np)
            # print(hetu.roll(x, list(shifts), list(dims)).numpy(force=True), "\n", gt)
            self.assertTrue(allclose(hetu.roll(x, list(shifts), list(dims)), gt))
            self.assertTrue(allclose(x.roll(list(shifts), list(dims)), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.roll(torch_in, shifts, dims)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.roll(hetu_in, list(shifts), list(dims))
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)
    
    def test_embedding_lookupop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_id in TestOtherOps._embedding_test_shapes:
            x_np = np.random.randn(*shape_x)
            id_np = np.random.randint(0, shape_x[0], size=shape_id)
            gt = torch.embedding(torch.from_numpy(x_np), torch.from_numpy(id_np)).numpy()
            x = hetu.from_numpy(x_np)
            id = hetu.from_numpy(id_np)
            self.assertTrue(allclose(hetu.embedding_lookup(x, id), gt))
            self.assertTrue(allclose(x.embedding_lookup(id), gt))

            if GRAD_TEST:
                torch_in = torch.tensor(x_np, requires_grad=True)
                torch_out = torch.embedding(torch_in, torch.from_numpy(id_np))
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_in], lr = 0.5)
                hetu_in = hetu.Tensor(x_np, requires_grad=True)
                hetu_out = hetu.embedding_lookup(hetu_in, id)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_in], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_in, torch_in.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    def test_onehotop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestOtherOps._onehot_test_shapes:
            x_np = np.random.randint(0, 16, size=shape_x)
            gt = torch.nn.functional.one_hot(torch.from_numpy(x_np), num_classes = 16).numpy()
            x = hetu.from_numpy(x_np)
            self.assertTrue(allclose(hetu.onehot(x, 16), gt))
            self.assertTrue(allclose(x.onehot(16), gt))
        print(sys._getframe().f_code.co_name)

    def test_whereop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x in TestOtherOps._onehot_test_shapes:
            cond_np = np.random.choice([0, 1], size=shape_x).astype(np.int64)
            x_np = np.random.randn(*shape_x)
            y_np = np.random.randn(*shape_x)
            gt = np.where(cond_np, x_np, y_np)
            cond = hetu.from_numpy(cond_np) 
            x = hetu.from_numpy(x_np)
            y = hetu.from_numpy(y_np)
            self.assertTrue(allclose(hetu.where(cond, x, y), gt))

            if GRAD_TEST:
                torch_cond = torch.tensor(cond_np, dtype=torch.bool)
                torch_x = torch.tensor(x_np, requires_grad=True)
                torch_y = torch.tensor(y_np)
                torch_out = torch.where(torch_cond, torch_x, torch_y)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_x], lr = 0.5)
                hetu_cond = hetu.Tensor(cond_np)
                hetu_x = hetu.Tensor(x_np, requires_grad=True)
                hetu_y = hetu.Tensor(y_np)
                hetu_out = hetu.where(hetu_cond, hetu_x, hetu_y)
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_x], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                self.assertTrue(allclose(hetu_x, torch_x.detach().numpy()))
        print(sys._getframe().f_code.co_name)

    # _attn_shapes = [
    #     ((8, 12, 16, 16),),
    # ]


    # def test_attnop(self):
    #     print(sys._getframe().f_code.co_name)
    #     for shape_x in TestOtherOps._attn_shapes:
    #         q_np = np.random.randn(*shape_x)
    #         k_np = np.random.randn(*shape_x)
    #         v_np = np.random.randn(*shape_x)

    #         q = hetu.from_numpy(q_np)
    #         k = hetu.from_numpy(k_np)
    #         v = hetu.from_numpy(v_np)
            
    #         query = q.transpose([0, 2, 1, 3])
    #         value = v.transpose([0, 2, 1, 3])
    #         # [micro_batch_size, num_heads, head_dim, seq_len]
    #         key_t = k.transpose([0, 2, 3, 1]) # k^T

    #         # self-attn, shape=[micro_batch_size, num_heads, seq_len, head_dim]
    #         attn_weights = hetu.bmm(query, key_t)

    #         # scale
    #         attn_weights = attn_weights / (float(value.global_shape[-1]) ** 0.5)

    #         # mask
    #         # causal_mask = ht.from_numpy_parallel(parallel_data_provider(np.tile(self.bias[:, :, :seq_len, :seq_len], 
    #         #                                                                     (micro_batch_size, num_heads, 1, 1)),
    #         #                                                             attn_weights.distributed_states, device_index),
    #         #                                     attn_weights.distributed_states, device_group=self.device_group, name='causal_mask')
    #         # mask = ht.from_numpy_parallel(parallel_data_provider(np.full(attn_weights.global_shape, self.masked_value, dtype=np.float32),
    #         #                                                     attn_weights.distributed_states, device_index), 
    #         #                             attn_weights.distributed_states, device_group=self.device_group, name='mask')
    #         # attn_weights = ht.where(causal_mask, attn_weights, mask)
    #         # if attention_mask is not None:
    #         #     # attn_weights: shape=[micro_batch_size, num_heads, seq_len, seq_len]
    #         #     # attention_mask: shape=[micro_batch_size, 1, 1, seq_len], 注意ds的设置
    #         #     # 被mask的<pad>位置上值为-1e4, 没有被mask的位置上值为0
    #         #     # todo: +-*/允许对应维度一个为n一个为1的情况下, n被切分
    #         #     # print(f'attn_weights global_shape={attn_weights.global_shape}, attention_mask.global_shape={attention_mask.global_shape}')
    #         #     # print(f'attn_weights shape={attn_weights.shape}, attention_mask.shape={attention_mask.shape}')
    #         #     attn_weights = attn_weights + attention_mask
    #         # softmax
    #         attn_weights = hetu.softmax(attn_weights, 3)
    #         # dropout
    #         # attn_weights = self.attn_dropout(attn_weights)
    #         # weight sum, shape=[micro_batch_size, num_heads, seq_len, head_dim]
    #         attn_output = hetu.bmm(attn_weights, value)

    #         # [micro_batch_size, seq_len, num_heads, head_dim]
    #         attn_output = attn_output.transpose([0, 2, 1, 3])
    #         print(attn_output.numpy(force=True).flatten()[:10])

    #         q = hetu.from_numpy(q_np)
    #         k = hetu.from_numpy(k_np)
    #         v = hetu.from_numpy(v_np)
    #         attn_output2 = hetu.attn(q, k, v)[0]
    #         print(attn_output2.numpy(force=True).flatten()[:10])
            
    #     print(sys._getframe().f_code.co_name)
    
    _rotary_test_shapes = [((16 * 16, 3 * 4 * 64), (16, 64)),
                           ((8 * 32, 3 * 32 * 128), (32, 128))]
    _rotary_test_packinig_shapes = [((16, 3 * 4 * 64), (16, 64), (0, 4, 16)),
                                    ((1024, 3 * 32 * 256), (1024, 256), (0, 128, 512, 1024))]
    def test_rotaryop(self):
        print(sys._getframe().f_code.co_name)

        def rotary_torch(qkv_np: np.ndarray, cos_np: np.ndarray, sin_np: np.ndarray):
            # qkv: (batch_size * seq_len, 3 * num_heads * head_dim)
            # cos, sin: (seq_len, head_dim)
            # packing情况下，qkv batch_size = 1, cos,sin 的seq_len应为当前的cur_seq_len
            s, d = cos_np.shape
            b = qkv_np.shape[0] // s
            h = qkv_np.shape[1] // d // 3
            qkv_t_ori = torch.tensor(qkv_np, requires_grad=GRAD_TEST)
            qkv_t = qkv_t_ori.reshape(b, s, 3 * h, d).transpose(0, 1)
            cos_t = torch.tensor(cos_np).reshape(s, 1, 1, d)
            sin_t = torch.tensor(sin_np).reshape(s, 1, 1, d)
            def _rotate_half(x):
                from einops import rearrange
                x = rearrange(x, '... (j d) -> ... j d', j=2)
                x1, x2 = x.unbind(dim=-2)
                return torch.cat((-x2, x1), dim=-1)
            
            def apply_rotary_torch(x, cos, sin):
                # x shape: [seq_len, ..., head_dim]
                # cos,sin shape: [seq_len, ..., head_dim]
                return (x * cos) + (_rotate_half(x) * sin)
            q_t, k_t, v_t = torch.chunk(qkv_t, 3, dim=-2)
            q_t = apply_rotary_torch(q_t, cos_t, sin_t)
            k_t = apply_rotary_torch(k_t, cos_t, sin_t)
            qkv_gt = torch.concat([q_t, k_t, v_t], dim=-2).transpose(0, 1).reshape(*qkv_np.shape)
            return qkv_t_ori, qkv_gt

        for shape_qkv, shape_cos, cu_seqlens in TestOtherOps._rotary_test_packinig_shapes:
            qkv_np = np.random.randn(*shape_qkv)
            cos_np = np.random.randn(*shape_cos)
            sin_np = np.random.randn(*shape_cos)
            cu_seqlens_np = np.array(cu_seqlens, dtype=np.int32)  # 注意类型必须是int32

            qkv = hetu.Tensor(qkv_np, requires_grad=GRAD_TEST)
            cos = hetu.Tensor(cos_np)
            sin = hetu.Tensor(sin_np)
            cu_seqlens_q = hetu.Tensor(cu_seqlens_np)
            cu_seqlens_k = hetu.Tensor(cu_seqlens_np)
            multi_seq_lens_symbol = [[hetu.IntSymbol(1)]]
            multi_cp_group_symbol = [[hetu.IntSymbol(1)]]
            max_seqlen_symbol = hetu.IntSymbol(1)
            multi_seq_lens_symbol[0][0].set_data(shape_cos[0])
            multi_cp_group_symbol[0][0].set_data(0)
            max_seqlen_symbol.set_data(shape_cos[0])
            qkv_out = hetu.rotary(
                qkv, cos, sin,
                shape_cos[1],
                1,
                multi_seq_lens_symbol,
                multi_cp_group_symbol,
                True,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_symbol,
                max_seqlen_symbol,
                inplace=False
            )

            qkv_t_ori_list = []
            qkv_gt_list = []
            for i in range(len(cu_seqlens) - 1):
                cur_qkv_np = qkv_np[cu_seqlens[i]:cu_seqlens[i+1], :]
                cur_cos_np = cos_np[:cu_seqlens[i+1]-cu_seqlens[i], :]
                cur_sin_np = sin_np[:cu_seqlens[i+1]-cu_seqlens[i], :]

                cur_qkv_t_ori, cur_qkv_gt = rotary_torch(cur_qkv_np, cur_cos_np, cur_sin_np)
                qkv_t_ori_list.append(cur_qkv_t_ori)
                qkv_gt_list.append(cur_qkv_gt)
            qkv_t_ori = torch.concat(qkv_t_ori_list, dim=0)
            qkv_gt = torch.concat(qkv_gt_list, dim=0)

            self.assertTrue(allclose(qkv_out, qkv_gt.detach().numpy()))

            if GRAD_TEST:
                hetu_loss = qkv_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([qkv], lr = 0.5)
                hetu_optimizer.minimize(hetu_loss)

                torch_loss = qkv_gt.sum()
                torch_optimizer = optim.SGD([x for x in qkv_t_ori_list], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()

                self.assertTrue(allclose(qkv, torch.concat(qkv_t_ori_list, dim=0).detach().numpy()))


        for shape_qkv, shape_cos in TestOtherOps._rotary_test_shapes:
            qkv_np = np.random.randn(*shape_qkv)
            cos_np = np.random.randn(*shape_cos)
            sin_np = np.random.randn(*shape_cos)

            qkv = hetu.Tensor(qkv_np, requires_grad=GRAD_TEST)
            cos = hetu.Tensor(cos_np)
            sin = hetu.Tensor(sin_np)
            multi_seq_lens_symbol = [[hetu.IntSymbol(1)]]
            multi_cp_group_symbol = [[hetu.IntSymbol(1)]]
            max_seqlen_symbol = hetu.IntSymbol(1)
            multi_seq_lens_symbol[0][0].set_data(shape_cos[0])
            multi_cp_group_symbol[0][0].set_data(0)
            max_seqlen_symbol.set_data(shape_cos[0])
            qkv_out = hetu.rotary(
                qkv, cos, sin,
                shape_cos[1],
                1,
                multi_seq_lens_symbol,
                multi_cp_group_symbol,
                False,
                None,
                None,
                max_seqlen_symbol,
                max_seqlen_symbol,
                inplace=False
            )

            qkv_t_ori, qkv_gt = rotary_torch(qkv_np, cos_np, sin_np)

            self.assertTrue(allclose(qkv_out, qkv_gt.detach().numpy()))

            if GRAD_TEST:
                hetu_loss = qkv_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([qkv], lr = 0.5)
                hetu_optimizer.minimize(hetu_loss)

                torch_loss = qkv_gt.sum()
                torch_optimizer = optim.SGD([qkv_t_ori], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()

                self.assertTrue(allclose(qkv, qkv_t_ori.detach().numpy()))
        print(sys._getframe().f_code.co_name)


    _rms_test_shapes = [((16, 16, 4, 64), (16, 16))]
    def test_rmsop(self):
        print(sys._getframe().f_code.co_name)
        for shape_x, shape_cos in TestOtherOps._rms_test_shapes:
            batch_size =8
            seqlen = 512 
            hidden_size = 64
            x0_np = np.random.randn(batch_size, seqlen, hidden_size).astype(np.float32)
            residual_np = np.random.randn(batch_size, seqlen, hidden_size).astype(np.float32)
            weight_np = np.random.randn(hidden_size).astype(np.float32)
            bias_np = np.random.randn(hidden_size).astype(np.float32)
            x0 = hetu.from_numpy(x0_np)
            residual = hetu.from_numpy(residual_np)
            weight = hetu.from_numpy(weight_np)
            bias = hetu.from_numpy(bias_np)
            from flash_attn.ops.rms_norm import dropout_add_rms_norm
            x0_t = torch.from_numpy(x0_np).to("cuda:1")
            residual_t = torch.from_numpy(residual_np).to("cuda:1")
            weight_t = torch.from_numpy(weight_np).to("cuda:1")
            bias_t = torch.from_numpy(bias_np).to("cuda:1")
            pt = dropout_add_rms_norm(x0_t, residual_t, weight_t, bias_t, 0, 1e-3, rowscale=None,
                                      layerscale=None, prenorm=False, residual_in_fp32=False,
                                      return_dropout_mask=False)
            gt = pt.cpu().detach().numpy()
            print("Shape:", pt.shape)
            
            # self.assertTrue(allclose(hetu.rms_norm(x0, residual, weight, bias, None, None, None, None,
            #                                        dropout_p=0., epsilon=1e-3,
            #                                        rowscale_const=1., z_numrows=0,
            #                                        residual_in_fp32=False, prenorm=False, 
            #                                        is_rms_norm=True, return_dmask=False)[0], gt))

            if GRAD_TEST:
                torch_x = torch.tensor(x0_np, requires_grad=True, device = "cuda:1")
                torch_out = dropout_add_rms_norm(torch_x, residual_t, weight_t, bias_t, 0, 1e-3, rowscale=None,
                                                 layerscale=None, prenorm=False, residual_in_fp32=False,
                                                 return_dropout_mask=False)
                torch_loss = torch_out.sum()
                torch_optimizer = optim.SGD([torch_x], lr = 0.5)
                hetu_x = hetu.Tensor(x0_np, requires_grad=True)
                hetu_out = hetu.rms_norm(hetu_x, residual, weight, bias, None, None, None, None,
                                         dropout_p=0., epsilon=1e-3,
                                         rowscale_const=1., z_numrows=0,
                                         residual_in_fp32=False, prenorm=False, 
                                         is_rms_norm=True, return_dmask=False)[0]
                hetu_loss = hetu_out.sum()
                hetu_optimizer = hetu.SGDOptimizer([hetu_x], lr = 0.5)
                torch_loss.backward()
                torch_optimizer.step()
                hetu_optimizer.minimize(hetu_loss)
                # self.assertTrue(allclose(hetu_x, torch_x.cpu().detach().numpy()))
        print(sys._getframe().f_code.co_name)
                
if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    with hetu.graph("eager"):
        with hetu.context(eager_device="cuda:0"):
            unittest.main()
