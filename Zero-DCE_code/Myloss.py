import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
from paddle.vision.models import vgg16
import numpy as np
import paddle


class L_color(nn.Layer):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = paddle.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = paddle.split(mean_rgb, 3, axis=1)
        Drg = paddle.pow(mr - mg, 2)
        Drb = paddle.pow(mr - mb, 2)
        Dgb = paddle.pow(mb - mg, 2)
        k = paddle.pow(paddle.pow(Drg, 2) + paddle.pow(Drb, 2) + paddle.pow
        (Dgb, 2), 0.5)
        return k


class L_spa(nn.Layer):

    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left = paddle.to_tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype='float32').unsqueeze(0).unsqueeze(0)
        kernel_right = paddle.to_tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype='float32').unsqueeze(0).unsqueeze(0)
        kernel_up = paddle.to_tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype='float32').unsqueeze(0).unsqueeze(0)
        kernel_down = paddle.to_tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype='float32').unsqueeze(0).unsqueeze(0)
        self.weight_left = paddle.create_parameter(shape=kernel_left.shape,
                                                   dtype=str(kernel_left.numpy().dtype), default_initializer= \
                                                       paddle.nn.initializer.Assign(kernel_left))
        self.weight_left.stop_gradient = False
        self.weight_right = paddle.create_parameter(shape=kernel_right.
                                                    shape, dtype=str(kernel_right.numpy().dtype),
                                                    default_initializer=paddle.nn.initializer.Assign(kernel_right))
        self.weight_right.stop_gradient = False
        self.weight_up = paddle.create_parameter(shape=kernel_up.shape,
                                                 dtype=str(kernel_up.numpy().dtype), default_initializer=paddle.
                                                 nn.initializer.Assign(kernel_up))
        self.weight_up.stop_gradient = False
        self.weight_down = paddle.create_parameter(shape=kernel_down.shape,
                                                   dtype=str(kernel_down.numpy().dtype), default_initializer= \
                                                       paddle.nn.initializer.Assign(kernel_down))
        self.weight_down.stop_gradient = False
        self.pool = nn.AvgPool2D(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape
        org_mean = paddle.mean(org, axis=1, keepdim=True)
        enhance_mean = paddle.mean(enhance, axis=1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        weight_diff = paddle.maximum(
            paddle.to_tensor([1.0]) + 10000.0 *
            paddle.minimum(org_pool - paddle.to_tensor([0.3]), paddle.to_tensor([0], dtype='float32')),
            paddle.to_tensor([0.5]))
        E_1 = paddle.multiply(paddle.sign(enhance_pool - paddle.to_tensor([
            0.5])), enhance_pool - org_pool)
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)
        D_left = paddle.pow(D_org_letf - D_enhance_letf, 2)
        D_right = paddle.pow(D_org_right - D_enhance_right, 2)
        D_up = paddle.pow(D_org_up - D_enhance_up, 2)
        D_down = paddle.pow(D_org_down - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down
        return E


class L_exp(nn.Layer):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2D(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = paddle.mean(x, axis=1, keepdim=True)
        mean = self.pool(x)
        d = paddle.mean(paddle.square(mean - paddle.to_tensor([self.mean_val])))
        return d


class L_TV(nn.Layer):

    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):

        batch_size = x.shape[0]

        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = (x.shape[2] - 1) * x.shape[3]
        count_w = x.shape[2] * (x.shape[3] - 1)
        h_tv = paddle.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = paddle.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w
                                         ) / batch_size


class Sa_Loss(nn.Layer):

    def __init__(self):
        super(Sa_Loss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        r, g, b = paddle.split(x, 1, axis=1)
        mean_rgb = paddle.mean(x, axis=[2, 3], keepdim=True)
        mr, mg, mb = paddle.split(mean_rgb, 1, axis=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = paddle.sqrt(paddle.square(Dr) + paddle.square(Db) + paddle.square(Dg))
        k = paddle.mean(k)
        return k


class perception_loss(nn.Layer):

    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        return h_relu_4_3
