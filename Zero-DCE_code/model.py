import paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class enhance_net_nopool(nn.Layer):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU()
        number_f = 32
        self.e_conv1 = nn.Conv2D(3, number_f, 3, 1, 1, bias_attr=True)
        self.e_conv2 = nn.Conv2D(number_f, number_f, 3, 1, 1, bias_attr=True)
        self.e_conv3 = nn.Conv2D(number_f, number_f, 3, 1, 1, bias_attr=True)
        self.e_conv4 = nn.Conv2D(number_f, number_f, 3, 1, 1, bias_attr=True)
        self.e_conv5 = nn.Conv2D(number_f * 2, number_f, 3, 1, 1, bias_attr
            =True)
        self.e_conv6 = nn.Conv2D(number_f * 2, number_f, 3, 1, 1, bias_attr
            =True)
        self.e_conv7 = nn.Conv2D(number_f * 2, 24, 3, 1, 1, bias_attr=True)
        self.maxpool = nn.MaxPool2D(2, stride=2, return_mask=False,
            ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2D(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(paddle.concat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(paddle.concat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(paddle.concat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = paddle.split(x_r, 8, axis=1)
        x = x + r1 * (paddle.pow(x, 2) - x)
        x = x + r2 * (paddle.pow(x, 2) - x)
        x = x + r3 * (paddle.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (paddle.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (paddle.pow(enhance_image_1, 2) -
            enhance_image_1)
        x = x + r6 * (paddle.pow(x, 2) - x)
        x = x + r7 * (paddle.pow(x, 2) - x)
        enhance_image = x + r8 * (paddle.pow(x, 2) - x)
        r = paddle.concat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r
