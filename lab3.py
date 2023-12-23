import numpy as np
from PIL import Image
import torch


class MyTransposeConv:
    def __init__(self, stride=1, padding=0, output_padding=0, dilation=1):
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

    def __call__(self, x, weights):
        size_inp_chanel, size_inp_height, size_inp_width = x.shape
        weights = weights.reshape(weights.shape[1], weights.shape[0], weights.shape[2], weights.shape[3])
        size_out_chanel = weights.shape[0]
        kernel_size = weights.shape[-1]
        submatrix_len = kernel_size + (kernel_size - 1) * (self.dilation - 1)
        result = np.zeros(shape=(size_out_chanel, (size_inp_height-1) * self.stride + submatrix_len, (size_inp_width-1) * self.stride + submatrix_len))
        for out_c in range(0, size_out_chanel):
            for row in range(0, size_inp_height):
                for column in range(0, size_inp_width):
                    start_row = row * self.stride
                    start_column = column * self.stride
                    finish_row = start_row + submatrix_len
                    finish_column = start_column + submatrix_len

                    result[out_c, start_row:finish_row:self.dilation, start_column:finish_column:self.dilation] += np.sum(x[:, row, column][:,None, None] * weights[out_c, :, :, :], axis=0)

        npad = ((0, 0), (0, self.output_padding), (0, self.output_padding))
        result = np.pad(array=result, pad_width=npad, mode='constant')

        obj = list(range(self.padding)) + list(range(-self.padding, 0))
        axis = 1
        result = np.delete(result, obj, axis)
        axis = 2
        result = np.delete(result, obj, axis)

        return result

def test_transpose_conv_2d(img, filter_weights,  stride=1, padding=1, output_padding=0, dilation=1):
    img_for_torch = img.reshape(1, count_colors, height, width)

    img_for_torch = torch.tensor(img_for_torch, dtype=torch.float32)

    out_torch = torch.nn.functional.conv_transpose2d(input=img_for_torch, weight=filter_weights, stride=stride, padding=padding,
                                                     output_padding=output_padding, dilation=dilation)

    convT_1 = MyTransposeConv(stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)
    out_my = convT_1(img, np.array(filter_weights))

    out_torch = np.array(out_torch).squeeze()
    out_torch = np.round(out_torch, 2)
    out_my = np.float32(np.array(out_my).squeeze())
    out_my = np.round(out_my, 2)

    return (out_torch == out_my).all()




img = Image.open('ford-mustang.jpg')
img = np.array(img)
height, width, count_colors = img.shape
img = img.reshape(count_colors, height, width)
img = img / 255.0

filter_weights = torch.tensor([[[[0.0, 1.0, -1.0],
                                [0.0, 1.0, -1.0],
                                [0.0, 1.0, -1.0]]],

                               [[[0.0, 1.0, -1.0],
                               [0.0, 1.0, -1.0],
                               [0.0, 1.0, -1.0]]],

                               [[[0.0, 1.0, -1.0],
                               [0.0, 1.0, -1.0],
                               [0.0, 1.0, -1.0]]]
                               ])


assert test_transpose_conv_2d(img=img, filter_weights=filter_weights, stride=1, padding=1, output_padding=0, dilation=1)
assert test_transpose_conv_2d(img=img, filter_weights=filter_weights, stride=3, padding=2, output_padding=2, dilation=1)
assert test_transpose_conv_2d(img=img, filter_weights=filter_weights, stride=1, padding=2, output_padding=0, dilation=4)