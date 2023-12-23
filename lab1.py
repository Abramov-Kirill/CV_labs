import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt


class MyConv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        #self.filters = np.ones((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

    def set_filters(self, filters):
        self.filters = filters

    def __call__(self, x):
        npad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
        x = np.pad(array=x, pad_width=npad, mode='constant')
        submatrix_len = self.kernel_size + (self.kernel_size-1) * (self.dilation-1)
        output_maps = []
        for filter in self.filters:
            output_map = np.zeros(((x.shape[1] - submatrix_len) // self.stride + 1, (x.shape[2] - submatrix_len) // self.stride + 1))
            for row in range(0, len(x[0]) - submatrix_len +1, self.stride):
                for column in range(0, len(x[0][0]) - submatrix_len +1, self.stride):
                    finish_row = row + submatrix_len
                    finish_column = column + submatrix_len
                    submatrix = x[:, row:finish_row:self.dilation, column:finish_column:self.dilation]
                    output_map[row//self.stride][column//self.stride] = np.sum(submatrix * filter)
            output_maps.append(np.array(output_map))


        return np.array(output_maps)

def test_conv_2d(img, filter_weights, in_channels, out_channels, kernel_size, padding, dilation, stride):
    conv1 = MyConv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride)
    conv1.set_filters(np.array(filter_weights))
    my_out = conv1(img)
    my_output = my_out.squeeze()
    my_out = np.float32(np.round(my_out, 1))

    plt.imshow(my_output, cmap='gray')
    plt.title('MY Convolved Image')
    plt.show()

    img_for_torch = img.reshape(1, count_colors, height, width)
    img_for_torch = torch.tensor(img_for_torch, dtype=torch.float32)

    out_torch = torch.nn.functional.conv2d(input=img_for_torch, weight=filter_weights, padding=padding, dilation=dilation, stride=stride)
    out_torch = np.array(out_torch)
    out_torch = out_torch.squeeze()
    out_torch = np.round(out_torch, 1)

    plt.imshow(out_torch, cmap='gray')
    plt.title('Torch Convolved Image')
    plt.show()

    return (my_out == out_torch).all()



img = Image.open('ford-mustang.jpg')
img = np.array(img)
height, width, count_colors = img.shape
img = img.reshape(count_colors, height, width)
img = img / 255.0

filter_weights = torch.tensor([[[[0.0, 1.0, -1.0],
                                [0.0, 1.0, -1.0],
                                [0.0, 1.0, -1.0]],

                               [[0.0, 1.0, -1.0],
                               [0.0, 1.0, -1.0],
                               [0.0, 1.0, -1.0]],

                               [[0.0, 1.0, -1.0],
                               [0.0, 1.0, -1.0],
                               [0.0, 1.0, -1.0]]
                               ]])
print(filter_weights.shape)

assert test_conv_2d(img, filter_weights, 3, 1, 3,5,4, 3)
assert test_conv_2d(img, filter_weights, 3, 1, 3,2,1, 6)
assert test_conv_2d(img, filter_weights, 3, 1, 3,1,2, 2)
assert test_conv_2d(img, filter_weights, 3, 1, 3,0,1, 1)