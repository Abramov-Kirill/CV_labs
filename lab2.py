import numpy as np
import torch



class MyConv3d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.filters = np.ones(shape=(out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size))


    def set_filters(self, filters):
        self.filters = filters

    def __call__(self, x):
        npad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding))
        x = np.pad(array=x, pad_width=npad, mode='constant')
        submatrix_len = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
        output_maps = []
        for filter in self.filters:
            output_map = np.zeros(
                ((x.shape[1] - submatrix_len) // self.stride + 1, (x.shape[2] - submatrix_len) // self.stride + 1, (x.shape[3] - submatrix_len) // self.stride + 1))
            for feature_map in range(0, len(x[0]) - submatrix_len + 1, self.stride):
                for row in range(0, len(x[0][0]) - submatrix_len + 1, self.stride):
                    for column in range(0, len(x[0][0][0]) - submatrix_len + 1, self.stride):
                        finish_row = row + submatrix_len
                        finish_column = column + submatrix_len
                        finish_feature_map = feature_map + submatrix_len
                        submatrix = x[:, feature_map:finish_feature_map:self.dilation, row:finish_row:self.dilation, column:finish_column:self.dilation]
                        output_map[feature_map // self.stride][row // self.stride][column // self.stride] = np.sum(submatrix * filter)
            output_maps.append(np.array(output_map))

        return np.array(output_maps)

def test_conv_3d(volume_tensor, in_channels, out_channels, kernel_size, padding, dilation, stride):
    volume_np = np.array(volume_tensor).squeeze()

    kernel_tensor = torch.rand(1, in_channels, kernel_size, kernel_size, kernel_size).to(torch.float32)
    kernel_np = np.array(kernel_tensor)

    out_torch = torch.nn.functional.conv3d(volume_tensor, kernel_tensor, stride=stride, padding=padding, dilation=dilation)

    conv3d_1 = MyConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    conv3d_1.set_filters(kernel_np)
    out_my = conv3d_1(volume_np)

    out_torch = np.array(out_torch).squeeze()
    out_torch = np.round(out_torch, 0)
    out_my = np.float32(np.array(out_my).squeeze())
    out_my = np.round(out_my, 0)

    return (out_torch == out_my).all()



volume_tensor = torch.rand(1, 5, 3, 100, 150).to(torch.float32)
assert test_conv_3d(volume_tensor, 5, 1, 3, 3, 2, 2)
volume_tensor = torch.rand(1, 5, 3, 100, 150).to(torch.float32)
assert test_conv_3d(volume_tensor, 5, 1, 2, 2, 2, 2)
volume_tensor = torch.rand(1, 5, 3, 100, 150).to(torch.float32)
assert test_conv_3d(volume_tensor, 5, 1, 3, 0, 1, 1)