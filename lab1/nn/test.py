import tensor
import numpy as np
class Conv2d():

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=False):
        # 假定padding一直为0
        # 假定只有一个卷积核
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.

        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.kernel = tensor.random((self.kernel_size, self.kernel_size, in_channels))

        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.

        B, C_in, H_in, W_in = x.shape
        # compute output shape
        W_out = int((W_in - self.kernel_size)/self.stride + 1)
        H_out = int((H_in - self.kernel_size)/self.stride + 1)
        # Img2Col and merge channels
        img_col = Conv2d_im2col.forward(self, x)
        W_img_col = img_col.shape[3]
        img_col = img_col.reshape(B, -1, W_img_col)
        print('转换后的数组\n', img_col)
        # kelnel2Col and merge
        kernel_col = self.kernel.reshape(1, self.kernel_size**2 * self.in_channels)

        out_col = np.dot(self.kernel, img_col)
        out = out_col.reshape(B, 1, H_out, W_out)

        # End of todo
class Conv2d_im2col(Conv2d):

    def forward(self, x):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.
        """
        Args:
            x: images of shape(B, C_in, H_in, W_in)
        Returns:
            out: column of shape(B, C_in, Kernel_size**2, W_out)
        """
        B, C_in, H_in, W_in= x.shape
        W_out = ((W_in - self.kernel_size)/self.stride + 1) * \
                ((H_in - self.kernel_size)/self.stride + 1)
        """W_out 的计算用到了除法，是float类型，需要转换成int"""
        W_out = int(W_out)
        out = tensor.zeros((B, C_in, self.kernel_size**2, W_out))
        convhIdx = 0
        convwIdx = 0
        for i in range(B):
            for j in range(C_in):
        # scan from left_top
                convhIdx = 0
                convwIdx = 0
                for k in range(B):
                    if convwIdx + self.kernel_size > W_in:
                        convwIdx = 0
                        convhIdx += self.stride
                    out[i, j, :, k] = x[i, j, convhIdx:convhIdx+self.kernel_size,
                                      convwIdx:convwIdx+self.kernel_size].flatten().reshape(
                        (self.kernel_size**2))
                    convwIdx += 1
        return out

        # End of todo
x = np.arange(1, 73).reshape(2, 3, 4, 3)
print('原数组')
print(x)
conv2d = Conv2d(3, 1, kernel_size=3)
print(Conv2d.forward(conv2d, x))
