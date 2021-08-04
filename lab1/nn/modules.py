import numpy as np
from itertools import product
from . import tensor


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int, x):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # TODO Initialize the weight
        # of linear module.

        self.w = tensor.zeros((in_length, out_length))
        self.x = x


        # End of todo

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.


        out = np.dot(x, self.w)
        return out
        # End of todo


    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.
        self.w.grad = np.dot(np.transpose(self.x), dy)
        dx = np.dot(dy, np.transpose(self.w))
        return dx
        # End of todo


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.

        self.length = length
        self.momentum = self.momentum


        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.

        out = None
        N = x.shape[0]
        # 求均值
        ave_ope = 1/N * tensor.ones((1, N))
        aves = np.dot(ave_ope, x)
        self.aves = aves
        # 求方差
        x_squares = np.square(x)
        vars = 1/N * np.dot(ave_ope, x_squares) - np.square(aves)
        self.vars = vars
        # nolmalization
        out = (x - aves)/vars
        return out
        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.

        dx = dy/self.vars
        return dx

        # End of todo


class Conv2d(Module):

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

        self.C_in = in_channels
        self.C_out = channels
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
        # kelnel2Col and merge
        kernel_col = self.kernel.reshape(1, self.kernel_size**2 * self.in_channels)

        out_col = np.dot(self.kernel, img_col) + self.bias.reshape(-1, 1)
        out = out_col.reshape(B, 1, H_out, W_out)


        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.

        kernel_size = self.kernel_size
        stride = self.stride
        C_in = self.C_in
        C_out = self.C_out

        B, C_in, H_in, W_in = input.shape
        #################################################################################
        B, C_out, H_out, W_out = dy.shape
        """
           compute b_grad
        """
        b_grad = np.sum(dy, axis=(0, 2, 3))
        b_grad = b_grad.reshape(C_out)

        # pad zero to input
        pad_input = np.pad(input,
                           ((0, 0), (0, 0), (0, 0), (0, 0)),
                           'constant', constant_values=0)
        # Img2Col
        col_input = Conv2d_im2col.forward(pad_input, dy)
        # merge channel
        col_input = col_input.reshape(col_input.shape[0], -1, col_input.shape[3])

        # transpose and reshape col_input to 2D matrix
        X_hat = col_input.transpose(1, 2, 0).reshape(C_in * self.kernel_size**2, -1)
        # transpose and reshape out_grad
        out_grad_reshape = dy.transpose(1, 2, 3, 0).reshape(C_out, -1)

        """
            compute w_grad
        """
        w_grad = out_grad_reshape @ X_hat.T
        w_grad = w_grad.reshape(W_out.shape)

        """
            compute in_grad
        """
        # reshape kernel
        W = W_out.reshape(C_out, -1)
        in_grad_column = W.T @ out_grad_reshape

        # Split batch dimension and transpose batch to first dimension
        in_grad_column = in_grad_column.reshape(in_grad_column.shape[0], -1, B).transpose(2, 0, 1)

        in_grad = Conv2d_col2im.forward(in_grad_column, dy)
        #################################################################################
        return in_grad, w_grad, b_grad


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
                for k in range():
                    if convwIdx + self.kernel_size > W_in:
                        convwIdx = 0
                        convhIdx += self.stride
                    out[i, j, :, k] = x[i, j, convhIdx:convhIdx+self.kernel_size,
                                      convwIdx:convwIdx+self.kernel_size].flatten().reshape((self.kernel_size**2, 1))
                    convwIdx += 1
        return out

        # End of todo

class Conv2d_col2im(Conv2d):
    def forward(self, x):
        # TODO Implement forward propogation of
        # 2d convolution module using col2im method.
        """
        Args:
            x: column of shape(B, C_in, Kernel_size**2, W_in)
        Returns:
            out: images of shape(B, C_in, H_out, W_out)
        """
        B = x.shape[0]
        C_in = x.shape[1]
        out = np.zeros((B, C_in, 0, 0))
        # unchannel input, get shape (batch, channel, kernel_h*kernel_w, out_h*out_w)
        unchannel_x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[2])
        col_idx = 0
        for i in range(B):
            for j in range(C_in):
                widx = 0
                hidx = 0
                # for each column in one channel
                for col_idx in range(unchannel_x.shape[-1]):
                    #                 print(i, j, hidx, widx)
                    out[i, j, hidx:hidx + self.kernel_size, widx:widx + self.kernel_size] += unchannel_x[i, j, :,
                                                                                 col_idx].reshape(
                        self.kernel_size, -1)
                    widx += self.stride
                    if widx + self.kernel_size > 0:
                        widx = 0
                        hidx += self.stride

        return out

        # End of todo
class AvgPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.

        B, C, H_in, W_in = x.shape
        # compute output shape
        H_out = int((H_in - self.kernel_size)/self.stride + 1)
        W_out = int((W_in - self.kernel_size)/self.stride + 1)
        # Img2col
        img_col = Conv2d_im2col.forward(self, x)
        out = np.average(img_col, axis=2).reshape(B, C, H_out, W_out)
        return out

        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.

        pool_size = self.kernel_size
        stride = self.stride

        B, C, H_in, W_in = dy.shape
        out_height = 1 + (H_in - pool_size) // stride
        out_width = 1 + (W_in - pool_size) // stride

        input_pad = np.pad(dy, pad_width=((0, 0), (0, 0), 0, 0),
                           mode='constant', constant_values=0)

        recep_fields_h = [stride * i for i in range(out_height)]
        recep_fields_w = [stride * i for i in range(out_width)]

        input_pool = Conv2d_im2col.forward(input_pad, recep_fields_h,
                                recep_fields_w, pool_size, pool_size)
        input_pool = input_pool.reshape(
            B, C, -1, out_height, out_width)

        scale = 1 / pool_size**2
        input_pool_grad = scale * \
                          np.repeat(dy[:, :, np.newaxis, :, :],
                                    pool_size**2, axis=2)

        input_pool_grad = input_pool_grad.reshape(
            B, C, -1, out_height * out_width)

        input_pad_grad = np.zeros(input_pad.shape)
        idx = 0
        for i in recep_fields_h:
            for j in recep_fields_w:
                input_pad_grad[:, :, i:i + pool_size, j:j + pool_size] += \
                    input_pool_grad[:, :, :, idx].reshape(
                        B, C, pool_size, pool_size)
                idx += 1
        in_grad = input_pad_grad[:, :, 0: H_in, 0: W_in]
        return in_grad


        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.

        B, C, H_in, W_in = x.shape
        # compute output shape
        H_out = int((H_in - self.kernel_size) / self.stride + 1)
        W_out = int((W_in - self.kernel_size) / self.stride + 1)
        # Img2col
        img_col = Conv2d_im2col.forward(self, x)
        out = img_col.max(axis=2).reshape(B, C, H_out, W_out)
        return out

        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.

        pool_size = self.kernel_size
        stride = self.stride

        B, C, H_in, W_in = dy.shape
        out_height = 1 + (H_in - pool_size) // stride
        out_width = 1 + (W_in - pool_size) // stride

        input_pad = np.pad(dy, pad_width=((0, 0), (0, 0), 0, 0),
                           mode='constant', constant_values=0)

        recep_fields_h = [stride * i for i in range(out_height)]
        recep_fields_w = [stride * i for i in range(out_width)]

        input_pool = Conv2d_im2col.forward(input_pad, recep_fields_h,
                                           recep_fields_w, pool_size, pool_size)
        input_pool = input_pool.reshape(
            B, C, -1, out_height, out_width)

        input_pool_grad = (input_pool == np.max(input_pool, axis=2, keepdims=True)) * \
                          dy[:, :, np.newaxis, :, :]

        input_pad_grad = np.zeros(input_pad.shape)
        idx = 0
        for i in recep_fields_h:
            for j in recep_fields_w:
                input_pad_grad[:, :, i:i + pool_size, j:j + pool_size] += \
                    input_pool_grad[:, :, :, idx].reshape(
                        B, C, pool_size, pool_size)
                idx += 1
        in_grad = input_pad_grad[:, :, 0: H_in, 0: W_in]
        return in_grad


        # End of todo


class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.

        self.p = 0.5

        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.

        H1 = forward(x)

        U1 = np.random.rand(*H1.shape) < self.p  # first dropout mask
        H1 *= U1  # drop!
        H2 = forward(H1)
        U2 = np.random.rand(*H2.shape) < self.p  # second dropout mask
        H2 *= U2  # drop!
        out = forward(H2)

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of dropout module.

        ...

        # End of todo


if __name__ == '__main__':
    import pdb; pdb.set_trace()
