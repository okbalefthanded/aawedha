import math

def calc_conv2d_output(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Calculates the output size of a Conv2d layer.
    Supports integers or (h, w) tuples.
    """
    def _calc(i, k, s, p, d):
        # The numerator represents the total space available for the kernel to slide
        # The denominator is the step size
        return math.floor(((i + 2 * p - d * (k - 1) - 1) / s) + 1)

    if isinstance(input_size, int): input_size = (input_size, input_size)
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)

    h_out = _calc(input_size[0], kernel_size[0], stride[0], padding[0], dilation[0])
    w_out = _calc(input_size[1], kernel_size[1], stride[1], padding[1], dilation[1])

    return h_out, w_out

def calc_pool2d_output(input_size, kernel_size, stride, padding=0):
    """
    Calculates the output dimension for a square or rectangular input.
    Accepts integers or tuples for all arguments.
    """
    # Ensure inputs are treated as tuples (Height, Width)
    if isinstance(input_size, int): input_size = (input_size, input_size)
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)

    h_out = math.floor(((input_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1)
    w_out = math.floor(((input_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1)

    return h_out, w_out