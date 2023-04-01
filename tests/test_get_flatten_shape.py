import pytest
from little_helpers.equinox_helpers import get_flatten_shape


def test_flatten_shape_no_pooling():
    width = 5
    height = 5
    kernel_size = 3
    stride = 1
    padding = 0
    out_channels = 1
    pool_kernel_size = 1
    pool_stride = 1

    result = get_flatten_shape(
        width,
        height,
        kernel_size,
        stride,
        padding,
        out_channels,
        pool_kernel_size,
        pool_stride,
    )

    assert result == 9, f"Expected 9, but got {result}"


def test_flatten_shape_with_pooling():
    width = 5
    height = 5
    kernel_size = 3
    stride = 1
    padding = 0
    out_channels = 1
    pool_kernel_size = 2
    pool_stride = 1

    result = get_flatten_shape(
        width,
        height,
        kernel_size,
        stride,
        padding,
        out_channels,
        pool_kernel_size,
        pool_stride,
    )

    assert result == 4, f"Expected 4, but got {result}"


def test_flatten_shape_with_channels():
    width = 5
    height = 5
    kernel_size = 3
    stride = 1
    padding = 0
    out_channels = 3
    pool_kernel_size = 1
    pool_stride = 1

    result = get_flatten_shape(
        width,
        height,
        kernel_size,
        stride,
        padding,
        out_channels,
        pool_kernel_size,
        pool_stride,
    )

    assert result == 27, f"Expected 27, but got {result}"


def test_flatten_shape_with_padding():
    width = 5
    height = 5
    kernel_size = 3
    stride = 1
    padding = 1
    out_channels = 1
    pool_kernel_size = 1
    pool_stride = 1

    result = get_flatten_shape(
        width,
        height,
        kernel_size,
        stride,
        padding,
        out_channels,
        pool_kernel_size,
        pool_stride,
    )

    assert result == 25, f"Expected 25, but got {result}"


def test_flatten_shape_negative():
    width = 5
    height = 5
    kernel_size = 6
    stride = 1
    padding = 0
    out_channels = 1
    pool_kernel_size = 1
    pool_stride = 1

    with pytest.raises(ValueError):
        get_flatten_shape(
            width,
            height,
            kernel_size,
            stride,
            padding,
            out_channels,
            pool_kernel_size,
            pool_stride,
        )
