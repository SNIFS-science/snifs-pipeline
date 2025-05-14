import numpy as np
import pytest
from conftest import dict_parametrize

from pipeline.tasks.preprocessing.binary_offset import count_bits, sum_binary_index


@dict_parametrize(
    {
        "empty_data": {
            "x": np.array([], dtype=np.uint16).reshape(0, 0),
            "expected": np.array([], dtype=np.uint16).reshape(0, 0),
        },
        "single_value_0": {
            "x": np.array([[0]], dtype=np.uint16),
            "expected": np.array([[0]], dtype=np.uint16),
        },
        "single_value_1": {
            "x": np.array([[1]], dtype=np.uint16),
            "expected": np.array([[0]], dtype=np.uint16),
        },
        "single_value_2": {
            "x": np.array([[2]], dtype=np.uint16),
            "expected": np.array([[1]], dtype=np.uint16),
        },
        "single_value_11": {
            "x": np.array([[11]], dtype=np.uint16),
            "expected": np.array([[4]], dtype=np.uint16),
        },
        "single_value_259": {
            "x": np.array([[259]], dtype=np.uint16),
            "expected": np.array([[9]], dtype=np.uint16),
        },
        "multiple_values": {
            "x": np.array([[0, 1], [2, 11], [259, 0]], dtype=np.uint16),
            "expected": np.array([[0, 0], [1, 4], [9, 0]], dtype=np.uint16),
        },
    }
)
def test_sum_binary_index(x: np.ndarray, expected: np.ndarray):
    result = sum_binary_index(x)
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)


@dict_parametrize(
    {
        "data is not uint16": {
            "x": np.array([[1, 2], [3, 4]], dtype=np.uint8),
        },
        "data is not 2D": {
            "x": np.array([1, 2, 3, 4], dtype=np.uint16),
        },
    }
)
def test_sum_binary_index_failures(x: np.ndarray):
    with pytest.raises(AssertionError):
        sum_binary_index(x)


@dict_parametrize(
    {
        "empty_data": {
            "x": np.array([], dtype=np.uint32),
            "expected": np.array([], dtype=np.uint32),
        },
        "single_value_0": {
            "x": np.array([[0]], dtype=np.uint16),
            "expected": np.array([[0]], dtype=np.uint16),
        },
        "single_value_1": {
            "x": np.array([[1]], dtype=np.uint16),
            "expected": np.array([[1]], dtype=np.uint16),
        },
        "single_value_2": {
            "x": np.array([[2]], dtype=np.uint8),
            "expected": np.array([[1]], dtype=np.uint8),
        },
        "single_value_11": {
            "x": np.array([[11]], dtype=np.uint16),
            "expected": np.array([[3]], dtype=np.uint16),
        },
        "single_value_259": {
            "x": np.array([[259]], dtype=np.uint32),
            "expected": np.array([[3]], dtype=np.uint32),
        },
    }
)
def test_count_bits(x: np.ndarray, expected: np.ndarray):
    result = count_bits(x)
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)


@dict_parametrize(
    {
        "data is not unsigned int": {
            "x": np.array([[1, 2], [3, 4]], dtype=np.int8),
        },
    }
)
def test_count_bits_failures(x: np.ndarray):
    with pytest.raises(AssertionError):
        count_bits(x)
