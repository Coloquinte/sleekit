from sleekit import Sleekit

import torch
import torch.nn as nn


def test_linear():
    layer = nn.Linear(10, 5)
    stats = Sleekit(layer)
    stats.add_batch(torch.randn(10))
    assert stats.count == 1
    stats.add_batch(torch.randn(3, 10))
    assert stats.count == 4
    stats.add_batch(torch.randn(3, 3, 10))
    assert stats.count == 13


def test_conv2d():
    layer = nn.Conv2d(10, 5, 3)
    stats = Sleekit(layer)
    stats.add_batch(torch.randn(10, 3, 3))
    assert stats.count == 1
    stats.add_batch(torch.randn(5, 10, 3, 3))
    assert stats.count == 6
    stats.add_batch(torch.randn(10, 7, 7))
    assert stats.count == 31

    layer = nn.Conv2d(10, 5, 3, padding=1)
    stats = Sleekit(layer)
    stats.add_batch(torch.randn(10, 3, 3))
    assert stats.count == 9
    stats.add_batch(torch.randn(5, 10, 3, 3))
    assert stats.count == 54
    stats.add_batch(torch.randn(10, 5, 5))
    assert stats.count == 79


def test_conv1d():
    layer = nn.Conv1d(10, 5, 3)
    stats = Sleekit(layer)
    stats.add_batch(torch.randn(10, 3))
    assert stats.count == 1
    stats.add_batch(torch.randn(5, 10, 3))
    assert stats.count == 6
    stats.add_batch(torch.randn(10, 7))
    assert stats.count == 11


def test_quantize_linear():
    layer = nn.Linear(10, 5)
    stats = Sleekit(layer)
    stats.add_batch(torch.randn(20, 10))
    stats.quantize_sleekit_light(3)


def test_quantize_conv1d():
    layer = nn.Conv1d(10, 5, 3)
    stats = Sleekit(layer)
    stats.add_batch(torch.randn(20, 10, 5))
    stats.quantize_sleekit_light(3)


def test_quantize_conv2d():
    layer = nn.Conv2d(10, 5, 3)
    stats = Sleekit(layer)
    stats.add_batch(torch.randn(20, 10, 5, 5))
    stats.quantize_sleekit_light(3)
