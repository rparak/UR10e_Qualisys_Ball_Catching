## =========================================================================== ##
# MIT License
# Copyright (c) 2026 Roman Parak
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
## =========================================================================== ##
# Author   : Roman Parak
# File Name: ../test/Learning/test_models.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# PyTorch (Tensors and automatic differentiation) [pip3 install torch]
import torch
# Custom Lib.: the network under test
#   ../Learning/Models
from Learning.Models import Network_Cls, Chebyshev, Query
#   ../Learning/Core - the basis degree and the horizon
from Learning.Core import CONST_BASIS_DEGREE, CONST_HORIZON

"""
Description:
    The network of the learned predictor (SPEC §7.8; F8): expectations come from NumPy's Chebyshev module,
    reverse-mode automatic differentiation and finite differences, never from the functions under test.
"""


def test_the_chebyshev_polynomials_and_their_derivatives_are_numpys():
    """
    Description:
        The basis of the basis decoder: T_0 ... T_6 at 200 arguments equal NumPy's Chebyshev Vandermonde matrix,
        and their derivatives the derivative series of each polynomial, within 1e-12.
    """

    s = np.random.default_rng(2).uniform(-1.0, 1.0, 200)
    values, derivatives = Chebyshev(torch.tensor(s), CONST_BASIS_DEGREE)
    assert values.shape == (200, CONST_BASIS_DEGREE + 1)
    assert np.allclose(values.numpy(), np.polynomial.chebyshev.chebvander(s, CONST_BASIS_DEGREE), rtol=0.0, atol=1e-12)
    for k in range(CONST_BASIS_DEGREE + 1):
        expected = np.polynomial.chebyshev.chebval(s, np.polynomial.chebyshev.chebder(np.eye(CONST_BASIS_DEGREE + 1)[k])) if k else np.zeros(200)
        assert np.allclose(derivatives.numpy()[:, k], expected, rtol=0.0, atol=1e-11), k


@pytest.mark.parametrize('name', ['QT_BASIS'])
def test_a_query_time_models_velocity_is_the_derivative_of_its_position(name):
    """
    Description:
        The velocity of a query-time model is the exact time derivative of its predicted
        position, never a separate output. In double precision the velocity of Query equals the derivative of
        its position by reverse-mode automatic differentiation within 1e-10 m/s and a central finite difference
        of the position (step 1e-6 s) within 1e-6 m/s.
    """

    torch.manual_seed(3)
    network = Network_Cls(name, 16).double()
    encoding = torch.randn(4, 16, dtype=torch.float64)
    tau = (torch.rand(4, 9, dtype=torch.float64) * 1.5).requires_grad_(True)
    shape = (3, CONST_BASIS_DEGREE + 1)
    mean, scale = torch.randn(shape, dtype=torch.float64), torch.rand(shape, dtype=torch.float64) + 0.5
    position, velocity = Query(network, encoding, tau, mean, scale)
    assert position.shape == velocity.shape == (4, 9, 3)
    reverse = torch.stack([torch.autograd.grad(position[..., a].sum(), tau, retain_graph=True)[0] for a in range(3)], dim=2)
    assert float((velocity - reverse).detach().abs().max()) < 1e-10
    h = 1e-6
    difference = (Query(network, encoding, tau + h, mean, scale)[0] - Query(network, encoding, tau - h, mean, scale)[0]) / (2.0 * h)
    assert float((velocity - difference).detach().abs().max()) < 1e-6 and float(velocity.detach().abs().max()) > 1e-3


def test_the_basis_decoders_position_is_its_chebyshev_series():
    """
    Description:
        QT_BASIS returns, per axis, the Chebyshev series of the coefficients mean + scale * output in
        s = 2 tau / 2.0 s - 1: equal to NumPy's evaluation of the same coefficients within 1e-12 m.
    """

    torch.manual_seed(4)
    network = Network_Cls('QT_BASIS', 16).double()
    encoding, tau = torch.randn(2, 16, dtype=torch.float64), torch.rand(2, 7, dtype=torch.float64) * CONST_HORIZON
    mean = torch.randn(3, CONST_BASIS_DEGREE + 1, dtype=torch.float64)
    scale = torch.rand(3, CONST_BASIS_DEGREE + 1, dtype=torch.float64)
    position, _ = Query(network, encoding, tau, mean, scale)
    coefficients = (mean + scale * network.Layers(encoding).reshape(2, 3, CONST_BASIS_DEGREE + 1)).detach().numpy()
    for b in range(2):
        expected = np.polynomial.chebyshev.chebval(2.0 * tau[b].numpy() / CONST_HORIZON - 1.0, coefficients[b].T)
        assert np.allclose(position[b].detach().numpy(), expected.T, rtol=0.0, atol=1e-12)


def test_the_encoding_of_a_prefix_ignores_what_follows_its_samples():
    """
    Description:
        The padding follows a prefix's samples and is never read (SPEC §2.13.4): the encoding of a prefix of 5
        samples in a batch of width 12 is the same bit for bit whatever the 7 entries after them hold, and
        differs when its fifth sample changes.
    """

    torch.manual_seed(5)
    network = Network_Cls('QT_BASIS', 16)
    sequence, context, count = torch.randn(2, 12, 9), torch.randn(2, 1), torch.tensor([5, 12])
    other = sequence.clone()
    other[0, 5:] = 99.0
    with torch.no_grad():
        first, second = network.Encode(sequence, context, count), network.Encode(other, context, count)
        other[0, 4, 0] += 1.0
        third = network.Encode(other, context, count)
    assert torch.equal(first, second) and not torch.equal(first[0], third[0]) and torch.equal(first[1], third[1])


@pytest.mark.parametrize('name, last', [('QT_BASIS', (3 * (CONST_BASIS_DEGREE + 1), 32))])
def test_the_exported_weights_hold_every_parameter(name, last):
    """
    Description:
        Network_Cls.Weights names every parameter once with its shape (the arrays weights.npz holds, SPEC §9.6):
        for a state of 16, the GRU's matrices 48 x 9 and 48 x 16, the joining layer 16 x 17, the model's last
        layer; their sizes sum to the network's parameter count, and the last layer does not start at zero.
    """

    network = Network_Cls(name, 16)
    weights = network.Weights()
    assert weights['gru_input'].shape == (48, 9) and weights['gru_state'].shape == (48, 16) and weights['context'].shape == (16, 17)
    layers = sorted(key for key in weights if key.startswith('layer_') and not key.endswith('_bias'))
    assert weights[layers[-1]].shape == last and all(value.dtype == np.float32 for value in weights.values())
    assert sum(value.size for value in weights.values()) == sum(p.numel() for p in network.parameters())
    assert weights[layers[-1]].any()
