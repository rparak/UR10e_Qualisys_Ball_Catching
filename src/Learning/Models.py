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
# File Name: ../Learning/Models.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# PyTorch (Tensors and automatic differentiation; the training only) [pip3 install torch]
import torch
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Learning/Core - the number of inputs per sample and per prefix, the basis degree, the horizon, the width factor
from Learning.Core import Layer_Sizes, CONST_SAMPLE_FEATURES, CONST_PREFIX_FEATURES, CONST_BASIS_DEGREE, CONST_HORIZON

"""
Description:
    Initialization of constants.

    The network of the learned predictor in PyTorch, for the training only (SPEC §7.8; F8):
    an encoder over the samples of a prefix, and after it a head giving the coefficients of a
    Chebyshev basis per axis (QT_BASIS). The velocity of the query-time model is the exact time derivative of
    its position, the derivative of the polynomials; there is no velocity output. Inference does not use this
    module (Learning.Predictor, NumPy).
"""


class Network_Cls(torch.nn.Module):
    """
    Description:
        A learned predictor's network: a GRU over the samples of a prefix in time order (each with its own time
        inputs, so irregular samples need no resampling), a layer joining its last state with the per-prefix
        inputs, and the model's layers after it (Learning.Core.CONST_LAYERS), SiLU between them.

    Initialization of the Class:
        Args:
            (1) name [string]: The model, one of Learning.Core.CONST_LAYERS.
            (2) hidden [int]: The size of the encoder's state (Model.Hidden).

        Example:
            Initialization:
                # Initialization of the class.
                Cls = Network_Cls('QT_BASIS', 128)

            Features:
                # Functions of the class.
                Cls.Encode(sequence, context, count); Cls.Layers(x); Cls.Weights(); Cls.Load(weights)
    """

    def __init__(self, name: str, hidden: int) -> None:
        super().__init__()
        # << PRIVATE >> #
        self.__name = name
        self.__gru = torch.nn.GRU(CONST_SAMPLE_FEATURES, hidden, batch_first=True)
        self.__context = torch.nn.Linear(hidden + CONST_PREFIX_FEATURES, hidden)
        self.__layers = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in Layer_Sizes(name, hidden)])

    @property
    def Name(self) -> str:
        """
        Description:
            The model's name.

        Returns:
            (1) parameter [string]: One of Learning.Core.CONST_LAYERS.
        """

        return self.__name

    def Encode(self, sequence: torch.Tensor, context: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """
        Description:
            The encoding of B prefixes: the GRU's state after each prefix's last sample (the padding follows
            the samples, so that state has seen only them), joined with the per-prefix inputs.

        Args:
            (1) sequence [Tensor<float> BxMx9]: The inputs per sample (Learning.Core.Prefix_Features).
            (2) context [Matrix<float> Bx1]: The inputs per prefix.
            (3) count [Vector<int> 1xB]: The samples of each prefix.

        Returns:
            (1) parameter [Matrix<float> BxH]: The encodings.
        """

        states, _ = self.__gru(sequence)
        last = states[torch.arange(len(sequence), device=sequence.device), count - 1]
        return torch.nn.functional.silu(self.__context(torch.cat((last, context), dim=1)))

    def Layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
            The layers after the encoder, SiLU between them.

        Args:
            (1) x [Tensor<float> ...xI]: The input.

        Returns:
            (1) parameter [Tensor<float> ...xO]: The output.
        """

        for i, layer in enumerate(self.__layers):
            x = layer(x)
            if i < len(self.__layers) - 1:
                x = x * torch.sigmoid(x)
        return x

    def Weights(self) -> tp.Dict[str, np.ndarray]:
        """
        Description:
            Every parameter as an array, by the names Learning.Predictor reads: the GRU's input and state matrices
            and biases (PyTorch's gate order: reset, update, new), the joining layer, and each layer after it.

        Returns:
            (1) parameter [Dict<string, Matrix<float>>]: The arrays (float32).
        """

        arrays = {'gru_input': self.__gru.weight_ih_l0, 'gru_state': self.__gru.weight_hh_l0, 'gru_input_bias': self.__gru.bias_ih_l0,
                  'gru_state_bias': self.__gru.bias_hh_l0, 'context': self.__context.weight, 'context_bias': self.__context.bias}
        for i, layer in enumerate(self.__layers):
            arrays['layer_%d' % i], arrays['layer_%d_bias' % i] = layer.weight, layer.bias
        return {key: value.detach().cpu().numpy().copy() for key, value in arrays.items()}

    def Load(self, weights: tp.Dict[str, np.ndarray]) -> None:
        """
        Description:
            Set every parameter from the arrays Weights gives (a network restored from its weights file).

        Args:
            (1) weights [Dict<string, Matrix<float>>]: The arrays, by Weights's names.
        """

        parameters = {'gru_input': self.__gru.weight_ih_l0, 'gru_state': self.__gru.weight_hh_l0, 'gru_input_bias': self.__gru.bias_ih_l0,
                      'gru_state_bias': self.__gru.bias_hh_l0, 'context': self.__context.weight, 'context_bias': self.__context.bias}
        for i, layer in enumerate(self.__layers):
            parameters['layer_%d' % i], parameters['layer_%d_bias' % i] = layer.weight, layer.bias
        with torch.no_grad():
            for key, parameter in parameters.items():
                parameter.copy_(torch.as_tensor(np.asarray(weights[key]), dtype=parameter.dtype))


def Chebyshev(s: torch.Tensor, degree: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
        The Chebyshev polynomials of the first kind T_0 ... T_degree at s and their derivatives, by the
        recurrences T_{k+1} = 2 s T_k - T_{k-1} and T'_{k+1} = 2 T_k + 2 s T'_k - T'_{k-1}.

    Args:
        (1) s [Tensor<float> ...]: The arguments, in [-1, 1] over the basis's span.
        (2) degree [int]: The highest degree, >= 1.

    Returns:
        (1) parameter 1 [Tensor<float> ...x(degree + 1)]: The values.
        (2) parameter 2 [Tensor<float> ...x(degree + 1)]: The derivatives with respect to s.
    """

    values, derivatives = [torch.ones_like(s), s], [torch.zeros_like(s), torch.ones_like(s)]
    for _ in range(degree - 1):
        values.append(2.0 * s * values[-1] - values[-2])
        derivatives.append(2.0 * values[-2] + 2.0 * s * derivatives[-1] - derivatives[-2])
    return torch.stack(values, dim=-1), torch.stack(derivatives, dim=-1)


def Query(network: Network_Cls, encoding: torch.Tensor, tau: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor,
          velocity: bool = True) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
    """
    Description:
        The query-time model's displacement from the last observed position and its velocity at Q query times per
        prefix; each query is answered from its own time only. The coefficients mean + scale * output of a
        Chebyshev series in s = 2 tau / CONST_HORIZON - 1 per axis; the velocity is the series of the derivatives.

    Args:
        (1) network [Network_Cls(object)]: The network.
        (2) encoding [Matrix<float> BxH]: The encodings (Network_Cls.Encode).
        (3) tau [Matrix<float> BxQ]: The query times after t_0 [s].
        (4) mean [Tensor<float>]: The outputs' mean (Normalisation_Str.Output_Mean).
        (5) scale [Tensor<float>]: The outputs' scale (Normalisation_Str.Output_Scale).
        (6) velocity [bool]: False leaves the velocity out (a position-only loss does not pay for it).

    Returns:
        (1) parameter 1 [Tensor<float> BxQx3]: The displacements [m].
        (2) parameter 2 [Tensor<float> BxQx3]: The velocities [m/s], or None.
    """

    coefficients = mean + scale * network.Layers(encoding).reshape(len(encoding), 3, CONST_BASIS_DEGREE + 1)
    values, derivatives = Chebyshev(2.0 * tau / CONST_HORIZON - 1.0, CONST_BASIS_DEGREE)
    return (torch.einsum('bak,bqk->bqa', coefficients, values),
            (2.0 / CONST_HORIZON) * torch.einsum('bak,bqk->bqa', coefficients, derivatives) if velocity else None)
