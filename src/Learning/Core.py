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
# File Name: ../Learning/Core.py
## =========================================================================== ##

# CSV (epochs.csv, read back for the figures)
import csv
# JSON (JSON encoder and decoder)
import json
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp

"""
Description:
    Initialization of constants.

    The inputs of a learned predictor (SPEC §7.8; F8): each observed sample of a
    prefix enters as observed, with its own time stamp, never resampled, and relative to the last sample only
    (no absolute position: a predictor must not learn where the generator throws from). And the normalisation,
    fitted on training prefixes only, with its one reader and writer (normalisation.json, SPEC §9.6), and the
    constants the training (Learning.Models, Learning.Training) and the inference (Learning.Predictor) share.
    NumPy only: inference imports no heavy runtime. Pure computation apart from the two file functions.
"""
# The farthest time after t_0 a query-time trajectory is defined for, and the span of the basis decoder's
# polynomials: the longest simulated flight lasts 1.52 s from its release (the Default dataset's largest floor
# time, F8 Part A), so 2.0 s holds every remaining flight with a margin. *Assumed* margin.
#   Unit [float, s]
CONST_HORIZON = 2.0
# The degree of the basis decoder's Chebyshev polynomials per axis: the smallest whose own
# error is below 0.5 mm. Measured on the simulated truth of the Default dataset's 4 377 validation prefixes
# (least squares over CONST_HORIZON from t_0, the largest position error over the remaining flight; F8 Part
# B, 2026-11-17): degree 4 3.07 mm, 5 0.97 mm, 6 0.20 mm, 7 0.12 mm (3 000 training prefixes alike: 0.96 and
# 0.20 mm at 5 and 6).
#   Unit [int]
CONST_BASIS_DEGREE = 6
# The width of the layers after the encoder, as a multiple of the encoder's state (Model.Hidden): F8 Part
# A's pilots (128 and 256). *Assumed*.
#   Unit [int]
CONST_WIDTH_FACTOR = 2
# The scale of a sample's time to t_0 and of its step to the previous sample: the inputs are of order 1 for
# prefixes of 0.03-1 s at 100-300 Hz (SPEC §4.5). *Assumed* (F8 Part A's pilots).
#   Unit [float, s]
CONST_TIME_SCALE = 0.1
CONST_STEP_SCALE = 0.01
# The values per sample: the time to t_0, the position relative to the last sample (3), the finite-difference
# velocity to the previous sample (3), the frames lost before it, the step to the previous sample.
#   Unit [int]
CONST_SAMPLE_FEATURES = 9
# The values per prefix: the frame period.
#   Unit [int]
CONST_PREFIX_FEATURES = 1
# normalisation.json's format version; the reader refuses another.
#   Unit [int]
CONST_NORMALISATION_VERSION = 1
# The layers after the model's encoder, as (inputs, outputs) in the encoder's state H, the width W =
# CONST_WIDTH_FACTOR * H and literal sizes: QT_BASIS gives 3 * (CONST_BASIS_DEGREE + 1) coefficients. A weights
# file names its model, so the table is by name (SPEC §9.6).
#   Unit [Dict<string, function>]
CONST_LAYERS = {'QT_BASIS': lambda H, W: [(H, W), (W, 3 * (CONST_BASIS_DEGREE + 1))]}
# The columns of epochs.csv (SPEC §9.6), written by Learning.Training and read by Read_Epochs.
#   Unit [Tuple<string>]
CONST_EPOCH_COLUMNS = ('epoch', 'train_loss_m', 'validation_loss_m', 'validation_position_m')


class Features_Error_Cls(ValueError):
    """
    Description:
        The inputs of a prefix cannot be built, or a normalisation file cannot be read: fewer than 2 samples,
        another format version, a missing or misshapen entry. The message names the value and what is expected.
    """


@dataclass
class Normalisation_Str:
    """
    Description:
        The normalisation of a learned predictor's inputs and outputs (SPEC §7.8), fitted on the training
        prefixes only (Learning.Training.Fit_Normalisation). A network output o stands for
        Output_Mean + Output_Scale * o.
    """

    # The deviation of the sample positions relative to the last sample, per axis.
    #   Unit [Vector<float> 1x3, m]
    Relative_Std: np.ndarray = field(default_factory=lambda: np.ones(3))
    # The mean and the deviation of the true velocity at t_0, per axis: the scale of the finite-difference
    # velocities.
    #   Unit [Vector<float> 1x3, m/s]
    Velocity_Mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Velocity_Std: np.ndarray = field(default_factory=lambda: np.ones(3))
    # The mean and the scale of the network's outputs: the basis coefficients per axis (QT_BASIS).
    #   Unit [Matrix<float> 3x(degree + 1), m]
    Output_Mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Output_Scale: np.ndarray = field(default_factory=lambda: np.ones(3))


def Prefix_Features(time: np.ndarray, position: np.ndarray, count: np.ndarray, rate: np.ndarray, *,
                    Normalisation_Str: Normalisation_Str) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
        The inputs of B prefixes, padded after their samples (SPEC §7.8): per sample its time to t_0, its
        position relative to the last sample, the finite-difference velocity from the previous sample (the
        first sample takes the second's), the frames lost before it as log(1 + frames) and its step to the
        previous sample; per prefix the frame period. A sample is used at its own time stamp; nothing is
        resampled, and a padded entry is 0.

    Args:
        (1) time [Matrix<float> BxM]: The sample times [s], increasing within each prefix's count.
        (2) position [Tensor<float> BxMx3]: The observed positions [m].
        (3) count [Vector<int> 1xB]: The samples of each prefix, 2 <= count <= M.
        (4) rate [Vector<float> 1xB]: The frame rate of each prefix [Hz].
        (5) Normalisation_Str [Normalisation_Str(object)]: REQUIRED, keyword-only: the normalisation.

    Returns:
        (1) parameter 1 [Tensor<float> BxMx9]: The inputs per sample [-].
        (2) parameter 2 [Matrix<float> Bx1]: The inputs per prefix [-].
        (3) parameter 3 [Matrix<float> Bx3]: The last observed position of each prefix [m].
    """

    time = np.asarray(time, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    count = np.asarray(count, dtype=np.int64).reshape(-1)
    rate = np.asarray(rate, dtype=np.float64).reshape(-1)
    B, M = time.shape
    if count.min() < 2 or count.max() > M:
        raise Features_Error_Cls('A prefix has %d sample(s); the inputs need 2 to %d samples (a velocity needs two).'
                                 % (int(count.min() if count.min() < 2 else count.max()), M))
    n = Normalisation_Str
    rows = np.arange(B)
    present = np.arange(M)[None, :] < count[:, None]
    last = position[rows, count - 1]
    step = np.zeros_like(time)
    step[:, 1:] = time[:, 1:] - time[:, :-1]
    step = np.where(present, step, 0.0)
    safe = np.where(step > 0.0, step, 1.0)
    velocity = np.zeros_like(position)
    velocity[:, 1:] = (position[:, 1:] - position[:, :-1]) / safe[:, 1:, None]
    velocity[:, 0] = velocity[:, 1]
    lost = np.where(step > 0.0, np.maximum(np.rint(step * rate[:, None]) - 1.0, 0.0), 0.0)
    sequence = np.concatenate(((time - time[rows, count - 1][:, None])[:, :, None] / CONST_TIME_SCALE,
                               (position - last[:, None, :]) / n.Relative_Std, (velocity - n.Velocity_Mean) / n.Velocity_Std,
                               np.log1p(lost)[:, :, None], step[:, :, None] / CONST_STEP_SCALE), axis=2)
    return np.where(present[:, :, None], sequence, 0.0), (1.0 / (rate * CONST_STEP_SCALE))[:, None], last


def Layer_Sizes(name: str, hidden: int) -> tp.List[tp.Tuple[int, int]]:
    """
    Description:
        The (inputs, outputs) of the layers after a model's encoder (CONST_LAYERS).

    Args:
        (1) name [string]: The model.
        (2) hidden [int]: The size of the encoder's state.

    Returns:
        (1) parameter [List<Tuple<int, int>>]: The sizes, first layer first.
    """

    return CONST_LAYERS[name](hidden, CONST_WIDTH_FACTOR * hidden)


def Chebyshev_Fit(tau: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Description:
        The least-squares coefficients of a Chebyshev series of degree CONST_BASIS_DEGREE per axis, in
        s = 2 tau / CONST_HORIZON - 1: the basis of the basis decoder fitted to a trajectory (its own
        approximation error, and the scale of its coefficients).

    Args:
        (1) tau [Vector<float> 1xK]: The times after t_0 [s], in [0, CONST_HORIZON].
        (2) values [Matrix<float> Kx3]: The values there [m].

    Returns:
        (1) parameter [Matrix<float> 3x(CONST_BASIS_DEGREE + 1)]: The coefficients per axis [m].
    """

    return np.polynomial.chebyshev.chebfit(2.0 * np.asarray(tau, dtype=np.float64) / CONST_HORIZON - 1.0,
                                           np.asarray(values, dtype=np.float64), CONST_BASIS_DEGREE).T


def Chebyshev_Series(coefficients: np.ndarray, tau: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        A Chebyshev series per axis at the times tau and its time derivative (the derivative series times
        2 / CONST_HORIZON): element by element, so each time's answer depends on that time only.

    Args:
        (1) coefficients [Matrix<float> 3x(D + 1)]: The coefficients per axis [m].
        (2) tau [Vector<float> 1xQ]: The times after t_0 [s].

    Returns:
        (1) parameter 1 [Matrix<float> Qx3]: The values [m].
        (2) parameter 2 [Matrix<float> Qx3]: The derivatives with respect to the time [m/s].
    """

    s = 2.0 * np.asarray(tau, dtype=np.float64) / CONST_HORIZON - 1.0
    c = np.asarray(coefficients, dtype=np.float64).T
    return (np.polynomial.chebyshev.chebval(s, c).T,
            (2.0 / CONST_HORIZON) * np.polynomial.chebyshev.chebval(s, np.polynomial.chebyshev.chebder(c)).T)


def Write_Normalisation(path: str, normalisation: Normalisation_Str) -> None:
    """
    Description:
        normalisation.json (SPEC §9.6): the format version and every array as nested lists, sorted keys; the
        same normalisation gives the same bytes (a float is written by its shortest exact text).

    Args:
        (1) path [string]: The file.
        (2) normalisation [Normalisation_Str(object)]: The normalisation.
    """

    document = {'format_version': CONST_NORMALISATION_VERSION,
                'relative_std': normalisation.Relative_Std.tolist(), 'velocity_mean': normalisation.Velocity_Mean.tolist(),
                'velocity_std': normalisation.Velocity_Std.tolist(), 'output_mean': normalisation.Output_Mean.tolist(),
                'output_scale': normalisation.Output_Scale.tolist()}
    with open(path, 'w', encoding='ascii') as file:
        file.write(json.dumps(document, sort_keys=True, indent=2) + '\n')


def Read_Normalisation(path: str) -> Normalisation_Str:
    """
    Description:
        The normalisation of a normalisation.json of format version CONST_NORMALISATION_VERSION; another
        version, a missing entry or an input scale that is not three positive numbers is refused.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Normalisation_Str(object)]: The normalisation.
    """

    with open(path, 'r', encoding='ascii') as file:
        document = json.load(file)
    if not isinstance(document, dict) or document.get('format_version') != CONST_NORMALISATION_VERSION:
        raise Features_Error_Cls('%s is not a normalisation file of format version %d (SPEC §9.6). Train the run again.'
                                 % (path, CONST_NORMALISATION_VERSION))
    keys = ('relative_std', 'velocity_mean', 'velocity_std', 'output_mean', 'output_scale')
    missing = [key for key in keys if key not in document]
    if missing:
        raise Features_Error_Cls('%s lacks %s (SPEC §9.6). Train the run again.' % (path, ', '.join(missing)))
    arrays = [np.asarray(document[key], dtype=np.float64) for key in keys]
    if any(a.shape != (3,) for a in arrays[:3]) or not (np.all(arrays[0] > 0.0) and np.all(arrays[2] > 0.0)) \
            or arrays[3].shape != arrays[4].shape:
        raise Features_Error_Cls('%s holds input scales that are not three positive numbers per entry, or output arrays of '
                                 'different shapes (SPEC §9.6). Train the run again.' % path)
    return Normalisation_Str(Relative_Std=arrays[0], Velocity_Mean=arrays[1], Velocity_Std=arrays[2], Output_Mean=arrays[3],
                             Output_Scale=arrays[4])


def Read_Epochs(path: str) -> np.ndarray:
    """
    Description:
        epochs.csv (SPEC §9.6; Learning.Training.Write_Epochs) read back for the figures (scripts/plots.py): one row
        per epoch; a file with another header is refused.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [Matrix<float> Ex4]: Per epoch its number, the train loss, the validation loss and its position
                                           term [-, m, m, m], in the order of CONST_EPOCH_COLUMNS.
    """

    with open(path, 'r', encoding='ascii', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None or tuple(header) != CONST_EPOCH_COLUMNS:
            raise Features_Error_Cls('%s does not have the columns %s (SPEC §9.6); it was not written by train.' % (path,
                                                                                                                    ', '.join(CONST_EPOCH_COLUMNS)))
        return np.array([[float(value) for value in row] for row in reader], dtype=np.float64).reshape(-1, len(CONST_EPOCH_COLUMNS))
