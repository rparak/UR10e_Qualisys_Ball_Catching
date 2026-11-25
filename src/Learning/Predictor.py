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
# File Name: ../Learning/Predictor.py
## =========================================================================== ##

# System (Default)
import os
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Configuration/Core - the learned predictor's name
from Configuration.Core import CONST_LEARNED_MODEL_NAMES
#   ../Flight/Core - the one flight model (the numerical predictor's)
from Flight.Core import Flight_Model_Str
#   ../Dataset/Core - the floor height a trajectory ends at
from Dataset.Core import CONST_FLOOR_HEIGHT
#   ../Observation/Core - M4 (a numerical predictor's weights)
from Observation.Core import Observation_Model_Str
#   ../Prediction/Core - the prefix check, the predictors' error and the numerical predictor
from Prediction.Core import Check_Prefix, Prediction_Error_Cls, Predict
#   ../Learning/Core - the inputs, the normalisation, the layers and the basis
from Learning.Core import (Prefix_Features, Normalisation_Str, Read_Normalisation, Layer_Sizes, Chebyshev_Series, Features_Error_Cls,
                           CONST_SAMPLE_FEATURES, CONST_PREFIX_FEATURES, CONST_BASIS_DEGREE, CONST_HORIZON)

"""
Description:
    Initialization of constants.

    A trained learned predictor at work (SPEC §2.13.13, §7.8; F8), in NumPy and double
    precision, without the training's runtime: the weights file with its one writer and reader, the encoder of
    a prefix, and the trajectory. The query-time model (QT_BASIS) answers each query by evaluating its
    decoder at that time, with no grid and no interpolation; its velocity is the exact time derivative of its
    position (the derivative of the basis); its end is the root of its height at the floor. Every query is
    computed on its own (element-wise functions of its own time), so its answer does not depend on the other
    times asked.
"""
# weights.npz's format version; the reader refuses another.
#   Unit [int]
CONST_WEIGHTS_VERSION = 1
# The file names of a trained seed (SPEC §9.6).
#   Unit [string]
CONST_WEIGHTS_FILE = 'weights.npz'
CONST_NORMALISATION_FILE = 'normalisation.json'
# The step of the scan for a query-time trajectory's descent through the floor: a ball does not dip below the
# floor and return within 0.1 s, and the root is then refined inside the interval found. *Assumed*.
#   Unit [float, s]
CONST_FLOOR_SCAN = 0.1
# The refinement of the floor time inside its interval: Newton steps with the trajectory's own vertical
# velocity, halving the interval when a step leaves it, until a Newton step or the interval is below
# CONST_ROOT_TOLERANCE (0.1 ps: 1e-12 m at the 5-8 m/s of a landing). Newton approaches the root of a falling
# height from one side, so the interval alone would not close (measured, F8 Part B: without the step's own
# test a third of the trajectories took about 200 decoder evaluations).
#   Unit [int], [float, s]
CONST_ROOT_ITERATIONS = 100
CONST_ROOT_TOLERANCE = 1e-13
# The step of the scan of a query-time trajectory for a surface's crossing (Evaluation's catch plane): above
# a surface for less than 0.01 s means an apex within 0.12 mm of it. *Assumed*.
#   Unit [float, s]
CONST_CROSSING_SCAN = 0.01


@dataclass
class Predictor_Str:
    """
    Description:
        A trained learned predictor (one seed of a run, SPEC §9.6): its model, the size of its encoder's
        state, its weights in double precision and its normalisation.
    """

    # The model (Configuration.Core.CONST_LEARNED_MODEL_NAMES) and the size of the encoder's state.
    #   Unit [string], [int]
    Name: str = ''
    Hidden: int = 0
    # The weights by name (Learning.Models.Network_Cls.Weights).
    #   Unit [Dict<string, Matrix<float>>]
    Weights: tp.Dict[str, np.ndarray] = field(default_factory=dict)
    # The normalisation of the inputs and the outputs.
    #   Unit [Normalisation_Str]
    Normalisation: Normalisation_Str = field(default_factory=Normalisation_Str)


@dataclass
class Query_Trajectory_Str:
    """
    Description:
        The trajectory of the query-time model (SPEC §2.13.13): fixed before any query (the encoding of the prefix
        is in Decoder), answered in [T_0, T_End] only by evaluating the decoder at the times asked.
    """

    # The last observed time and the predicted end, the ball centre at CONST_FLOOR_HEIGHT.
    #   Unit [float, s]
    T_0: float = 0.0
    T_End: float = 0.0
    # The time of the prefix's first sample (predictions.npz records it).
    #   Unit [float, s]
    Grid_Start: float = 0.0
    # The last observed position, which the decoder's displacement is added to.
    #   Unit [Vector<float> 1x3, m]
    Origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # The decoder: the displacement [m] and the velocity [m/s] at times after T_0 [s], Qx3 each.
    #   Unit [function]
    Decoder: tp.Optional[tp.Callable[[np.ndarray], tp.Tuple[np.ndarray, np.ndarray]]] = None

    def State(self, t: np.ndarray) -> np.ndarray:
        """
        Description:
            The state at each time: the decoder at t - T_0, the position from Origin, the velocity its exact
            time derivative. Each answer depends on its own time only (query independence, SPEC §2.13.13). A
            time outside [T_0, T_End] is refused.

        Args:
            (1) t [Vector<float> 1xM]: The times [s], in any order.

        Returns:
            (1) parameter [Matrix<float> Mx6]: The states [m, m/s].
        """

        t = np.asarray(t, dtype=np.float64).reshape(-1)
        outside = t[~np.isfinite(t) | (t < self.T_0) | (t > self.T_End)]
        if outside.size:
            raise Prediction_Error_Cls('The time %r s is outside the trajectory\'s range [%r, %r] s: a trajectory is answered from its '
                                       'last observed sample to its predicted end only (SPEC §2.13.13).'
                                       % (float(outside[0]), self.T_0, self.T_End))
        displacement, velocity = self.Decoder(t - self.T_0)
        return np.hstack((self.Origin + displacement, velocity))

    def Scan_Times(self, low: float) -> np.ndarray:
        """
        Description:
            The times from low to T_End at which a crossing of a surface is looked for (Evaluation), every
            CONST_CROSSING_SCAN.

        Args:
            (1) low [float]: The first time, in [T_0, T_End] [s].

        Returns:
            (1) parameter [Vector<float> 1xK]: The times, increasing, from low to T_End [s].
        """

        inner = low + CONST_CROSSING_SCAN * np.arange(1, int(np.ceil((self.T_End - low) / CONST_CROSSING_SCAN)) + 1)
        return np.concatenate(([low], inner[inner < self.T_End], [self.T_End]))


def _Names(name: str, hidden: int) -> tp.Dict[str, tp.Tuple[int, ...]]:
    """
    Description:
        The arrays a model's weights file holds, with their shapes.

    Args:
        (1) name [string]: The model.
        (2) hidden [int]: The size of the encoder's state.

    Returns:
        (1) parameter [Dict<string, Tuple<int>>]: The name -> shape.
    """

    shapes: tp.Dict[str, tp.Tuple[int, ...]] = {'gru_input': (3 * hidden, CONST_SAMPLE_FEATURES), 'gru_state': (3 * hidden, hidden),
                                                 'gru_input_bias': (3 * hidden,), 'gru_state_bias': (3 * hidden,),
                                                 'context': (hidden, hidden + CONST_PREFIX_FEATURES), 'context_bias': (hidden,)}
    for i, (inputs, outputs) in enumerate(Layer_Sizes(name, hidden)):
        shapes['layer_%d' % i], shapes['layer_%d_bias' % i] = (outputs, inputs), (outputs,)
    return shapes


def Write_Weights(path: str, name: str, hidden: int, weights: tp.Dict[str, np.ndarray]) -> None:
    """
    Description:
        weights.npz (SPEC §9.6), np.savez uncompressed: the format version, the model,
        the size of the encoder's state and every weight array (float32) under its name. The same weights give
        the same bytes.

    Args:
        (1) path [string]: The file.
        (2) name [string]: The model.
        (3) hidden [int]: The size of the encoder's state.
        (4) weights [Dict<string, Matrix<float>>]: The arrays (Learning.Models.Network_Cls.Weights).
    """

    np.savez(path, format_version=np.int64(CONST_WEIGHTS_VERSION), model=np.array(name), hidden=np.int64(hidden),
             **{key: np.asarray(weights[key], dtype=np.float32) for key in _Names(name, hidden)})


def Read_Weights(path: str) -> tp.Tuple[str, int, tp.Dict[str, np.ndarray]]:
    """
    Description:
        The model, the size of its encoder's state and its weights (double precision) of a weights.npz of
        format version CONST_WEIGHTS_VERSION; another version, an unknown model, a missing array, a wrong
        shape or a value that is not finite is refused.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter 1 [string]: The model.
        (2) parameter 2 [int]: The size of the encoder's state.
        (3) parameter 3 [Dict<string, Matrix<float>>]: The weights.
    """

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if 'format_version' not in arrays or int(arrays['format_version']) != CONST_WEIGHTS_VERSION:
        raise Prediction_Error_Cls('%s is not a weights file of format version %d (SPEC §9.6). Train the run again.'
                                   % (path, CONST_WEIGHTS_VERSION))
    name, hidden = str(arrays.get('model', '')), int(arrays.get('hidden', 0))
    if name not in CONST_LEARNED_MODEL_NAMES or hidden < 1:
        raise Prediction_Error_Cls('%s names the model %r with a state of %d; a learned predictor is one of %s with a state >= 1 '
                                   '(SPEC §9.6). Train the run again.' % (path, name, hidden, ', '.join(CONST_LEARNED_MODEL_NAMES)))
    weights = {}
    for key, shape in _Names(name, hidden).items():
        if key not in arrays or arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
            raise Prediction_Error_Cls('%s: the array %s is missing, is not of shape %s or holds a value that is not finite '
                                       '(SPEC §9.6). Train the run again.' % (path, key, 'x'.join(str(n) for n in shape)))
        weights[key] = arrays[key].astype(np.float64)
    return name, hidden, weights


def Load_Predictor(folder: str) -> Predictor_Str:
    """
    Description:
        The trained predictor of one seed's folder of a run (SPEC §9.6): its weights.npz and
        normalisation.json.

    Args:
        (1) folder [string]: The folder <run>/seed_<s>.

    Returns:
        (1) parameter [Predictor_Str(object)]: The predictor.
    """

    try:
        name, hidden, weights = Read_Weights(os.path.join(folder, CONST_WEIGHTS_FILE))
        normalisation = Read_Normalisation(os.path.join(folder, CONST_NORMALISATION_FILE))
    except (OSError, Features_Error_Cls) as error:
        raise Prediction_Error_Cls('%s holds no trained predictor (%s): train writes %s and %s per seed (SPEC §9.6).'
                                   % (folder, error, CONST_WEIGHTS_FILE, CONST_NORMALISATION_FILE)) from error
    return Predictor_Str(Name=name, Hidden=hidden, Weights=weights, Normalisation=normalisation)


def _Logistic(a: np.ndarray) -> np.ndarray:
    """
    Description:
        The logistic function 1 / (1 + exp(-a)), written with tanh so that no value overflows.

    Args:
        (1) a [Matrix<float>]: The arguments.

    Returns:
        (1) parameter [Matrix<float>]: The values.
    """

    return 0.5 * (1.0 + np.tanh(0.5 * a))


def _Rows(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Description:
        The product of each row of x with the transposed weight matrix, as one 1 x I by I x O product per row:
        every row is multiplied in the same way whatever the other rows are, so a row's result is the same bit
        for bit alone or among others (one product of all the rows is not: measured, F8 Part B).

    Args:
        (1) x [Matrix<float> QxI]: The rows.
        (2) weight [Matrix<float> OxI]: The layer's matrix.

    Returns:
        (1) parameter [Matrix<float> QxO]: The products.
    """

    return np.matmul(x[:, None, :], weight.T)[:, 0, :]


def Encode(predictor: Predictor_Str, time: np.ndarray, position: np.ndarray, rate: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Description:
        The encoding of one prefix: its inputs (Learning.Core.Prefix_Features; every sample at its own time
        stamp), the GRU over its samples in time order (PyTorch's gates: reset, update, new), and the layer
        joining the last state with the per-prefix inputs.

    Args:
        (1) predictor [Predictor_Str(object)]: The predictor.
        (2) time [Vector<float> 1xN]: The sample times [s].
        (3) position [Matrix<float> Nx3]: The observed positions [m].
        (4) rate [float]: The frame rate [Hz].

    Returns:
        (1) parameter 1 [Vector<float> 1xH]: The encoding.
        (2) parameter 2 [Vector<float> 1x3]: The last observed position [m].
    """

    w, H = predictor.Weights, predictor.Hidden
    sequence, context, last = Prefix_Features(time[None, :], position[None, :, :], np.array([len(time)]), np.array([rate]),
                                              Normalisation_Str=predictor.Normalisation)
    state = np.zeros(H)
    for x in sequence[0]:
        a = w['gru_input'] @ x + w['gru_input_bias']
        b = w['gru_state'] @ state + w['gru_state_bias']
        reset, update = _Logistic(a[:H] + b[:H]), _Logistic(a[H:2 * H] + b[H:2 * H])
        new = np.tanh(a[2 * H:] + reset * b[2 * H:])
        state = (1.0 - update) * new + update * state
    joined = w['context'] @ np.concatenate((state, context[0])) + w['context_bias']
    return joined * _Logistic(joined), last[0]


def _Layers(predictor: Predictor_Str, x: np.ndarray) -> np.ndarray:
    """
    Description:
        The layers after the encoder, row by row (_Rows), SiLU between them (Learning.Models.Network_Cls.Layers
        does the same on tensors).

    Args:
        (1) predictor [Predictor_Str(object)]: The predictor.
        (2) x [Matrix<float> QxI]: The inputs of the first layer.

    Returns:
        (1) parameter [Matrix<float> QxO]: The output.
    """

    for i in range(len(Layer_Sizes(predictor.Name, predictor.Hidden))):
        if i > 0:
            x = x * _Logistic(x)
        x = _Rows(x, predictor.Weights['layer_%d' % i]) + predictor.Weights['layer_%d_bias' % i]
    return x


def _Decoder(predictor: Predictor_Str, encoding: np.ndarray) -> tp.Callable[[np.ndarray], tp.Tuple[np.ndarray, np.ndarray]]:
    """
    Description:
        The decoder of the query-time model for one encoding: a function of the times after t_0 giving the
        displacement and the velocity, the series of the coefficients the head gives, fixed here.

    Args:
        (1) predictor [Predictor_Str(object)]: The predictor.
        (2) encoding [Vector<float> 1xH]: The encoding of the prefix.

    Returns:
        (1) parameter [function]: tau [Vector<float> 1xQ, s] -> (displacement [Qx3, m], velocity [Qx3, m/s]).
    """

    n = predictor.Normalisation
    coefficients = n.Output_Mean + n.Output_Scale * _Layers(predictor, encoding[None, :])[0].reshape(3, CONST_BASIS_DEGREE + 1)
    return lambda tau: Chebyshev_Series(coefficients, tau)


def _Floor_Time(decoder: tp.Callable[[np.ndarray], tp.Tuple[np.ndarray, np.ndarray]], height: float, t_0: float) -> float:
    """
    Description:
        The time after t_0 at which a query-time trajectory first descends through the floor height: a scan
        every CONST_FLOOR_SCAN up to CONST_HORIZON for the first interval that starts at or above the floor and
        ends below it, then Newton steps with the trajectory's own vertical velocity inside that interval (a
        step that leaves it halves it instead), until a step is below CONST_ROOT_TOLERANCE: the end is then
        within that tolerance of the root, the predicted height within about 1e-12 m of the floor.

    Args:
        (1) decoder [function]: The decoder.
        (2) height [float]: The height of the last observed position above the floor height [m].
        (3) t_0 [float]: The last observed time, named in a refusal [s].

    Returns:
        (1) parameter [float]: The time after t_0 [s].
    """

    scan = CONST_FLOOR_SCAN * np.arange(int(round(CONST_HORIZON / CONST_FLOOR_SCAN)) + 1)
    z = height + decoder(scan)[0][:, 2]
    if z[0] < 0.0:
        raise Prediction_Error_Cls('The predicted trajectory is below the floor height %.3f m at its last observed time %r s: the '
                                   'prefix gives a flight that has already landed. Check the samples.' % (CONST_FLOOR_HEIGHT, float(t_0)))
    down = np.where((z[:-1] >= 0.0) & (z[1:] < 0.0))[0]
    if not down.size:
        raise Prediction_Error_Cls('The predicted trajectory does not descend to the floor height %.3f m within %.1f s of its last '
                                   'observed time (CONST_HORIZON); the prefix gives no flight that lands. Check the samples.'
                                   % (CONST_FLOOR_HEIGHT, CONST_HORIZON))
    low, high = float(scan[down[0]]), float(scan[down[0] + 1])
    x = 0.5 * (low + high)
    for _ in range(CONST_ROOT_ITERATIONS):
        displacement, velocity = decoder(np.array([x]))
        value, slope = height + float(displacement[0, 2]), float(velocity[0, 2])
        if value >= 0.0:
            low = x
        else:
            high = x
        step = x - value / slope if slope != 0.0 else low
        if abs(step - x) < CONST_ROOT_TOLERANCE or high - low < CONST_ROOT_TOLERANCE:
            break
        x = step if low < step < high else 0.5 * (low + high)
    return x


def Predict_Learned(predictor: Predictor_Str, time: np.ndarray, position: np.ndarray, rate: float) -> Query_Trajectory_Str:
    """
    Description:
        The predicted trajectory of a prefix by the learned predictor (SPEC §7.8). Only the prefix is read. The
        encoding fixes the decoder, and the end is the root of the predicted height at the floor.

    Args:
        (1) predictor [Predictor_Str(object)]: The trained predictor.
        (2) time [Vector<float> 1xN]: The sample times [s], strictly increasing, as observed (never resampled).
        (3) position [Matrix<float> Nx3]: The positions [m].
        (4) rate [float]: The rate of the samples' frames [Hz].

    Returns:
        (1) parameter [Query_Trajectory_Str(object)]: The trajectory from the last sample to the floor.
    """

    time = np.asarray(time, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    Check_Prefix(time, position, float(rate))
    encoding, last = Encode(predictor, time, position, float(rate))
    t_0 = float(time[-1])
    decoder = _Decoder(predictor, encoding)
    end = _Floor_Time(decoder, float(last[2]) - CONST_FLOOR_HEIGHT, t_0)
    return Query_Trajectory_Str(T_0=t_0, T_End=t_0 + end, Grid_Start=float(time[0]), Origin=last, Decoder=decoder)


def Run_Predictor(name: str, run: str, seed: tp.Optional[int], *, Flight_Model_Str: Flight_Model_Str,
                  Observation_Model_Str: Observation_Model_Str) -> tp.Callable[[np.ndarray, np.ndarray, float], tp.Any]:
    """
    Description:
        A run's predictor as a function of a prefix (times [s], positions [m], rate [Hz]) giving its trajectory: the
        learned predictor (CONST_LEARNED_MODEL_NAMES) from its seed's folder of the run (Predict_Learned), the numerical
        one (Prediction.Core.Predict) under the flight model and M4, without a seed.

    Args:
        (1) name [string]: The run's Model.Name.
        (2) run [string]: The run folder (a learned predictor's seeds).
        (3) seed [int or None]: The training seed of a learned predictor; None for a numerical one.
        (4) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the one flight model (the numerical
                                                         predictor's).
        (5) Observation_Model_Str [Observation_Model_Str(object)]: REQUIRED, keyword-only: M4 (the numerical predictor's).

    Returns:
        (1) parameter [Callable]: The function of a prefix.
    """

    if name in CONST_LEARNED_MODEL_NAMES:
        predictor = Load_Predictor(os.path.join(run, 'seed_%d' % seed))
        return lambda time, position, rate: Predict_Learned(predictor, time, position, rate)
    return lambda time, position, rate: Predict(name, time, position, rate, Flight_Model_Str=Flight_Model_Str,
                                                Observation_Model_Str=Observation_Model_Str)
