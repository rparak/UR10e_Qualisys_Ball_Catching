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
# File Name: ../test/Learning/test_predictor.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# PyTorch (Tensors and automatic differentiation) [pip3 install torch]
import torch
# Custom Lib.: the trained predictor at work, under test
#   ../Learning/Predictor
from Learning.Predictor import Predictor_Str, Predict_Learned, Encode, Write_Weights, Read_Weights, Load_Predictor
#   ../Learning/Core, ../Learning/Models - the normalisation, the inputs and the network the predictor must equal
from Learning.Core import Normalisation_Str, Prefix_Features, Write_Normalisation, CONST_BASIS_DEGREE
from Learning.Models import Network_Cls, Query
#   ../Prediction/Core, ../Flight/Core - the predictors' error, the one flight model (the samples of the prefix)
from Prediction.Core import Prediction_Error_Cls
from Flight.Core import Flight_Model_Str, Simulate

"""
Description:
    Initialization of constants.

    The trajectory contract of SPEC §2.13.13 for the learned predictor (the query-time model with the basis
    decoder), on untrained networks built here: the contract holds for any weights.
    Expectations come from the contract, from a parabola written out, from finite differences and from the
    tensors' network, never from the predictor under test.
"""
# The learned predictor ('QT_MLP' and 'HYBRID' removed).
#   Unit [Tuple<string>]
CONST_PREDICTORS = ('QT_BASIS',)
# The flight model: g = 9.81 m/s^2 along -z and F3's k (SPEC §4.2).
#   Unit [Flight_Model_Str]
CONST_MODEL = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.00930496257)
# A throw from the throw area towards the robot (robot frame) and an irregular 200 Hz prefix of 12 samples
# with dropouts of 1, 2, 5 and 9 frames.
#   Unit [Vector<float> 1x6, m, m/s], [Vector<float> 1x12, s]
CONST_RELEASE = np.array([-0.15, -4.78, 1.17, 0.1, 4.3, 5.2])
CONST_PREFIX_TIME = np.array([0, 1, 2, 4, 5, 6, 9, 10, 16, 17, 27, 28]) / 200.0
# The floor height (SPEC §14.6), and the velocity of the parabola the basis predictor is built to follow.
#   Unit [float, m], [Vector<float> 1x3, m/s]
CONST_FLOOR = 0.030
CONST_PARABOLA_VELOCITY = np.array([0.2, 4.0, 3.0])
# The size of the encoder's state in these tests.
#   Unit [int]
CONST_HIDDEN = 16


def _Prefix():
    return CONST_PREFIX_TIME, Simulate(CONST_RELEASE, CONST_PREFIX_TIME, Flight_Model_Str=CONST_MODEL)[:, :3]


def _Predictor(name, seed=7):
    """
    Description:
        An untrained predictor that lands: random weights, the head's last layer zeroed and its bias set to the
        Chebyshev coefficients of the parabola v tau + g tau^2 / 2 over the 2 s span (tau = s + 1).
    """

    torch.manual_seed(seed)
    weights = {key: value.astype(np.float64) for key, value in Network_Cls(name, CONST_HIDDEN).Weights().items()}
    normalisation = Normalisation_Str(Relative_Std=np.array([0.3, 0.6, 0.5]), Velocity_Mean=np.array([0.1, 4.3, 4.5]),
                                      Velocity_Std=np.array([1.0, 0.5, 1.0]), Output_Mean=np.zeros(3), Output_Scale=np.array([0.5, 1.0, 1.0]))
    coefficients = np.zeros((3, CONST_BASIS_DEGREE + 1))
    for a in range(3):
        g = CONST_MODEL.Gravity[a]
        series = np.polynomial.chebyshev.poly2cheb([CONST_PARABOLA_VELOCITY[a] + 0.5 * g, CONST_PARABOLA_VELOCITY[a] + g, 0.5 * g])
        coefficients[a, :len(series)] = series
    weights['layer_1'][:] = 0.0
    weights['layer_1_bias'] = coefficients.ravel()
    normalisation.Output_Mean, normalisation.Output_Scale = np.zeros((3, CONST_BASIS_DEGREE + 1)), np.ones((3, CONST_BASIS_DEGREE + 1))
    return Predictor_Str(Name=name, Hidden=CONST_HIDDEN, Weights=weights, Normalisation=normalisation)


def _Predict(name, time=None, position=None):
    default_time, default_position = _Prefix()
    return Predict_Learned(_Predictor(name), default_time if time is None else time, default_position if position is None else position, 200.0)


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_a_query_at_times_off_every_grid_is_answered(name):
    """
    Description:
        SPEC §2.13.13: a learned trajectory starts at the last observed time and is answered at times off every
        sample and frame grid, position [m] and velocity [m/s], up to its own end.
    """

    trajectory = _Predict(name)
    assert trajectory.T_0 == CONST_PREFIX_TIME[-1] and trajectory.T_End > trajectory.T_0 + 0.1
    t = trajectory.T_0 + (trajectory.T_End - trajectory.T_0) * np.array([0.0, 0.0137, 0.2718, 0.5, 0.7071, 0.999, 1.0])
    state = trajectory.State(t)
    assert state.shape == (7, 6) and np.all(np.isfinite(state)) and np.all(np.abs(state[:, 3:]).max(axis=1) > 0.1)


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_a_time_outside_the_range_is_refused_naming_it_and_the_range(name):
    """
    Description:
        SPEC §2.13.13: a time before t_0 or after the predicted end is refused, the message naming the time and
        the range; the two ends themselves are answered.
    """

    trajectory = _Predict(name)
    for bad in (trajectory.T_0 - 1e-6, trajectory.T_End + 1e-6):
        with pytest.raises(Prediction_Error_Cls) as error:
            trajectory.State([trajectory.T_0, bad])
        assert repr(bad) in str(error.value) and repr(trajectory.T_0) in str(error.value) and repr(trajectory.T_End) in str(error.value)
    assert trajectory.State([trajectory.T_0, trajectory.T_End]).shape == (2, 6)


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_queries_are_independent_of_the_other_times_asked(name):
    """
    Description:
        SPEC §2.13.13 (query independence; by construction): querying all times, a random
        subset, the subset reversed and one time alone gives the same values bit for bit, in 40 draws of 1 to
        300 times, so the query times cannot leak a flight's dropouts or its end.
    """

    trajectory = _Predict(name)
    rng = np.random.default_rng(11)
    for _ in range(40):
        t = rng.uniform(trajectory.T_0, trajectory.T_End, int(rng.integers(1, 300)))
        state = trajectory.State(t)
        subset = rng.permutation(len(t))[:max(1, len(t) // 3)]
        assert np.array_equal(trajectory.State(t[subset]), state[subset])
        assert np.array_equal(trajectory.State(t[subset][::-1]), state[subset][::-1])
        assert np.array_equal(trajectory.State(t[subset[:1]]), state[subset[:1]])


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_every_trajectory_ends_at_the_floor_height(name):
    """
    Description:
        SPEC §2.13.13, §14.6: a learned trajectory's own end is where its height reaches 0.030 m, within 1e-9 m
        (the root of the network's height), and 0.1 ms earlier it is above the floor.
    """

    trajectory = _Predict(name)
    assert trajectory.State([trajectory.T_End])[0, 2] == pytest.approx(CONST_FLOOR, abs=1e-9)
    assert trajectory.State([trajectory.T_End - 1e-4])[0, 2] > CONST_FLOOR + 1e-5


@pytest.mark.parametrize('name', ['QT_BASIS'])
def test_the_velocity_is_the_time_derivative_of_the_predicted_position(name):
    """
    Description:
        For the query-time model the velocity is the exact time derivative of the
        predicted position, never a separate output. At 60 times the velocity of State equals the central finite
        difference of State's own positions (step 1e-6 s) within 1e-6 m/s.
    """

    trajectory = _Predict(name)
    h = 1e-6
    t = np.random.default_rng(12).uniform(trajectory.T_0 + h, trajectory.T_End - h, 60)
    difference = (trajectory.State(t + h)[:, :3] - trajectory.State(t - h)[:, :3]) / (2.0 * h)
    velocity = trajectory.State(t)[:, 3:]
    assert np.max(np.abs(velocity - difference)) < 1e-6 and np.max(np.abs(velocity[:, 2])) > 1.0


def test_the_basis_decoder_follows_its_series_to_the_floor():
    """
    Description:
        The basis decoder is a polynomial evaluation: with the coefficients of the parabola
        v tau + g tau^2 / 2 the trajectory is that parabola from the last observed position within 1e-12 m, its
        velocity v + g tau within 1e-12 m/s, and its end the parabola's time at 0.030 m within 1e-10 s.
    """

    time, position = _Prefix()
    trajectory = _Predict('QT_BASIS')
    v, g, z = CONST_PARABOLA_VELOCITY, CONST_MODEL.Gravity, position[-1, 2] - CONST_FLOOR
    assert trajectory.T_End - trajectory.T_0 == pytest.approx((v[2] + np.sqrt(v[2] ** 2 + 2.0 * 9.81 * z)) / 9.81, abs=1e-10)
    tau = np.linspace(0.0, trajectory.T_End - trajectory.T_0, 23)
    state = trajectory.State(trajectory.T_0 + tau)
    assert np.max(np.abs(state[:, :3] - (position[-1] + np.outer(tau, v) + 0.5 * np.outer(tau ** 2, g)))) < 1e-12
    assert np.max(np.abs(state[:, 3:] - (v + np.outer(tau, g)))) < 1e-12


@pytest.mark.parametrize('name', ['QT_BASIS'])
def test_a_trajectory_below_the_floor_or_that_never_lands_is_refused(name):
    """
    Description:
        Like the numerical predictor, the query-time model refuses a prefix whose predicted height
        at t_0 is below the floor, and one whose trajectory does not descend to the floor within the 2 s it is
        defined for (the last sample 30 m up), each naming the floor height.
    """

    time, position = _Prefix()
    with pytest.raises(Prediction_Error_Cls, match='below the floor height 0.030 m'):
        _Predict(name, position=position - np.array([0.0, 0.0, 3.0]))
    with pytest.raises(Prediction_Error_Cls, match='does not descend to the floor height 0.030 m within 2.0 s'):
        _Predict(name, position=position + np.array([0.0, 0.0, 30.0]))


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_the_samples_are_used_at_their_own_time_stamps(name):
    """
    Description:
        SPEC §2.7 (irregular input): the prefix shifted by 0.123 s gives the same trajectory shifted (within
        1e-9 m: only time differences enter), and one sample's stamp moved by half a frame changes the encoding of
        the prefix, so the time stamps are read and not replaced by a fixed step. (This predictor's head is
        zeroed to follow its parabola, so the change is checked on the encoding the head receives; the removed
        MLP decoder's prediction was checked itself.)
    """

    time, position = _Prefix()
    trajectory, shifted = _Predict(name), _Predict(name, time=time + 0.123)
    h = trajectory.T_0 + np.array([0.0, 0.05, 0.2])
    assert np.max(np.abs(shifted.State(h + 0.123) - trajectory.State(h))) < 1e-9
    moved = time.copy()
    moved[5] += 0.5 / 200.0
    predictor = _Predictor(name)
    assert np.max(np.abs(Encode(predictor, moved, position, 200.0)[0] - Encode(predictor, time, position, 200.0)[0])) > 1e-6


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_the_predictor_computes_what_the_tensors_network_computes(name):
    """
    Description:
        The inference in NumPy and the training in PyTorch are one model: for random weights (in double
        precision) the encoding of a prefix equals Network_Cls.Encode within 1e-12, and the displacement and the
        velocity of the query-time model equal Query's within 1e-10 at 9 times.
    """

    torch.manual_seed(21)
    network = Network_Cls(name, CONST_HIDDEN).double()
    time, position = _Prefix()
    shape = (3, CONST_BASIS_DEGREE + 1)
    rng = np.random.default_rng(22)
    normalisation = Normalisation_Str(Relative_Std=np.array([0.3, 0.6, 0.5]), Velocity_Mean=np.array([0.1, 4.3, 4.5]),
                                      Velocity_Std=np.array([1.0, 0.5, 1.0]), Output_Mean=rng.normal(size=shape),
                                      Output_Scale=rng.uniform(0.5, 2.0, shape))
    predictor = Predictor_Str(Name=name, Hidden=CONST_HIDDEN, Weights={k: v.astype(np.float64) for k, v in network.Weights().items()},
                              Normalisation=normalisation)
    sequence, context, last = Prefix_Features(time[None, :], position[None, :, :], np.array([len(time)]), np.array([200.0]),
                                              Normalisation_Str=normalisation)
    with torch.no_grad():
        encoding = network.Encode(torch.tensor(sequence), torch.tensor(context), torch.tensor([len(time)]))
        ours, origin = Encode(predictor, time, position, 200.0)
        assert np.max(np.abs(ours - encoding.numpy()[0])) < 1e-12 and np.array_equal(origin, position[-1])
        mean, scale = torch.tensor(normalisation.Output_Mean), torch.tensor(normalisation.Output_Scale)
        from Learning.Predictor import _Decoder
        tau = np.linspace(0.0, 1.9, 9)
        expected_position, expected_velocity = Query(network, encoding, torch.tensor(tau)[None, :], mean, scale)
        displacement, velocity = _Decoder(predictor, ours)(tau)
        assert np.max(np.abs(displacement - expected_position.numpy()[0])) < 1e-10
        assert np.max(np.abs(velocity - expected_velocity.numpy()[0])) < 1e-10


def test_the_weights_file_round_trips_and_is_deterministic(tmp_path):
    """
    Description:
        weights.npz (SPEC §9.6): written twice the bytes are the same (torch.save's were
        not, F8 Part A); read back, the model, the state's size and every array (float32 values) are the written
        ones; Load_Predictor reads it with its normalisation.json and predicts like the predictor it was written from.
    """

    predictor = _Predictor('QT_BASIS')
    for name in ('a', 'b'):
        (tmp_path / name).mkdir()
        Write_Weights(str(tmp_path / name / 'weights.npz'), 'QT_BASIS', CONST_HIDDEN, predictor.Weights)
    assert (tmp_path / 'a' / 'weights.npz').read_bytes() == (tmp_path / 'b' / 'weights.npz').read_bytes()
    name, hidden, weights = Read_Weights(str(tmp_path / 'a' / 'weights.npz'))
    assert (name, hidden, sorted(weights)) == ('QT_BASIS', CONST_HIDDEN, sorted(predictor.Weights))
    assert all(np.array_equal(weights[key], predictor.Weights[key].astype(np.float32).astype(np.float64)) for key in weights)
    Write_Normalisation(str(tmp_path / 'a' / 'normalisation.json'), predictor.Normalisation)
    loaded = Load_Predictor(str(tmp_path / 'a'))
    time, position = _Prefix()
    first = Predict_Learned(loaded, time, position, 200.0)
    second = Predict_Learned(Predictor_Str(Name='QT_BASIS', Hidden=CONST_HIDDEN, Weights=weights, Normalisation=predictor.Normalisation),
                             time, position, 200.0)
    assert first.T_End == second.T_End and np.array_equal(first.State([first.T_0 + 0.1]), second.State([first.T_0 + 0.1]))


@pytest.mark.parametrize('change, reason', [('version', 'format version 1'), ('model', 'names the model'), ('missing', 'layer_1 is missing'),
                                            ('shape', 'gru_state is missing, is not of shape 48x16'), ('nan', 'context_bias'),
                                            ('removed', "names the model 'HYBRID' with a state of 16; a learned predictor is one of QT_BASIS ")])
def test_a_weights_file_of_another_version_model_or_shape_is_refused(tmp_path, change, reason):
    """
    Description:
        The reader refuses another format version, an unknown model, a missing array, an array of another shape
        and a value that is not finite, naming the file and the array; a weights file of a removed predictor (the
        hybrid) is refused like an unknown one, naming the predictor that remains; a seed's
        folder without its files is refused by Load_Predictor naming both.
    """

    path = str(tmp_path / 'weights.npz')
    Write_Weights(path, 'QT_BASIS', CONST_HIDDEN, _Predictor('QT_BASIS').Weights)
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    if change == 'version':
        arrays['format_version'] = np.int64(2)
    elif change == 'model':
        arrays['model'] = np.array('FFNN')
    elif change == 'removed':
        arrays['model'] = np.array('HYBRID')
    elif change == 'missing':
        del arrays['layer_1']
    elif change == 'shape':
        arrays['gru_state'] = arrays['gru_state'][:, :-1]
    else:
        arrays['context_bias'][3] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(Prediction_Error_Cls, match=reason):
        Read_Weights(path)
    with pytest.raises(Prediction_Error_Cls, match='weights.npz and normalisation.json'):
        Load_Predictor(str(tmp_path / 'absent'))


def test_the_floor_time_is_found_in_few_decoder_evaluations():
    """
    Description:
        The end of a query-time trajectory is found by a scan and Newton steps (inference
        speed matters). For 40 falling parabolas (heights 0.2-3 m, vertical velocities -6 to +5 m/s) the floor
        time is the parabola's root within 1e-12 s, and the decoder is evaluated at most 12 times per
        trajectory, the scan being one of them: Newton approaches the root from one side, so the search stops on
        the size of its step, not only on its interval (without that, a third of them took about 200).
    """

    from Learning.Predictor import _Floor_Time
    rng = np.random.default_rng(31)
    worst = 0
    for _ in range(40):
        height, v = rng.uniform(0.2, 3.0), rng.uniform(-6.0, 5.0)
        calls = []

        def Decoder(tau):
            calls.append(len(tau))
            z = v * tau - 0.5 * 9.81 * tau * tau
            return np.stack((0.0 * tau, 0.0 * tau, z), axis=1), np.stack((0.0 * tau, 0.0 * tau, v - 9.81 * tau), axis=1)

        end = _Floor_Time(Decoder, height, 0.0)
        assert end == pytest.approx((v + np.sqrt(v * v + 2.0 * 9.81 * height)) / 9.81, abs=1e-12)
        worst = max(worst, len(calls))
    assert worst <= 12, worst
