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
# File Name: ../test/Learning/test_training.py
## =========================================================================== ##

# Functools (a dataset built once per process)
import functools
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the training under test
#   ../Learning/Training
from Learning.Training import (Prepare_Flights, Merge_Flights, Fixed_Prefixes, Drawn_Prefixes, Fit_Normalisation, Validation_Loss, Train,
                               Write_Epochs, Training_Error_Cls)
#   ../Learning/Models, ../Learning/Core - the network restored from the returned weights, the basis
from Learning.Models import Network_Cls
from Learning.Core import Chebyshev_Series, CONST_BASIS_DEGREE
#   ../Configuration/Core, ../Dataset/Core, ../Flight/Core - the training configuration, a dataset, the one flight model
from Configuration.Core import Training_Configuration_Str
from Dataset.Core import Dataset_Str, Floor_Time
from Flight.Core import Flight_Model_Str, Simulate, CONST_SIMULATION_STEP

"""
Description:
    Initialization of constants.

    The training of the learned predictor (SPEC §2.13.7, §7.8; F8) on a dataset of 24 simulated throws built
    here, in a few epochs of a network with a state of 8: expectations come from Simulate, from the
    definitions written out and from the validation loss recomputed on the returned weights.
"""
# The flight model: g = 9.81 m/s^2 along -z and F3's k (SPEC §4.2).
#   Unit [Flight_Model_Str]
CONST_MODEL = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.00930496257)
# The values of N of these tests, and the samples kept per flight.
#   Unit [List<int>], [int]
CONST_N = [4, 8, 12]
CONST_SAMPLES = 12
# The learned predictor ('QT_MLP' and 'HYBRID' removed).
#   Unit [Tuple<string>]
CONST_PREDICTORS = ('QT_BASIS',)


@functools.lru_cache(maxsize=None)
def _Dataset(flights=24, seed=5, short=()):
    """
    Description:
        A dataset of simulated throws: releases around the throw area, rates 100 / 200 / 300 Hz, observations at
        frame times with about a third of the frames dropped and 2 mm of noise; the flights in `short` keep 3
        samples only.
    """

    rng = np.random.default_rng(seed)
    release = np.array([-0.15, -4.78, 1.17, 0.1, 4.3, 5.2]) + rng.normal(0.0, [0.5, 0.3, 0.1, 0.4, 0.3, 0.3], (flights, 6))
    rate = rng.choice([100.0, 200.0, 300.0], flights)
    floor = Floor_Time(release, np.full(flights, 1.0), Flight_Model_Str=CONST_MODEL)
    time, position, offset = [], [], [0]
    for i in range(flights):
        frames = np.arange(int(np.floor(floor[i] * rate[i])) + 1)
        keep = frames[(rng.random(len(frames)) > 0.33) | (frames == 0)]
        if i in short:
            keep = keep[:3]
        t = keep / rate[i]
        time.append(t)
        position.append(Simulate(release[i], t, Flight_Model_Str=CONST_MODEL)[:, :3] + rng.normal(0.0, 0.002, (len(t), 3)))
        offset.append(offset[-1] + len(t))
    return Dataset_Str(Partition=np.zeros(flights, dtype=np.int8), Rate=rate, Release_State=release, Floor_Time=floor,
                       Observed_Offset=np.array(offset, dtype=np.int64), Observed_Time=np.concatenate(time),
                       Observed_Position=np.concatenate(position))


def _Flights(first=0, last=24, **kwargs):
    return Prepare_Flights(_Dataset(**kwargs), np.arange(first, last), CONST_SAMPLES, Flight_Model_Str=CONST_MODEL)


def _Configuration(name, **kwargs):
    values = dict(Model_Name=name, Observed_Samples=CONST_N, Hidden=8, Velocity_Weight=0.0, Learning_Rate=0.003, Batch_Size=16, Max_Epochs=4,
                  Weight_Decay=0.0, Patience=10, Min_Delta=0.0, Device='cpu')
    values.update(kwargs)
    return Training_Configuration_Str(**values)


def test_the_flights_are_prepared_as_observed_with_their_exact_truth():
    """
    Description:
        SPEC §2.13.4, §7.8: a flight's samples are taken as observed (its first 12 times and positions, bit for
        bit, with their count), never resampled; its truth is Simulate of its release state at every grid time
        (within 1e-6 m in single precision) to at least 0.1 s past its floor; the floor index is the last grid
        time at or before the floor time.
    """

    data = _Dataset()
    flights = Prepare_Flights(data, np.array([3, 7, 20]), CONST_SAMPLES, Flight_Model_Str=CONST_MODEL)
    time, position = data.Observations(7)
    assert flights.Samples[1] == min(len(time), CONST_SAMPLES) == 12 and np.array_equal(flights.Time[1], time[:12])
    assert np.array_equal(flights.Position[1], position[:12]) and np.unique(np.round(np.diff(time[:12]) * data.Rate[7])).size > 1
    grid = np.arange(flights.Truth.shape[1]) * CONST_SIMULATION_STEP
    assert np.max(np.abs(flights.Truth[2] - Simulate(data.Release_State[20], grid, Flight_Model_Str=CONST_MODEL))) < 1e-5
    assert grid[-1] >= data.Floor_Time[[3, 7, 20]].max() + 0.1
    assert np.all(flights.Floor_Index * CONST_SIMULATION_STEP <= data.Floor_Time[[3, 7, 20]] + 1e-12)
    assert np.all((flights.Floor_Index + 1) * CONST_SIMULATION_STEP > data.Floor_Time[[3, 7, 20]])


def test_the_normalisation_is_that_of_the_training_flights_given():
    """
    Description:
        SPEC §7.8: the normalisation comes from the flights given and nothing else. Over the prefixes of N = 4, 8
        and 12 the deviation of the samples relative to the last one equals the value computed here sample by
        sample; the outputs' scale has the shape of the basis coefficients; and the
        normalisation of 16 flights does not change when 8 other flights are prepared beside them.
    """

    flights = _Flights(0, 16)
    relative = []
    for row in range(16):
        for n in CONST_N:
            if flights.Samples[row] >= n:
                relative.extend(flights.Position[row, :n] - flights.Position[row, n - 1])
    normalisation = Fit_Normalisation(flights, CONST_N, Flight_Model_Str=CONST_MODEL)
    assert np.allclose(normalisation.Relative_Std, np.std(relative, axis=0), rtol=1e-12)
    assert normalisation.Output_Scale.shape == (3, CONST_BASIS_DEGREE + 1)
    both = Merge_Flights([flights, _Flights(16, 24)])
    again = Fit_Normalisation(Prepare_Flights(_Dataset(), np.arange(16), CONST_SAMPLES, Flight_Model_Str=CONST_MODEL), CONST_N,
                              Flight_Model_Str=CONST_MODEL)
    assert np.array_equal(again.Relative_Std, normalisation.Relative_Std) and len(both.Rate) == 24
    assert not np.array_equal(Fit_Normalisation(both, CONST_N, Flight_Model_Str=CONST_MODEL).Relative_Std, normalisation.Relative_Std)


def test_an_epochs_prefixes_take_every_n_from_the_smallest_to_the_largest():
    """
    Description:
        One model is trained on every N from the smallest to the largest configured. An
        epoch holds each flight with at least 4 samples three times (the number of configured values), its N
        between 4 and 12; over 30 epochs every N from 4 to 12 occurs; a flight of 3 samples never does; the same
        generator state gives the same prefixes.
    """

    flights = _Flights(short=(2, 9))
    seen = set()
    for epoch in range(30):
        rows, n = Drawn_Prefixes(flights, CONST_N, np.random.default_rng([1, epoch]))
        assert len(rows) == 3 * 22 and 2 not in rows and 9 not in rows and n.min() >= 4 and n.max() <= 12
        assert np.all(np.bincount(rows, minlength=24)[[0, 1, 3, 23]] == 3)
        seen.update(n.tolist())
    assert seen == set(range(4, 13))
    again = Drawn_Prefixes(flights, CONST_N, np.random.default_rng([1, 29]))
    assert np.array_equal(again[0], rows) and np.array_equal(again[1], n)
    fixed_rows, fixed_n = Fixed_Prefixes(flights, CONST_N)
    assert len(fixed_rows) == 3 * 22 and sorted(set(fixed_n.tolist())) == CONST_N


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_training_lowers_the_loss_and_returns_the_weights_of_the_best_epoch(name):
    """
    Description:
        SPEC §2.13.7: a training of 4 epochs on 16 flights, early-stopped on 8 others, lowers the train loss, and
        the weights returned are those of the epoch with the lowest validation loss: restored into a network,
        their validation loss is that epoch's, as recorded.
    """

    train, validation = _Flights(0, 16), _Flights(16, 24)
    configuration = _Configuration(name)
    normalisation = Fit_Normalisation(train, CONST_N, Flight_Model_Str=CONST_MODEL)
    weights, epochs = Train(configuration, train, validation, normalisation, 0)
    assert [e[0] for e in epochs] == [1, 2, 3, 4] and epochs[-1][1] < epochs[0][1]
    network = Network_Cls(name, 8)
    network.Load(weights)
    value, position = Validation_Loss(network, validation, CONST_N, normalisation, 0.0)
    assert value == pytest.approx(min(e[2] for e in epochs), rel=1e-6) and position == value


def test_the_training_stops_after_the_patience_without_an_improvement():
    """
    Description:
        Early stopping (SPEC §8.3): with Min_Delta 1 m no epoch after the first improves by that much, so a
        training of at most 6 epochs with Patience 2 ends after 3; with Patience 6 it runs all 6. The bound
        is tested on both sides.
    """

    train, validation = _Flights(0, 16), _Flights(16, 24)
    normalisation = Fit_Normalisation(train, CONST_N, Flight_Model_Str=CONST_MODEL)
    stopped = Train(_Configuration('QT_BASIS', Max_Epochs=6, Patience=2, Min_Delta=1.0), train, validation, normalisation, 0)[1]
    full = Train(_Configuration('QT_BASIS', Max_Epochs=6, Patience=6, Min_Delta=1.0), train, validation, normalisation, 0)[1]
    assert len(stopped) == 3 and len(full) == 6


def test_one_seed_gives_the_same_weights_bit_for_bit_and_another_seed_others():
    """
    Description:
        SPEC §7.8 (the CPU as reference device): two trainings of the same seed give the same weights bit for
        bit and the same epochs; another seed gives other weights. The seed never changes the data.
    """

    train, validation = _Flights(0, 16), _Flights(16, 24)
    normalisation = Fit_Normalisation(train, CONST_N, Flight_Model_Str=CONST_MODEL)
    runs = [Train(_Configuration('QT_BASIS', Max_Epochs=2), train, validation, normalisation, seed) for seed in (3, 3, 4)]
    assert all(np.array_equal(runs[0][0][key], runs[1][0][key]) for key in runs[0][0]) and runs[0][1] == runs[1][1]
    assert not np.array_equal(runs[0][0]['layer_0'], runs[2][0]['layer_0'])


def test_the_velocity_weight_adds_the_velocity_term_to_the_loss():
    """
    Description:
        The velocity loss: the loss is the position term plus Loss.Velocity_Weight times the
        velocity term. For one network the validation loss at weights 0, 0.5 and 1 s is p, p + 0.5 v and p + v
        with the same position term p and a velocity term v > 0; and a training with the weight 0.5 s gives
        other weights than one without.
    """

    train, validation = _Flights(0, 16), _Flights(16, 24)
    normalisation = Fit_Normalisation(train, CONST_N, Flight_Model_Str=CONST_MODEL)
    weights, _ = Train(_Configuration('QT_BASIS', Max_Epochs=2), train, validation, normalisation, 0)
    network = Network_Cls('QT_BASIS', 8)
    network.Load(weights)
    values = [Validation_Loss(network, validation, CONST_N, normalisation, w) for w in (0.0, 0.5, 1.0)]
    p, v = values[0][0], values[2][0] - values[0][0]
    assert v > 1e-3 and values[1][0] == pytest.approx(p + 0.5 * v, rel=1e-6) and all(value[1] == pytest.approx(p, rel=1e-9) for value in values)
    other, _ = Train(_Configuration('QT_BASIS', Max_Epochs=2, Velocity_Weight=0.5), train, validation, normalisation, 0)
    assert not np.array_equal(other['layer_0'], weights['layer_0'])


def test_the_basis_models_loss_is_the_rms_of_its_series_against_the_truth():
    """
    Description:
        A query-time model's validation loss, on the basis decoder with a zeroed head: its coefficients are then
        the normalisation's mean, so the predicted displacement is that Chebyshev series for every prefix, and the
        loss equals the value computed here on 8 flights against Simulate at the same times (within 1e-4 relative).
    """

    flights = _Flights(0, 8)
    normalisation = Fit_Normalisation(flights, CONST_N, Flight_Model_Str=CONST_MODEL)
    network = Network_Cls('QT_BASIS', 8)
    weights = network.Weights()
    weights['layer_1'][:], weights['layer_1_bias'][:] = 0.0, 0.0
    network.Load(weights)
    value, _ = Validation_Loss(network, flights, CONST_N, normalisation, 0.0)
    data, expected = _Dataset(), []
    for row in range(8):
        for n in CONST_N:
            if flights.Samples[row] < n:
                continue
            t_0 = flights.Time[row, n - 1]
            first, last = int(np.ceil(t_0 / 0.01 - 1e-9)), int(np.floor(data.Floor_Time[row] / 0.01 + 1e-9))
            t = 0.01 * np.arange(first, max(last, first) + 1)
            predicted = flights.Position[row, n - 1] + Chebyshev_Series(normalisation.Output_Mean, t - t_0)[0]
            truth = Simulate(data.Release_State[row], t, Flight_Model_Str=CONST_MODEL)[:, :3]
            expected.append(np.sqrt(np.mean(np.sum((predicted - truth) ** 2, axis=1))))
    assert value == pytest.approx(np.mean(expected), rel=1e-4)


def test_flights_without_enough_samples_are_refused():
    """
    Description:
        A training whose flights all have fewer samples than the smallest N is refused naming N, for the
        normalisation, the training and the validation loss alike; nothing is trained on padding.
    """

    short = _Flights(0, 6, short=tuple(range(24)))
    good = _Flights(0, 16)
    normalisation = Fit_Normalisation(good, CONST_N, Flight_Model_Str=CONST_MODEL)
    with pytest.raises(Training_Error_Cls, match='No training flight has 4 samples'):
        Fit_Normalisation(short, CONST_N, Flight_Model_Str=CONST_MODEL)
    with pytest.raises(Training_Error_Cls, match='No training flight has 4 samples'):
        Train(_Configuration('QT_BASIS'), short, good, normalisation, 0)
    with pytest.raises(Training_Error_Cls, match='No validation flight has 4 samples'):
        Validation_Loss(Network_Cls('QT_BASIS', 8), short, CONST_N, normalisation, 0.0)


def test_the_epochs_file_holds_one_row_per_epoch(tmp_path):
    """
    Description:
        epochs.csv (SPEC §9.6): a header and one row per epoch, the losses in metres with 9 decimals, LF.
    """

    Write_Epochs(str(tmp_path / 'epochs.csv'), [(1, 0.5, 0.25, 0.125), (2, 0.0421234567891, 0.04, 0.03)])
    assert (tmp_path / 'epochs.csv').read_bytes() == (b'epoch,train_loss_m,validation_loss_m,validation_position_m\n'
                                                      b'1,0.500000000,0.250000000,0.125000000\n2,0.042123457,0.040000000,0.030000000\n')
