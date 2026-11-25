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
# File Name: ../test/Catch/test_robustness.py
## =========================================================================== ##


# Types (A stand-in for a dataset)
import types
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Flight/Core - the flight model and the true trajectory
from Flight.Core import Flight_Model_Str, Simulate
#   ../Prediction/Core - a query outside a trajectory's range
from Prediction.Core import Prediction_Error_Cls
#   ../Catch/Core - the catch's error
from Catch.Core import Catch_Error_Cls
#   ../Catch/Robustness
from Catch.Robustness import Error_Table, Injected_Predictor, Range_Labels

"""
Description:
    Initialization of constants.

    The robustness sweep's prediction (SPEC §7.11, §9.14; F11): the error table from
    constructed rows of a stored evaluation, and the injected trajectory against the flight model.
"""
# The horizon ranges of a stored evaluation (SPEC §9.7) and their labels in per_horizon.csv.
#   Unit [Tuple<Tuple<float, float>>, s], [List<string>]
CONST_RANGES = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0))
CONST_LABELS = ['[0, 0.1)', '[0.1, 0.2)', '[0.2, 0.3)', '[0.3, 0.5)', '[0.5, 0.8)', '[0.8, 1.2)', '[1.2, 2]']
# A flight model (F3's drag) and two throws: their release states and floor times.
#   Unit [Flight_Model_Str], [Matrix<float> 2x6], [Vector<float> 1x2, s]
CONST_FLIGHT = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.00930496257, Drag_Std=0.0)
CONST_RELEASE = np.array([[0.0, -5.0, 1.2, 0.4, 5.5, 5.0], [0.2, -5.2, 1.3, 0.1, 5.8, 4.6]])
CONST_FLOOR = np.array([1.2, 1.15])


def _Rows(flights=12, last=12, seeds=('0', '1')):
    """
    Description:
        Rows of a per_horizon.csv: per flight f, N = 20 and 40, range r and seed, an error of (10 + f + 10 r) mm at
        N 20 and half of it at N 40, the second seed's 2 mm above the first's (their mean 1 mm above); the last range
        only for the first `last` flights; one row without an error, one at N 10 and one fixed horizon, all to be ignored.
    """

    rows = [dict(flight='0', n='20', horizon='[0, 0.1)', position_error_mm=''), dict(flight='0', n='10', horizon='[0, 0.1)', position_error_mm='500'),
            dict(flight='0', n='20', horizon='0.1', position_error_mm='700')]
    for flight in range(flights):
        for n, factor in ((20, 1.0), (40, 0.5)):
            for index, label in enumerate(CONST_LABELS):
                if index == len(CONST_LABELS) - 1 and flight >= last:
                    continue
                for k, seed in enumerate(seeds):
                    rows.append(dict(flight='%d' % flight, n='%d' % n, horizon=label, seed=seed,
                                     position_error_mm=repr(factor * (10.0 + flight + 10.0 * index) + 2.0 * k)))
    return rows


def test_the_error_table_holds_each_flights_mean_error_per_range_sorted_and_a_thin_range_takes_the_one_before():
    """
    Description:
        From constructed rows: the labels are those of per_horizon.csv; per range the table holds each flight's error,
        the mean over its seeds, sorted, in metres, for N 20 and N 40 (rows without an error, at another N or at a fixed
        horizon are ignored); the mid-points are the ranges'. With only 9 flights in the last range it takes the
        range before it; with 9 flights in every range the table is refused naming the count.
    """

    assert Range_Labels(CONST_RANGES) == CONST_LABELS
    table = Error_Table(_Rows(), CONST_RANGES)
    assert np.allclose(table.Mid, [0.05, 0.15, 0.25, 0.4, 0.65, 1.0, 1.6]) and len(table.Short) == len(table.Long) == 7
    for index in range(7):
        assert np.allclose(table.Short[index], (10.0 + np.arange(12) + 10.0 * index + 1.0) * 1e-3)
        assert np.allclose(table.Long[index], (0.5 * (10.0 + np.arange(12) + 10.0 * index) + 1.0) * 1e-3)
    thin = Error_Table(_Rows(last=9), CONST_RANGES)
    assert np.array_equal(thin.Short[6], thin.Short[5]) and np.array_equal(thin.Long[6], thin.Long[5]) and len(thin.Short[6]) == 12
    with pytest.raises(Catch_Error_Cls, match='holds 9 flights with a position error at N = 20 in the horizon range \\[0, 0.1\\)'):
        Error_Table(_Rows(flights=9), CONST_RANGES)


def test_the_injected_prediction_is_the_true_trajectory_plus_the_tables_error_along_one_direction():
    """
    Description:
        For a throw the injected trajectory answers from the prefix's last time to the true floor time; its position
        differs from Flight.Core.Simulate's by an error along one unit direction, the same for every prefix of the
        throw, whose size at a range's mid-point is the scale times one quantile of the table's errors there (the same
        quantile in every range: the ratio to the range's smallest and largest error is one number), N 20's table below
        40 samples and N 40's (half the size here) from there on, linear between the mid-points and constant beyond
        them; the velocity carries the error's rate of change. At scale 0 it is the true trajectory exactly. Another
        throw draws another direction; the same throw gives the same error again; a time outside the range is refused.
    """

    data = types.SimpleNamespace(Release_State=CONST_RELEASE, Floor_Time=CONST_FLOOR)
    table = Error_Table(_Rows(), CONST_RANGES)
    predict = Injected_Predictor(data, CONST_FLIGHT, table=table, scale=2.0)(0)
    time = 0.004 * np.arange(60)
    short, long = predict(time[:20], None, 250.0), predict(time[:45], None, 250.0)
    assert (short.T_0, short.T_End, long.T_0) == (time[19], 1.2, time[44])
    errors = {}
    for name, trajectory in (('short', short), ('long', long)):
        at = trajectory.T_0 + table.Mid[:5]
        state = trajectory.State(at)
        truth = Simulate(CONST_RELEASE[0], at, Flight_Model_Str=CONST_FLIGHT)
        errors[name] = state[:, :3] - truth[:, :3]
        size = np.linalg.norm(errors[name], axis=1)
        low = 2.0 * np.array([table.Short[k][0] if name == 'short' else table.Long[k][0] for k in range(5)])
        high = 2.0 * np.array([table.Short[k][-1] if name == 'short' else table.Long[k][-1] for k in range(5)])
        quantile = (size - low) / (high - low)
        assert np.all(size > low) and np.all(size < high) and np.ptp(quantile) < 1e-9
        assert np.allclose(errors[name] / size[:, None], errors[name][0] / size[0], atol=1e-9)
        # Between two mid-points the size is linear, and the velocity carries its slope along the direction.
        between = trajectory.State(np.array([trajectory.T_0 + 0.10]))[0]
        truth_between = Simulate(CONST_RELEASE[0], np.array([trajectory.T_0 + 0.10]), Flight_Model_Str=CONST_FLIGHT)[0]
        assert abs(np.linalg.norm(between[:3] - truth_between[:3]) - 0.5 * (size[0] + size[1])) < 1e-9
        assert np.allclose(between[3:] - truth_between[3:], (size[1] - size[0]) / 0.1 * errors[name][0] / size[0], atol=1e-9)
        # Before the first mid-point the size is the first one's, and the velocity is the truth's.
        early = trajectory.State(np.array([trajectory.T_0 + 0.01]))[0]
        truth_early = Simulate(CONST_RELEASE[0], np.array([trajectory.T_0 + 0.01]), Flight_Model_Str=CONST_FLIGHT)[0]
        assert abs(np.linalg.norm(early[:3] - truth_early[:3]) - size[0]) < 1e-9 and np.allclose(early[3:], truth_early[3:], atol=1e-12)
    assert np.allclose(errors['short'][0] / np.linalg.norm(errors['short'][0]), errors['long'][0] / np.linalg.norm(errors['long'][0]), atol=1e-9)
    assert 1.5 < np.linalg.norm(errors['short'][0]) / np.linalg.norm(errors['long'][0]) < 2.1
    exact = Injected_Predictor(data, CONST_FLIGHT, table=table, scale=0.0)(0)(time[:20], None, 250.0)
    at = np.array([0.3, 0.9, 1.2])
    assert np.array_equal(exact.State(at), Simulate(CONST_RELEASE[0], at, Flight_Model_Str=CONST_FLIGHT))
    again = Injected_Predictor(data, CONST_FLIGHT, table=table, scale=2.0)(0)(time[:20], None, 250.0)
    assert np.array_equal(again.State(short.T_0 + table.Mid[:5]), short.State(short.T_0 + table.Mid[:5]))
    other = Injected_Predictor(data, CONST_FLIGHT, table=table, scale=2.0)(1)(time[:20], None, 250.0)
    other_error = other.State(np.array([0.5]))[0, :3] - Simulate(CONST_RELEASE[1], np.array([0.5]), Flight_Model_Str=CONST_FLIGHT)[0, :3]
    cosine = float(other_error @ errors['short'][0]) / (np.linalg.norm(other_error) * np.linalg.norm(errors['short'][0]))
    assert other.T_End == 1.15 and abs(cosine) < 0.999
    for outside in (short.T_0 - 0.001, 1.2001):
        with pytest.raises(Prediction_Error_Cls, match='outside the trajectory\'s range'):
            short.State(np.array([outside]))
