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
# File Name: ../test/Observation/test_estimation.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the estimation and the statistics under test, and the model that generates known residuals
#   ../Observation/Estimation, Core
from Observation.Estimation import (Aggregate_Error, Aggregate_Pattern, Bootstrap_Bounds, Error_Parts, Estimate_Observation_Model,
                                    Pattern_Parts, Residual_Extremes, Residual_Str)
from Observation.Core import Observation_Error, Observation_Model_Str

"""
Description:
    Initialization of constants.

    The estimator is checked on residuals generated from a known model; the statistics on constructed
    flights whose values are worked out by hand in each test's docstring.
"""
# The known model of the recovery test: the sizes of F6 Part A, both time constants on the grids.
#   Unit [Observation_Model_Str]
CONST_KNOWN = Observation_Model_Str(White_Std=np.array([0.4e-3, 0.8e-3, 0.3e-3]), White_Dof=6.0,
                                    Drift_Std=np.array([2e-3, 3e-3, 3e-3]), Drift_Time=0.08,
                                    Offset_Std=np.array([9e-3, 12e-3, 6e-3]), Offset_Time=0.1, Offset_Dof=6.0)


def _Renewal_Frames(rng, n):
    """
    Description:
        Frames of a 200 Hz flight of n frames: present runs and dropouts of geometric lengths (means 20
        and 8 frames, i.e. 100 and 40 ms: runs long enough for the drift to show at lags up to 100 ms).
    """

    frames, j, present = [], 0, True
    while j < n:
        length = int(rng.geometric(1 / 20 if present else 1 / 8))
        if present:
            frames.extend(range(j, min(j + length, n)))
        j += length
        present = not present
    return np.array(frames)


def _Structure(model, tau):
    return model.White_Std ** 2 + model.Drift_Std ** 2 * (1.0 - np.exp(-tau / model.Drift_Time))


def _Jump_Variance(std, time, lost):
    return 2.0 * std ** 2 * (1.0 - np.exp(-lost / time))


def test_the_estimator_recovers_what_the_data_identify():
    """
    Description:
        Recovery when the data are rich (long runs, nu = 6), not at the
        size of the identification flights (that is the slow replicate test of test/Dataset/test_calibration.py).
        On 200 flights of 1.2 s whose residual is the known model's error, the estimator finds the drift's
        time constant on its grid value or the next (0.08 or 0.1 s) and its standard deviations within
        15 %; per axis, the within-run structure s_w^2 + s_c^2 (1 - exp(-tau / T_c)) at tau = 5, 20 and
        80 ms within 15 %; and the offset's jump variance 2 s_o^2 (1 - exp(-L / T_o)) at a lost time L of
        20 and 100 ms within 20 %. Not identified at this size, and not asserted (F6 Part B, KNOWN_ISSUES):
        the white noise of an axis apart from the drift's shortest increment (z came out 0.11 against
        0.3 mm), the offset's s and T apart from each other, and both degrees of freedom.
    """

    rng = np.random.default_rng(20261111)
    flights = []
    for _ in range(200):
        frames = _Renewal_Frames(rng, 241)
        flights.append(Residual_Str(Frame=frames, Rate=200.0,
                                    Residual=Observation_Error(frames / 200.0, 200.0, Observation_Model_Str=CONST_KNOWN, scale=1.0, rng=rng)))
    model = Estimate_Observation_Model(flights)
    assert model.Drift_Time in (0.08, 0.1)
    assert model.Drift_Std == pytest.approx(CONST_KNOWN.Drift_Std, rel=0.15)
    for tau in (0.005, 0.020, 0.080):
        assert _Structure(model, tau) == pytest.approx(_Structure(CONST_KNOWN, tau), rel=0.15), tau
    for lost in (0.020, 0.100):
        assert _Jump_Variance(model.Offset_Std, model.Offset_Time, lost) == pytest.approx(
            _Jump_Variance(CONST_KNOWN.Offset_Std, CONST_KNOWN.Offset_Time, lost), rel=0.20), lost


def test_the_estimate_does_not_depend_on_the_order_of_the_flights():
    """
    Description:
        Order invariance: 20 synthetic flights in their order and
        shuffled give the same estimate, bit for bit, in every parameter.
    """

    rng = np.random.default_rng(20261112)
    flights = []
    for _ in range(20):
        frames = _Renewal_Frames(rng, 241)
        flights.append(Residual_Str(Frame=frames, Rate=200.0,
                                    Residual=Observation_Error(frames / 200.0, 200.0, Observation_Model_Str=CONST_KNOWN, scale=1.0, rng=rng)))
    order = rng.permutation(20)
    estimates = [Estimate_Observation_Model(f) for f in (flights, [flights[i] for i in order])]
    for name in ('White_Std', 'White_Dof', 'Drift_Std', 'Drift_Time', 'Offset_Std', 'Offset_Time', 'Offset_Dof'):
        assert np.array_equal(getattr(estimates[0], name), getattr(estimates[1], name)), name


def test_the_error_statistics_of_a_constructed_flight():
    """
    Description:
        A 100 Hz flight at frames 0, 1, 2, 4, 5, 9 with x residuals 1, 1, 1, 3, 3, 7 mm: RMS sqrt(70 / 6)
        mm, max 7 mm; every run is constant, so the offset share is 1; the jump over one missing frame
        (1 -> 3 mm) is sqrt(4 / 2) mm, over three (3 -> 7 mm) sqrt(16 / 2) mm; at 10 ms (one frame) inside
        the runs the difference is 0; the edge samples (frames 2, 4, 5, 9) give sqrt(68 / 4) mm, the
        interior ones (0, 1) 1 mm; no 5 ms pair at 100 Hz.
    """

    r = np.zeros((6, 3))
    r[:, 0] = np.array([1, 1, 1, 3, 3, 7]) * 1e-3
    s = Aggregate_Error(Error_Parts([Residual_Str(Frame=np.array([0, 1, 2, 4, 5, 9]), Rate=100.0, Residual=r)]))
    assert s['rms_median'] == pytest.approx(np.sqrt(70.0 / 6.0) * 1e-3, rel=1e-12)
    assert s['max_median'] == pytest.approx(7e-3, rel=1e-12)
    assert s['offset_share'] == pytest.approx(1.0, abs=1e-12)
    assert s['jump_1'] == pytest.approx(np.sqrt(2.0) * 1e-3, rel=1e-12)
    assert s['jump_2_4'] == pytest.approx(np.sqrt(8.0) * 1e-3, rel=1e-12)
    assert s['within_10ms'] == 0.0 and np.isnan(s['within_5ms'])
    assert s['edge_rms'] == pytest.approx(np.sqrt(17.0) * 1e-3, rel=1e-12)
    assert s['interior_rms'] == pytest.approx(1e-3, rel=1e-12)


def test_the_pattern_statistics_of_a_constructed_pattern():
    """
    Description:
        A 100 Hz pattern at frames 0, 1, 2, 5, 6, 10 (11 frames): coverage 6 / 11; dropouts of 20 and
        30 ms; present runs of 30, 20, 10 ms; lag-1 autocorrelation 63 / 330 (worked out from the
        deviations 5/11 and -6/11); no dropout longer than 0.375 s; presence 0.5 in [0, 0.1) s (frames 0-9)
        and 1 in [0.1, 0.2) s (frame 10).
    """

    s = Aggregate_Pattern(Pattern_Parts([np.array([0, 1, 2, 5, 6, 10])], [100.0]))
    assert s['coverage_median'] == pytest.approx(6.0 / 11.0, rel=1e-12)
    assert s['missing_p50'] == pytest.approx(0.025, rel=1e-12) and s['present_p50'] == pytest.approx(0.02, rel=1e-12)
    assert s['lag1_median'] == pytest.approx(63.0 / 330.0, rel=1e-12)
    assert s['long_share'] == 0.0
    assert s['phase_00'] == pytest.approx(0.5, rel=1e-12) and s['phase_01'] == 1.0 and np.isnan(s['phase_02'])


def test_the_bootstrap_bounds_hold_the_statistic_and_are_reproducible():
    """
    Description:
        The 95 % bootstrap bounds of the median of 200 standard normal values contain the median, have
        about the width 2 x 1.96 x 1.2533 / sqrt(200) = 0.347 (within 30 %), and the same seed gives the
        same bounds.
    """

    values = list(np.random.default_rng(5).standard_normal(200))

    def Median(sample):
        return {'median': float(np.median(sample))}

    low, high = Bootstrap_Bounds(values, Median, seed=7)['median']
    assert low <= np.median(values) <= high
    assert high - low == pytest.approx(0.347, rel=0.3)
    assert Bootstrap_Bounds(values, Median, seed=7)['median'] == (low, high)


def test_the_largest_values_of_constructed_residuals():
    """
    Description:
        Two flights with a residual along x only, in mm. Flight 1 at frames 0-3, 10, 11, 30: 1, 3, 1, 1 | -4, -4 |
        2. Flight 2 at frames 0, 1: 0, 2.5. The largest residual is 4 mm; the largest run mean 4 mm (the runs'
        means are 1.5, -4, 2 and 1.25); the largest jump across a dropout 6 mm (-4 to 2; the other is 1 to -4);
        the largest step inside a run 2.5 mm (flight 2; flight 1 has 2, 2, 0 and 0). A flight of one sample has
        a residual and a run mean, no jump and no step.
    """

    one = Residual_Str(Frame=np.array([0, 1, 2, 3, 10, 11, 30]), Rate=200.0,
                       Residual=np.c_[np.array([1.0, 3.0, 1.0, 1.0, -4.0, -4.0, 2.0]) * 1e-3, np.zeros(7), np.zeros(7)])
    two = Residual_Str(Frame=np.array([0, 1]), Rate=200.0, Residual=np.c_[np.array([0.0, 2.5]) * 1e-3, np.zeros(2), np.zeros(2)])
    extremes = Residual_Extremes([one, two])
    assert extremes == pytest.approx({'residual_max': 4e-3, 'run_mean_max': 4e-3, 'jump_max': 6e-3, 'step_max': 2.5e-3}, rel=1e-12)
    single = Residual_Str(Frame=np.array([0]), Rate=200.0, Residual=np.array([[0.0, 3e-3, 4e-3]]))
    assert Residual_Extremes([single]) == pytest.approx({'residual_max': 5e-3, 'run_mean_max': 5e-3, 'jump_max': 0.0, 'step_max': 0.0}, rel=1e-12)
