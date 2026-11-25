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
# File Name: ../test/Prediction/test_core.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the predictors and the trajectory contract under test
#   ../Prediction/Core
from Prediction.Core import Predict, Prediction_Error_Cls, Trajectory_From_State
#   ../Flight/Core, ../Observation/Core, ../Dataset/Core - the models, the generated error, the floor time
from Flight.Core import Fit_Initial_States, Flight_Model_Str, Simulate
from Observation.Core import Observation_Error, Observation_Model_Str
from Dataset.Core import Floor_Time

"""
Description:
    Initialization of constants.

    The trajectory contract of SPEC §2.13.13 for the numerical predictor, on flights simulated with the drag model:
    expectations come from Simulate (the one flight model), the floor height 0.030 m (SPEC §14.6) and the
    F7 Part A measurements, never from the predictors.
"""
# The numerical predictor shipped (the unweighted 'LS' removed).
#   Unit [Tuple<string>]
CONST_PREDICTORS = ('GLS_M4',)
# The flight model: g = 9.81 m/s^2 along -z and F3's k (SPEC §4.2).
#   Unit [Flight_Model_Str]
CONST_MODEL = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.00930496257)
# M4 with the configured values (Config/ball_catching.yaml, F6).
#   Unit [Observation_Model_Str]
CONST_M4 = Observation_Model_Str(White_Std=np.array([0.370011195, 0.803520496, 0.242729141]) * 1e-3, White_Dof=2.1,
                                 Drift_Std=np.array([1.9717258, 3.19042089, 2.97464373]) * 1e-3, Drift_Time=0.08,
                                 Offset_Std=np.array([9.32578562, 12.4212851, 5.84368324]) * 1e-3, Offset_Time=0.1, Offset_Dof=2.5)
# A throw from the throw area towards the robot (robot frame): its release state.
#   Unit [Vector<float> 1x6, m, m/s]
CONST_RELEASE = np.array([-0.15, -4.78, 1.17, 0.1, 4.3, 5.2])
# An irregular 200 Hz prefix of 12 samples with dropouts of 1, 2, 5 and 9 frames.
#   Unit [Vector<float> 1x12, s]
CONST_PREFIX_TIME = np.array([0, 1, 2, 4, 5, 6, 9, 10, 16, 17, 27, 28]) / 200.0
# The floor height (SPEC §14.6).
#   Unit [float, m]
CONST_FLOOR = 0.030


def _Prefix():
    return CONST_PREFIX_TIME, Simulate(CONST_RELEASE, CONST_PREFIX_TIME, Flight_Model_Str=CONST_MODEL)[:, :3]


def _Predict(name, time=None, position=None, rate=200.0):
    t, p = _Prefix()
    return Predict(name, t if time is None else time, p if position is None else position, rate, Flight_Model_Str=CONST_MODEL,
                   Observation_Model_Str=CONST_M4)


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_a_query_off_every_grid_is_answered_with_the_flight_model(name):
    """
    Description:
        SPEC §2.13.13: from a noise-free prefix, a query at 97 times off every sample and frame grid (and off
        the 1/600 s grid) is answered; position and velocity equal Simulate's within 1e-9 m and 1e-8 m/s.
    """

    trajectory = _Predict(name)
    t = trajectory.T_0 + (trajectory.T_End - trajectory.T_0) * (np.arange(97) + 0.3183) / 97.0
    truth = Simulate(CONST_RELEASE, t, Flight_Model_Str=CONST_MODEL)
    state = trajectory.State(t)
    assert np.max(np.abs(state[:, :3] - truth[:, :3])) < 1e-9 and np.max(np.abs(state[:, 3:] - truth[:, 3:])) < 1e-8


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_a_time_outside_the_range_is_refused_naming_it_and_the_range(name):
    """
    Description:
        A time before t_0 (the last observed sample) or after t_end is refused, the message naming the time
        and the range; t_0 and t_end themselves are answered.
    """

    trajectory = _Predict(name)
    assert trajectory.T_0 == CONST_PREFIX_TIME[-1]
    trajectory.State([trajectory.T_0, trajectory.T_End])
    for bad in (trajectory.T_0 - 1e-6, trajectory.T_End + 1e-6):
        with pytest.raises(Prediction_Error_Cls) as error:
            trajectory.State([trajectory.T_0, bad])
        assert repr(bad) in str(error.value) and repr(trajectory.T_0) in str(error.value) and repr(trajectory.T_End) in str(error.value)


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_queries_are_independent_of_the_other_times_asked(name):
    """
    Description:
        Query independence (SPEC §2.13.13): all 500 times, a random subset and the subset reversed give
        identical values bit for bit, so the query times cannot leak the true flight's end.
    """

    trajectory = _Predict(name)
    rng = np.random.default_rng(20261114)
    t = np.sort(rng.uniform(trajectory.T_0, trajectory.T_End, 500))
    every = trajectory.State(t)
    subset = np.sort(rng.choice(500, 37, replace=False))
    assert np.array_equal(trajectory.State(t[subset]), every[subset])
    assert np.array_equal(trajectory.State(t[subset][::-1]), every[subset][::-1])
    assert np.array_equal(trajectory.State(t[subset[:1]]), every[subset[:1]])


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_every_trajectory_ends_at_the_floor_height(name):
    """
    Description:
        The predicted end is where the ball centre descends to 0.030 m (SPEC §14.6): the height at t_end is
        0.030 m within 1e-9 m and, from a noise-free prefix, t_end is Dataset.Core.Floor_Time's within 1e-9 s;
        from a noisy prefix the end is at the floor too.
    """

    trajectory = _Predict(name)
    assert trajectory.State([trajectory.T_End])[0, 2] == pytest.approx(CONST_FLOOR, abs=1e-9)
    floor = Floor_Time(CONST_RELEASE[None], np.array([1.0]), Flight_Model_Str=CONST_MODEL)[0]
    assert trajectory.T_End == pytest.approx(floor, abs=1e-9)
    t, p = _Prefix()
    noisy = _Predict(name, position=p + np.random.default_rng(3).normal(0.0, 5e-3, p.shape))
    assert noisy.State([noisy.T_End])[0, 2] == pytest.approx(CONST_FLOOR, abs=1e-9)


def test_the_truth_trajectory_of_a_release_state_ends_at_its_floor_time():
    """
    Description:
        Trajectory_From_State of a release state from t = 0 (the truth Evaluation builds) ends at
        Dataset.Core.Floor_Time's floor time within 1e-9 s and equals Simulate within 1e-12 m.
    """

    trajectory = Trajectory_From_State(CONST_RELEASE, 0.0, 0.0, Flight_Model_Str=CONST_MODEL)
    floor = Floor_Time(CONST_RELEASE[None], np.array([1.0]), Flight_Model_Str=CONST_MODEL)[0]
    assert trajectory.T_End == pytest.approx(floor, abs=1e-9)
    t = np.linspace(0.0, trajectory.T_End, 301)
    assert np.max(np.abs(trajectory.State(t) - Simulate(CONST_RELEASE, t, Flight_Model_Str=CONST_MODEL))) < 1e-12


@pytest.mark.parametrize('name', CONST_PREDICTORS)
def test_the_samples_are_used_at_their_own_time_stamps(name):
    """
    Description:
        Irregular input is used as observed, never resampled (SPEC §2.7): the prefix shifted in time by
        0.123 s gives the same trajectory shifted (within 1e-9 m); one sample's stamp moved by one frame
        changes the prediction at t_0 + 0.5 s by more than 1 mm.
    """

    t, p = _Prefix()
    trajectory = _Predict(name)
    shifted = _Predict(name, time=t + 0.123)
    h = trajectory.T_0 + np.array([0.1, 0.5, 0.9])
    assert np.max(np.abs(shifted.State(h + 0.123) - trajectory.State(h))) < 1e-9
    moved = t.copy()
    moved[7] += 1.0 / 200.0
    assert np.max(np.abs(_Predict(name, time=moved).State(h[1:2]) - trajectory.State(h[1:2]))) > 1e-3


def test_an_invalid_prefix_or_predictor_is_refused_naming_the_problem():
    """
    Description:
        An unknown predictor, the removed predictor 'LS', a single sample, times not strictly
        increasing, positions of the wrong shape and a rate of 0 are each refused, the message naming the problem.
    """

    t, p = _Prefix()
    cases = (('Kalman', t, p, 200.0, '"Kalman" is not one of GLS_M4.'), ('LS', t, p, 200.0, '"LS" is not one of GLS_M4.'),
             ('GLS_M4', t[:1], p[:1], 200.0, 'at least 2 samples'),
             ('GLS_M4', t[[0, 2, 1]], p[:3], 200.0, 'not strictly increasing'), ('GLS_M4', t, p[:, :2], 200.0, 'Mx3 positions'),
             ('GLS_M4', t, p, 0.0, 'is not > 0'))
    for name, time, position, rate, message in cases:
        with pytest.raises(Prediction_Error_Cls) as error:
            Predict(name, time, position, rate, Flight_Model_Str=CONST_MODEL, Observation_Model_Str=CONST_M4)
        assert message in str(error.value), message


@pytest.mark.slow
def test_gls_m4_predicts_better_than_the_unweighted_fit_under_m4_errors():
    """
    Description:
        On 40 flights of 20 irregular samples at 200 Hz with M4's configured error (seed 20261114), GLS_M4's
        median position error at t_0 + 0.5 s is below that of the unweighted least-squares fit (F7 Part A: 29.1
        against 33.8 mm at N = 20 on the simulated validation partition), and both are finite and above 1 mm (the
        error is present). The unweighted fit is built here from Flight.Core.Fit_Initial_States without whitening
        and Trajectory_From_State (the removed predictor 'LS' computed the same): the only
        test that M4's weighting helps.
    """

    rng = np.random.default_rng(20261114)
    errors = {name: [] for name in ('LS', 'GLS_M4')}
    for _ in range(40):
        frames = np.cumsum(np.concatenate(([0], 1 + rng.geometric(0.5, 19) - 1 + (rng.random(19) < 0.2) * rng.integers(1, 12, 19))))
        t = frames / 200.0
        release = CONST_RELEASE + np.concatenate((rng.normal(0.0, 0.2, 3), rng.normal(0.0, 0.3, 3)))
        p = Simulate(release, t, Flight_Model_Str=CONST_MODEL)[:, :3] + Observation_Error(t, 200.0, Observation_Model_Str=CONST_M4,
                                                                                          scale=1.0, rng=rng)
        truth = Simulate(release, [t[-1] + 0.5], Flight_Model_Str=CONST_MODEL)[0, :3]
        x, _ = Fit_Initial_States([t - t[0]], [p], Flight_Model_Str=CONST_MODEL)
        trajectories = {'LS': Trajectory_From_State(x[0], float(t[0]), float(t[-1]), Flight_Model_Str=CONST_MODEL),
                        'GLS_M4': Predict('GLS_M4', t, p, 200.0, Flight_Model_Str=CONST_MODEL, Observation_Model_Str=CONST_M4)}
        for name, trajectory in trajectories.items():
            errors[name].append(np.linalg.norm(trajectory.State([t[-1] + 0.5])[0, :3] - truth))
    ls, gls = np.median(errors['LS']), np.median(errors['GLS_M4'])
    assert 1e-3 < gls < ls < 1.0
