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
# File Name: ../test/Evaluation/test_core.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the metrics under test
#   ../Evaluation/Core
from Evaluation.Core import (Catch_Height, Crossing_Time, Evaluate_Real, Flight_Result_Str, Real_Result, Simulated_Result, Statistics,
                             Summary_Rows, Velocity_Errors, _Ranges)
#   ../Prediction/Core, ../Flight/Core, ../Configuration/Core - trajectories, the model, the shipped areas
from Prediction.Core import Predict, Prediction_Error_Cls, Trajectory_From_State
from Flight.Core import Flight_Model_Str, Simulate
from Observation.Core import Observation_Model_Str
from Configuration.Core import Load_Configuration

"""
Description:
    Initialization of constants.

    The metrics on constructed flights: vacuum trajectories (closed-form heights), predictions that are the
    truth or the truth moved by a known amount, and sets of known per-flight errors (SPEC §2.13.11).
    Expectations are worked out by hand from the definitions of SPEC §7.12.
"""
# A vacuum model (closed-form crossings) and F3's drag model.
#   Unit [Flight_Model_Str]
CONST_VACUUM = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.0)
CONST_MODEL = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.00930496257)
# A release state (robot frame) and the catch height of the shipped areas (0.3607 + 1.0 / 2 m).
#   Unit [Vector<float> 1x6, m, m/s], [float, m]
CONST_RELEASE = np.array([-0.15, -4.78, 1.17, 0.1, 4.3, 5.2])
CONST_HEIGHT = 0.8607


def _Truth(x=CONST_RELEASE, model=CONST_MODEL):
    truth = Trajectory_From_State(x, 0.0, 0.0, Flight_Model_Str=model)
    frames = np.arange(int(np.floor(truth.T_End * 200.0)) + 1) / 200.0
    return truth, frames, truth.State(frames)[:, :3]


def _Result(errors, n=20, partition='validation', dataset='S0', rate=200.0, prefix=0.1, ranges=(), crossing='both'):
    result = Flight_Result_Str(Flight='0', Partition=partition, Dataset=dataset, N=n, Prefix_Duration=prefix, Rate=rate,
                               Remainder_RMS=errors * 1e-3, End_Mismatch=errors * 1e-3, Inference_Time=0.05, Crossing=crossing,
                               Catch_Point_Error=errors * 1e-3, Time_To_Catch_Error=-errors * 1e-3)
    result.Fixed['0.5'] = errors * 1e-3
    result.Ranges = [(r, errors * 1e-3, 3) for r in ranges]
    return result


def test_the_catch_surface_is_derived_from_the_robot_area(root):
    """
    Description:
        The catch surface is z = Areas.Robot centre + r/2: 0.3607 + 0.5 = 0.8607 m for
        the shipped configuration.
    """

    configuration = Load_Configuration(root + '/Config/ball_catching.yaml')
    assert Catch_Height(configuration) == pytest.approx(0.8607, abs=1e-12)


def test_a_crossing_time_is_the_closed_form_descent():
    """
    Description:
        In vacuum the descent through 0.8607 m is at t = (v_z + sqrt(v_z^2 + 2 g (z_0 - h))) / g; Crossing_Time
        gives it within 1e-9 s, and None after it or for a height above the apex.
    """

    truth, _, _ = _Truth(model=CONST_VACUUM)
    v, z = CONST_RELEASE[5], CONST_RELEASE[2]
    expected = (v + np.sqrt(v ** 2 + 2.0 * 9.81 * (z - CONST_HEIGHT))) / 9.81
    assert Crossing_Time(truth, CONST_HEIGHT, 0.0) == pytest.approx(expected, abs=1e-9)
    assert Crossing_Time(truth, CONST_HEIGHT, expected + 0.01) is None
    assert Crossing_Time(truth, z + v ** 2 / (2.0 * 9.81) + 0.01, 0.0) is None


def test_a_perfect_prediction_has_zero_error_and_both_cross():
    """
    Description:
        A prediction equal to the truth from t_0 = 0.1 s: every fixed horizon, range and the remainder RMS below
        1e-9 m, the end mismatch below 1e-9 s, both cross with errors below 1e-9 m and 1e-9 s; the horizons past the
        truth's end are not scored, 'end' is.
    """

    truth, frames, positions = _Truth()
    predicted = Trajectory_From_State(CONST_RELEASE, 0.0, 0.1, Flight_Model_Str=CONST_MODEL)
    after = np.arange(21, len(frames), 7)
    result = Simulated_Result(predicted, truth, truth.T_End, frames, positions, after, positions[after], CONST_HEIGHT)
    assert set(result.Fixed) == {'0.05', '0.1', '0.2', '0.3', '0.5', '0.8', 'end'} and max(result.Fixed.values()) < 1e-9
    assert result.Remainder_RMS < 1e-9 and abs(result.End_Mismatch) < 1e-9 and max(v for _, v, _ in result.Ranges) < 1e-9
    assert result.Crossing == 'both' and result.Catch_Point_Error < 1e-9 and abs(result.Time_To_Catch_Error) < 1e-9


def test_an_early_landing_counts_as_error_through_the_held_state():
    """
    Description:
        A real flight's recorded samples continue 30, 60 and 90 mm below the predicted end point after t_end (and
        equal the prediction before): the remainder RMS is sqrt((30^2 + 60^2 + 90^2) / n) over the n samples, the
        held state scores them, and the end mismatch is t_end minus the last sample.
    """

    predicted = Trajectory_From_State(CONST_RELEASE, 0.0, 0.1, Flight_Model_Str=CONST_MODEL)
    before = np.linspace(0.12, predicted.T_End - 0.01, 17)
    end = predicted.State([predicted.T_End])[0, :3]
    later = predicted.T_End + np.array([0.01, 0.02, 0.03])
    t = np.concatenate((before, later))
    p = np.vstack((predicted.State(before)[:, :3], end - np.array([[0.0, 0.0, 0.03], [0.0, 0.0, 0.06], [0.0, 0.0, 0.09]])))
    result = Real_Result(predicted, t, p)
    assert result.Remainder_RMS == pytest.approx(np.sqrt((0.03 ** 2 + 0.06 ** 2 + 0.09 ** 2) / 20.0), abs=1e-9)
    assert result.End_Mismatch == pytest.approx(-0.03, abs=1e-12)


def test_a_real_flight_has_no_catch_metric_and_no_fixed_horizon():
    """
    Description:
        SPEC §2.14: a real flight's result has no crossing category, NaN catch-point and time-to-catch errors and
        no fixed horizon; its horizon ranges and remainder RMS are computed.
    """

    predicted = Trajectory_From_State(CONST_RELEASE, 0.0, 0.1, Flight_Model_Str=CONST_MODEL)
    t = np.array([0.12, 0.25, 0.6])
    result = Real_Result(predicted, t, predicted.State(t)[:, :3] + 0.002)
    assert result.Crossing == '' and np.isnan(result.Catch_Point_Error) and np.isnan(result.Time_To_Catch_Error)
    assert result.Fixed == {} and [r for r, _, _ in result.Ranges] == [0, 1, 4]
    assert result.Remainder_RMS == pytest.approx(0.002 * np.sqrt(3.0), abs=1e-12)


def test_the_ranges_take_one_median_per_flight_and_close_the_last():
    """
    Description:
        Samples at horizons 0.05, 0.15, 0.16, 0.17 s (errors 1, 2, 9, 3 mm) and 2.0 s give range [0, 0.1) 1 mm,
        [0.1, 0.2) the median 3 mm of three samples, and [1.2, 2.0] (closed) its sample; one at 2.0001 s is in no
        range.
    """

    predicted = Trajectory_From_State(CONST_RELEASE, 0.0, 0.0, Flight_Model_Str=CONST_VACUUM)
    t = np.array([0.05, 0.15, 0.16, 0.17])
    p = predicted.State(t)[:, :3] + np.array([[0.001, 0, 0], [0.002, 0, 0], [0.009, 0, 0], [0.003, 0, 0]])
    ranges = Real_Result(predicted, t, p).Ranges
    assert [(r, round(v * 1e3, 9), k) for r, v, k in ranges] == [(0, 1.0, 1), (1, 3.0, 3)]
    assert [r for r, _, _ in _Ranges(np.array([2.0, 2.0001]), np.array([1.0, 1.0]))] == [6]


def test_the_crossing_categories_passed_and_disagree():
    """
    Description:
        A prefix ending after the truth's descent through the catch surface is 'passed' (no catch metric); a
        prediction whose apex stays below the surface while the truth crosses is 'disagree'.
    """

    truth, frames, positions = _Truth()
    crossing = Crossing_Time(truth, CONST_HEIGHT, 0.0)
    late = Trajectory_From_State(CONST_RELEASE, 0.0, crossing + 0.01, Flight_Model_Str=CONST_MODEL)
    result = Simulated_Result(late, truth, truth.T_End, frames, positions, np.array([], dtype=int), np.zeros((0, 3)), CONST_HEIGHT)
    assert result.Crossing == 'passed' and np.isnan(result.Catch_Point_Error)
    low = Trajectory_From_State(np.array([-0.15, -4.78, 0.5, 0.1, 4.3, 1.0]), 0.0, 0.01, Flight_Model_Str=CONST_MODEL)
    result = Simulated_Result(low, truth, truth.T_End, frames, positions, np.array([], dtype=int), np.zeros((0, 3)), CONST_HEIGHT)
    assert result.Crossing == 'disagree' and np.isnan(result.Time_To_Catch_Error)


def test_statistics_are_count_mean_median_p90_max():
    """
    Description:
        The values 1..10 (and a NaN, left out): count 10, mean 5.5, median 5.5, p90 9.1 (linear interpolation),
        max 10.
    """

    assert Statistics(list(range(1, 11)) + [float('nan')]) == (10, 5.5, 5.5, pytest.approx(9.1, abs=1e-12), 10.0)


def test_a_range_with_9_flights_reports_no_statistics_and_one_with_10_does():
    """
    Description:
        SPEC §2.13.11: errors 1..10 mm; range [0, 0.1) reached by all 10 flights reports mean 5.5 and p90 9.1 mm,
        range [0.1, 0.2) reached by 9 reports its count 9 and no statistics; the remainder RMS of all 10 is
        summarised as usual.
    """

    results = [_Result(e, ranges=(0, 1) if e < 10 else (0,)) for e in range(1, 11)]
    rows = {(r['group'], r['metric']): r for r in Summary_Rows(results) if r['group'] == 'all'}
    full, short = rows[('all', 'range_0_0.1_mm')], rows[('all', 'range_0.1_0.2_mm')]
    assert (full['count'], full['mean'], full['p90']) == (10, pytest.approx(5.5), pytest.approx(9.1))
    assert short['count'] == 9 and all(np.isnan(short[k]) for k in ('mean', 'median', 'p90', 'max'))
    assert rows[('all', 'remainder_rms_mm')]['median'] == pytest.approx(5.5)


def test_the_summary_groups_by_rate_and_prefix_and_counts_the_crossings():
    """
    Description:
        Two flights at 100 Hz (prefix 0.03 s) and one at 300 Hz (prefix 0.25 s): the rate groups hold 2 and 1, the
        prefix groups [0, 0.05) and [0.2, 0.4) hold 2 and 1; with one 'disagree' among three crossings the
        disagreement rate is 1/3 and the catch-point error is summarised over the two 'both' flights only.
    """

    results = [_Result(1.0, rate=100.0, prefix=0.03), _Result(3.0, rate=100.0, prefix=0.03, crossing='disagree'),
               _Result(5.0, rate=300.0, prefix=0.25)]
    rows = Summary_Rows(results)
    count = {(r['group'], r['value']): r['count'] for r in rows if r['metric'] == 'remainder_rms_mm'}
    assert count == {('all', ''): 3, ('rate', '100'): 2, ('rate', '300'): 1, ('prefix', '[0, 0.05)'): 2, ('prefix', '[0.2, 0.4)'): 1}
    every = {r['metric']: r for r in rows if r['group'] == 'all'}
    assert every['crossing_disagreement']['mean'] == pytest.approx(1.0 / 3.0)
    assert (every['catch_point_mm']['count'], every['catch_point_mm']['mean']) == (2, pytest.approx(3.0))


def test_the_spread_over_datasets_is_min_max_and_sd_of_each_statistic():
    """
    Description:
        Two datasets whose remainder RMS medians are 2 and 4 mm: the spread rows hold min 2, max 4 and sample sd
        sqrt(2) for the median; a partition with one dataset gets no spread rows.
    """

    results = [_Result(e, dataset='A') for e in (1.0, 2.0, 3.0)] + [_Result(e, dataset='B') for e in (3.0, 4.0, 5.0)]
    results += [_Result(7.0, partition='real', dataset='')]
    rows = [r for r in Summary_Rows(results) if r['metric'] == 'remainder_rms_mm' and r['group'] == 'all']
    spread = {r['dataset']: r['median'] for r in rows if r['dataset'].startswith('spread')}
    assert spread == {'spread_min': 2.0, 'spread_max': 4.0, 'spread_sd': pytest.approx(np.sqrt(2.0))}
    assert not [r for r in rows if r['partition'] == 'real' and r['dataset'].startswith('spread')]


def test_a_refused_prediction_is_its_own_outcome_and_the_evaluation_goes_on():
    """
    Description:
        A flight whose 12-sample prefix ends below the floor (0.3 m under a landing; a trajectory that has
        already landed) is refused by the predictor; Evaluate_Real records it as 'refused' with the predictor's
        reason and every metric empty, returns no prediction for it, and goes on to score the next flight.
    """

    truth = Trajectory_From_State(CONST_RELEASE, 0.0, 0.0, Flight_Model_Str=CONST_MODEL)
    good = np.linspace(0.0, 0.6, 13)
    landed = np.linspace(truth.T_End - 0.3, truth.T_End, 13)
    sunk = truth.State(landed)[:, :3] - np.array([0.0, 0.0, 0.3])
    m4 = Observation_Model_Str(White_Std=np.full(3, 1e-3), White_Dof=6.0, Drift_Std=np.full(3, 1e-3), Drift_Time=0.08,
                               Offset_Std=np.full(3, 1e-3), Offset_Time=0.1, Offset_Dof=6.0)
    predictor = lambda time, position, rate: Predict('GLS_M4', time, position, rate, Flight_Model_Str=CONST_MODEL, Observation_Model_Str=m4)
    results, predictions = Evaluate_Real([('sunk', 200.0, landed - landed[0], sunk), ('good', 200.0, good, truth.State(good)[:, :3])],
                                         [('', predictor)], [12])
    assert [(r.Flight, r.Crossing) for r in results] == [('sunk', 'refused'), ('good', '')]
    assert 'below the floor height' in results[0].Refusal and np.isnan(results[0].Remainder_RMS) and np.isnan(results[0].Inference_Time)
    assert predictions[0] is None and predictions[1] is not None and results[1].Remainder_RMS < 1e-9


def test_the_summary_counts_refused_predictions_and_scores_only_the_others():
    """
    Description:
        Two refused predictions among five: 'refused' has count 2, 'refused_share' mean 0.4 over
        all five, the remainder RMS is summarised over the three predicted flights only, and a partition without a
        refusal states count 0.
    """

    results = [_Result(e) for e in (1.0, 2.0, 3.0)] + [Flight_Result_Str(Flight=str(k), Partition='validation', Dataset='S0', N=20,
                                                                         Rate=200.0, Crossing='refused', Refusal='x') for k in (7, 8)]
    results += [_Result(4.0, partition='test')]
    rows = {(r['partition'], r['metric']): r for r in Summary_Rows(results) if r['group'] == 'all'}
    assert rows[('validation', 'refused')]['count'] == 2 and rows[('validation', 'refused_share')]['mean'] == pytest.approx(0.4)
    assert rows[('validation', 'remainder_rms_mm')]['count'] == 3 and rows[('test', 'refused')]['count'] == 0


def test_the_velocity_errors_are_the_angle_between_the_velocities_and_the_speed_difference():
    """
    Description:
        SPEC §7.12: the direction error is the angle between the predicted and the true
        velocity, the speed error the predicted speed minus the true one. Perpendicular velocities of equal speed:
        90 degrees, 0 m/s; the same direction at 2 and 1 m/s: 0 degrees, +1 m/s; a velocity turned by 3 degrees
        about z and 5 % slower: 3 degrees, -5 % of the speed; opposite velocities: 180 degrees; a zero velocity
        has no direction (NaN).
    """

    direction, speed = Velocity_Errors(np.array([0.0, 3.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    assert np.degrees(direction) == pytest.approx(90.0, abs=1e-12) and speed == 0.0
    assert Velocity_Errors(np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])) == (0.0, 1.0)
    true = np.array([0.1, 4.3, -5.2])
    c, s = np.cos(np.radians(3.0)), np.sin(np.radians(3.0))
    turned = 0.95 * np.array([c * true[0] - s * true[1], s * true[0] + c * true[1], true[2]])
    direction, speed = Velocity_Errors(turned, true)
    expected = np.degrees(np.arccos((np.hypot(true[0], true[1]) ** 2 * c + true[2] ** 2) / np.dot(true, true)))
    assert np.degrees(direction) == pytest.approx(expected, abs=1e-9) and speed == pytest.approx(-0.05 * np.linalg.norm(true), abs=1e-12)
    assert np.degrees(Velocity_Errors(-true, true)[0]) == pytest.approx(180.0, abs=1e-6)
    assert np.isnan(Velocity_Errors(np.zeros(3), true)[0]) and Velocity_Errors(np.zeros(3), true)[1] == pytest.approx(-np.linalg.norm(true))


def test_a_simulated_flight_has_velocity_errors_at_the_fixed_horizons_and_the_catch_plane():
    """
    Description:
        SPEC §7.12: on a simulated flight the velocity's errors are taken at the fixed horizons
        reached and at the catch plane, against the exact truth. A prediction that is the truth has zero errors
        at every horizon and at the catch plane. A prediction from the true state at t_0 = 0.1 s with the
        velocity 2 % faster has, at the horizon 0.3 s, the errors of the two states Simulate gives there
        (computed here from Simulate), and a positive speed error at the catch plane; the truth's end has no
        velocity error.
    """

    truth, frames, positions = _Truth()
    after = np.where(frames > 0.1)[0]
    perfect = Simulated_Result(Trajectory_From_State(CONST_RELEASE, 0.0, 0.1, Flight_Model_Str=CONST_MODEL), truth, truth.T_End, frames,
                               positions, after, positions[after], CONST_HEIGHT)
    assert set(perfect.Direction) == set(perfect.Speed) == {'0.05', '0.1', '0.2', '0.3', '0.5', '0.8', 'catch'}
    assert max(abs(v) for v in perfect.Speed.values()) < 1e-9 and max(perfect.Direction.values()) < 1e-6
    state = Simulate(CONST_RELEASE, np.array([0.1]), Flight_Model_Str=CONST_MODEL)[0]
    state[3:] *= 1.02
    result = Simulated_Result(Trajectory_From_State(state, 0.1, 0.1, Flight_Model_Str=CONST_MODEL), truth, truth.T_End, frames, positions,
                              after, positions[after], CONST_HEIGHT)
    predicted = Simulate(state, np.array([0.3]), Flight_Model_Str=CONST_MODEL)[0, 3:]
    true = Simulate(CONST_RELEASE, np.array([0.4]), Flight_Model_Str=CONST_MODEL)[0, 3:]
    angle = np.arccos(np.dot(predicted, true) / (np.linalg.norm(predicted) * np.linalg.norm(true)))
    assert result.Direction['0.3'] == pytest.approx(angle, abs=1e-9) and angle > 1e-3
    assert result.Speed['0.3'] == pytest.approx(np.linalg.norm(predicted) - np.linalg.norm(true), abs=1e-9)
    assert result.Crossing == 'both' and result.Speed['catch'] > 0.01 and 'end' not in result.Direction


def test_the_summary_holds_the_velocity_metrics_in_degrees_and_none_for_real_flights():
    """
    Description:
        SPEC §2.13.11, §2.14: the velocity's errors are summarised as distributions like every
        metric: three simulated flights with direction errors of 1, 2 and 6 degrees and speed errors of -0.1, 0.2
        and 0.3 m/s at 0.5 s give the direction's median 2 and max 6 [deg], the signed speed's mean 0.4 / 3 and
        the absolute speed's median 0.2 [m/s], and at the catch plane alike; a real flight has no velocity metric.
    """

    results = []
    for direction, speed in ((1.0, -0.1), (2.0, 0.2), (6.0, 0.3)):
        r = _Result(10.0)
        r.Direction, r.Speed = {'0.5': np.radians(direction), 'catch': np.radians(2.0 * direction)}, {'0.5': speed, 'catch': -speed}
        results.append(r)
    results.append(_Result(10.0, partition='real', dataset='', crossing=''))
    rows = {(r['partition'], r['metric']): r for r in Summary_Rows(results) if r['group'] == 'all'}
    assert (rows[('validation', 'direction_0.5_deg')]['median'], rows[('validation', 'direction_0.5_deg')]['max']) == pytest.approx((2.0, 6.0))
    assert rows[('validation', 'speed_0.5_m_s')]['mean'] == pytest.approx(0.4 / 3.0) and rows[('validation', 'speed_abs_0.5_m_s')]['median'] == 0.2
    assert rows[('validation', 'direction_catch_deg')]['median'] == pytest.approx(4.0) and rows[('validation', 'speed_catch_m_s')]['count'] == 3
    assert not any(metric.startswith(('direction', 'speed')) for partition, metric in rows if partition == 'real')


def test_the_spread_over_training_seeds_is_min_max_and_sd_of_each_statistic():
    """
    Description:
        SPEC §2.13.11 (variance over at least 5 seeds): a dataset evaluated with 3 training
        seeds whose median remainder RMS is 10, 12 and 17 mm gets the rows seed 'spread_min' 10, 'spread_max' 17
        and 'spread_sd' (the sample sd, sqrt(13)); each seed keeps its own row; one seed alone gets no spread row,
        and a numerical predictor (no seed) none over seeds.
    """

    results = []
    for seed, error in (('0', 10.0), ('1', 12.0), ('2', 17.0)):
        r = _Result(error)
        r.Seed = seed
        results.append(r)
    rows = [r for r in Summary_Rows(results) if r['group'] == 'all' and r['metric'] == 'remainder_rms_mm']
    by_seed = {r['seed']: r['median'] for r in rows}
    assert {k: by_seed[k] for k in ('0', '1', '2')} == pytest.approx({'0': 10.0, '1': 12.0, '2': 17.0})
    assert (by_seed['spread_min'], by_seed['spread_max']) == pytest.approx((10.0, 17.0)) and by_seed['spread_sd'] == pytest.approx(np.sqrt(13.0))
    assert all(r['dataset'] == 'S0' for r in rows) and len(rows) == 6
    assert not any(r['seed'].startswith('spread') for r in Summary_Rows(results[:1]) + Summary_Rows([_Result(10.0), _Result(12.0, dataset='S1')]))


def test_every_predictor_is_scored_on_the_same_prefix_under_its_seed():
    """
    Description:
        A predictor is a function of a prefix giving a trajectory (SPEC §7.12), and an evaluation takes a list of
        them with their training seeds: each is given the same prefix (the first N samples, nothing after them),
        the results come by flight, then N, then predictor, under their seeds; a predictor that refuses does not
        stop the others.
    """

    truth = Trajectory_From_State(CONST_RELEASE, 0.0, 0.0, Flight_Model_Str=CONST_MODEL)
    t = np.linspace(0.0, 0.6, 13)
    seen = []

    def Exact(time, position, rate):
        seen.append((len(time), float(time[-1]), rate))
        return Trajectory_From_State(CONST_RELEASE, 0.0, float(time[-1]), Flight_Model_Str=CONST_MODEL)

    def Refusing(time, position, rate):
        seen.append((len(time), float(time[-1]), rate))
        raise Prediction_Error_Cls('no prediction')

    results, predictions = Evaluate_Real([('one', 200.0, t, truth.State(t)[:, :3])], [('0', Exact), ('1', Refusing), ('2', Exact)], [4, 8])
    assert [(r.N, r.Seed, r.Crossing) for r in results] == [(4, '0', ''), (4, '1', 'refused'), (4, '2', ''), (8, '0', ''), (8, '1', 'refused'),
                                                            (8, '2', '')]
    assert seen == [(4, t[3], 200.0)] * 3 + [(8, t[7], 200.0)] * 3 and [p is None for p in predictions] == [False, True, False] * 2
    assert results[0].Remainder_RMS < 1e-9 and results[1].Refusal == 'no prediction'
