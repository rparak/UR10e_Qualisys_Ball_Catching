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
# File Name: ../test/Dataset/test_calibration.py
## =========================================================================== ##

# System (Default)
import os
# Math (Mathematical functions)
import math
# Dataclasses (Data Classes)
from dataclasses import replace
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the calibration under test, the configurations, the models and the shooting
#   ../Dataset/Calibration, Core, ../Configuration/Core, ../Flight/Core, ../Recording/Identification, ../Observation/*
from Dataset.Calibration import (Calibration_Error_Cls, Calibration_Str, Coverage, Derived_Bound, Robot_States, Shape_Values, Summary_Rows,
                                 Synthetic_Residuals, Write_Observation_Summary, _Check_Transform)
from Dataset.Core import Identification_Flights, Shoot
from Recording.Identification import Fit_Initial_States
from Observation.Estimation import Estimate_Observation_Model
from Configuration.Core import Load_Configuration, Load_Dataset_Configuration
from Flight.Core import Flight_Model_Str
from Recording.Identification import Fitted_Flight_Str
from Observation.Core import Observation_Model_Str
from Observation.Estimation import CONST_ERROR_DIAGNOSTICS, CONST_ERROR_STATISTICS

"""
Description:
    Initialization of constants.

    The calibration's parts on constructed inputs; the calibration of the real identification flights is
    pinned through scripts/catch.py observation (test/scripts/test_catch.py, slow).
"""
# A vacuum model with g = 9.81 m/s^2.
#   Unit [Flight_Model_Str]
CONST_VACUUM = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]))


@pytest.fixture(scope='module')
def configurations(root):
    return (Load_Configuration(os.path.join(root, 'Config', 'ball_catching.yaml')),
            Load_Dataset_Configuration(os.path.join(root, 'Config', 'dataset.yaml')))


def test_robot_states_transform_the_position_and_rotate_the_velocity(configurations):
    """
    Description:
        By the assumed transform Rz(180 deg) + (0.827, -4.670, 0) m (SPEC §7.3), a QTM state (1, 2, 3 m;
        4, 5, 6 m/s) is (-0.173, -6.670, 3) m and (-4, -5, 6) m/s in the robot frame.
    """

    state = Robot_States([Fitted_Flight_Str(X_0=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))], configurations[0])
    assert state[0] == pytest.approx([-0.173, -6.670, 3.0, -4.0, -5.0, 6.0], abs=1e-12)


def test_the_shape_of_a_vacuum_throw_is_its_closed_form():
    """
    Description:
        A vacuum throw from z 1.2 m with v = (0, 4, 5) m/s: speed sqrt(41) m/s, elevation atan2(5, 4), apex
        1.2 + 25 / (2 g) m (on the 1/600 s grid, within 1e-5 m), floor time (5 + sqrt(25 + 2 g (1.2 - 0.03)))
        / g within 1e-9 s.
    """

    values = Shape_Values(np.array([[0.0, -5.0, 1.2, 0.0, 4.0, 5.0]]), Flight_Model_Str=CONST_VACUUM)
    g = 9.81
    assert values['speed'][0] == pytest.approx(math.sqrt(41.0), rel=1e-12)
    assert values['elevation'][0] == pytest.approx(math.degrees(math.atan2(5.0, 4.0)), rel=1e-12)
    assert values['apex'][0] == pytest.approx(1.2 + 25.0 / (2.0 * g), abs=1e-5)
    assert values['floor_time'][0] == pytest.approx((5.0 + math.sqrt(25.0 + 2.0 * g * (1.2 - 0.03))) / g, abs=1e-9)


def test_coverage_names_each_reason_a_release_is_outside(configurations):
    """
    Description:
        The support of the generated throws (SPEC §2.13.6): a throw shot from inside Areas.Throw at 6.7 m/s
        into the target area is inside; the same state moved out of Areas.Throw, one 10 m/s fast, and one
        aimed 1.2 m behind the centre (+y: it is still at +y when the flight-time window opens, so it never
        enters the half facing the thrower, r 1.5 m) are outside, each with its reason named.
    """

    configuration, dataset = configurations
    flight = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=configuration.Flight.Drag)
    release = np.array([[0.0, -4.8, 1.2], [0.0, -4.8, 1.2]])
    aim = np.array([[0.0, -0.5, 0.9], [0.0, 1.2, 0.9]])
    v, converged = Shoot(release, aim, np.array([1.15, 1.15]), Flight_Model_Str=flight)
    assert converged.all()
    states = np.vstack([np.hstack((release[0], v[0])), np.hstack(([0.0, -8.0, 1.2], v[0])),
                        np.hstack((release[0], v[0] * 10.0 / np.linalg.norm(v[0]))), np.hstack((release[1], v[1]))])
    result = dict(Coverage(['in', 'far', 'fast', 'behind'], states, configuration, dataset, Flight_Model_Str=flight))
    assert 5.5 <= np.linalg.norm(v[0]) <= 8.0 and result['in'] == ''
    assert 'release outside Areas.Throw' in result['far']
    assert 'release speed 10.00 m/s outside Throw.Release_Speed' in result['fast']
    assert result['behind'] == 'not in the target area during Throw.Flight_Time'


def test_coverage_of_a_catch_dataset_uses_the_catch_area_grown_by_its_margin(configurations):
    """
    Description:
        With Throw.Target_Catch_Area the support's target area is Areas.Catch grown by the
        margin (x 0.55-0.85, y -0.55-0.20, z 0.65-1.15 m at margin 0). A throw aimed at (0.7, 0.1, 0.9) m, inside
        Areas.Catch beyond the quarter sphere's y <= 0, is inside the catch dataset's support; one aimed at
        (0, -0.5, 0.9) m, inside the quarter sphere and 0.55 m beside the area in x, is not; with the margin 0.6 m it is.
    """

    configuration, dataset = configurations
    flight = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=configuration.Flight.Drag)
    release = np.array([[0.0, -4.8, 1.2], [0.0, -4.8, 1.2]])
    v, converged = Shoot(release, np.array([[0.7, 0.1, 0.9], [0.0, -0.5, 0.9]]), np.array([1.15, 1.15]), Flight_Model_Str=flight)
    assert converged.all()
    states = np.hstack((release, v))
    catch = replace(dataset, Throw=replace(dataset.Throw, Target_Shape='catch_area', Target_Margin=0.0))
    result = dict(Coverage(['area', 'beside'], states, configuration, catch, Flight_Model_Str=flight))
    assert result == {'area': '', 'beside': 'not in the target area during Throw.Flight_Time'}
    wide = replace(dataset, Throw=replace(dataset.Throw, Target_Shape='catch_area', Target_Margin=0.6))
    assert dict(Coverage(['area', 'beside'], states, configuration, wide, Flight_Model_Str=flight)) == {'area': '', 'beside': ''}


# The identification flights outside the support of the generated throws with the targets of Throw.Target_Area
# (r 1.5 m; Part A, 2026-11-15: 42 of 45 inside), and those outside with F7's targets (r 1.0 m, the half of
# Areas.Robot; F6 Part B: 39 of 45).
#   Unit [Set<string>]
CONST_UNCOVERED = {'Frequency_100_Hz_Active_Filtering_OFF/Experiment_5.txt', 'Frequency_100_Hz_Active_Filtering_ON/Experiment_6.txt',
                   'Frequency_200_Hz_Active_Filtering_OFF/Experiment_27.txt'}
CONST_UNCOVERED_F7 = CONST_UNCOVERED | {'Frequency_200_Hz_Active_Filtering_OFF/Experiment_3.txt',
                                        'Frequency_200_Hz_Active_Filtering_OFF/Experiment_11.txt',
                                        'Frequency_200_Hz_Active_Filtering_OFF/Experiment_31.txt'}


def test_42_of_the_45_identification_flights_lie_in_the_support_of_the_generated_throws(root, configurations):
    """
    Description:
        The coverage of SPEC §2.13.6 on the identification flights with the shipped configurations (measured on these
        flights only): 42 of 45 inside, the three outside named (two cross only
        the far side of the target area, one is a late start). With the radius of the target area at 1.0 m, F7's
        targets, it is the 39 of 45 of F6 with the same six flights outside: the wider area is what covers the
        three. The validation flights are not read.
    """

    configuration, dataset = configurations
    flight = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -configuration.Flight.Gravity]), Drag=configuration.Flight.Drag,
                              Drag_Std=configuration.Flight.Drag_Std)
    flights = Identification_Flights(os.path.join(root, 'Data', 'Recordings'))
    names = ['%s/Experiment_%d.txt' % (f.Folder, f.Experiment) for f in flights]
    states = Robot_States(Fit_Initial_States(flights, Flight_Model_Str=flight), configuration)
    assert len(flights) == 45
    coverage = Coverage(names, states, configuration, dataset, Flight_Model_Str=flight)
    assert {name for name, reason in coverage if reason} == CONST_UNCOVERED
    former = replace(dataset, Throw=replace(dataset.Throw, Target_Area=replace(dataset.Throw.Target_Area, Radius=1.0)))
    assert {name for name, reason in Coverage(names, states, configuration, former, Flight_Model_Str=flight) if reason} == CONST_UNCOVERED_F7


def test_a_transform_that_turns_the_vertical_is_refused(configurations):
    """
    Description:
        The calibration needs a transform about z only (the error model per axis, gravity along -z): a 10 deg
        turn about x is refused, naming the orientation.
    """

    configuration = configurations[0]
    _Check_Transform(configuration)
    turned = replace(configuration, Transform=replace(configuration.Transform, Orientation=[math.radians(10.0), 0.0, math.pi]))
    with pytest.raises(Calibration_Error_Cls, match='not a multiple of 90 deg about z'):
        _Check_Transform(turned)


def test_a_bound_is_the_largest_value_with_its_margin_rounded_up_to_5_mm():
    """
    Description:
        The error model's bounds: the largest value of the identification flights x 1.25
        (assumed), rounded up to 0.005 m. The measured 55.9 mm gives 69.9 -> 0.070 m and 29.5 mm gives 36.9 ->
        0.040 m; 56.0 mm x 1.25 = 70.0 mm stays 0.070 m, 56.1 mm goes to 0.075 m (both sides of a step); the
        summary carries the largest values and the two bounds in its section 'bound'.
    """

    assert Derived_Bound(0.0559) == 0.070 and Derived_Bound(0.0295) == 0.040
    assert Derived_Bound(0.0560) == 0.070 and Derived_Bound(0.0561) == 0.075
    keys = CONST_ERROR_STATISTICS + CONST_ERROR_DIAGNOSTICS
    spread = {k: (0.0, 1.0) for k in ('White_Std_x', 'White_Std_y', 'White_Std_z', 'White_Dof', 'Drift_Std_x', 'Drift_Std_y', 'Drift_Std_z',
                                      'Drift_Time', 'Offset_Std_x', 'Offset_Std_y', 'Offset_Std_z', 'Offset_Time', 'Offset_Dof')}
    calibration = Calibration_Str(Spread=spread, Extremes={'residual_max': 0.0559, 'run_mean_max': 0.0552, 'jump_max': 0.0894, 'step_max': 0.0295},
                                  Offset_Bound=Derived_Bound(0.0559), White_Bound=Derived_Bound(0.0295), Error_Real={k: 1.0 for k in keys},
                                  Error_Bounds={k: (0.5, 2.0) for k in keys}, Error_Generated={k: 1.0 for k in keys},
                                  Error_Generated_Range={k: (0.75, 1.25) for k in keys}, Releases=1, Above_Real_Of=1)
    rows = {row[1]: row for row in Summary_Rows(calibration) if row[0] == 'bound'}
    assert [rows[key][3] for key in ('residual_max', 'run_mean_max', 'jump_max', 'step_max', 'Offset_Bound', 'White_Bound')] == \
        ['0.0559', '0.0552', '0.0894', '0.0295', '0.07', '0.04']


def test_the_summary_marks_each_statistic_inside_or_outside_its_bound(tmp_path):
    """
    Description:
        The observation summary (SPEC §9.9): a generated value (the median over the synthetic sets) inside its
        bound is 'true', outside 'false', and a NaN has no verdict; every row has 10 columns, the error rows
        the sets' range; the file starts with the header of SPEC §9.9.
    """

    keys = CONST_ERROR_STATISTICS + CONST_ERROR_DIAGNOSTICS
    model = Observation_Model_Str(White_Std=np.ones(3), White_Dof=3.0, Drift_Std=np.ones(3), Drift_Time=0.1, Offset_Std=np.ones(3),
                                  Offset_Time=0.1, Offset_Dof=3.0)
    spread = {k: (0.0, 1.0) for k in ('White_Std_x', 'White_Std_y', 'White_Std_z', 'White_Dof', 'Drift_Std_x', 'Drift_Std_y', 'Drift_Std_z',
                                      'Drift_Time', 'Offset_Std_x', 'Offset_Std_y', 'Offset_Std_z', 'Offset_Time', 'Offset_Dof')}
    generated = {k: 1.0 for k in keys}
    generated['rms_median'], generated['jump_1'] = 3.0, float('nan')
    calibration = Calibration_Str(Estimate=model, Spread=spread, Error_Real={k: 1.0 for k in keys}, Error_Bounds={k: (0.5, 2.0) for k in keys},
                                  Error_Generated=generated, Error_Generated_Range={k: (0.75, 1.25) for k in keys}, Releases=1, Above_Real_Of=1)
    rows = {row[1]: row for row in Summary_Rows(calibration) if row[0] == 'error'}
    assert rows['rms_median'][7] == 'false' and rows['rms_p90'][7] == 'true' and rows['jump_1'][7] == ''
    assert rows['rms_p90'][8:] == ['0.75', '1.25'] and all(len(row) == 10 for row in Summary_Rows(calibration))
    path = str(tmp_path / 'summary.csv')
    Write_Observation_Summary(path, calibration)
    assert open(path, encoding='utf-8').readline() == 'section,statistic,rate,real,low,high,generated,inside,generated_min,generated_max\n'


# The quantities the estimator identifies at 45 flights (F6 closure, 2026-11-11): on 20 replicate sets of the 45
# identification patterns with the configured model as the truth, refitted, their p10-p90 lay within +-15 % of the
# truth (the other parameters and derived quantities did not; KNOWN_ISSUES): the within-run structure
# s_w^2 + s_c^2 (1 - exp(-tau / T_c)) at (tau [s], axis).
#   Unit [Tuple<Tuple<float, int>>]
CONST_IDENTIFIED = ((0.020, 0), (0.020, 2), (0.080, 2))


@pytest.mark.slow
def test_the_estimator_recovers_what_45_flights_identify(root, configurations):
    """
    Description:
        The replicate study: 20 synthetic sets of the 45 identification
        flights' own frames (seeds [20261111, 9, r]), the truth the configured Observation.*, refitted like the
        real residuals; for each quantity identified at this size (CONST_IDENTIFIED) the median estimate lies
        within 10 % of the truth.
    """

    configuration = configurations[0]
    o = configuration.Observation
    truth = Observation_Model_Str(White_Std=np.array(o.White_Std), White_Dof=o.White_Dof, Drift_Std=np.array(o.Drift_Std),
                                  Drift_Time=o.Drift_Time, Offset_Std=np.array(o.Offset_Std), Offset_Time=o.Offset_Time,
                                  Offset_Dof=o.Offset_Dof)
    flight = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -configuration.Flight.Gravity]), Drag=configuration.Flight.Drag)
    flights = Identification_Flights(os.path.join(root, 'Data', 'Recordings'))
    fitted = Fit_Initial_States(flights, Flight_Model_Str=flight)
    estimates = [Estimate_Observation_Model(Synthetic_Residuals(flights, fitted, [20261111, 9, r], Flight_Model_Str=flight,
                                                                Observation_Model_Str=truth)) for r in range(20)]

    def Structure(model, tau, axis):
        return model.White_Std[axis] ** 2 + model.Drift_Std[axis] ** 2 * (1.0 - np.exp(-tau / model.Drift_Time))

    for tau, axis in CONST_IDENTIFIED:
        median = np.median([Structure(m, tau, axis) for m in estimates])
        assert median == pytest.approx(Structure(truth, tau, axis), rel=0.10), (tau, axis)
