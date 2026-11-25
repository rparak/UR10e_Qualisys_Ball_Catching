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
# File Name: ../test/Plot/test_core.py
## =========================================================================== ##

# System (Default)
import os
# JSON (a dataset's manifest of the fixture)
import json
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Flight/Core - the shipped flight model
from Flight.Core import Flight_Model_Str
#   ../Prediction/Core - an exact trajectory, the predictor of a constructed real flight
from Prediction.Core import Trajectory_From_State
#   ../Robot/Core - the robot in the scene and its tool centre point
from Robot.Core import Forward_Kinematics
from Scene.Core import Scene_Robot
#   ../Catch/Interception, Physical - the rows a record carries
from Catch.Interception import CONST_PER_THROW_COLUMNS as CONST_INTERCEPTION_COLUMNS, CONST_MOVING_COLUMNS as CONST_INTERCEPTION_MOVING
from Catch.Physical import CONST_PER_THROW_COLUMNS as CONST_CATCH_COLUMNS, CONST_MOVING_COLUMNS as CONST_CATCH_MOVING
#   ../View/Core - a recorded throw
from View.Core import Throw_Record_Str
#   ../Plot/Core
from Plot.Core import (Load_Comparison, Load_Sweep, Load_Throw, Load_Horizon_Ranges, Choose_Throw, Real_Predictions, Motion_From_Records,
                       Training_Curves, Read_Captions, Write_Captions, Plot_Error_Cls)
#   test/scripts/Script_Helpers - the constructed stored results
from Script_Helpers import _Plot_Results

"""
Description:
    Initialization of constants.

    The figures' data (the visualisation, delivery 2; SPEC §7.14): what the loaders refuse (a missing file, a table
    without a value the figure shows, a throw outside the validation partition before its dataset's file is opened), the
    predictions on a real flight, the motion from two records, the captions' file. Expectations come from constructed
    files, flights and records.
"""
# Per comparison set its table, partition and dataset, as scripts/plots.py names them.
#   Unit [Dict<string, Tuple<string, string, string>>]
CONST_TABLES = {'validation': ('Final_Validation', 'validation', 'Default'), 'held_out': ('Final_Test', 'test', 'Default'),
                'real': ('Final_Test', 'real', '')}
# The ready pose (Robot.Ready_Pose, SPEC §8.1).
#   Unit [rad]
CONST_READY = np.radians([29.972673, -104.916569, -88.564625, -57.375802, -61.349913, 0.0])


def test_a_missing_stored_file_is_refused_naming_it(tmp_path):
    """
    Description:
        A figure's file that does not exist is a Plot_Error_Cls naming its path, not a traceback.
    """

    with pytest.raises(Plot_Error_Cls, match='Comparisons.Final_Validation.summary.csv does not exist'):
        Load_Comparison(str(tmp_path / 'Comparisons'), {'QT_BASIS': 'Learned', 'GLS_M4': 'Run'}, CONST_TABLES)


def test_a_comparison_without_a_method_at_every_n_is_refused_and_a_catch_summary_is_not_the_sweeps(tmp_path):
    """
    Description:
        A run name absent from compare's tables is refused naming the table, the run and the N; the static catch's
        summary.csv read as the sweep's is refused (it has no error source).
    """

    _Plot_Results(str(tmp_path))
    with pytest.raises(Plot_Error_Cls, match=r'holds no remainder_rms_median_mm of Missing on validation Default at N \[10, 20, 40\]'):
        Load_Comparison(str(tmp_path / 'Comparisons'), {'QT_BASIS': 'Learned', 'GLS_M4': 'Missing'}, CONST_TABLES)
    with pytest.raises(Plot_Error_Cls, match='is not the sweep'):
        Load_Sweep(str(tmp_path / 'Catches' / 'Learned' / 'validation' / 'summary.csv'))


@pytest.mark.parametrize('index', [0, 4, 9])
def test_a_throw_outside_the_validation_partition_is_refused_before_the_datasets_file_is_opened(tmp_path, index):
    """
    Description:
        The partition comes from the manifest's partition sizes (5 / 3 / 2 here): a train throw and a test throw are refused
        naming the validation flights, and flights.npz (absent here) is never opened.
    """

    (tmp_path / 'manifest.json').write_text(json.dumps({'partition_sizes': [5, 3, 2]}), encoding='utf-8')
    with pytest.raises(Plot_Error_Cls, match='is not in its validation partition .the flights 5 to 7.'):
        Load_Throw(str(tmp_path), index, {})


def test_the_real_predictions_of_an_exact_predictor_have_no_error_and_the_path_starts_at_its_prefix(configuration):
    """
    Description:
        A flight sampled from the flight model and predicted by the same model's trajectory from its first sample: the error
        at every N is zero (to 1e-6 mm), its horizons are the times after the prefix's last sample, the path shown runs from
        the shown N's t_0 to the floor; a flight too short for the largest N is refused.
    """

    model = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -configuration.Flight.Gravity]), Drag=configuration.Flight.Drag,
                             Drag_Std=configuration.Flight.Drag_Std)
    truth = Trajectory_From_State(np.array([0.0, -4.0, 1.2, 0.1, 5.0, 5.0]), 0.0, 0.0, Flight_Model_Str=model)
    t = np.arange(0.0, 0.8, 0.01)
    position = truth.State(t)[:, :3]

    def Exact(time, observed, rate):
        return Trajectory_From_State(truth.State([time[-1]])[0], time[-1], time[-1], Flight_Model_Str=model)

    real = Real_Predictions('F', 100.0, t, position, {'GLS_M4': Exact}, [10, 20], 20)
    for n in (10, 20):
        horizon, error = real.Error[('GLS_M4', n)]
        assert np.allclose(horizon, t[n:] - t[n - 1]) and np.max(error) < 1e-6
    path_t, path = real.Path['GLS_M4']
    assert path_t[0] == pytest.approx(t[19]) and np.allclose(path[0], position[19]) and path[-1, 2] < position[-1, 2]
    with pytest.raises(Plot_Error_Cls, match='has 15 samples'):
        Real_Predictions('F', 100.0, t[:15], position[:15], {'GLS_M4': Exact}, [10, 20], 20)


def _Record(scene, velocity, moving=True, decision='go'):
    """
    Description:
        A constructed record: the robot holding the ready pose, the ball flying at a constant velocity from (0.5, -2, 1.5),
        states every 0.00025 s for 1 s, a stroke's interception row (the first plan at 0.1 s, the arrival 0.3 s, the stroke
        0.5-0.9 s) and a catch row (the first cup contact at 0.7 s, a relative speed of 5.1 m/s).
    """

    t = np.arange(0.0, 1.0, 0.00025)
    robot = Scene_Robot(scene)
    states = np.zeros((len(t), scene.nq + 3))
    states[:, robot.Position_Address] = CONST_READY
    states[:, scene.nq:] = np.array([0.5, -2.0, 1.5]) + np.outer(t, velocity)
    interception = dict.fromkeys(CONST_INTERCEPTION_COLUMNS + CONST_INTERCEPTION_MOVING, '')
    interception.update(arrival_s=0.3, stroke_start_s=0.5, stroke_end_s=0.9)
    catch = dict.fromkeys(CONST_CATCH_COLUMNS + CONST_CATCH_MOVING, '')
    catch.update(cup_contact_s=0.7, contact_speed_m_s=5.1, first_contact='cup_bottom')
    return Throw_Record_Str(Dataset='D', Flight=3, Decision=decision, Stroke=moving, Class='caught', Plan_Time=np.array([0.1]), State_Time=t,
                            State_Position=states, Robot_Address=robot.Position_Address, Ball_Address=scene.nq,
                            Interception_Row=tuple(interception.values()), Catch_Row=tuple(catch.values()))


def test_the_motion_of_a_cup_at_rest_and_a_ball_at_constant_velocity_and_its_phases_from_the_rows(scene):
    """
    Description:
        At every control period (0.002 s): the cup's speed along the ball 0 (the robot at rest), the ball's speed its
        velocity's norm, the distance from the mouth (the tool centre point at the ready pose) to the ball; the phases from
        the rows: move 0.1-0.3, wait 0.3-0.5, the stroke from 0.5 to the cup's largest speed along the ball (within the
        stroke; at rest here, so anywhere in it), the braking from there to the stroke's end 0.9, the rest to the last state;
        the contact, its part and the relative speed from the catch row; at each mode's contact (the backward difference of
        the two states before it) the ball's speed, the cup's 0 and their relative speed; the ball's pass nearest the mouth.
    """

    velocity = np.array([0.0, 5.0, -1.0])
    data = mujoco.MjData(scene)
    robot = Scene_Robot(scene)
    motion = Motion_From_Records({'moving': _Record(scene, velocity), 'static': _Record(scene, velocity, moving=False)}, scene, data, robot)
    t = motion.Time['moving']
    assert np.allclose(np.diff(t), 0.002) and t[0] == 0.0
    mouth = Forward_Kinematics(scene, data, CONST_READY, Robot_Str=robot)[0]
    ball = np.array([0.5, -2.0, 1.5]) + np.outer(t, velocity)
    for mode in ('moving', 'static'):
        assert np.allclose(motion.Cup_Speed[mode], 0.0, atol=1e-9) and np.allclose(motion.Ball_Speed[mode], np.linalg.norm(velocity))
        assert np.allclose(motion.Distance[mode], np.linalg.norm(ball - mouth, axis=1))
    phases = [(n, round(a, 6), round(b, 6)) for n, a, b in motion.Phases]
    assert phases[:3] == [('observe', 0.0, 0.1), ('move', 0.1, 0.3), ('wait', 0.3, 0.5)] and phases[5] == ('rest', 0.9, round(t[-1], 6))
    assert phases[3][:2] == ('stroke', 0.5) and phases[4][0::2] == ('braking', 0.9) and 0.5 <= phases[3][2] == phases[4][1] <= 0.9
    for mode in ('moving', 'static'):
        time, ball_speed, cup, relative = motion.At_Contact[mode]
        assert time == pytest.approx(0.7 - 0.00025) and ball_speed == pytest.approx(np.linalg.norm(velocity)) and cup == pytest.approx(0.0, abs=1e-9)
        assert relative == pytest.approx(np.linalg.norm(velocity))
    distance = np.linalg.norm(ball - mouth, axis=1)
    assert motion.Mouth_Pass == pytest.approx(t[t <= 0.7][np.argmin(distance[t <= 0.7])]) and motion.Contact_Part == 'cup_bottom'
    assert motion.Contact == 0.7 and motion.Relative_Speed == 5.1 and motion.Class == 'caught'
    assert np.allclose(motion.Joints, CONST_READY) and np.allclose(motion.Joint_Speeds, 0.0)


def test_the_motion_is_refused_for_a_go_without_a_stroke_and_for_a_no_go(scene):
    """
    Description:
        The figure shows a go caught with a stroke: a moving record without one, or a no-go, is refused naming the throw.
    """

    data, robot = mujoco.MjData(scene), Scene_Robot(scene)
    with pytest.raises(Plot_Error_Cls, match='The throw 3 of D is a go without a stroke'):
        Motion_From_Records({'moving': _Record(scene, [0.0, 5.0, 0.0], moving=False)}, scene, data, robot)
    with pytest.raises(Plot_Error_Cls, match='The throw 3 of D is a no-go'):
        Motion_From_Records({'moving': _Record(scene, [0.0, 5.0, 0.0], decision='no-go')}, scene, data, robot)


def test_a_run_without_a_seeds_epochs_is_refused(tmp_path):
    """
    Description:
        A folder without seed_<s>/ (a numerical run) has no learning curves.
    """

    with pytest.raises(Plot_Error_Cls, match='holds no seed_<s>/epochs.csv'):
        Training_Curves(str(tmp_path))


def test_the_captions_are_read_back_as_written_and_another_file_is_refused(tmp_path):
    """
    Description:
        captions.csv round trip (a caption with commas and a quote), and a file with another header refused.
    """

    rows = [('comparison_dark.png', 'QT_BASIS against GLS_M4: (a), (b) and "c".'), ('README/readme_motion_light.png', 'The motion.')]
    Write_Captions(str(tmp_path / 'captions.csv'), rows)
    assert Read_Captions(str(tmp_path / 'captions.csv')) == rows
    (tmp_path / 'other.csv').write_text('name,text\n', encoding='utf-8')
    with pytest.raises(Plot_Error_Cls, match='does not have the columns file, caption'):
        Read_Captions(str(tmp_path / 'other.csv'))


def _Errors(rms, end):
    """
    Description:
        Per flight the N values 10, 20, 40 with a remainder RMS and an error at the end [mm].
    """

    return {n: (r, e) for n, r, e in zip((10, 20, 40), rms, end)}


def test_the_prediction_throw_is_the_ordered_one_nearest_the_test_medians_and_none_is_refused():
    """
    Description:
        Throws whose remainder RMS or end error does not fall from each N to the next for both methods are left out (7: the
        learned predictor's RMS rises at 40; 8: the numerical one's end error rises; 9: no end error, no prediction), and so
        is a throw with a gap between its observed samples over 0.1 s (4, otherwise the nearest); of the ordered throws
        the one nearest the medians (sum of |ln(RMS / median)|) is chosen (5 against 6, which is twice as far); its
        score is returned; without an ordered throw the choice is refused.
    """

    medians = {'QT_BASIS': {10: 60.0, 20: 36.0, 40: 22.0}, 'GLS_M4': {10: 58.0, 20: 35.0, 40: 21.0}}
    rms = {'QT_BASIS': {4: [60, 36, 22], 5: [61, 37, 23], 6: [120, 72, 44], 7: [60, 36, 40], 8: [60, 36, 22], 9: [60, 36, 22]},
           'GLS_M4': {4: [58, 35, 21], 5: [58, 35, 21], 6: [58, 35, 21], 7: [58, 35, 21], 8: [58, 35, 21], 9: [58, 35, 21]}}
    end = {'QT_BASIS': {f: [90, 40, 20] for f in (4, 5, 6, 7, 8)}, 'GLS_M4': {4: [90, 40, 20], 5: [90, 40, 20], 6: [90, 40, 20], 7: [90, 40, 20],
                                                                              8: [90, 40, 45]}}
    flat = lambda table: {m: {(f, n): v for f, values in per.items() for n, v in zip((10, 20, 40), values)} for m, per in table.items()}
    gaps = {4: 0.11, 5: 0.1, 6: 0.02, 7: 0.02, 8: 0.02, 9: 0.02}
    index, score = Choose_Throw(flat(rms), flat(end), gaps, medians, [10, 20, 40], 'D')
    assert index == 5 and score == pytest.approx(np.sum(np.abs(np.log(np.array([61, 37, 23]) / np.array([60.0, 36.0, 22.0])))))
    del rms['QT_BASIS'][5], rms['QT_BASIS'][6]
    with pytest.raises(Plot_Error_Cls, match='No validation throw of D without a gap over 0.1 s has its errors falling'):
        Choose_Throw(flat(rms), flat(end), gaps, medians, [10, 20, 40], 'D')


def test_a_horizon_set_without_its_rows_is_refused(tmp_path):
    """
    Description:
        The final evaluation's summary.csv without a set's rows at the N asked for is refused naming the partition and N.
    """

    _Plot_Results(str(tmp_path))
    with pytest.raises(Plot_Error_Cls, match=r'holds no horizon range of test Default at N 40'):
        Load_Horizon_Ranges({'QT_BASIS': str(tmp_path / 'Evaluations' / 'Learned' / 'test')}, {'held_out': ('test', 'Default')}, 40)

