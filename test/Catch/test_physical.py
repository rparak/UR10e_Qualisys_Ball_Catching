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
# File Name: ../test/Catch/test_physical.py
## =========================================================================== ##


# Dataclasses (Comparing two results)
import dataclasses
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Flight/Core - the flight model and the reference flight
from Flight.Core import Flight_Model_Str, Simulate
#   ../Dataset/Core - a release velocity that reaches a point at a time
from Dataset.Core import Shoot
#   ../Robot/Core - the cup's pose at the ready pose and the joint rates of a motion along a line
from Robot.Core import Forward_Kinematics, Line_Rates
#   ../Robot/Motion - a profile, a stroke and the interception's tracking
from Robot.Motion import Profile, Stroke, Track
#   ../Catch/Core - the catch's error
from Catch.Core import Catch_Error_Cls
#   ../Catch/Interception - a plan, its ticks and the interception's columns
from Catch.Interception import (Plan_Str, Plan_Ticks, CONST_PER_THROW_COLUMNS as CONST_INTERCEPTION_COLUMNS,
                                CONST_MOVING_COLUMNS as CONST_INTERCEPTION_MOVING_COLUMNS)
#   ../Catch/Physical
from Catch.Physical import (Rebound, Mean_Rebound, Catch_Scene, Set_Contact, Simulate_Catch, Classify, Catch_Row, Catch_Chunk, Ball_In_Cup,
                            Marker_Orientation,
                            Summary_Rows,
                            Write_Per_Throw, Write_Summary, Contact_Str, Catch_Result_Str, _Read_Contacts, CONST_CLASSES, CONST_PER_THROW_COLUMNS,
                            CONST_SUMMARY_COLUMNS, CONST_SWEEP_COLUMNS, CONST_MOVING_COLUMNS, CONST_MOVING_SUMMARY_COLUMNS)

"""
Description:
    Initialization of constants.

    The physical catch (SPEC §2.10, §7.11; F11): the catch's scene, the ball's
    contact, F11's own loop on constructed shots into the cup held at the ready pose, the classes, the summary and
    the files. Expectations come from the geometry of the shots, the flight model and constructed rows.
"""
# The ready pose (Robot.Ready_Pose, SPEC §8.1), the robot's speed limits and nominal acceleration.
#   Unit [rad], [rad/s], [rad/s^2]
CONST_READY = np.radians([29.972673, -104.916569, -88.564625, -57.375802, -61.349913, 0.0])
CONST_SPEED = np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0])
CONST_ACCELERATION = np.radians(800.0)
# The catch's shipped settings (Contact, SPEC §8.1): the step, the nominal contact, the friction; and the levels with the
# restitution each gives at 7 m/s on the floor (the addendum of F11's Part A: the mean over the phase of the first
# contact within a step and the range over it; the bands below hold them with 0.01 to spare).
#   Unit [s], [Tuple<float, float>: s, -], [Vector<float> 1x3], [Dict<Tuple<float, float>, Tuple<float, float>>]
CONST_STEP = 0.00025
CONST_NOMINAL = (0.004, 0.7)
CONST_FRICTION = (0.5, 0.005, 0.0001)
CONST_RESTITUTION = {(0.004, 0.7): (0.17, 0.22), (0.00157, 0.9): (0.10, 0.17), (0.004, 0.5): (0.25, 0.32), (0.004, 0.3): (0.40, 0.48)}
# The ball's radius, the cup's length from the flange to the mouth and the height of its bottom's face (SPEC §4.4).
#   Unit [m]
CONST_BALL_RADIUS = 0.030
CONST_CUP_LENGTH = 0.190
CONST_CUP_BOTTOM = 0.005


def _Contact(settle=1.0):
    return Contact_Str(Step=CONST_STEP, Friction=CONST_FRICTION, Settle_Time=settle, Rest_Speed=0.05, Rest_Time=0.2, Levels=[CONST_NOMINAL])


@pytest.fixture(scope='module')
def flight(configuration):
    """
    Description:
        The shipped flight model.
    """

    return Flight_Model_Str(Gravity=np.array([0.0, 0.0, -configuration.Flight.Gravity]), Drag=configuration.Flight.Drag,
                            Drag_Std=configuration.Flight.Drag_Std)


@pytest.fixture(scope='module')
def catch_scene(configuration, flight):
    """
    Description:
        The catch's scene with the shipped step and friction, built once for this module (0.3 s); a test sets its
        contact first.
    """

    return Catch_Scene(configuration.Areas, _Contact(), Flight_Model_Str=flight)


@pytest.fixture(scope='module')
def mouth(catch_scene):
    """
    Description:
        The centre of the cup's mouth at the ready pose, the cup's axis out of the mouth and a horizontal direction
        across it.
    """

    point, orientation = Forward_Kinematics(catch_scene.Model, mujoco.MjData(catch_scene.Model), CONST_READY, Robot_Str=catch_scene.Robot)
    axis = orientation[:, 2]
    across = np.cross(axis, [0.0, 0.0, 1.0])
    return point, axis, across / np.linalg.norm(across)


def _Shot(catch_scene, mouth, lateral, contact=CONST_NOMINAL, settle=1.0, speed=3.0):
    """
    Description:
        A ball released 50 mm outside the mouth plane, `lateral` beside the cup's axis, at `speed` into the cup along
        its axis; the robot holds the ready pose (no plan), the catch time is the release.
    """

    point, axis, across = mouth
    Set_Contact(catch_scene, *contact)
    state = np.concatenate([point + 0.05 * axis + lateral * across, -speed * axis])
    return Simulate_Catch(catch_scene, state, Plan_Str(), CONST_READY, 0.0, Contact_Str=_Contact(settle))


def test_a_ball_shot_along_the_cups_axis_lands_on_the_bottom_and_is_caught(catch_scene, mouth, flight):
    """
    Description:
        A ball shot at 3 m/s along the axis of the cup held at the ready pose, with the nominal contact: it meets the
        cup's bottom first, after the 0.205 m from 50 mm outside the mouth to the bottom's face under gravity's part
        along the axis (0.0629 s), was inside, is retained at the end of the 0.9 s window and has come to rest; the
        robot touches nothing, no servo is at its limit. It is classified caught.
    """

    result = _Shot(catch_scene, mouth, 0.0, settle=0.9)
    assert result.Retained and result.At_Rest and result.Entered and result.Robot_Contact == '' and result.Force_Limit_Ticks == 0
    assert result.First_Contact == 'cup_bottom' and result.Cup_Contact_Time == result.First_Contact_Time
    along = -flight.Gravity[2] * mouth[1][2]
    distance = 0.05 + CONST_CUP_LENGTH - CONST_CUP_BOTTOM - CONST_BALL_RADIUS
    expected = (-3.0 + np.sqrt(9.0 + 2.0 * along * distance)) / along
    assert abs(result.First_Contact_Time - expected) < 0.001
    assert 0.25 < result.Rest_Time < 0.9 and result.End_Speed < 0.01 and 0.002 < result.Penetration < 0.008
    assert result.Entry_Lateral < 0.002
    assert Classify('go', True, False, result) == 'caught'


def test_a_bouncy_ball_leaves_the_cup_and_a_short_window_separates_retained_from_at_rest(catch_scene, mouth):
    """
    Description:
        The same shot with a contact of restitution 0.70 (time constant 0.004 s, damping ratio 0.12): the ball was
        inside, touched the cup and is not retained, bounced out. With the nominal contact and a settle window of 0.5 s
        the ball is retained but not yet at rest: the two measures are separate, and the window ends the simulation.
    """

    bouncy = _Shot(catch_scene, mouth, 0.0, contact=(0.004, 0.12))
    assert not bouncy.Retained and not bouncy.At_Rest and bouncy.Entered and bouncy.First_Contact == 'cup_bottom'
    assert Classify('go', True, True, bouncy) == 'bounced_out'
    short = _Shot(catch_scene, mouth, 0.0, settle=0.5)
    assert short.Retained and not short.At_Rest and np.isnan(short.Rest_Time)


def test_a_ball_on_the_lip_on_a_link_on_the_stand_and_beside_the_robot_is_classified(catch_scene, mouth):
    """
    Description:
        Shots parallel to the cup's axis: 80 mm beside it the ball meets the lip first, after the 20 mm to the mouth
        plane (0.0066 s), is never inside and is on the rim; 150 mm beside it it meets a link of the robot first and
        never the cup; 0.6 m beside it towards the robot's base it meets the stand first; 0.6 m beside it on the other
        side it meets the floor only and is missed. None is retained.
    """

    rim = _Shot(catch_scene, mouth, 0.080)
    assert rim.First_Contact == 'cup_lip' and not rim.Entered and not rim.Retained and abs(rim.Cup_Contact_Time - 0.0066) < 0.001
    assert Classify('go', True, True, rim) == 'rim'
    link = _Shot(catch_scene, mouth, 0.150)
    assert link.First_Contact == 'robot' and np.isnan(link.Cup_Contact_Time) and not link.Retained
    assert Classify('go', True, True, link) == 'hit_robot'
    stand = _Shot(catch_scene, mouth, 0.6)
    assert stand.First_Contact == 'stand' and np.isnan(stand.Cup_Contact_Time) and not stand.Retained
    assert Classify('go', True, True, stand) == 'hit_stand'
    beside = _Shot(catch_scene, mouth, -0.6)
    assert beside.First_Contact == 'floor' and np.isnan(beside.Cup_Contact_Time) and not beside.Entered and not beside.Retained
    assert Classify('go', True, True, beside) == 'missed'


def test_the_step_callback_sees_the_start_and_every_step_and_changes_nothing(catch_scene, mouth):
    """
    Description:
        The per-step callback of the catch's loop (the view records
        through it): for the shot along the axis with a settle window of 0.3 s it is called once at the start and once after
        each of the 1 200 steps of 0.00025 s, with the time and a copy of the scene's 13 positions, each call its own
        (none shares memory with the scene's data or with another call's); the first copy has the ball at its release
        point, the second 0.75 mm further along the cup's axis (one step at 3 m/s), the last 0.205 m further within
        15 mm (on the bottom), and the robot's six joints stay at the ready pose within 2 mrad. The result is the
        same, field for field, with and without the callback.
    """

    point, axis, across = mouth
    Set_Contact(catch_scene, *CONST_NOMINAL)
    state = np.concatenate([point + 0.05 * axis, -3.0 * axis])
    plain = Simulate_Catch(catch_scene, state, Plan_Str(), CONST_READY, 0.0, Contact_Str=_Contact(0.3))
    seen = []
    watched = Simulate_Catch(catch_scene, state, Plan_Str(), CONST_READY, 0.0, Contact_Str=_Contact(0.3),
                             on_step=lambda t, q: seen.append((t, q)))
    assert plain.Retained and dataclasses.asdict(plain) == dataclasses.asdict(watched)
    times = np.array([t for t, _ in seen])
    assert len(seen) == 1201 and times[0] == 0.0 and np.allclose(np.diff(times), CONST_STEP, rtol=0.0, atol=1e-12)
    assert all(q.shape == (13,) for _, q in seen)
    assert not any(np.shares_memory(q, catch_scene.Data.qpos) for _, q in seen) and not np.shares_memory(seen[1][1], seen[2][1])
    ball, robot = catch_scene.Ball.Position_Address, catch_scene.Robot.Position_Address
    assert np.allclose(seen[0][1][ball:ball + 3], state[:3], rtol=0.0, atol=1e-12)
    assert abs((seen[1][1][ball:ball + 3] - state[:3]) @ -axis - 3.0 * CONST_STEP) < 5e-6
    travelled = (seen[-1][1][ball:ball + 3] - state[:3]) @ -axis
    assert abs(travelled - (0.05 + CONST_CUP_LENGTH - CONST_CUP_BOTTOM - CONST_BALL_RADIUS)) < 0.015
    assert all(np.max(np.abs(q[robot] - CONST_READY)) < 0.002 for _, q in seen[::100])


def test_the_ball_is_in_the_cup_below_the_mouth_above_the_flange_and_within_the_opening(catch_scene, mouth):
    """
    Description:
        The one definition of *inside* (the catch's loop and the view's overlay), with the robot at the ready pose and
        the ball placed by hand: its centre 10 mm below the mouth's plane on the axis is inside, 10 mm above it is
        not; 100 mm below the plane it is inside up to the opening's 62.5 mm from the axis (62 mm inside, 63 mm not),
        and the distance from the axis is returned; 185 mm below the plane (5 mm above the flange) is inside, 195 mm
        (below the flange) is not.
    """

    point, axis, across = mouth
    model, data = catch_scene.Model, catch_scene.Data
    mujoco.mj_resetData(model, data)
    data.qpos[catch_scene.Robot.Position_Address] = CONST_READY
    address = catch_scene.Ball.Position_Address

    def At(along, lateral):
        data.qpos[address:address + 3] = point - along * axis + lateral * across
        mujoco.mj_kinematics(model, data)
        return Ball_In_Cup(catch_scene)

    assert At(0.010, 0.0)[0] and not At(-0.010, 0.0)[0]
    assert At(0.100, 0.062)[0] and not At(0.100, 0.063)[0]
    assert abs(At(0.100, 0.062)[1] - 0.062) < 1e-9 and abs(At(0.050, 0.020)[1] - 0.020) < 1e-9
    assert At(0.185, 0.0)[0] and not At(0.195, 0.0)[0]


def test_a_catch_row_classifies_a_no_go_by_its_truth_and_a_go_by_its_simulation(catch_scene, mouth):
    """
    Description:
        The catch's row from an interception row (one function for 'catch' and for the view's stored check): a no-go
        takes its class from the truth (catchable: no_go_catchable, missed; not catchable: no_go, outside the
        workspace), is neither retained nor at rest and leaves the 11 columns of a simulation empty; a go takes the
        simulation's values, the rest time counted from the catch time. Every row has the 25 columns of the file, the
        dataset, the flight, the seed, the acceleration and the contact first.
    """

    def Interception(decision, catchable, catch_time=''):
        row = dict.fromkeys(CONST_INTERCEPTION_COLUMNS, '')
        row.update(dataset='Catch_In', flight='7', seed='0', acceleration_deg_s2='800', catchable=catchable, decision=decision,
                   catch_time_s=catch_time, intercepted='true' if decision == 'go' else '')
        return tuple(row[name] for name in CONST_INTERCEPTION_COLUMNS)

    column = {name: k for k, name in enumerate(CONST_PER_THROW_COLUMNS)}
    missed = Catch_Row(Interception('no-go', 'true'), 0.004, 0.7, False, None)
    outside = Catch_Row(Interception('no-go', 'false'), 0.004, 0.7, False, None)
    assert len(missed) == len(outside) == len(CONST_PER_THROW_COLUMNS) == 25
    assert missed[:10] == ('Catch_In', '7', '0', '800', 0.004, 0.7, 'true', 'no-go', 'false', '')
    assert (missed[column['class']], missed[column['outcome']]) == ('no_go_catchable', 'missed')
    assert (outside[column['class']], outside[column['outcome']]) == ('no_go', 'outside the workspace')
    assert missed[column['retained']:column['at_rest'] + 1] == ('false', 'false') and missed[column['at_rest'] + 1:] == ('',) * 11
    result = _Shot(catch_scene, mouth, 0.0, settle=0.9)
    caught = Catch_Row(Interception('go', 'true', 0.25), 0.004, 0.7, True, result)
    assert len(caught) == 25 and caught[column['intercepted']] == 'true' and caught[column['catch_time_s']] == 0.25
    assert (caught[column['class']], caught[column['outcome']], caught[column['retained']], caught[column['at_rest']]) == \
        ('caught', 'caught', 'true', 'true')
    assert caught[column['rest_s']] == result.Rest_Time - 0.25 and caught[column['first_contact']] == 'cup_bottom'
    assert caught[column['first_contact_s']] == result.First_Contact_Time and caught[column['penetration_m']] == result.Penetration
    assert caught[column['entered']] == 'true' and caught[column['robot_contact']] == '' and caught[column['force_limit_ticks']] == 0


def _Arriving(catch_scene, mouth, flight, lateral, time=0.40, speed=7.0):
    """
    Description:
        A throw under the shipped flight model that passes, `time` after its release, the point `lateral` beside the
        centre of the mouth of the cup at the ready pose, at about `speed` against the cup's axis (released where a ball
        without drag would have come from; the release velocity by Dataset.Core.Shoot): its release state, and its
        state at that time.
    """

    point, axis, across = mouth
    aim = point + lateral * across
    release = aim + speed * axis * time + 0.5 * flight.Gravity * time ** 2
    velocity, converged = Shoot(release[None], aim[None], np.array([time]), Flight_Model_Str=flight)
    assert converged[0]
    state = np.concatenate([release, velocity[0]])
    return state, Simulate(state, np.array([time]), Flight_Model_Str=flight)[0]


def test_a_cup_moving_with_the_ball_holds_a_ball_that_the_cup_at_rest_loses_at_a_restitution_of_0_29(catch_scene, mouth, flight):
    """
    Description:
        F15, with its forced failure. A throw arrives at 6.9 m/s within 0.1 deg of the axis
        of the cup at the ready pose, 20 mm beside it, 0.4 s after its release. The cup at rest (no plan) retains it
        with the nominal contact (restitution 0.20: the checks of F11 pass on it) and loses it at 0.29, bounced out: the
        case the moving catch exists for. On a stroke built here from the robot's functions alone (the joint rates of
        the arrival direction at the ready pose, the largest speed the joint speeds allow, about 1.9 m/s; the peak
        0.155 / (v - w) after the arrival, at the joints a cup moving at w would have then; braking at once), the robot
        starting at rest at the stroke's start, the ball is retained at 0.20 and at 0.29. The relative speed at the
        first cup contact is the ball's own at rest (7.0 to 7.2 m/s: it gains under gravity after the arrival) and on
        the stroke between v - w and 0.35 m/s above it (the wall is met a little before the peak, the cup not yet at
        w), more than 1.5 m/s below the cup at rest; the robot touches nothing but the ball, no servo is at its limit
        and the joints stay within 5 mrad of the stroke. A ball that never meets the cup has no contact speed (NaN).
    """

    state, arriving = _Arriving(catch_scene, mouth, flight, 0.020)
    v = float(np.linalg.norm(arriving[3:]))
    assert 6.8 < v < 7.0 and np.degrees(np.arccos(-arriving[3:] @ mouth[1] / v)) < 0.1
    rates = np.append(Line_Rates(catch_scene.Model, mujoco.MjData(catch_scene.Model), CONST_READY, arriving[3:], 5, Robot_Str=catch_scene.Robot), 0.0)
    w = float(np.min(CONST_SPEED[:5] / np.abs(rates[:5])))
    inside = (CONST_CUP_LENGTH - CONST_CUP_BOTTOM - CONST_BALL_RADIUS) / (v - w)
    stroke = Stroke(CONST_READY + rates * w * inside, rates * w, CONST_SPEED, CONST_ACCELERATION, 0.40 + inside)
    moving = Plan_Str(Profiles=[stroke], Stroke=True, Scale=1.0, Poses=[CONST_READY])
    assert 1.7 < w < 2.1 and stroke.Start_Time > 0.1
    results = {}
    for name, contact in (('0.20', CONST_NOMINAL), ('0.29', (0.004, 0.5))):
        Set_Contact(catch_scene, *contact)
        results[name] = (Simulate_Catch(catch_scene, state, Plan_Str(), CONST_READY, 0.40, Contact_Str=_Contact()),
                         Simulate_Catch(catch_scene, state, moving, CONST_READY, 0.40, Contact_Str=_Contact()))
    assert results['0.20'][0].Retained and results['0.20'][1].Retained
    assert not results['0.29'][0].Retained and Classify('go', True, True, results['0.29'][0]) == 'bounced_out'
    assert results['0.29'][1].Retained and Classify('go', True, True, results['0.29'][1]) == 'caught'
    for at_rest, on_stroke in results.values():
        assert at_rest.First_Contact.startswith('cup_') and on_stroke.First_Contact.startswith('cup_')
        assert 7.0 < at_rest.Contact_Speed < 7.2 and v - w < on_stroke.Contact_Speed < v - w + 0.35
        assert on_stroke.Contact_Speed < at_rest.Contact_Speed - 1.5
        assert on_stroke.Robot_Contact == '' and on_stroke.Force_Limit_Ticks == 0 and on_stroke.Tracking < 0.005
    assert np.isnan(_Shot(catch_scene, mouth, -0.6).Contact_Speed)


def test_a_moving_catchs_row_ends_with_the_plans_mode_the_strokes_scale_and_the_contact_speed(catch_scene, mouth, tmp_path):
    """
    Description:
        F15: from an interception row with the moving columns, the catch's row of a moving catch
        has the 25 columns of the static row and then the plan's mode and the stroke's scale (the interception's) and the
        relative speed at the first cup contact (the simulation's; for the shot of 3 m/s along the axis the speed at
        the bottom under gravity's part along the axis, within 0.03 m/s) and whether the row is the markers variant's:
        29 columns; a no-go leaves the first three empty; a static row has 25 whatever the interception row holds. The
        summary of moving rows has one row per contact and variant and ends with the goes simulated with a stroke, the
        median and 95th percentile of the contact speed and the variant, and the two files' headers end with the
        further columns only when written as a moving catch's.
    """

    def Interception(decision, mode='', scale=''):
        row = dict.fromkeys(CONST_INTERCEPTION_COLUMNS + CONST_INTERCEPTION_MOVING_COLUMNS, '')
        row.update(dataset='Catch_In', flight='7', seed='0', acceleration_deg_s2='800', catchable='true', decision=decision,
                   catch_time_s=0.25 if decision == 'go' else '', intercepted='true' if decision == 'go' else '', plan_mode=mode, stroke_scale=scale)
        return tuple(row[name] for name in CONST_INTERCEPTION_COLUMNS + CONST_INTERCEPTION_MOVING_COLUMNS)

    result = _Shot(catch_scene, mouth, 0.0, settle=0.9)
    fallen = np.sqrt(9.0 + 2.0 * 9.81 * mouth[1][2] * (0.05 + CONST_CUP_LENGTH - CONST_CUP_BOTTOM - CONST_BALL_RADIUS))
    assert CONST_MOVING_COLUMNS == ('plan_mode', 'stroke_scale', 'contact_speed_m_s', 'markers') and abs(result.Contact_Speed - fallen) < 0.03
    stroke = Catch_Row(Interception('go', 'moving', 0.75), 0.004, 0.7, True, result, moving=True)
    fallback = Catch_Row(Interception('go', 'static', 0.0), 0.004, 0.7, True, result, moving=True)
    no_go = Catch_Row(Interception('no-go'), 0.004, 0.7, False, None, moving=True)
    static = Catch_Row(Interception('go', 'moving', 0.75)[:len(CONST_INTERCEPTION_COLUMNS)], 0.004, 0.7, True, result)
    assert len(stroke) == len(fallback) == len(no_go) == 29 and len(static) == 25 and stroke[:25] == static
    assert stroke[25:] == ('moving', 0.75, result.Contact_Speed, 'false') and fallback[25:] == ('static', 0.0, result.Contact_Speed, 'false')
    assert no_go[25:] == ('', '', '', 'false')
    assert no_go[:25] == Catch_Row(Interception('no-go')[:len(CONST_INTERCEPTION_COLUMNS)], 0.004, 0.7, False, None)
    marked = Catch_Row(Interception('go', 'moving', 0.75), 0.004, 0.7, True, result, moving=True, markers=True)
    assert marked[25:] == ('moving', 0.75, result.Contact_Speed, 'true') and marked[:25] == stroke[:25]
    faster = dataclasses.replace(result, Contact_Speed=result.Contact_Speed + 1.0)
    rows = [stroke, fallback, no_go, Catch_Row(Interception('go', 'moving', 1.0), 0.004, 0.7, True, faster, moving=True)]
    both = Summary_Rows(rows + [marked], 'validation')
    assert len(both) == 2 and [row[-1] for row in both] == ['false', 'true'] and both[1][CONST_SUMMARY_COLUMNS.index('throws')] == 1
    summary = both[:1]
    assert len(summary[0]) == len(CONST_SUMMARY_COLUMNS) + len(CONST_MOVING_SUMMARY_COLUMNS)
    named = dict(zip(CONST_SUMMARY_COLUMNS + CONST_MOVING_SUMMARY_COLUMNS, summary[0]))
    assert (named['throws'], named['go'], named['retained'], named['strokes'], named['markers']) == (4, 3, 3, 2, 'false')
    assert named['contact_speed_p50_m_s'] == pytest.approx(result.Contact_Speed, abs=1e-12)
    assert named['contact_speed_p95_m_s'] == pytest.approx(result.Contact_Speed + 0.9, abs=1e-12)
    assert len(Summary_Rows([row[:25] for row in rows], 'validation')[0]) == len(CONST_SUMMARY_COLUMNS)
    Write_Per_Throw(str(tmp_path / 'moving.csv'), rows, False, True)
    Write_Per_Throw(str(tmp_path / 'static.csv'), [row[:25] for row in rows])
    Write_Summary(str(tmp_path / 'summary.csv'), summary, False, True)
    headers = [(tmp_path / name).read_text(encoding='ascii').splitlines()[0] for name in ('moving.csv', 'static.csv', 'summary.csv')]
    assert headers[1] == ','.join(CONST_PER_THROW_COLUMNS) and headers[0] == headers[1] + ',plan_mode,stroke_scale,contact_speed_m_s,markers'
    assert headers[2] == ','.join(CONST_SUMMARY_COLUMNS) + ',strokes,contact_speed_p50_m_s,contact_speed_p95_m_s,markers'


def test_the_markers_variant_puts_six_shapes_on_the_ball_in_memory_and_turns_it_per_throw(catch_scene, mouth, configuration, flight):
    """
    Description:
        F15: the markers variant's scene has the shipped scene's geoms and six more, on the
        ball: at the markers measured on its mesh (38 mm out on the axes, radius 8 mm, each within 0.1 mm), of
        negligible length, on the ball's contact bits (3 / 2), with the ball's friction, contact dimension and
        priority, and after Set_Contact the ball's contact on all seven; the ball's mass (0.122 kg) and inertia are
        the scene's. The scene without the variant has the one sphere, as before. A ball shot at 3 m/s along the cup's
        axis as the scene holds it (a marker on each axis) meets the cone's wall first instead of the bottom and comes
        to rest in the cup with its centre more than 5 mm higher than the bare ball's 30 mm above the bottom's face;
        both are retained. The orientation of a throw is a unit quaternion, the same for the same dataset and throw,
        another for another throw and for another dataset, and Simulate_Catch starts the ball turned so. The variant
        in a static catch is refused before anything is read.
    """

    marked = Catch_Scene(configuration.Areas, _Contact(), Flight_Model_Str=flight, markers=True)
    model, bare = marked.Model, catch_scene.Model
    assert model.ngeom == bare.ngeom + 6 and len(marked.Ball_Geoms) == 7 and catch_scene.Ball_Geoms == [catch_scene.Ball_Geom]
    assert marked.Ball_Geom == marked.Ball_Geoms[0] and model.geom_type[marked.Ball_Geom] == mujoco.mjtGeom.mjGEOM_SPHERE
    markers = marked.Ball_Geoms[1:]
    assert np.linalg.norm(model.geom_pos[markers], axis=1) == pytest.approx(0.038, abs=1e-4)
    assert sorted(int(np.argmax(np.abs(row))) for row in model.geom_pos[markers]) == [0, 0, 1, 1, 2, 2]
    assert model.geom_size[markers, 0] == pytest.approx(0.008, abs=1e-4) and np.all(model.geom_size[markers, 1] <= 1e-5)
    Set_Contact(marked, 0.004, 0.5)
    for geom in marked.Ball_Geoms:
        assert (model.geom_contype[geom], model.geom_conaffinity[geom], model.geom_condim[geom], model.geom_priority[geom]) == (3, 2, 6, 1)
        assert tuple(model.geom_friction[geom]) == CONST_FRICTION and tuple(model.geom_solref[geom]) == (0.004, 0.5)
    assert model.body_mass[marked.Ball.Body] == bare.body_mass[catch_scene.Ball.Body] == 0.122
    assert np.array_equal(model.body_inertia[marked.Ball.Body], bare.body_inertia[catch_scene.Ball.Body])
    point, axis, _ = mouth
    state = np.concatenate([point + 0.05 * axis, -3.0 * axis])
    Set_Contact(marked, *CONST_NOMINAL)
    Set_Contact(catch_scene, *CONST_NOMINAL)
    ends = {}
    for name, scene in (('bare', catch_scene), ('marked', marked)):
        seen = []
        result = Simulate_Catch(scene, state, Plan_Str(), CONST_READY, 0.0, Contact_Str=_Contact(0.9), on_step=lambda t, q: seen.append(q))
        address = scene.Ball.Position_Address
        ends[name] = (result, CONST_CUP_LENGTH - float((point - seen[-1][address:address + 3]) @ axis) - CONST_CUP_BOTTOM)
    assert ends['bare'][0].Retained and ends['bare'][0].First_Contact == 'cup_bottom' and ends['bare'][1] == pytest.approx(0.030, abs=1e-3)
    assert ends['marked'][0].Retained and ends['marked'][0].First_Contact == 'cup_wall' and ends['marked'][1] > 0.035
    turned = Marker_Orientation('Catch_In', 7)
    assert np.linalg.norm(turned) == pytest.approx(1.0, abs=1e-12) and np.array_equal(turned, Marker_Orientation('Catch_In', 7))
    assert not np.allclose(turned, Marker_Orientation('Catch_In', 8)) and not np.allclose(turned, Marker_Orientation('Catch_Near', 7))
    seen = []
    Simulate_Catch(marked, state, Plan_Str(), CONST_READY, 0.0, Contact_Str=_Contact(0.01), on_step=lambda t, q: seen.append(q), orientation=turned)
    address = marked.Ball.Position_Address
    assert np.array_equal(seen[0][address + 3:address + 7], turned)
    from Catch.Interception import Interception_Setup_Str
    with pytest.raises(Catch_Error_Cls, match='markers variant is a moving catch'):
        Catch_Chunk((Interception_Setup_Str(), 'Set', 'absent.npz', np.arange(2), '0', dataclasses.replace(_Contact(), Markers=True), None))


def test_a_throws_result_does_not_depend_on_the_throws_before_it(catch_scene, mouth):
    """
    Description:
        The held robot's state is kept and restored between throws: the shot along the axis gives the same result,
        field for field, first (the hold computed, nothing kept yet), after a shot on the lip (the hold restored), and
        after a throw with another contact.
    """

    catch_scene.Held.clear()
    first = _Shot(catch_scene, mouth, 0.0, settle=0.3)
    _Shot(catch_scene, mouth, 0.080, settle=0.3)
    second = _Shot(catch_scene, mouth, 0.0, settle=0.3)
    _Shot(catch_scene, mouth, 0.0, contact=(0.004, 0.3), settle=0.3)
    third = _Shot(catch_scene, mouth, 0.0, settle=0.3)
    assert first.Retained and first.Penetration > 0.002 and dataclasses.asdict(first) == dataclasses.asdict(second) == dataclasses.asdict(third)


def test_the_balls_contact_gives_the_levels_restitution_on_the_floor(catch_scene):
    """
    Description:
        The ball's contact is the one of everything it touches (its priority): dropped on the floor at 7 m/s the
        rebound ratio is 0.19 with the nominal contact, 0.13 with the stiffer pair, 0.29 and 0.44 with the two
        bouncier levels (the addendum's values, each band holding the scatter over the contact's phase within a
        step), not the floor's default (0.05). At 1 m/s the nominal contact gives less, 0.14 to 0.19 (0.16: gravity
        acts during the 10 ms contact).
    """

    for contact, (low, high) in CONST_RESTITUTION.items():
        Set_Contact(catch_scene, *contact)
        assert low < Rebound(catch_scene, 7.0) < high, contact
    Set_Contact(catch_scene, *CONST_NOMINAL)
    assert 0.14 < Rebound(catch_scene, 1.0) < 0.19


def test_a_step_that_does_not_divide_the_control_period_and_a_time_constant_below_two_steps_are_refused(catch_scene, configuration, flight):
    """
    Description:
        The robot is commanded once per control period of 0.002 s, so a catch step of 0.0003 s is refused naming both;
        a time constant below two steps (0.0004 s at 0.00025 s) is refused, because MuJoCo would raise it to two steps
        silently, and so is a damping ratio of 0. The shipped step gives 8 steps per control period.
    """

    with pytest.raises(Catch_Error_Cls, match='Contact.Step 0.0003 s is not a whole fraction of the scene\'s control period 0.002 s'):
        Catch_Scene(configuration.Areas, dataclasses.replace(_Contact(), Step=0.0003), Flight_Model_Str=flight)
    with pytest.raises(Catch_Error_Cls, match='time constant 0.0004 s'):
        Set_Contact(catch_scene, 0.0004, 0.7)
    with pytest.raises(Catch_Error_Cls, match='damping ratio 0.0'):
        Set_Contact(catch_scene, 0.004, 0.0)
    assert catch_scene.Steps == 8 and catch_scene.Tick == 0.002 and catch_scene.Model.opt.timestep == CONST_STEP


def _Plan(start_time):
    goal = CONST_READY + np.radians([15.0, -10.0, 12.0, 8.0, -6.0, 0.0])
    return Plan_Str(Profiles=[Profile(CONST_READY, np.zeros(6), goal, CONST_SPEED, CONST_ACCELERATION, start_time)])


def test_at_the_scenes_own_step_the_loop_tracks_a_plan_as_the_interceptions_tracking_does(configuration, flight):
    """
    Description:
        With the catch's step set to the scene's 0.002 s (one step per control tick) and a ball that never comes near,
        the largest joint error against a planned motion is the one of Robot.Motion.Track (F10's tracking) on the same
        ticks, within 1e-12 rad: the hold, the command of each tick's planned state and the step are the same.
    """

    contact = dataclasses.replace(_Contact(settle=0.1), Step=0.002)
    scene = Catch_Scene(configuration.Areas, contact, Flight_Model_Str=flight)
    Set_Contact(scene, *CONST_NOMINAL)
    plan = _Plan(0.1037)
    end = plan.Profiles[0].Start_Time + plan.Profiles[0].Duration
    result = Simulate_Catch(scene, np.array([0.0, -5.0, 1.0, 0.0, 0.0, 7.0]), plan, CONST_READY, end, Contact_Str=contact)
    _, q, qd, qdd = Plan_Ticks(plan, end + 0.002, 0.002)
    model, data = scene.Model, mujoco.MjData(scene.Model)
    data.qpos[scene.Ball.Position_Address:scene.Ball.Position_Address + 3] = (0.0, -3.0, CONST_BALL_RADIUS)
    data.qpos[scene.Robot.Position_Address] = q[0]
    tracked = Track(model, data, scene.Servos, q, qd, qdd, 250, Robot_Str=scene.Robot)
    reference = float(np.max(np.abs(tracked.Positions - q)))
    assert 0.0005 < reference < 0.003 and abs(result.Tracking - reference) < 1e-12
    assert result.First_Contact == '' and result.Force_Limit_Ticks == tracked.Force_Limit_Ticks == 0


def test_the_ball_starts_at_its_state_at_the_first_control_tick_and_flies_as_the_flight_model(catch_scene, flight):
    """
    Description:
        The control ticks are the plan's: for a plan starting at 0.1037 s the first tick at or after the release is at
        0.0017 s, where the ball is placed at the flight model's state; a simulation to 0.35 s ends at 0.0017 s plus
        175 control periods, and the ball is there where Flight.Core.Simulate has it, within 5 um (F5's agreement, at
        the catch's step).
    """

    Set_Contact(catch_scene, *CONST_NOMINAL)
    release = np.array([0.0, -5.0, 1.0, 0.3, 0.5, 6.0])
    result = Simulate_Catch(catch_scene, release, _Plan(0.1037), CONST_READY, 0.25, Contact_Str=_Contact(settle=0.1))
    end = 0.0017 + 175 * 0.002
    assert result.First_Contact == '' and abs(catch_scene.Data.time - end) < 1e-9
    position = catch_scene.Data.qpos[catch_scene.Ball.Position_Address:catch_scene.Ball.Position_Address + 3]
    assert np.linalg.norm(position - Simulate(release, np.array([end]), Flight_Model_Str=flight)[0, :3]) < 5e-6


def test_every_throw_has_one_class_by_its_decision_its_plan_and_what_the_ball_met():
    """
    Description:
        The classes (F11 Q7) on constructed results: a no-go by its truth; a go caught when the ball is retained,
        whatever it met first and even without a plan; late without an accepted plan; then by the first contact, a link
        or the stand; then bounced out when it was inside the cup and touched it, on the rim when it touched the cup
        only, missed otherwise. Each class maps to one of SPEC §2.10's five, and a go without a simulation is refused.
    """

    make = (lambda **values: Catch_Result_Str(**values))
    assert Classify('no-go', True, False, None) == 'no_go_catchable' and Classify('no-go', False, False, None) == 'no_go'
    assert Classify('go', True, True, make(Retained=True, First_Contact='robot')) == 'caught'
    assert Classify('go', False, False, make(Retained=True)) == 'caught'
    assert Classify('go', True, False, make(First_Contact='cup_wall', Cup_Contact_Time=0.9, Entered=True)) == 'late'
    assert Classify('go', True, True, make(First_Contact='robot', Cup_Contact_Time=0.9, Entered=True)) == 'hit_robot'
    assert Classify('go', True, True, make(First_Contact='stand')) == 'hit_stand'
    assert Classify('go', True, True, make(First_Contact='cup_wall', Cup_Contact_Time=0.9, Entered=True)) == 'bounced_out'
    assert Classify('go', True, True, make(First_Contact='cup_lip', Cup_Contact_Time=0.9)) == 'rim'
    assert Classify('go', True, True, make(First_Contact='floor', Entered=True)) == 'missed'
    assert Classify('go', True, True, make()) == 'missed'
    with pytest.raises(Catch_Error_Cls, match='A go has no simulation'):
        Classify('go', True, True, None)
    assert CONST_CLASSES == {'caught': 'caught', 'bounced_out': 'bounced off', 'rim': 'bounced off', 'hit_robot': 'bounced off',
                             'missed': 'missed', 'hit_stand': 'missed', 'late': 'late', 'no_go_catchable': 'missed',
                             'no_go': 'outside the workspace'}


def test_a_contact_of_the_robot_with_anything_but_the_ball_is_reported(catch_scene):
    """
    Description:
        R2's reading: a set of contacts with a link against the floor names the two geoms as the robot's contact, once;
        the ball against the cup's wall and the floor in the same set are the ball's contacts, the wall first, and the
        set is one with the cup.
    """

    model = catch_scene.Model
    floor, ball = model.geom('floor').id, model.geom('ball_collision').id
    wall = model.geom('ur10e/cup_wall_07').id
    link = [geom for geom in range(model.ngeom) if catch_scene.Names[geom] == 'ur10e/forearm_link' and model.geom_contype[geom]][0]
    result, seen = Catch_Result_Str(), set()
    cup = _Read_Contacts(catch_scene, np.array([[ball, wall], [floor, link], [floor, ball], [link, floor]]), 0.75, seen, result)
    assert cup and result.Robot_Contact == 'floor + ur10e/forearm_link'
    assert (result.First_Contact, result.First_Contact_Time, result.Cup_Contact_Time) == ('cup_wall', 0.75, 0.75)
    assert sorted(seen) == [3, 7]
    assert not _Read_Contacts(catch_scene, np.array([[floor, ball]]), 0.9, seen, result) and result.First_Contact_Time == 0.75


def _Row(**values):
    row = dict(dataset='Catch_In', flight='7', seed='0', acceleration_deg_s2='800', time_constant_s=0.004, damping_ratio=0.7, catchable='true',
               decision='go', intercepted='true', catch_time_s=0.9, outcome='', retained='false', at_rest='false', rest_s=float('nan'),
               first_contact='', first_contact_s=float('nan'), cup_contact_s=float('nan'), entered='false', entry_lateral_m=float('nan'),
               end_speed_m_s=float('nan'), penetration_m=0.0, robot_contact='', tracking_joint_rad=0.002, force_limit_ticks=0)
    row['class'] = 'missed'
    row.update(values)
    return tuple(row[name] for name in CONST_PER_THROW_COLUMNS)


def test_the_summary_counts_the_classes_and_gives_the_retained_shares_with_their_intervals(tmp_path):
    """
    Description:
        Constructed rows of one group: 4 goes on catchable throws, 3 retained (2 at rest, 0.6 and 1.0 s after the catch
        time), 1 bounced out with the robot touching the floor; 1 false go, missed; 1 no-go on a catchable throw and 2
        not catchable. Retained 3 of 5 goes (60 %) and 3 of the 5 catchable (60 %), each with the Clopper-Pearson
        interval of 3 / 5, (0.1466, 0.9473); a second group with another contact is summarised on its own; the files
        hold the columns of SPEC §9.14, the sweep's with the error's source and scale first.
    """

    caught = dict(retained='true', penetration_m=0.008)
    caught['class'] = 'caught'
    out, false_go = dict(robot_contact='floor + ur10e/forearm_link', penetration_m=0.011), dict(catchable='false', intercepted='false')
    out['class'] = 'bounced_out'
    rows = [_Row(at_rest='true', rest_s=0.6, **caught), _Row(at_rest='true', rest_s=1.0, **caught), _Row(**caught), _Row(**out),
            _Row(**false_go), _Row(decision='no-go', intercepted='false', **{'class': 'no_go_catchable'}),
            _Row(decision='no-go', catchable='false', intercepted='false', **{'class': 'no_go'}),
            _Row(decision='no-go', catchable='false', intercepted='false', **{'class': 'no_go'}),
            _Row(damping_ratio=0.3, **out)]
    summary = Summary_Rows(rows, 'validation')
    assert len(summary) == 2 and all(len(row) == len(CONST_SUMMARY_COLUMNS) for row in summary)
    first = dict(zip(CONST_SUMMARY_COLUMNS, summary[0]))
    assert (first['dataset'], first['partition'], first['seed'], first['acceleration_deg_s2'], first['damping_ratio']) == \
        ('Catch_In', 'validation', '0', '800', 0.7)
    assert (first['throws'], first['catchable'], first['go'], first['false_go'], first['intercepted']) == (8, 5, 5, 1, 4)
    assert (first['retained'], first['retained_catchable'], first['at_rest'], first['robot_contacts']) == (3, 3, 2, 1)
    for name in ('retained_of_go', 'retained_of_catchable'):
        assert first[name] == pytest.approx(0.6) and first[name + '_lower'] == pytest.approx(0.1466, abs=5e-4)
        assert first[name + '_upper'] == pytest.approx(0.9473, abs=5e-4)
    assert {name: first[name] for name in CONST_CLASSES} == {'caught': 3, 'bounced_out': 1, 'rim': 0, 'hit_robot': 0, 'missed': 1, 'hit_stand': 0,
                                                             'late': 0, 'no_go_catchable': 1, 'no_go': 2}
    assert (first['rest_p50_s'], first['rest_max_s'], first['penetration_max_m']) == (pytest.approx(0.8), 1.0, 0.011)
    second = dict(zip(CONST_SUMMARY_COLUMNS, summary[1]))
    assert (second['damping_ratio'], second['throws'], second['retained'], second['bounced_out']) == (0.3, 1, 0, 1)
    assert second['retained_of_go'] == 0.0 and np.isnan(second['rest_p50_s'])
    Write_Per_Throw(str(tmp_path / 'per_throw.csv'), rows)
    Write_Summary(str(tmp_path / 'summary.csv'), summary)
    Write_Per_Throw(str(tmp_path / 'sweep.csv'), [('real', 1.5) + row for row in rows], sweep=True)
    lines = (tmp_path / 'per_throw.csv').read_text().splitlines()
    assert lines[0] == ','.join(CONST_PER_THROW_COLUMNS) and len(lines) == 10
    assert lines[1].startswith('Catch_In,7,0,800,0.004,0.7,true,go,true,0.9,caught,,true,true,0.6,')
    assert (tmp_path / 'summary.csv').read_text().splitlines()[0] == ','.join(CONST_SUMMARY_COLUMNS)
    sweep = (tmp_path / 'sweep.csv').read_text().splitlines()
    assert sweep[0] == ','.join(CONST_SWEEP_COLUMNS + CONST_PER_THROW_COLUMNS) and sweep[1].startswith('real,1.5,Catch_In,7,0,800,')


def test_the_catchs_decisions_and_interceptions_summaries_are_read_back_by_their_own_readers(tmp_path):
    """
    Description:
        The readers the figures use (the visualisation, delivery 2): each stage's summary.csv (a moving catch's, the sweep's
        among them) and an interception's per_throw.csv read back as written, by column; a file of another stage is refused
        naming its SPEC section.
    """

    from Catch.Decision import Write_Summary as Write_Decision, Read_Summary as Read_Decision, CONST_SUMMARY_COLUMNS as CONST_DECISION
    from Catch.Interception import (Write_Per_Throw as Write_Rows, Read_Per_Throw as Read_Rows, CONST_PER_THROW_COLUMNS as CONST_ROWS,
                                    CONST_MOVING_COLUMNS as CONST_ROWS_MOVING)
    from Catch.Physical import Write_Summary, Read_Summary, CONST_SUMMARY_COLUMNS, CONST_MOVING_SUMMARY_COLUMNS, CONST_SWEEP_COLUMNS
    row = tuple(range(len(CONST_DECISION)))
    Write_Decision(str(tmp_path / 'decision.csv'), [row])
    assert Read_Decision(str(tmp_path / 'decision.csv')) == [dict(zip(CONST_DECISION, ['%d' % v for v in row]))]
    rows = [tuple(range(len(CONST_ROWS + CONST_ROWS_MOVING)))]
    Write_Rows(str(tmp_path / 'per_throw.csv'), rows, moving=True)
    assert Read_Rows(str(tmp_path / 'per_throw.csv'))[0]['in_stroke'] == '%d' % (len(rows[0]) - 1)
    for name, columns, options in (('moving', CONST_SUMMARY_COLUMNS + CONST_MOVING_SUMMARY_COLUMNS, {'moving': True}),
                                   ('sweep', CONST_SWEEP_COLUMNS + CONST_SUMMARY_COLUMNS, {'sweep': True})):
        Write_Summary(str(tmp_path / ('%s.csv' % name)), [tuple([0.5] * len(columns))], **options)
        assert Read_Summary(str(tmp_path / ('%s.csv' % name))) == [dict.fromkeys(columns, '0.5')]
    with pytest.raises(Catch_Error_Cls, match=r'does not have the columns its writer writes \(SPEC §9.14\)'):
        Read_Summary(str(tmp_path / 'decision.csv'))
    with pytest.raises(Catch_Error_Cls, match=r'\(SPEC §9.12\)'):
        Read_Decision(str(tmp_path / 'moving.csv'))


def test_the_mean_restitution_over_the_phases_is_the_records_measure(catch_scene):
    """
    Description:
        SPEC §4.14's and F15's measure, the mean over 8 positions of the first contact within a step at 7 m/s: the nominal
        contact 0.194 (§4.14), the stiffer pair 0.135, ratio 0.5 0.284, ratio 0.3 0.438 (§4.14; its scratch scene had the
        cone's top at 66.8 mm, so 0.005 to spare) and ratio 0.4 0.347, ratio 0.2 0.560 (F15's Part A, SPEC §8.1, 0.002 to
        spare); a single phase differs from the mean.
    """

    for contact, recorded, spare in (((0.004, 0.7), 0.194, 0.005), ((0.00157, 0.9), 0.135, 0.005), ((0.004, 0.5), 0.284, 0.005),
                                     ((0.004, 0.3), 0.438, 0.005), ((0.004, 0.4), 0.347, 0.002), ((0.004, 0.2), 0.560, 0.002)):
        Set_Contact(catch_scene, *contact)
        assert abs(Mean_Rebound(catch_scene, 7.0) - recorded) < spare, contact
    Set_Contact(catch_scene, *CONST_NOMINAL)
    assert Mean_Rebound(catch_scene, 7.0) != Rebound(catch_scene, 7.0, 0.5)

