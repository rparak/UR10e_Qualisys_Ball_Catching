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
# File Name: ../test/Scene/test_ball.py
## =========================================================================== ##

# System (Default)
import os
# Copy (Copies of the shared model)
import copy
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Scene/Ball
from Scene.Ball import Find_Ball, Find_Markers, Place_Ball, Apply_Drag, Ball_State, Ball_Error_Cls
#   ../Flight/Core - the reference integrator, independently of the scene
from Flight.Core import Flight_Model_Str, Simulate
#   ../Dataset/Core - the floor height of the datasets and the throws of S1
from Dataset.Core import CONST_FLOOR_HEIGHT, Draw_Throws
#   ../Configuration/Core - the dataset configuration of S1
from Configuration.Core import Load_Dataset_Configuration

"""
Description:
    Initialization of constants.

    The ball's flight in the MuJoCo scene (src/Scene/Ball.py; F5). Expectations come from the
    reference integrator (Flight.Core.Simulate), the flight model of the configuration, the SPEC's ball (§7.4) and
    the bound of SPEC §4.13, never from the module under test.
"""
# The bound (SPEC §4.13): the ball's position and velocity against the reference integrator
# at every step to the floor at the scene's 0.002 s, a third of the recordings' white noise in flight per axis
# (0.3 mm, §4.2), 0.012-0.017 % of the speed at the catch plane (5.87-8.09 m/s).
#   Unit [float, m], [float, m/s]
CONST_POSITION_BOUND = 1e-4
CONST_VELOCITY_BOUND = 1e-3
# The regression pin of the position (F5 Q3): the largest error measured on the throws below and on the train
# partitions was 0.98 um (SPEC §4.13).
#   Unit [float, m]
CONST_POSITION_PIN = 2e-6
# The ball (SPEC §7.4): mass and radius.
#   Unit [float, kg], [float, m]
CONST_BALL_MASS = 0.122
CONST_BALL_RADIUS = 0.030
# Six release states of the train partition of the dataset Default (flights 4872, 5810, 2912, 2475, 4278, 6657): the
# fastest (7.95 m/s), the slowest (5.52 m/s), the highest (1.54 m), the lowest (0.82 m), the leftmost and the rightmost
# release whose flight touches nothing in the scene before the floor (F5 Part B, 2026-11-22).
#   Unit [Matrix<float> 6x6, m, m/s]
CONST_STATES = ((-0.9769343671607507, -5.303545200014166, 1.1485684374650695, 1.644929165787861, 5.632235838114116, 5.361432774891346),
                (-0.6077384918611882, -4.322559055982516, 1.1952543759263305, 0.13200195675425674, 2.787666037194898, 4.764533855068842),
                (0.4759953359310625, -5.035678386092973, 1.5416336645586979, -1.1764477293020068, 3.7189579937613355, 4.759295266437062),
                (0.28457604964754785, -4.996709681936893, 0.8211753981173493, 0.7699958275604265, 3.8597300728999624, 6.214100438058826),
                (-1.959248126269807, -4.639884447669243, 1.2254062546395426, 1.0880350539661436, 4.259149605582344, 5.533675650226395),
                (1.709063095059238, -4.718321255824863, 1.0238528551417816, -1.225195042301372, 2.996970395848995, 6.135219232414046))
# The throws of S1: their count and the seed of their generators.
#   Unit [int]
CONST_DRAWN_THROWS = 1000
CONST_DRAWN_SEED = 20261122


@pytest.fixture(scope='module')
def model(configuration):
    """
    Description:
        The flight model of the shipped configuration.
    """

    return Flight_Model_Str(Gravity=np.array([0.0, 0.0, -configuration.Flight.Gravity]), Drag=configuration.Flight.Drag)


def _Fly(scene, state, model, steps=None):
    """
    Description:
        The ball's states from the state given, one per step of the scene, until the step at which its first contact
        appears (or the steps given); returns the states and the contact's step and other geom (None without one).
    """

    ball = Find_Ball(scene)
    data = mujoco.MjData(scene)
    Place_Ball(scene, data, state, Ball_Str=ball, Flight_Model_Str=model)
    states, contact = [Ball_State(scene, data, Ball_Str=ball, Flight_Model_Str=model)], None
    while contact is None and len(states) < (steps or 1000):
        Apply_Drag(scene, data, Ball_Str=ball, Flight_Model_Str=model)
        mujoco.mj_step(scene, data)
        states.append(Ball_State(scene, data, Ball_Str=ball, Flight_Model_Str=model))
        for c in (data.contact[:data.ncon] if steps is None else ()):
            if ball.Body in (scene.geom_bodyid[c.geom1], scene.geom_bodyid[c.geom2]):
                contact = (len(states) - 1, c.geom2 if scene.geom_bodyid[c.geom1] == ball.Body else c.geom1)
                break
    return np.array(states), contact


def _Errors(states, reference, above):
    """
    Description:
        The largest position and velocity error over the steps at which the reference ball is above the floor.
    """

    return (float(np.max(np.linalg.norm(states[above, :3] - reference[above, :3], axis=1))),
            float(np.max(np.linalg.norm(states[above, 3:] - reference[above, 3:], axis=1))))


def test_the_ball_follows_the_flight_model_within_the_bound_at_every_step_to_the_floor(scene, model):
    """
    Description:
        Placed at a release state and driven by the drag at every step of the scene's 0.002 s, the ball agrees with
        the reference integrator within the bound (0.1 mm, 1 mm/s) and the pin (2 um) at every step while the reference
        ball is above the floor, for six releases at the edges of the train partition's range; its first contact is
        the floor, two steps after the reference's last step above it (MuJoCo finds a contact at the start of a step).
    """

    floor = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    h = scene.opt.timestep
    for state in CONST_STATES:
        states, contact = _Fly(scene, state, model)
        reference = Simulate(np.array(state), np.arange(len(states)) * h, Flight_Model_Str=model)
        above = reference[:, 2] > CONST_FLOOR_HEIGHT
        last = int(np.flatnonzero(above)[-1])
        position, velocity = _Errors(states, reference, above)
        assert position <= CONST_POSITION_BOUND and velocity <= CONST_VELOCITY_BOUND, (state, position, velocity)
        assert position <= CONST_POSITION_PIN, (state, position)
        assert contact == (last + 2, floor), (state, contact, last)


def test_the_drag_is_the_flight_models_at_the_balls_centre_of_mass(scene, model):
    """
    Description:
        The applied force is -m k |v| v at the placed velocity (within the half-step estimate's 1e-6 of it), with the
        mass the model holds (a model whose ball is twice as heavy gets twice the force), and no torque (one left in
        the data is cleared): the ball does not start to spin over 200 steps.
    """

    v = np.array(CONST_STATES[0][3:])
    expected = -CONST_BALL_MASS * model.Drag * np.linalg.norm(v) * v
    data = mujoco.MjData(scene)
    ball = Find_Ball(scene)
    Place_Ball(scene, data, CONST_STATES[0], Ball_Str=ball, Flight_Model_Str=model)
    data.xfrc_applied[ball.Body, 3:] = 1.0
    Apply_Drag(scene, data, Ball_Str=ball, Flight_Model_Str=model)
    force = data.xfrc_applied[ball.Body]
    assert np.allclose(force[:3], expected, rtol=1e-6, atol=0.0) and np.all(force[3:] == 0.0), (force, expected)
    heavy = copy.deepcopy(scene)
    heavy.body_mass[ball.Body] *= 2.0
    other = mujoco.MjData(heavy)
    Place_Ball(heavy, other, CONST_STATES[0], Ball_Str=Find_Ball(heavy), Flight_Model_Str=model)
    Apply_Drag(heavy, other, Ball_Str=Find_Ball(heavy), Flight_Model_Str=model)
    assert np.allclose(other.xfrc_applied[ball.Body, :3], 2.0 * expected, rtol=1e-6, atol=0.0)
    for _ in range(200):
        Apply_Drag(scene, data, Ball_Str=ball, Flight_Model_Str=model)
        mujoco.mj_step(scene, data)
    assert np.max(np.abs(data.qvel[ball.Velocity_Address + 3:ball.Velocity_Address + 6])) <= 1e-12


def test_a_placed_ball_lags_half_a_step_in_its_joint_and_reads_back_its_state(scene, model):
    """
    Description:
        A placed ball's centre is the state's position, without rotation or spin; its free joint's velocity is the
        state's velocity half a step back, v - h/2 (g - k |v| v); Ball_State reads the state back, the velocity within
        1e-5 m/s (the half-step estimate; the joint's lag is 0.0098 m/s).
    """

    h = scene.opt.timestep
    state = np.array(CONST_STATES[0])
    v = state[3:]
    data = mujoco.MjData(scene)
    ball = Find_Ball(scene)
    Place_Ball(scene, data, state, Ball_Str=ball, Flight_Model_Str=model)
    p, d = ball.Position_Address, ball.Velocity_Address
    assert data.qpos[p:p + 3].tolist() == state[:3].tolist() and data.qpos[p + 3:p + 7].tolist() == [1.0, 0.0, 0.0, 0.0]
    lagged = v - 0.5 * h * (model.Gravity - model.Drag * np.linalg.norm(v) * v)
    assert np.allclose(data.qvel[d:d + 3], lagged, rtol=0.0, atol=1e-15) and np.all(data.qvel[d + 3:d + 6] == 0.0)
    read = Ball_State(scene, data, Ball_Str=ball, Flight_Model_Str=model)
    assert read[:3].tolist() == state[:3].tolist() and np.max(np.abs(read[3:] - v)) <= 1e-5, read - state


def test_the_ball_is_the_scenes_free_sphere_of_the_datasets_floor_height(scene):
    """
    Description:
        Find_Ball gives the scene's ball: the free joint "ball"'s addresses, the assumed mass 0.122 kg (SPEC §7.4) and
        the sphere's radius 0.030 m, the height at which a dataset's flight ends (CONST_FLOOR_HEIGHT).
    """

    ball = Find_Ball(scene)
    joint = scene.joint('ball')
    assert (ball.Body, ball.Position_Address, ball.Velocity_Address) == (scene.body('ball').id, int(joint.qposadr[0]), int(joint.dofadr[0]))
    assert ball.Mass == CONST_BALL_MASS and ball.Radius == CONST_BALL_RADIUS == CONST_FLOOR_HEIGHT


@pytest.mark.parametrize('state, words', [((0.0, 0.0, 1.0, 1.0, 1.0), 'six finite numbers'),
                                          ((0.0, 0.0, 1.0, float('nan'), 1.0, 1.0), 'six finite numbers')])
def test_a_state_that_is_not_six_finite_numbers_is_refused(scene, model, state, words):
    """
    Description:
        A state of five numbers, or with a NaN, is refused naming it; the data stays as it was.
    """

    data = mujoco.MjData(scene)
    with pytest.raises(Ball_Error_Cls, match=words):
        Place_Ball(scene, data, state, Ball_Str=Find_Ball(scene), Flight_Model_Str=model)
    assert np.all(data.qvel == 0.0)


@pytest.mark.parametrize('body, words', [('<body name="other"><freejoint/><geom type="sphere" size="0.03"/></body>', 'no body "ball"'),
                                         ('<body name="ball"><joint type="hinge"/><geom type="sphere" size="0.03"/></body>', 'no single free joint'),
                                         ('<body name="ball"><freejoint/><geom type="box" size="0.03 0.03 0.03"/></body>', '0 sphere geoms')])
def test_a_model_without_the_scenes_free_ball_is_refused(body, words):
    """
    Description:
        A model without the body "ball", whose ball has no free joint, or no sphere, is refused naming what is missing.
    """

    with pytest.raises(Ball_Error_Cls, match=words):
        Find_Ball(mujoco.MjModel.from_xml_string('<mujoco><worldbody>%s</worldbody></mujoco>' % (body,)))


def test_the_markers_are_measured_on_the_balls_mesh_six_of_16_mm_on_the_axes_38_mm_out(scene):
    """
    Description:
        F15: the ball's markers are read from the scene's own mesh of the ball, in the ball's
        frame: six, one on each side of each axis, their centres 38 mm from the ball's centre and their radius 8 mm
        (SPEC §4.3, §4.4: a 16 mm marker on the R 30 mm ball, 91.9 mm over the markers), each within 0.1 mm; so the
        markers reach 45.9 mm from the centre, half of the 91.9 mm. A ball without a mesh is refused.
    """

    centres, radius = Find_Markers(scene)
    assert centres.shape == (6, 3) and radius == pytest.approx(0.008, abs=1e-4)
    assert np.linalg.norm(centres, axis=1) == pytest.approx(0.038, abs=1e-4)
    directions = np.round(centres / np.linalg.norm(centres, axis=1)[:, None], 3)
    assert sorted(tuple(row) for row in directions.tolist()) == sorted(tuple(sign * np.eye(3)[axis]) for axis in range(3) for sign in (-1.0, 1.0))
    assert 2.0 * (np.linalg.norm(centres, axis=1).max() + radius) == pytest.approx(0.0919, abs=2e-4)
    bare = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="ball"><freejoint/><geom type="sphere" size="0.03"/></body>'
                                          '</worldbody></mujoco>')
    with pytest.raises(Ball_Error_Cls, match='0 mesh geoms'):
        Find_Markers(bare)


@pytest.mark.parametrize('field, value, words', [('gravity', (0.0, 0.0, -9.80), 'gravity'), ('density', 1.225, 'medium'),
                                                 ('viscosity', 1.8e-5, 'medium'), ('wind', (0.0, 1.0, 0.0), 'medium')])
def test_a_model_whose_gravity_or_medium_differs_from_the_flight_model_is_refused(scene, model, field, value, words):
    """
    Description:
        A model whose gravity is not the flight model's, or with a medium whose fluid forces would add to the drag, is
        refused when the ball is placed.
    """

    other = copy.deepcopy(scene)
    setattr(other.opt, field, value)
    with pytest.raises(Ball_Error_Cls, match=words):
        Place_Ball(other, mujoco.MjData(other), CONST_STATES[0], Ball_Str=Find_Ball(other), Flight_Model_Str=model)


@pytest.mark.slow
def test_the_ball_follows_the_flight_model_within_the_bound_on_a_thousand_drawn_throws(root, scene, model, configuration):
    """
    Description:
        S1 (F5 Q6): 1 000 throws drawn by the dataset generator from the shipped dataset configuration (seed
        20261122; nothing read from Data/Generated/), each flown to the floor in the scene with the ball's contacts
        switched off in a copy: every step within the bound and the pin. About 25 s alone.
    """

    dataset = Load_Dataset_Configuration(os.path.join(root, 'Config', 'dataset.yaml'))
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(CONST_DRAWN_SEED).spawn(CONST_DRAWN_THROWS)]
    throws = Draw_Throws(rngs, dataset, configuration, Flight_Model_Str=model)
    free = copy.deepcopy(scene)
    sphere = [g for g in range(free.ngeom) if free.geom_bodyid[g] == Find_Ball(free).Body and free.geom_contype[g]]
    free.geom_contype[sphere], free.geom_conaffinity[sphere] = 0, 0
    h = free.opt.timestep
    worst = [0.0, 0.0]
    for state, floor in zip(throws['release_state'], throws['floor_time']):
        steps = int(np.ceil(floor / h)) + 1
        states, _ = _Fly(free, state, model, steps)
        t = np.arange(len(states)) * h
        position, velocity = _Errors(states, Simulate(state, t, Flight_Model_Str=model), t < floor)
        worst = [max(worst[0], position), max(worst[1], velocity)]
    assert worst[0] <= CONST_POSITION_BOUND and worst[1] <= CONST_VELOCITY_BOUND, worst
    assert worst[0] <= CONST_POSITION_PIN, worst
