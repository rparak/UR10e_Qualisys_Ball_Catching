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
# File Name: ../test/Flight/test_core.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the flight model and the integrators under test
#   ../Flight/Core
from Flight.Core import (Acceleration, Fit_Initial_States, Flight_Error_Cls, Flight_Model_Str, Integrate, Simulate, Simulate_At,
                         State_Derivative)

"""
Description:
    Initialization of constants.

    Under a constant acceleration every one of the four integrators is exact (SPEC §4.1), so the
    expectation is the closed form r(t) = r_0 + v_0 t + g t^2 / 2, v(t) = v_0 + g t, never the code.
"""
# A vacuum model with g = 9.81 m/s^2, z up.
#   Unit [m/s^2]
CONST_GRAVITY = np.array([0.0, 0.0, -9.81])
# A throw from the legacy throw area towards the robot: position and velocity.
#   Unit [m], [m/s]
CONST_X_0 = np.array([-1.66, 6.13, 0.87, 0.6, -5.1, 4.2])
# The step of the 2022 motion capture (370 Hz) and the number of states (1 s).
#   Unit [s], [int]
CONST_H = 1.0 / 370.0
CONST_N = 371
# The bound on the accumulated rounding over 370 steps of a state of a few metres: 370 steps x
# 2.2e-16 relative x 10 m is about 1e-12; a wrong step is off by about 1e-2 m after 1 s.
#   Unit [m]
CONST_ROUNDING = 1e-11
# A drag constant of the size F3 identified on the real flights (about 0.009 1/m, SPEC §4.2).
#   Unit [1/m]
CONST_DRAG = 0.00905


def _Closed_Form(t):
    return np.hstack((CONST_X_0[:3] + CONST_X_0[3:] * t + 0.5 * CONST_GRAVITY * t ** 2, CONST_X_0[3:] + CONST_GRAVITY * t))


@pytest.mark.parametrize('method', ['Midpoint', 'Heun', 'RK2', 'RK4'])
def test_each_integrator_is_exact_under_constant_gravity(method):
    """
    Description:
        371 states over 1 s follow the closed form within the rounding: a wrong coefficient in a
        step leaves an error of order h^2 g per step.
    """

    x = Integrate(CONST_X_0, CONST_N, CONST_H, method, Flight_Model_Str=Flight_Model_Str(Gravity=CONST_GRAVITY))
    expected = np.array([_Closed_Form(i * CONST_H) for i in range(CONST_N)])
    error = np.max(np.abs(x - expected))
    assert error <= CONST_ROUNDING, '%s: max |error| %.3e > %.0e' % (method, error, CONST_ROUNDING)


def test_integrate_returns_n_states_starting_with_x_0():
    """
    Description:
        n states, the first exactly x_0 and the last at (n - 1) h: no step lost or added.
    """

    x = Integrate(CONST_X_0, 5, CONST_H, 'RK4', Flight_Model_Str=Flight_Model_Str(Gravity=CONST_GRAVITY))
    assert x.shape == (5, 6)
    assert np.array_equal(x[0], CONST_X_0)
    assert np.max(np.abs(x[-1] - _Closed_Form(4 * CONST_H))) <= CONST_ROUNDING


def test_the_state_derivative_is_the_velocity_and_gravity():
    """
    Description:
        The vacuum model: dr/dt = v, dv/dt = g, whatever the velocity.
    """

    v = np.array([1.0, -2.0, 3.0])
    assert np.array_equal(State_Derivative(v, Flight_Model_Str=Flight_Model_Str(Gravity=CONST_GRAVITY)),
                          [1.0, -2.0, 3.0, 0.0, 0.0, -9.81])


def test_an_unknown_integrator_is_refused_naming_the_known_ones():
    """
    Description:
        A misspelt method is an error naming it and the integrators, never a silent default.
    """

    with pytest.raises(Flight_Error_Cls, match='Euler.*Midpoint, Heun, RK2, RK4'):
        Integrate(CONST_X_0, 3, CONST_H, 'Euler', Flight_Model_Str=Flight_Model_Str(Gravity=CONST_GRAVITY))


def test_the_flight_model_must_be_named_by_keyword():
    """
    Description:
        The model is keyword-only with no default (SPEC §7.4): leaving it out or passing it
        by position is an error, so no caller integrates with a model it did not name.
    """

    model = Flight_Model_Str(Gravity=CONST_GRAVITY)
    with pytest.raises(TypeError, match='Flight_Model_Str'):
        Integrate(CONST_X_0, 3, CONST_H, 'RK4')
    with pytest.raises(TypeError):
        Integrate(CONST_X_0, 3, CONST_H, 'RK4', model)


def test_drag_decelerates_along_the_velocity():
    """
    Description:
        a = g - k |v| v: for v = (3, -4, 0) m/s (|v| = 5 m/s) and k = 0.01 1/m the drag is -0.05 v, so
        a = (-0.15, 0.2, -9.81) m/s^2; a batch of velocities gives each row its own value.
    """

    model = Flight_Model_Str(Gravity=CONST_GRAVITY, Drag=0.01)
    v = np.array([3.0, -4.0, 0.0])
    assert State_Derivative(v, Flight_Model_Str=model) == pytest.approx([3.0, -4.0, 0.0, -0.15, 0.2, -9.81], abs=1e-15)
    batch = Acceleration(np.array([v, [0.0, 0.0, -2.0]]), Flight_Model_Str=model)
    assert batch == pytest.approx(np.array([[-0.15, 0.2, -9.81], [0.0, 0.0, -9.81 + 0.04]]), abs=1e-15)


def test_zero_drag_takes_the_2022_vacuum_path_bit_for_bit():
    """
    Description:
        With Drag = 0 the derivative is exactly (v, g) and RK4 integrates exactly as the gravity-only
        model of F1, so the 2022 baseline is unchanged (SPEC §2.1; the slow baseline tests).
    """

    model = Flight_Model_Str(Gravity=CONST_GRAVITY, Drag=0.0)
    for v in (np.array([0.6, -5.1, 4.2]), np.array([-0.0, 0.0, -7.3])):
        assert np.array_equal(State_Derivative(v, Flight_Model_Str=model), np.hstack((v, CONST_GRAVITY)))
    x = Integrate(CONST_X_0, CONST_N, CONST_H, 'RK4', Flight_Model_Str=model)
    assert np.array_equal(x[-1], Integrate(CONST_X_0, CONST_N, CONST_H, 'RK4', Flight_Model_Str=Flight_Model_Str(Gravity=CONST_GRAVITY))[-1])
    assert np.abs(x[-1] - _Closed_Form((CONST_N - 1) * CONST_H)).max() < CONST_ROUNDING


def test_simulate_without_drag_is_the_vacuum_parabola():
    """
    Description:
        Without drag, Simulate gives the closed form at any time, on and off its internal grid,
        within the rounding of a few hundred steps.
    """

    t = np.array([0.0, 1.0 / 600.0, 0.0123, 0.5, 0.98765, 1.5])
    x = Simulate(CONST_X_0, t, Flight_Model_Str=Flight_Model_Str(Gravity=CONST_GRAVITY))
    assert x.shape == (6, 6)
    for i, t_i in enumerate(t):
        assert np.abs(x[i] - _Closed_Form(t_i)).max() < 1e-12


def test_simulate_has_converged_at_its_step():
    """
    Description:
        With drag, Simulate agrees within 1e-9 m and 1e-9 m/s over 2 s with RK4 at a step 8 times
        finer (1/4800 s), at times off Simulate's 1/600 s grid.
    """

    model = Flight_Model_Str(Gravity=CONST_GRAVITY, Drag=CONST_DRAG)
    h = 1.0 / 4800.0
    reference = Integrate(CONST_X_0, 9601, h, 'RK4', Flight_Model_Str=model)
    index = np.arange(0, 9601, 37)
    x = Simulate(CONST_X_0, index * h, Flight_Model_Str=model)
    assert np.abs(x - reference[index]).max() < 1e-9
    # The drag acts: after 2 s the ball is far from the vacuum parabola (about 0.1 m).
    assert np.linalg.norm(x[-1, :3] - _Closed_Form(index[-1] * h)[:3]) > 0.05


def test_simulate_answers_each_time_independently_of_the_others():
    """
    Description:
        Query independence (SPEC §2.13.13): the state at a time is identical, bit for bit, whether it
        is asked alone, in a random subset or in reversed order; a batch of initial states gives each
        row what it gives alone.
    """

    model = Flight_Model_Str(Gravity=CONST_GRAVITY, Drag=CONST_DRAG)
    rng = np.random.default_rng(20261110)
    t = np.sort(rng.uniform(0.0, 1.5, 200))
    full = Simulate(CONST_X_0, t, Flight_Model_Str=model)
    subset = np.sort(rng.choice(200, 37, replace=False))
    assert np.array_equal(Simulate(CONST_X_0, t[subset], Flight_Model_Str=model), full[subset])
    assert np.array_equal(Simulate(CONST_X_0, t[subset][::-1], Flight_Model_Str=model), full[subset][::-1])
    assert np.array_equal(Simulate(CONST_X_0, t[:1], Flight_Model_Str=model), full[:1])
    batch = Simulate(np.array([CONST_X_0, CONST_X_0 + 0.1]), t, Flight_Model_Str=model)
    assert batch.shape == (2, 200, 6) and np.array_equal(batch[0], full)
    assert np.array_equal(batch[1], Simulate(CONST_X_0 + 0.1, t, Flight_Model_Str=model))


def test_simulate_at_gives_each_row_what_simulate_gives_it_alone():
    """
    Description:
        Simulate_At (SPEC §7.6): row i at its own time t[i] equals Simulate(x_0[i], [t[i]]) bit for bit,
        for times on and off the simulation grid, a time of 0 included, in any order of the rows.
    """

    model = Flight_Model_Str(Gravity=CONST_GRAVITY, Drag=CONST_DRAG)
    rng = np.random.default_rng(20261111)
    x_0 = CONST_X_0 + rng.normal(0.0, 0.5, (9, 6))
    t = np.r_[0.0, 1.0 / 600.0, 2.0 / 600.0, rng.uniform(0.0, 1.6, 6)]
    at = Simulate_At(x_0, t, Flight_Model_Str=model)
    for i in range(9):
        assert np.array_equal(at[i], Simulate(x_0[i], t[i:i + 1], Flight_Model_Str=model)[0]), i
    assert np.array_equal(Simulate_At(x_0[::-1], t[::-1], Flight_Model_Str=model), at[::-1])


def test_simulate_at_refuses_a_negative_time_and_a_count_mismatch():
    """
    Description:
        Simulate_At refuses a negative time, naming it, and a number of times that differs from the
        number of initial states, naming both counts.
    """

    model = Flight_Model_Str(Gravity=CONST_GRAVITY)
    with pytest.raises(Flight_Error_Cls, match=r'-0\.002 s is outside'):
        Simulate_At(np.array([CONST_X_0, CONST_X_0]), np.array([0.5, -0.002]), Flight_Model_Str=model)
    with pytest.raises(Flight_Error_Cls, match=r'2x6 states and 3 times'):
        Simulate_At(np.array([CONST_X_0, CONST_X_0]), np.array([0.5, 0.6, 0.7]), Flight_Model_Str=model)


def test_simulate_refuses_a_negative_time():
    """
    Description:
        A flight is simulated forward only: a negative time is refused, naming it.
    """

    with pytest.raises(Flight_Error_Cls, match=r'-0\.001 s is outside'):
        Simulate(CONST_X_0, np.array([0.5, -0.001]), Flight_Model_Str=Flight_Model_Str(Gravity=CONST_GRAVITY))


def _Irregular_Flight(drag=CONST_DRAG):
    model = Flight_Model_Str(Gravity=CONST_GRAVITY, Drag=drag)
    t = np.array([0, 1, 2, 5, 6, 7, 8, 15, 16, 30, 31, 32]) / 200.0
    return model, t, Simulate(CONST_X_0, t, Flight_Model_Str=model)[:, :3]


def test_the_fit_recovers_a_known_state_from_irregular_noise_free_samples():
    """
    Description:
        Fit_Initial_States (Gauss-Newton under the drag model, SPEC §7.4) recovers the state that generated 12
        irregular noise-free samples within 1e-8 m and 1e-7 m/s, unweighted and whitened, and leaves a
        residual below 1e-9 m.
    """

    model, t, p = _Irregular_Flight()
    W = np.stack([np.diag(np.linspace(1.0, 3.0, len(t)))] * 3) * 1e3
    for whitening in (None, [W]):
        x, residuals = Fit_Initial_States([t], [p], Flight_Model_Str=model, whitening=whitening)
        assert np.max(np.abs(x[0, :3] - CONST_X_0[:3])) < 1e-8 and np.max(np.abs(x[0, 3:] - CONST_X_0[3:])) < 1e-7
        assert np.max(np.abs(residuals[0])) < 1e-9


def test_whitening_by_the_identity_is_least_squares():
    """
    Description:
        Whitening every axis by the identity gives the unweighted fit within 1e-12 (the same equations with the
        rows ordered by axis, so only the rounding differs): least squares is the generalised fit without
        weights.
    """

    model, t, p = _Irregular_Flight()
    rng = np.random.default_rng(1)
    p = p + rng.normal(0.0, 1e-3, p.shape)
    plain, _ = Fit_Initial_States([t], [p], Flight_Model_Str=model)
    white, _ = Fit_Initial_States([t], [p], Flight_Model_Str=model, whitening=[np.stack([np.eye(len(t))] * 3)])
    assert np.max(np.abs(plain - white)) < 1e-12 and np.max(np.abs(plain - CONST_X_0)) > 1e-4


def test_whitening_weights_each_sample_by_its_stated_error():
    """
    Description:
        One sample off by 0.1 m on every axis: unweighted it pulls the fitted position by millimetres; whitened
        with that sample's standard deviation 1000 times the others' (1 m against 1 mm), it moves the fit by less
        than 0.1 mm (generalised least squares, the GLS_M4 predictor).
    """

    model, t, p = _Irregular_Flight()
    p = p.copy()
    p[5] += 0.1
    std = np.full(len(t), 1e-3)
    std[5] = 1.0
    W = np.stack([np.diag(1.0 / std)] * 3)
    plain, _ = Fit_Initial_States([t], [p], Flight_Model_Str=model)
    white, _ = Fit_Initial_States([t], [p], Flight_Model_Str=model, whitening=[W])
    assert np.max(np.abs(plain[0, :3] - CONST_X_0[:3])) > 1e-3
    assert np.max(np.abs(white[0, :3] - CONST_X_0[:3])) < 1e-4


def test_a_batch_fits_each_flight_as_its_own_fit_when_the_start_is_given():
    """
    Description:
        Two flights fitted in one batch (the identification's way) and each alone give the same states within
        1e-12: each flight's normal equations use only its own samples, and Simulate's state at a time does not
        depend on the other times requested.
    """

    model, t, p = _Irregular_Flight()
    t2 = np.array([0, 3, 4, 9, 20, 21]) / 300.0
    p2 = Simulate(CONST_X_0 + np.array([0.1, 0.0, 0.0, 0.0, 0.2, 0.0]), t2, Flight_Model_Str=model)[:, :3]
    both, _ = Fit_Initial_States([t, t2], [p, p2], Flight_Model_Str=model)
    alone = np.vstack([Fit_Initial_States([a], [b], Flight_Model_Str=model)[0] for a, b in ((t, p), (t2, p2))])
    assert np.max(np.abs(both - alone)) < 1e-12
