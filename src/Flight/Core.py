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
# File Name: ../Flight/Core.py
## =========================================================================== ##

# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp

"""
Description:
    Initialization of constants.

    The flight model of the ball, gravity and quadratic drag a = g - k |v| v (SPEC §2.3, identified by
    F3), the fixed-step integrators (SPEC §7.4), ported from Ball/Simulation.py and
    Prediction/Numerical_Analysis.py of commit 0e59db2, the true state at any times (Simulate) and the
    fit of an initial state to positions by Gauss-Newton (Fit_Initial_States, F3 and F7). The
    state is x = (r_x, r_y, r_z, v_x, v_y, v_z) [m, m/s]. The model is always passed in, keyword-only
    and without a default, so no code integrates with a model it did not name (SPEC §7.4).
    Pure computation: no file, no configuration, no printing.

    Each step keeps the order of operations of the 2022 code (for example (1.0 / 2.0) * h * f), and a
    model without drag takes the 2022 path unchanged, so the 2022 baseline is reproduced bit for bit
    (SPEC §2.1).
"""
# The internal step of Simulate: RK4 with drag k = 0.00905 1/m from 7 m/s converges to 1.2e-13 m over
# 2 s against 1/19200 s (F3 Part A, 2026-11-10); 1/600 s also divides the frame periods of 100, 200 and
# 300 Hz, so a recorded sample time falls on the grid.
#   Unit [float, s]
CONST_SIMULATION_STEP = 1.0 / 600.0
# The Gauss-Newton fit of an initial state (Fit_Initial_States; F3, moved here for the
# predictors): the forward-difference step [m, m/s], the convergence of the update and the most
# iterations.
#   Unit [float], [float, m], [int]
CONST_DIFFERENCE_STEP = 1e-6
CONST_FIT_TOLERANCE = 1e-10
CONST_FIT_ITERATIONS = 10


class Flight_Error_Cls(ValueError):
    """
    Description:
        A flight cannot be computed as asked (an unknown integrator, a negative time). The message
        names the value and the legal ones; a script prints it verbatim as an [ERROR] line.
    """


@dataclass
class Flight_Model_Str:
    """
    Description:
        The forces on the ball in flight, as the acceleration they cause: a = g - k |v| v, gravity
        and quadratic drag (SPEC §2.3; F3 identified k on the identification flights, no further term
        passed its held-out test). The drag law is isotropic and gravity is along
        z, so the model is the same in the QTM and the robot frame (a rotation about z and a
        translation, SPEC §7.3). Drag 0 is the vacuum model of the 2022 code.
    """

    # The acceleration of gravity, in the frame of the state (z up: (0, 0, -g)).
    #   Unit [Vector<float> 1x3, m/s^2]
    Gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # The drag constant k = rho C_d A / (2 m) of a = g - k |v| v, identified directly (it does not
    # need the ball's mass).
    #   Unit [float, 1/m]
    Drag: float = 0.0
    # The uncertainty of Drag as one standard deviation (a robustness dataset draws from it, SPEC §7.6).
    #   Unit [float, 1/m]
    Drag_Std: float = 0.0


def Acceleration(v: np.ndarray, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        The model's acceleration at one or many velocities: g - k |v| v. The speed is computed
        element by element, so each row's value does not depend on the other rows.

    Args:
        (1) v [Matrix<float> ...x3]: The velocities [m/s], one per row.
        (2) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Matrix<float> ...x3]: The accelerations [m/s^2].
    """

    speed = np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])
    return Flight_Model_Str.Gravity - (Flight_Model_Str.Drag * speed)[..., None] * v


def State_Derivative(v: np.ndarray, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        The time derivative of the state: (v, a), with a the model's acceleration. The
        acceleration does not depend on the position, so only the velocity is passed. Without drag
        it is the 2022 vacuum derivative, bit for bit.

    Args:
        (1) v [Vector<float> 1x3]: The velocity [m/s].
        (2) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<float> 1x6]: (v, a) [m/s, m/s^2].
    """

    if Flight_Model_Str.Drag == 0.0:
        return np.hstack((v, Flight_Model_Str.Gravity))
    return np.hstack((v, Acceleration(v, Flight_Model_Str=Flight_Model_Str)))


def Midpoint_Step(x: np.ndarray, h: float, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        One step of the explicit midpoint method: x_mid = x + h/2 f(x), x_next = x + h f(x_mid).

    Args:
        (1) x [Vector<float> 1x6]: The state [m, m/s].
        (2) h [float]: The time step [s].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<float> 1x6]: The state after h [m, m/s].
    """

    x_mid = x + (1.0 / 2.0) * h * State_Derivative(x[3:], Flight_Model_Str=Flight_Model_Str)
    return x + h * State_Derivative(x_mid[3:], Flight_Model_Str=Flight_Model_Str)


def Heun_Step(x: np.ndarray, h: float, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        One step of Heun's method, an Euler predictor with one trapezoidal corrector (the "Euler +
        trapezoid / PECE" of SPEC §2.7): x_est = x + h f(x), x_next = x + h/2 (f(x) + f(x_est)).

    Args:
        (1) x [Vector<float> 1x6]: The state [m, m/s].
        (2) h [float]: The time step [s].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<float> 1x6]: The state after h [m, m/s].
    """

    f_0 = State_Derivative(x[3:], Flight_Model_Str=Flight_Model_Str)
    x_estimate = x + h * f_0
    return x + (1.0 / 2.0) * h * (f_0 + State_Derivative(x_estimate[3:], Flight_Model_Str=Flight_Model_Str))


def RK2_Step(x: np.ndarray, h: float, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        One step of the second-order Runge-Kutta method: k_1 = h f(x), k_2 = h f(x + k_1 / 2),
        x_next = x + k_2.

    Args:
        (1) x [Vector<float> 1x6]: The state [m, m/s].
        (2) h [float]: The time step [s].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<float> 1x6]: The state after h [m, m/s].
    """

    k_1 = h * State_Derivative(x[3:], Flight_Model_Str=Flight_Model_Str)
    k_2 = h * State_Derivative(x[3:] + (1.0 / 2.0) * k_1[3:], Flight_Model_Str=Flight_Model_Str)
    return x + k_2


def RK4_Step(x: np.ndarray, h: float, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        One step of the fourth-order Runge-Kutta method: k_1 = h f(x), k_2 = h f(x + k_1 / 2),
        k_3 = h f(x + k_2 / 2), k_4 = h f(x + k_3), x_next = x + (k_1 + 2 k_2 + 2 k_3 + k_4) / 6.

    Args:
        (1) x [Vector<float> 1x6]: The state [m, m/s].
        (2) h [float]: The time step [s].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Vector<float> 1x6]: The state after h [m, m/s].
    """

    k_1 = h * State_Derivative(x[3:], Flight_Model_Str=Flight_Model_Str)
    k_2 = h * State_Derivative(x[3:] + (1.0 / 2.0) * k_1[3:], Flight_Model_Str=Flight_Model_Str)
    k_3 = h * State_Derivative(x[3:] + (1.0 / 2.0) * k_2[3:], Flight_Model_Str=Flight_Model_Str)
    k_4 = h * State_Derivative(x[3:] + k_3[3:], Flight_Model_Str=Flight_Model_Str)
    return x + (1.0 / 6.0) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


"""
Description:
    The integrators by name.
"""
# The step function of each integrator (SPEC §2.7, §7.4).
#   Unit [Dict<string, function>]
CONST_INTEGRATORS = {'Midpoint': Midpoint_Step, 'Heun': Heun_Step, 'RK2': RK2_Step, 'RK4': RK4_Step}


def Integrator(method: str) -> tp.Callable[..., np.ndarray]:
    """
    Description:
        The step function of an integrator, by name.

    Args:
        (1) method [string]: One of the keys of CONST_INTEGRATORS.

    Returns:
        (1) parameter [function]: Its step function (x, h, *, Flight_Model_Str) -> x_next.
    """

    if method not in CONST_INTEGRATORS:
        raise Flight_Error_Cls('The integrator "%s" is not one of %s. Use one of them.'
                               % (method, ', '.join(CONST_INTEGRATORS)))
    return CONST_INTEGRATORS[method]


def Integrate(x_0: np.ndarray, n: int, h: float, method: str, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        n states at a fixed step h from x_0, the first being x_0 itself.

    Args:
        (1) x_0 [Vector<float> 1x6]: The first state [m, m/s].
        (2) n [int]: The number of states (n >= 1).
        (3) h [float]: The time step [s].
        (4) method [string]: One of the keys of CONST_INTEGRATORS.
        (5) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Matrix<float> nx6]: The states at 0, h, ..., (n - 1) h [m, m/s].
    """

    step = Integrator(method)
    x = np.zeros((n, len(x_0)))
    x[0] = x_0
    for i in range(0, n - 1):
        x[i + 1] = step(x[i, :], h, Flight_Model_Str=Flight_Model_Str)
    return x


def _RK4_Batch(x: np.ndarray, h: tp.Union[float, np.ndarray], Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        One RK4 step of many states at once, each with its own step (or one shared step).

    Args:
        (1) x [Matrix<float> ...x6]: The states [m, m/s].
        (2) h [float or Matrix<float> ...x1]: The steps [s].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: The flight model.

    Returns:
        (1) parameter [Matrix<float> ...x6]: The states after h [m, m/s].
    """

    r, v = x[..., :3], x[..., 3:]
    a_1 = Acceleration(v, Flight_Model_Str=Flight_Model_Str)
    v_2 = v + 0.5 * h * a_1
    a_2 = Acceleration(v_2, Flight_Model_Str=Flight_Model_Str)
    v_3 = v + 0.5 * h * a_2
    a_3 = Acceleration(v_3, Flight_Model_Str=Flight_Model_Str)
    v_4 = v + h * a_3
    a_4 = Acceleration(v_4, Flight_Model_Str=Flight_Model_Str)
    return np.concatenate((r + h / 6.0 * (v + 2.0 * v_2 + 2.0 * v_3 + v_4), v + h / 6.0 * (a_1 + 2.0 * a_2 + 2.0 * a_3 + a_4)), axis=-1)


def Simulate(x_0: np.ndarray, t: np.ndarray, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        The true state at any times under the flight model (SPEC §7.6): the one function shared by the
        dataset, the predictors and F5's MuJoCo check. RK4 runs on the grid of CONST_SIMULATION_STEP
        from x_0, and each time t_i takes one partial RK4 step from the grid state before it. The grid
        does not depend on the requested times, so the state at t_i is the same whatever other times
        are requested (query independence, SPEC §2.13.13). Many initial states run at once.

    Args:
        (1) x_0 [Vector<float> 1x6 or Matrix<float> Bx6]: The state(s) at t = 0 [m, m/s].
        (2) t [Vector<float> 1xM]: The times [s], each >= 0, in any order.
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Matrix<float> Mx6 or Tensor<float> BxMx6]: The state at each time [m, m/s].
    """

    t = np.asarray(t, dtype=np.float64).reshape(-1)
    invalid = t[~np.isfinite(t) | (t < 0.0)]
    if invalid.size:
        bad = invalid[0]
        raise Flight_Error_Cls('The time %r s is outside [0, inf) s: a flight is simulated forward from its initial state only. '
                               'Request times >= 0.' % (float(bad),))
    x_0 = np.asarray(x_0, dtype=np.float64)
    h = CONST_SIMULATION_STEP
    n = np.floor(t / h).astype(np.int64)
    grid = np.empty((int(n.max()) + 1 if t.size else 1,) + x_0.shape)
    grid[0] = x_0
    for i in range(1, grid.shape[0]):
        grid[i] = _RK4_Batch(grid[i - 1], h, Flight_Model_Str)
    # The partial step from the grid state before each time; batch axis first in the result.
    start = np.moveaxis(grid[n], 0, -2) if x_0.ndim == 2 else grid[n]
    return _RK4_Batch(start, (t - n * h)[:, None], Flight_Model_Str)


def Simulate_At(x_0: np.ndarray, t: np.ndarray, *, Flight_Model_Str: Flight_Model_Str) -> np.ndarray:
    """
    Description:
        The true state of each initial state at its own time: row i at t[i] (SPEC §7.6; the dataset's
        shooting and floor time). It takes Simulate's grid of CONST_SIMULATION_STEP and the same partial
        RK4 step, so row i equals Simulate(x_0[i], [t[i]]) bit for bit, whatever the other rows are.

    Args:
        (1) x_0 [Matrix<float> Bx6]: The states at t = 0 [m, m/s].
        (2) t [Vector<float> 1xB]: The time of each row [s], each >= 0.
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.

    Returns:
        (1) parameter [Matrix<float> Bx6]: The state of each row at its time [m, m/s].
    """

    x_0 = np.asarray(x_0, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if x_0.ndim != 2 or x_0.shape[1] != 6 or x_0.shape[0] != t.size:
        raise Flight_Error_Cls('Simulate_At needs one time per initial state: %s states and %d times were given. Pass a Bx6 matrix '
                               'and B times.' % ('x'.join(str(n) for n in x_0.shape), t.size))
    invalid = t[~np.isfinite(t) | (t < 0.0)]
    if invalid.size:
        raise Flight_Error_Cls('The time %r s is outside [0, inf) s: a flight is simulated forward from its initial state only. '
                               'Request times >= 0.' % (float(invalid[0]),))
    h = CONST_SIMULATION_STEP
    n = np.floor(t / h).astype(np.int64)
    x, start = x_0.copy(), np.empty_like(x_0)
    # Only the rows that have not reached their grid index are stepped (every operation is per row, so a
    # row's states do not depend on which other rows are stepped with it); the rows due at each index come
    # from one stable sort of the indices.
    order = np.argsort(n, kind='stable')
    bounds = np.searchsorted(n[order], np.arange(int(n.max()) + 2 if t.size else 1))
    active = np.arange(t.size)
    for i in range(int(n.max()) + 1 if t.size else 0):
        due = order[bounds[i]:bounds[i + 1]]
        start[due] = x[np.searchsorted(active, due)]
        if i < n.max():
            keep = n[active] > i
            active, x = active[keep], x[keep]
            x = _RK4_Batch(x, h, Flight_Model_Str)
    return _RK4_Batch(start, (t - n * h)[:, None], Flight_Model_Str)


def _Vacuum_State(time: np.ndarray, position: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """
    Description:
        The least-squares initial state without drag (closed form): the Gauss-Newton start.

    Args:
        (1) time [Vector<float> 1xM]: The sample times from the state's time [s].
        (2) position [Matrix<float> Mx3]: The positions [m].
        (3) gravity [Vector<float> 1x3]: The acceleration of gravity [m/s^2].

    Returns:
        (1) parameter [Vector<float> 1x6]: The state [m, m/s].
    """

    A = np.stack([np.ones_like(time), time], axis=1)
    c, *_ = np.linalg.lstsq(A, position - 0.5 * np.outer(time ** 2, gravity), rcond=None)
    return np.hstack((c[0], c[1]))


def Fit_Initial_States(times: tp.List[np.ndarray], positions: tp.List[np.ndarray], *, Flight_Model_Str: Flight_Model_Str,
                       X_0: tp.Optional[np.ndarray] = None,
                       whitening: tp.Optional[tp.List[np.ndarray]] = None) -> tp.Tuple[np.ndarray, tp.List[np.ndarray]]:
    """
    Description:
        Each flight's state at its time 0 under a given flight model, by Gauss-Newton on its positions
        (forward differences, every flight in one batch of Simulate on the union of their times). No model
        parameter is fitted. All flights iterate until the largest update of any is below
        CONST_FIT_TOLERANCE (at most CONST_FIT_ITERATIONS), so a batch of one flight is that flight's own
        fit. Without whitening the residual is weighted equally (least squares, the identification);
        with it, each axis of each flight is weighted by the inverse Cholesky factor of its
        error covariance (generalised least squares, the GLS_M4 predictor; SPEC §7.7).

    Args:
        (1) times [List<Vector<float> 1xM>]: Per flight its sample times [s], each >= 0.
        (2) positions [List<Matrix<float> Mx3>]: Per flight its positions [m].
        (3) Flight_Model_Str [Flight_Model_Str(object)]: REQUIRED, keyword-only: the flight model.
        (4) X_0 [Matrix<float> Fx6]: The start of each fit [m, m/s]; None: the vacuum fit.
        (5) whitening [List<Tensor<float> 3xMxM>]: Per flight, per axis, the inverse Cholesky factor L^-1 of
                                                    the error covariance L L^T [1/m]; None: least squares.

    Returns:
        (1) parameter 1 [Matrix<float> Fx6]: The fitted states [m, m/s].
        (2) parameter 2 [List<Matrix<float> Mx3>]: Per flight the residual, positions minus simulated [m].
    """

    F, P = len(times), 6
    x = (np.array([_Vacuum_State(t, p, Flight_Model_Str.Gravity) for t, p in zip(times, positions)]) if X_0 is None
         else np.array(X_0, dtype=np.float64))
    union = np.unique(np.concatenate(times))
    index = [np.searchsorted(union, t) for t in times]
    perturb = np.vstack((np.zeros(P), CONST_DIFFERENCE_STEP * np.eye(P)))

    def Positions(states: np.ndarray) -> np.ndarray:
        """
        Description:
            The simulated positions of many initial states at the flights' union of times.

        Args:
            (1) states [Matrix<float> Bx6]: The initial states [m, m/s].

        Returns:
            (1) parameter [Tensor<float> BxTx3]: The positions [m].
        """

        return Simulate(states, union, Flight_Model_Str=Flight_Model_Str)[..., :3]

    for _ in range(CONST_FIT_ITERATIONS):
        p = Positions((x[:, None, :] + perturb[None, :, :]).reshape(F * (P + 1), P)).reshape(F, P + 1, len(union), 3)
        largest = 0.0
        for i in range(F):
            base = p[i, 0, index[i]]
            J = np.stack([(p[i, j + 1, index[i]] - base).ravel() / CONST_DIFFERENCE_STEP for j in range(P)], axis=1)
            r = (positions[i] - base).ravel()
            if whitening is not None:
                J = np.concatenate([whitening[i][a] @ J[a::3] for a in range(3)], axis=0)
                r = np.concatenate([whitening[i][a] @ r[a::3] for a in range(3)])
            d, *_ = np.linalg.lstsq(J, r, rcond=None)
            x[i] += d
            largest = max(largest, float(np.abs(d).max()))
        if largest < CONST_FIT_TOLERANCE:
            break
    p = Positions(x)
    return x, [positions[i] - p[i, index[i]] for i in range(F)]
