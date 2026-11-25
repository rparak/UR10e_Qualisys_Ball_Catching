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
# File Name: ../test/Recording/test_identification.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the identification under test, the flight model it fits, the flights
#   ../Recording/Identification
from Recording.Identification import (Bootstrap_Drag_Sd, Edge_Interior_RMS, Estimated_Floor, Fit_Initial_States, Identify_Drag,
                                      Trim_Flight)
#   ../Flight/Core
from Flight.Core import Flight_Model_Str, Simulate
#   ../Recording/Core, Segmentation
from Recording.Core import Recording_Error_Cls
from Recording.Segmentation import Flight_Str

"""
Description:
    Initialization of constants.

    Synthetic flights are simulated with a known k and sampled with dropouts, so the expectation is the
    k they were made with, never the code's output. The real-data pin is the slow test of
    test/scripts/test_catch.py.
"""
# Gravity and the drag the synthetic flights are made with (about F3's identified value).
#   Unit [m/s^2], [1/m]
CONST_GRAVITY = 9.81
CONST_DRAG = 0.0093
# Release states of the synthetic flights (QTM-like: thrown towards -y).
#   Unit [m], [m/s]
CONST_RELEASES = ([0.1, 0.5, 1.0, 0.2, -4.5, 4.9], [-0.2, 0.3, 0.9, -0.1, -4.9, 4.5], [0.3, 0.8, 1.1, 0.0, -4.2, 5.2],
                  [0.0, 0.0, 1.0, 0.3, -5.2, 4.3])


def _Model(drag=CONST_DRAG):
    return Flight_Model_Str(Gravity=np.array([0.0, 0.0, -CONST_GRAVITY]), Drag=drag)


def _Flights(set_name='Identification', in_hand=False, rate=200.0, duration=0.8):
    """
    Description:
        Synthetic flights with irregular sampling (about half the frames dropped, seed 20261110). With
        in_hand, the samples of the first 40 ms are still in the hand: they lag the free flight by
        0.4 m/s x (0.04 s - t), up to 16 mm.
    """

    rng = np.random.default_rng(20261110)
    flights = []
    for i, x_0 in enumerate(CONST_RELEASES):
        frame = np.arange(int(duration * rate) + 1)
        keep = rng.random(len(frame)) < 0.5
        keep[0] = keep[-1] = True
        frame = frame[keep]
        p = Simulate(np.array(x_0), frame / rate, Flight_Model_Str=_Model())[:, :3]
        if in_hand:
            t = frame / rate
            p = p - np.outer(np.clip(0.04 - t, 0.0, None) * 0.4, [0.0, -0.7, 0.7])
        flights.append(Flight_Str(Folder='Synthetic', Experiment=i + 1, Rate=rate, Set=set_name, First_Line=2, Last_Line=2 + len(frame),
                                  Frame=frame, Position=p))
    return flights


def test_the_fit_refuses_a_validation_flight_naming_it():
    """
    Description:
        k is identified on the identification set only (SPEC §7.3): one validation flight among them
        refuses the whole call, naming that flight and its set.
    """

    flights = _Flights()
    flights[2].Set = 'Validation'
    with pytest.raises(Recording_Error_Cls, match=r'1 flight\(s\) are not of it: Synthetic/Experiment_3.txt \(Validation\)'):
        Identify_Drag(flights, gravity=CONST_GRAVITY, trim=0.0)


def test_a_known_drag_is_identified_from_irregular_flights():
    """
    Description:
        Flights made with k = 0.0093 1/m, half their frames dropped, give back k within 1e-8 1/m and
        residuals below 1 um.
    """

    drag, coefficients, fitted = Identify_Drag(_Flights(), gravity=CONST_GRAVITY, trim=0.0)
    assert drag == pytest.approx(CONST_DRAG, abs=1e-8)
    assert coefficients.shape == (4, 3) and max(f.Max for f in fitted) < 1e-6


def test_the_trim_drops_an_in_hand_start_that_biases_k():
    """
    Description:
        Flights whose first 40 ms are still in the hand: with the 40 ms trim k is recovered within
        1e-8 1/m; without it k is biased by more than 1e-4 1/m (F3 Q5).
    """

    flights = _Flights(in_hand=True)
    trimmed, _, _ = Identify_Drag(flights, gravity=CONST_GRAVITY, trim=0.040)
    untrimmed, _, _ = Identify_Drag(flights, gravity=CONST_GRAVITY, trim=0.0)
    assert trimmed == pytest.approx(CONST_DRAG, abs=1e-8)
    assert abs(untrimmed - CONST_DRAG) > 1e-4


def test_trim_flight_rebases_the_frames_and_moves_the_first_line():
    """
    Description:
        Trimming 40 ms at 200 Hz drops the samples before frame 8; the first kept frame becomes 0 and
        the first line moves by the samples dropped.
    """

    flight = Flight_Str(Folder='Synthetic', Experiment=1, Rate=200.0, Set='Identification', First_Line=10, Last_Line=15,
                        Frame=np.array([0, 3, 7, 8, 9, 12]), Position=np.arange(18, dtype=float).reshape(6, 3))
    trimmed = Trim_Flight(flight, 0.040)
    assert trimmed.Frame.tolist() == [0, 1, 4] and trimmed.First_Line == 13 and trimmed.Position[0].tolist() == [9.0, 10.0, 11.0]
    assert flight.Frame.tolist() == [0, 3, 7, 8, 9, 12]


def test_the_initial_state_is_recovered_under_a_given_model():
    """
    Description:
        Under the model the flights were made with, each flight's release state comes back within
        1e-8 (m, m/s); a validation flight is accepted, since no model parameter is fitted.
    """

    fitted = Fit_Initial_States(_Flights(set_name='Validation'), Flight_Model_Str=_Model())
    for f, x_0 in zip(fitted, CONST_RELEASES):
        assert np.abs(f.X_0 - np.array(x_0)).max() < 1e-8 and f.RMS < 1e-9


def test_the_bootstrap_is_reproducible_from_its_seed():
    """
    Description:
        The same per-flight quadratics and seed give the same sd; identical flights give 0 (every
        resample has the same minimum); differing flights give a positive sd that a new seed changes.
    """

    same = np.tile([2.0, -0.04, 1.0], (10, 1))
    assert Bootstrap_Drag_Sd(same, seed=1, resamples=200) == pytest.approx(0.0, abs=1e-15)
    rng = np.random.default_rng(3)
    differing = np.column_stack((rng.uniform(1.0, 3.0, 10), rng.uniform(-0.06, -0.02, 10), np.ones(10)))
    first = Bootstrap_Drag_Sd(differing, seed=1, resamples=200)
    assert first > 0.0 and first == Bootstrap_Drag_Sd(differing, seed=1, resamples=200)
    assert first != Bootstrap_Drag_Sd(differing, seed=2, resamples=200)


def test_the_estimated_floor_removes_a_cubic_and_not_a_quartic():
    """
    Description:
        The floor's correction is a per-axis cubic (F3 Q6): a cubic residual leaves nothing, a quartic
        leaves a part (degree 3 exactly).
    """

    t = np.linspace(0.0, 1.2, 150)
    u = t / 1.2
    cubic = np.column_stack((0.01 * u ** 3, -0.004 * u ** 2 + 0.002, 0.003 * u))
    assert Estimated_Floor(t, cubic) < 1e-12
    assert Estimated_Floor(t, np.column_stack((0.01 * u ** 4, 0 * u, 0 * u))) > 1e-5


def test_the_edge_and_interior_residuals_are_split_at_the_dropouts():
    """
    Description:
        Frames 0, 1, 2, 5, 6, 7: frames 2 and 5 border the dropout (edges, residual 3 and 4 mm), the
        others are interior (1 mm each).
    """

    residual = np.array([[0.001, 0, 0], [0.001, 0, 0], [0.003, 0, 0], [0, 0.004, 0], [0, 0, 0.001], [0, 0, 0.001]])
    edge, interior = Edge_Interior_RMS(np.array([0, 1, 2, 5, 6, 7]), residual)
    assert edge == pytest.approx(np.sqrt((0.003 ** 2 + 0.004 ** 2) / 2.0), abs=1e-15)
    assert interior == pytest.approx(0.001, abs=1e-15)
