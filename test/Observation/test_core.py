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
# File Name: ../test/Observation/test_core.py
## =========================================================================== ##

# Hashlib (Secure hashes; the pinned error of the unbounded model)
import hashlib
# Dataclasses (Data Classes)
from dataclasses import replace
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the observation model under test
#   ../Observation/Core
from Observation.Core import (Assign_Pools, Donor_Str, Draw_Pattern, Error_Covariance, Frame_Count, Observation_Error,
                              Observation_Error_Cls, Observation_Model_Str, Present_On_Grid)

"""
Description:
    Initialization of constants.

    The donors here are constructed; every expectation follows from the replay rule of SPEC §7.5 (a
    sample stands for its own frame period [t - T/2, t + T/2)) or from the model's stated variances.
"""
# A model with every component at zero.
#   Unit [Observation_Model_Str]
CONST_ZERO = Observation_Model_Str(White_Std=np.zeros(3), White_Dof=6.0, Drift_Std=np.zeros(3), Drift_Time=0.08,
                                   Offset_Std=np.zeros(3), Offset_Time=0.1, Offset_Dof=6.0)


def _Donor(name, rate, frames, pool='A'):
    return Donor_Str(Name=name, Rate=float(rate), Time=np.asarray(frames, dtype=float) / rate, Pool=pool)


def _Model(white=0.0, drift=0.0, offset=0.0, dof=30.0):
    return Observation_Model_Str(White_Std=np.full(3, white), White_Dof=dof, Drift_Std=np.full(3, drift), Drift_Time=0.08,
                                 Offset_Std=np.full(3, offset), Offset_Time=0.1, Offset_Dof=dof)


def test_a_donor_replayed_at_its_own_rate_gives_its_own_frames():
    """
    Description:
        A donor replayed on the grid of its own rate from its release gives exactly its own frames.
    """

    frames = [0, 1, 2, 5, 9, 10, 11, 20]
    present = Present_On_Grid([(_Donor('d', 200, frames), 0.0, np.inf)], 200.0, 21)
    assert np.flatnonzero(present).tolist() == frames


def test_a_sample_holds_its_own_frame_period_half_open():
    """
    Description:
        A 100 Hz donor with samples at 0, 10 and 40 ms replayed at 200 Hz: a 200 Hz frame is present when
        it lies in [t_k - 5 ms, t_k + 5 ms) of a sample, so frames 0, 1 (5 ms), 2, 7 (35 ms) and 8 are
        present and 3 (15 ms) and 9 (45 ms) are not; nothing after the last sample.
    """

    present = Present_On_Grid([(_Donor('d', 100, [0, 1, 4]), 0.0, np.inf)], 200.0, 12)
    assert np.flatnonzero(present).tolist() == [0, 1, 2, 7, 8]


def test_the_release_frame_is_always_present_and_frame_count_counts_to_the_end():
    """
    Description:
        Frame 0 (the release) carries a sample even when the segment's donor has none there, and a
        flight of duration D at rate R has every frame j with j / R <= D.
    """

    donor = _Donor('d', 200, [0, 50])
    present = Present_On_Grid([(donor, 0.1, np.inf)], 200.0, 60)
    assert present[0] and np.flatnonzero(present).tolist() == [0, 50]
    assert Frame_Count(1.0, 200.0) == 201 and Frame_Count(0.999, 200.0) == 200 and Frame_Count(1.0, 370.0) == 371


def test_composed_and_recorded_patterns_of_one_donor_are_its_replay():
    """
    Description:
        With one donor in the pool, a composed pattern (whatever its cut times) and a recorded one are
        the replay of that donor; 'none' keeps every frame.
    """

    donor = _Donor('d', 200, [0, 1, 2, 7, 8, 30, 31, 32, 60, 100, 101, 230, 240])
    rng = np.random.default_rng(1)
    for pattern in ('composed', 'recorded'):
        present, names = Draw_Pattern([donor], 200.0, 1.2, pattern, 'all', rng)
        assert np.flatnonzero(present).tolist() == [0, 1, 2, 7, 8, 30, 31, 32, 60, 100, 101, 230, 240] and set(names) == {'d'}
    present, names = Draw_Pattern([donor], 200.0, 1.2, 'none', 'all', rng)
    assert present.all() and present.size == 241 and names == []


def test_composed_segments_switch_donors_at_the_cut_mean():
    """
    Description:
        Two donors, one with a sample at every frame and one with none after the release: a composed
        pattern alternates their segments, which last CONST_CUT_MEAN = 0.25 s on average and pick either
        donor with probability 1/2, so a present run lasts 0.25 / (1 - 1/2) = 0.5 s on average (within 5 %
        over 200 patterns of 20 s).
    """

    full = _Donor('full', 200, np.arange(0, 6001))
    empty = _Donor('empty', 200, [0])
    rng = np.random.default_rng(20261111)
    lengths = []
    for _ in range(200):
        present, _ = Draw_Pattern([full, empty], 200.0, 20.0, 'composed', 'all', rng)
        edges = np.flatnonzero(np.diff(np.r_[0, present.astype(int), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if start > 0 and stop < present.size:
                lengths.append((stop - start) / 200.0)
    assert np.mean(lengths) == pytest.approx(0.5, rel=0.05)


def test_a_flight_draws_from_the_ten_donors_of_its_pool_nearest_in_duration():
    """
    Description:
        Twelve 200 Hz donors ending at 1.00, 1.02, ..., 1.22 s: a 1.0 s flight draws from the ten ending at
        1.00-1.18 s only. A 370 Hz flight draws from the 300 Hz donors only (F6 Q4).
    """

    donors = [_Donor('d%02d' % i, 200, np.arange(0, 201 + 4 * i, 4).tolist() + [200 + 4 * i]) for i in range(12)]
    rng = np.random.default_rng(3)
    names = set()
    for _ in range(400):
        names.update(Draw_Pattern(donors, 200.0, 1.0, 'recorded', 'all', rng)[1])
    assert names == {'d%02d' % i for i in range(10)}
    mixed = donors + [_Donor('t%d' % i, 300, [0, 3, 300]) for i in range(3)]
    names = set()
    for _ in range(100):
        names.update(Draw_Pattern(mixed, 370.0, 1.0, 'composed', 'all', rng)[1])
    assert names == {'t0', 't1', 't2'}


def test_an_unknown_rate_pattern_or_pool_is_refused_naming_it():
    """
    Description:
        A rate without donors (150 Hz), an unknown pattern and an unknown pool are refused, each named.
    """

    donors = [_Donor('d', 200, [0, 1, 200])]
    rng = np.random.default_rng(0)
    with pytest.raises(Observation_Error_Cls, match='150 Hz has no donors'):
        Draw_Pattern(donors, 150.0, 1.0, 'composed', 'all', rng)
    with pytest.raises(Observation_Error_Cls, match="pattern 'shuffled' is unknown"):
        Draw_Pattern(donors, 200.0, 1.0, 'shuffled', 'all', rng)
    with pytest.raises(Observation_Error_Cls, match="pool 'C' is unknown"):
        Draw_Pattern(donors, 200.0, 1.0, 'composed', 'C', rng)


def test_the_pools_split_each_rate_in_halves_from_the_names_only():
    """
    Description:
        Per rate the first half (len // 2) of the donors is pool A: 10, 30 and 5 donors give 5, 15 and 2
        in A; the same names in another order give the same pools.
    """

    donors = [_Donor('F%d/Experiment_%d.txt' % (rate, i), rate, [0, 1]) for rate, count in ((100, 10), (200, 30), (300, 5))
              for i in range(count)]
    pooled = Assign_Pools(donors)
    for rate, a in ((100, 5), (200, 15), (300, 2)):
        assert sum(d.Pool == 'A' for d in pooled if d.Rate == rate) == a
    again = {d.Name: d.Pool for d in Assign_Pools(donors[::-1])}
    assert {d.Name: d.Pool for d in pooled} == again


def test_the_error_vanishes_without_components_and_scales_exactly():
    """
    Description:
        A model with every component at zero adds no error; Observation.Noise.Scale multiplies the error
        of the same generator exactly.
    """

    time = np.array([0, 1, 2, 5, 6, 9]) / 200.0
    assert np.array_equal(Observation_Error(time, 200.0, Observation_Model_Str=CONST_ZERO, scale=1.0, rng=np.random.default_rng(1)),
                          np.zeros((6, 3)))
    model = _Model(white=4e-4, drift=3e-3, offset=9e-3, dof=3.0)
    one = Observation_Error(time, 200.0, Observation_Model_Str=model, scale=1.0, rng=np.random.default_rng(2))
    two = Observation_Error(time, 200.0, Observation_Model_Str=model, scale=2.0, rng=np.random.default_rng(2))
    assert np.array_equal(two, 2.0 * one) and np.any(one != 0.0)


def test_the_offset_changes_only_after_a_dropout():
    """
    Description:
        With the offset alone, the error is constant within a run of consecutive frames and changes at
        each dropout.
    """

    frames = np.array([0, 1, 2, 3, 10, 11, 12, 30, 31])
    error = Observation_Error(frames / 200.0, 200.0, Observation_Model_Str=_Model(offset=9e-3), scale=1.0, rng=np.random.default_rng(4))
    for run in ((0, 4), (4, 7), (7, 9)):
        assert np.all(error[run[0]:run[1]] == error[run[0]])
    assert np.all(error[3] != error[4]) and np.all(error[6] != error[7])


def test_each_component_has_its_stated_variance():
    """
    Description:
        The components meet their definitions on 60 000 samples (Student-t with 30 dof): the white noise
        has variance s^2 (within 5 %); the drift's increment over one 5 ms step has variance
        2 s^2 (1 - exp(-5 ms / 80 ms)) (within 5 %) and the drift itself s^2 (within 10 %: its samples are
        correlated); the offset's jump over a dropout of 20 ms lost time has variance
        2 s^2 (1 - exp(-20 ms / 100 ms)) (within 5 %).
    """

    rng = np.random.default_rng(20261111)
    frames = np.arange(60000)
    white = Observation_Error(frames / 200.0, 200.0, Observation_Model_Str=_Model(white=1e-3), scale=1.0, rng=rng)
    assert np.var(white) == pytest.approx(1e-6, rel=0.05)
    drift = Observation_Error(frames / 200.0, 200.0, Observation_Model_Str=_Model(drift=1e-3), scale=1.0, rng=rng)
    assert np.var(np.diff(drift, axis=0)) == pytest.approx(2e-6 * (1.0 - np.exp(-0.005 / 0.08)), rel=0.05)
    assert np.mean(drift ** 2) == pytest.approx(1e-6, rel=0.10)
    gapped = np.arange(60000) * 5
    offset = Observation_Error(gapped / 200.0, 200.0, Observation_Model_Str=_Model(offset=1e-3), scale=1.0, rng=rng)
    assert np.var(np.diff(offset, axis=0)) == pytest.approx(2e-6 * (1.0 - np.exp(-0.020 / 0.1)), rel=0.05)


def test_the_error_covariance_is_that_of_the_generated_error():
    """
    Description:
        Error_Covariance (the GLS_M4 weights, SPEC §7.7) equals the covariance of Observation_Error over
        20 000 independent flights of 12 irregular samples at 200 Hz (dropouts of 1, 4 and 10 frames;
        Student-t with 30 dof; every component on, a different size per axis): every element within 3 % of
        the largest variance of its axis.
    """

    frames = np.array([0, 1, 2, 4, 5, 6, 11, 12, 13, 14, 25, 26])
    time = frames / 200.0
    model = Observation_Model_Str(White_Std=np.array([1.0, 2.0, 0.5]) * 1e-3, White_Dof=30.0, Drift_Std=np.array([2.0, 1.0, 3.0]) * 1e-3,
                                  Drift_Time=0.02, Offset_Std=np.array([3.0, 4.0, 1.0]) * 1e-3, Offset_Time=0.03, Offset_Dof=30.0)
    rng = np.random.default_rng(20261114)
    errors = np.stack([Observation_Error(time, 200.0, Observation_Model_Str=model, scale=1.0, rng=rng) for _ in range(20000)])
    covariance = Error_Covariance(time, 200.0, Observation_Model_Str=model)
    assert covariance.shape == (3, 12, 12)
    for axis in range(3):
        empirical = np.cov(errors[:, :, axis], rowvar=False)
        assert np.max(np.abs(empirical - covariance[axis])) <= 0.03 * np.max(np.diag(covariance[axis]))


def test_the_error_covariance_of_a_run_without_dropouts_has_a_constant_offset_term():
    """
    Description:
        Within a run without dropouts the offset does not move: its term is s_o^2 for every pair; across a
        dropout of lost time L it is s_o^2 exp(-L / Offset_Time) (one 10 ms frame period at 100 Hz plus 20 ms
        lost: exp(-0.2) at Offset_Time 0.1 s), and the white noise only on the diagonal.
    """

    model = Observation_Model_Str(White_Std=np.zeros(3), White_Dof=6.0, Drift_Std=np.zeros(3), Drift_Time=0.08,
                                  Offset_Std=np.full(3, 2e-3), Offset_Time=0.1, Offset_Dof=6.0)
    covariance = Error_Covariance(np.array([0.0, 0.01, 0.02, 0.05]), 100.0, Observation_Model_Str=model)
    assert covariance[0, 0, 2] == pytest.approx(4e-6, rel=1e-12)
    assert covariance[1, 2, 3] == pytest.approx(4e-6 * np.exp(-0.2), rel=1e-12)
    assert covariance[2, 0, 3] == pytest.approx(4e-6 * np.exp(-0.2), rel=1e-12)
    white = Observation_Model_Str(White_Std=np.full(3, 1e-3), White_Dof=6.0, Drift_Std=np.zeros(3), Drift_Time=0.08,
                                  Offset_Std=np.zeros(3), Offset_Time=0.1, Offset_Dof=6.0)
    assert np.array_equal(Error_Covariance(np.array([0.0, 0.01, 0.05]), 100.0, Observation_Model_Str=white)[1], 1e-6 * np.eye(3))


# The sizes of the configured error model with its heavy tails (SPEC §8.1, rounded), unbounded: the model the
# bounds were asked for.
#   Unit [Observation_Model_Str]
CONST_HEAVY = Observation_Model_Str(White_Std=np.array([0.37e-3, 0.80e-3, 0.24e-3]), White_Dof=2.1, Drift_Std=np.array([2.0e-3, 3.2e-3, 3.0e-3]),
                                    Drift_Time=0.08, Offset_Std=np.array([9.3e-3, 12.4e-3, 5.8e-3]), Offset_Time=0.1, Offset_Dof=2.5)
# The SHA-256 of CONST_HEAVY's error on 600 gapped frames from the generator 20261115, computed with the code of
# commit 5845508 (before the bounds existed; its largest norm is 34.3 mm).
#   Unit [string]
CONST_UNBOUNDED_DIGEST = '20e6786bb31fb3561c58934f01c541afb744fbaf10c07761ddfd828df2e55250'


def _Parts(white=False, offset=False, **bounds):
    return replace(CONST_HEAVY, White_Std=CONST_HEAVY.White_Std if white else np.zeros(3), Drift_Std=np.zeros(3),
                   Offset_Std=CONST_HEAVY.Offset_Std if offset else np.zeros(3), **bounds)


def test_the_offset_stays_within_its_bound_and_a_value_beyond_it_is_drawn_again():
    """
    Description:
        The offset alone, a new one after every dropout of 0.1 s (5 000 runs, Student-t with 2.5 dof): unbounded
        it exceeds 70 mm somewhere (the reason for the bound); with Offset_Bound 0.070 m its norm never does. Up
        to the first value beyond the bound the two errors are the same numbers, and that value is another.
    """

    time = np.arange(5000) * 21 / 200.0
    free = Observation_Error(time, 200.0, Observation_Model_Str=_Parts(offset=True), scale=1.0, rng=np.random.default_rng(3))
    bounded = Observation_Error(time, 200.0, Observation_Model_Str=_Parts(offset=True, Offset_Bound=0.070), scale=1.0,
                                rng=np.random.default_rng(3))
    beyond = np.flatnonzero(np.linalg.norm(free, axis=1) > 0.070)
    assert beyond.size > 0
    assert np.linalg.norm(bounded, axis=1).max() <= 0.070
    assert np.array_equal(bounded[:beyond[0]], free[:beyond[0]]) and np.all(bounded[beyond[0]] != free[beyond[0]])


def test_the_white_noise_stays_within_its_bound_and_only_the_values_beyond_it_change():
    """
    Description:
        The white noise alone (Student-t with 2.1 dof) on 20 000 samples: unbounded, some vectors exceed 5 mm;
        with White_Bound 0.005 m none does, the vectors that were within keep their numbers and each of the
        others is drawn again.
    """

    time = np.arange(20000) / 200.0
    free = Observation_Error(time, 200.0, Observation_Model_Str=_Parts(white=True), scale=1.0, rng=np.random.default_rng(5))
    bounded = Observation_Error(time, 200.0, Observation_Model_Str=_Parts(white=True, White_Bound=0.005), scale=1.0,
                                rng=np.random.default_rng(5))
    beyond = np.linalg.norm(free, axis=1) > 0.005
    assert beyond.sum() > 0
    assert np.linalg.norm(bounded, axis=1).max() <= 0.005
    assert np.array_equal(bounded[~beyond], free[~beyond]) and np.all(np.any(bounded[beyond] != free[beyond], axis=1))


def test_a_flight_without_a_value_beyond_the_bounds_has_the_error_it_had_without_them():
    """
    Description:
        The bounds change only the values beyond them: on 600 gapped frames whose largest error is 34.3 mm, the
        error of the unbounded model, and of the same model with the configured bounds (0.070 and 0.040 m), is
        bit for bit the one computed before the bounds existed (commit 5845508), and the generator is left in
        the same state (its next number is the same).
    """

    frames = np.cumsum(np.tile([1, 1, 1, 4, 1, 1, 21, 1, 2, 1], 60))
    following = []
    for model in (CONST_HEAVY, replace(CONST_HEAVY, Offset_Bound=0.070, White_Bound=0.040)):
        rng = np.random.default_rng(20261115)
        error = Observation_Error(frames / 200.0, 200.0, Observation_Model_Str=model, scale=1.0, rng=rng)
        assert hashlib.sha256(error.tobytes()).hexdigest() == CONST_UNBOUNDED_DIGEST
        following.append(rng.standard_normal())
    assert following[0] == following[1]


def test_a_bounded_offset_is_drawn_again_and_never_put_on_the_bound():
    """
    Description:
        An offset beyond the bound is drawn again, not clipped. Of the 5 000 offsets
        bounded at 0.070 m none lies on the bound (within 1e-9 m), although the unbounded model puts some
        beyond it; clipping would leave each of those exactly there.
    """

    time = np.arange(5000) * 21 / 200.0
    free = Observation_Error(time, 200.0, Observation_Model_Str=_Parts(offset=True), scale=1.0, rng=np.random.default_rng(3))
    bounded = Observation_Error(time, 200.0, Observation_Model_Str=_Parts(offset=True, Offset_Bound=0.070), scale=1.0,
                                rng=np.random.default_rng(3))
    assert np.sum(np.linalg.norm(free, axis=1) > 0.070) > 0
    assert not np.any(np.abs(np.linalg.norm(bounded, axis=1) - 0.070) < 1e-9)


def test_the_error_covariance_does_not_depend_on_the_bounds():
    """
    Description:
        Error_Covariance, the weights of GLS_M4, holds the variances of the unbounded
        components. With no bound, the configured bounds (0.070 and 0.040 m) and bounds below the components'
        sizes (1 mm, 0.1 mm) it is the same array bit for bit, so the bounds cannot move GLS_M4's predictions.
    """

    time = np.array([0, 1, 2, 4, 5, 6, 11, 12, 13, 14, 25, 26]) / 200.0
    covariance = Error_Covariance(time, 200.0, Observation_Model_Str=CONST_HEAVY)
    for offset_bound, white_bound in ((0.070, 0.040), (1e-3, 1e-4)):
        model = replace(CONST_HEAVY, Offset_Bound=offset_bound, White_Bound=white_bound)
        assert np.array_equal(Error_Covariance(time, 200.0, Observation_Model_Str=model), covariance)
    assert covariance[1, 0, 0] == pytest.approx(0.80e-3 ** 2 + 3.2e-3 ** 2 + 12.4e-3 ** 2, rel=1e-12)


def test_a_bound_that_no_draw_meets_is_refused_naming_it():
    """
    Description:
        Limits fail explicitly: an offset of 9-12 mm bounded at 1e-9 m, or a white noise of 0.2-0.8 mm bounded
        at 1e-12 m, is not met in 1 000 redraws; the error names the bound's field, its value and the count,
        and nothing is clamped.
    """

    time = np.arange(5) * 21 / 200.0
    with pytest.raises(Observation_Error_Cls, match=r'Observation\.Offset_Bound = 1e-09 m in 1000 redraws'):
        Observation_Error(time, 200.0, Observation_Model_Str=_Parts(offset=True, Offset_Bound=1e-9), scale=1.0, rng=np.random.default_rng(1))
    with pytest.raises(Observation_Error_Cls, match=r'Observation\.White_Bound = 1e-12 m in 1000 redraws'):
        Observation_Error(time, 200.0, Observation_Model_Str=_Parts(white=True, White_Bound=1e-12), scale=1.0, rng=np.random.default_rng(1))
