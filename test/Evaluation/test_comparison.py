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
# File Name: ../test/Evaluation/test_comparison.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the comparison under test
#   ../Evaluation/Comparison
from Evaluation.Comparison import (Build_Table, Comparison_Error_Cls, Comparison_Paired, Comparison_Sim_To_Real, Comparison_Summary,
                                   Evaluation_Table_Str, Flight_Means, Group_Str, Report_Lines, Resampled_Medians, Seed_Statistic)

"""
Description:
    Initialization of constants.

    The comparison of stored evaluations (SPEC §7.12, §9.7; F14) on constructed rows and matrices. The rules are
    written out here from the decision (F14), never read from the module: the bootstrap's
    seed and its number of resamples, the interval's percentiles, the smallest group with statistics.
"""
# The bootstrap of an interval: its seed, its resamples and the interval's percentiles.
#   Unit [int], [int], [Tuple<float>, %]
CONST_SEED = 20261118
CONST_RESAMPLES = 10000
CONST_PERCENTILES = (2.5, 97.5)


def _Row(flight, seed='', n=10, partition='real', dataset='', rms='10.000000', end='-2.000000', inference='1.000000', crossing=''):
    return {'flight': flight, 'partition': partition, 'dataset': dataset, 'n': '%d' % n, 'prefix_duration_s': '0.100000', 'rate_hz': '100',
            'remainder_rms_mm': rms, 'end_mismatch_ms': end, 'catch_point_error_mm': '', 'time_to_catch_error_ms': '', 'crossing': crossing,
            'inference_ms': inference, 'refusal': '', 'seed': seed}


def _Range(flight, horizon, error, seed='', n=10, truth='recorded', partition='real', dataset=''):
    return {'flight': flight, 'partition': partition, 'dataset': dataset, 'n': '%d' % n, 'horizon': horizon, 'truth': truth,
            'position_error_mm': error, 'samples': '3', 'direction_error_deg': '', 'speed_error_m_s': '', 'seed': seed}


def _Group(values, flights=None, end=None, inference=None, ranges=None):
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    return Group_Str(Flights=flights or ['f%d' % i for i in range(values.shape[1])], Seeds=['%d' % s for s in range(values.shape[0])],
                     Remainder_RMS=values, End_Mismatch=values if end is None else np.atleast_2d(np.asarray(end, dtype=np.float64)),
                     Inference=values if inference is None else np.atleast_2d(np.asarray(inference, dtype=np.float64)),
                     Refused=np.zeros(values.shape, dtype=bool), Ranges=ranges or {})


def _Table(name, groups, first='Default'):
    return Evaluation_Table_Str(Name=name, First_Dataset=first, Groups=groups)


def _Pick(rows, **keys):
    found = [row for row in rows if all(row[k] == v for k, v in keys.items())]
    assert len(found) == 1, (keys, len(found))
    return found[0]


def test_a_table_holds_one_row_per_seed_and_one_column_per_flight():
    """
    Description:
        The stored rows of a learned run become, per partition, dataset and N, matrices with the seeds sorted in
        rows and the flights in the rows' order in columns; a refused prediction is marked and has no metric; a
        horizon range is read against the recording, never against the observed positions, and a fixed horizon
        is not a range.
    """

    per_flight = [_Row('b', '1', rms='12.500000'), _Row('a', '1', rms='', end='', inference='', crossing='refused'),
                  _Row('b', '0', rms='11.000000', end='-4.000000'), _Row('a', '0', rms='9.000000', inference='0.500000'),
                  _Row('7', '0', n=20, partition='test', dataset='Default', rms='30.000000', crossing='both')]
    per_horizon = [_Range('a', '[0, 0.1)', '5.500000', '0'), _Range('b', '[0, 0.1)', '6.500000', '1'),
                   _Range('a', '[0, 0.1)', '99.000000', '0', truth='observed'), _Range('a', '0.05', '77.000000', '0', truth='exact')]
    table = Build_Table('Run', 'Default', per_flight, per_horizon)
    assert (table.Name, table.First_Dataset) == ('Run', 'Default')
    assert sorted(table.Groups) == [('real', '', 10), ('test', 'Default', 20)]
    group = table.Groups[('real', '', 10)]
    assert (group.Flights, group.Seeds) == (['b', 'a'], ['0', '1'])
    assert np.array_equal(group.Remainder_RMS, [[11.0, 9.0], [12.5, np.nan]], equal_nan=True)
    assert np.array_equal(group.End_Mismatch, [[-4.0, -2.0], [-2.0, np.nan]], equal_nan=True)
    assert np.array_equal(group.Inference, [[1.0, 0.5], [1.0, np.nan]], equal_nan=True)
    assert np.array_equal(group.Refused, [[False, False], [False, True]])
    assert list(group.Ranges) == ['[0, 0.1)']
    assert np.array_equal(group.Ranges['[0, 0.1)'], [[np.nan, 5.5], [6.5, np.nan]], equal_nan=True)
    assert table.Groups[('test', 'Default', 20)].Remainder_RMS.tolist() == [[30.0]]


def test_a_prediction_stored_twice_is_refused():
    """
    Description:
        Two rows of one flight, N and seed contradict each other: the table is refused, naming the flight.
    """

    with pytest.raises(Comparison_Error_Cls, match='holds the flight a of real'):
        Build_Table('Run', 'Default', [_Row('a', '0'), _Row('a', '0', rms='11.000000')], [])


def test_a_seed_statistic_is_the_median_over_the_seeds_of_the_per_seed_statistic():
    """
    Description:
        Per seed the statistic over its flights, a value that does not exist left out; then the median over the
        seeds with the smallest and the largest seed. The p90 interpolates linearly (0 ... 10: 9.0).
    """

    values = np.array([[1.0, 2.0, 3.0, 4.0, 100.0], [2.0, 2.0, 2.0, 2.0, 2.0], [5.0, 6.0, 7.0, 8.0, np.nan]])
    assert Seed_Statistic(values, 'median') == (3.0, 2.0, 6.5)
    assert Seed_Statistic(np.arange(11.0)[None, :], 'p90') == (9.0, 9.0, 9.0)
    assert all(np.isnan(v) for v in Seed_Statistic(np.full((2, 3), np.nan), 'median'))
    assert all(np.isnan(v) for v in Seed_Statistic(np.zeros((2, 0)), 'median'))


def test_an_unknown_statistic_is_refused():
    """
    Description:
        Only the median and the p90 are statistics of a comparison; another name is refused, naming it.
    """

    with pytest.raises(Comparison_Error_Cls, match='"mean" is not known'):
        Seed_Statistic(np.ones((1, 3)), 'mean')


def test_the_flight_means_average_the_seeds_and_miss_where_a_seed_misses():
    """
    Description:
        A flight's error is its mean over the seeds, and does not exist when one seed has none.
    """

    assert np.array_equal(Flight_Means(np.array([[1.0, 2.0, 3.0], [3.0, 6.0, np.nan]])), [2.0, 4.0, np.nan], equal_nan=True)


def test_the_summary_reports_each_metric_on_the_flights_it_exists_on():
    """
    Description:
        One numerical run on 12 real flights, the last one refused: the remainder RMS over the 11 others (1 ... 11:
        median 6, p90 10), the end mismatch as the median of its absolute value, the inference time, a horizon
        range with its own flight count, and the refused predictions over all 12 flights.
    """

    rms = np.append(np.arange(1.0, 12.0), np.nan)[None, :]
    group = _Group(rms, end=-2.0 * rms, inference=0.1 * rms, ranges={'[0, 0.1)': np.append(np.arange(10.0, 21.0), np.nan)[None, :]})
    group.Refused[0, 11] = True
    rows = Comparison_Summary(_Table('Run', {('real', '', 20): group}))
    assert {(r['run'], r['partition'], r['dataset'], r['support'], r['n']) for r in rows} == {('Run', 'real', '', 'all', 20)}
    assert [r['metric'] for r in rows] == ['remainder_rms_median_mm', 'remainder_rms_p90_mm', 'end_mismatch_abs_median_ms',
                                           'inference_median_ms', 'inference_p90_ms', 'range_[0, 0.1)_mm', 'refused']
    median = _Pick(rows, metric='remainder_rms_median_mm')
    assert (median['flights'], median['value'], median['seed_min'], median['seed_max'], median['values']) == (11, 6.0, 6.0, 6.0, [])
    assert _Pick(rows, metric='remainder_rms_p90_mm')['value'] == 10.0
    assert _Pick(rows, metric='end_mismatch_abs_median_ms')['value'] == 12.0
    assert _Pick(rows, metric='inference_median_ms')['value'] == pytest.approx(0.6)
    assert _Pick(rows, metric='inference_p90_ms')['value'] == pytest.approx(1.0)
    assert (_Pick(rows, metric='range_[0, 0.1)_mm')['flights'], _Pick(rows, metric='range_[0, 0.1)_mm')['value']) == (11, 15.0)
    assert (_Pick(rows, metric='refused')['flights'], _Pick(rows, metric='refused')['value']) == (12, 1.0)


@pytest.mark.parametrize('flights, listed', [(9, True), (10, False)])
def test_a_metric_below_ten_flights_lists_its_values_and_reports_no_statistics(flights, listed):
    """
    Description:
        The minimum of SPEC §7.12 on both sides: a metric on 9 flights reports no statistic and lists the 9
        per-flight values, the seeds averaged; on 10 flights it reports the statistic and lists nothing.
    """

    values = np.array([np.arange(1.0, flights + 1.0), np.arange(3.0, flights + 3.0)])
    row = _Pick(Comparison_Summary(_Table('Run', {('real', '', 10): _Group(values)})), metric='remainder_rms_median_mm')
    assert row['flights'] == flights
    if listed:
        assert np.isnan(row['value']) and np.isnan(row['seed_min']) and row['values'] == list(np.arange(2.0, 11.0))
    else:
        assert (row['value'], row['seed_min'], row['seed_max'], row['values']) == (6.5, 5.5, 7.5, [])


def test_the_real_flights_are_split_by_the_coverage_and_a_simulated_partition_is_not():
    """
    Description:
        With a coverage, the real flights are reported for all, for the flights inside and for those outside the
        support of the generated throws (24 = 14 + 10, the medians of each part); a simulated partition has no
        support. A real flight the coverage does not name is refused.
    """

    flights = ['f%d' % i for i in range(24)]
    coverage = {flight: i < 14 for i, flight in enumerate(flights)}
    table = _Table('Run', {('real', '', 10): _Group(np.arange(24.0), flights), ('test', 'Default', 10): _Group(np.arange(24.0), flights)})
    rows = [r for r in Comparison_Summary(table, coverage) if r['metric'] == 'remainder_rms_median_mm']
    assert [(r['partition'], r['support'], r['flights'], r['value']) for r in rows] == [
        ('real', 'all', 24, 11.5), ('real', 'inside', 14, 6.5), ('real', 'outside', 10, 18.5), ('test', 'all', 24, 11.5)]
    assert [r['support'] for r in Comparison_Summary(table) if r['metric'] == 'refused'] == ['all', 'all']
    del coverage['f3']
    with pytest.raises(Comparison_Error_Cls, match='does not name 1 real flight'):
        Comparison_Summary(table, coverage)


def test_a_paired_difference_is_the_run_minus_the_reference_with_the_seeds_averaged():
    """
    Description:
        Per flight the run's error averaged over its two seeds minus the reference's, on the flights by name
        (the reference stores them in another order): the differences are 1, 2, ... 12 mm, their median 6.5 mm,
        positive because the run is worse; the interval encloses it.
    """

    flights = ['f%d' % i for i in range(12)]
    base = np.linspace(20.0, 64.0, 12)
    run = _Group(np.array([base + np.arange(1.0, 13.0) - 2.0, base + np.arange(1.0, 13.0) + 2.0]), flights)
    reference = _Group(base[::-1], flights[::-1])
    rows = Comparison_Paired(_Table('Run', {('real', '', 20): run}), _Table('Reference', {('real', '', 20): reference, ('real', '', 40): reference}))
    assert len(rows) == 1
    row = rows[0]
    assert (row['run'], row['against'], row['partition'], row['dataset'], row['support'], row['n']) == ('Run', 'Reference', 'real', '', 'all', 20)
    assert (row['flights'], row['left_out'], row['values']) == (12, 0, [])
    assert row['median_mm'] == pytest.approx(6.5) and row['low_mm'] < 6.5 < row['high_mm']


def test_the_interval_is_the_percentiles_of_ten_thousand_resampled_medians_from_the_fixed_seed():
    """
    Description:
        The bootstrap of the decision (F14), computed here on its own: the flights resampled
        with replacement 10 000 times by a generator started from the seed 20261118, the interval the 2.5th and
        97.5th percentiles of the resampled medians. Every resampled median is compared, not the interval alone:
        the median of 15 values takes few distinct values, so another seed can give the same two percentiles.
    """

    difference = np.array([-3.0, 1.5, 0.2, 4.0, -0.7, 2.2, 9.0, -1.1, 0.4, 3.3, 5.1, -2.4, 0.9, 1.0, 6.5])
    rng = np.random.default_rng(CONST_SEED)
    index = rng.integers(0, len(difference), size=(CONST_RESAMPLES, len(difference)))
    medians = [np.median(difference[index[i]]) for i in range(CONST_RESAMPLES)]
    expected = np.percentile(medians, CONST_PERCENTILES)
    run, reference = _Group(difference + 30.0), _Group(np.full(15, 30.0))
    row = Comparison_Paired(_Table('Run', {('real', '', 10): run}), _Table('Reference', {('real', '', 10): reference}))[0]
    assert row['median_mm'] == pytest.approx(1.0)
    assert (row['low_mm'], row['high_mm']) == pytest.approx(tuple(expected), abs=1e-12)
    assert np.array_equal(Resampled_Medians(difference), medians)


def test_the_same_flights_are_resampled_for_every_seed():
    """
    Description:
        A resample draws flights, not predictions: with a second seed 10 mm above the first on every flight, each
        resampled statistic (the median over the seeds of the per-seed median) is the first seed's resampled
        median plus 5 mm; seeds resampled apart would not give that.
    """

    first = np.array([3.0, 8.0, 1.0, 9.5, 4.0, 7.0, 2.5, 6.0, 5.0, 10.0, 0.5, 11.0])
    assert np.allclose(Resampled_Medians(np.array([first, first + 10.0])), Resampled_Medians(first) + 5.0, atol=1e-12)
    assert not np.array_equal(Resampled_Medians(first), Resampled_Medians(first, 1))


def test_a_flight_without_a_remainder_on_either_side_is_left_out_and_counted():
    """
    Description:
        A flight that one seed of the run refused, and a flight only the reference holds, are left out of the
        paired difference and counted; the other 10 flights give it.
    """

    flights = ['f%d' % i for i in range(11)]
    values = np.array([np.arange(11.0) + 1.0, np.arange(11.0) + 1.0])
    values[1, 4] = np.nan
    reference = _Group(np.zeros(12), flights + ['only_reference'])
    row = Comparison_Paired(_Table('Run', {('real', '', 10): _Group(values, flights)}), _Table('Reference', {('real', '', 10): reference}))[0]
    assert (row['flights'], row['left_out']) == (10, 2)
    assert row['median_mm'] == pytest.approx(np.median([1, 2, 3, 4, 6, 7, 8, 9, 10, 11]))


def test_a_paired_group_below_ten_flights_lists_its_differences_and_is_split_by_the_coverage():
    """
    Description:
        The support split of a paired difference: 19 real flights, 10 inside and 9 outside; inside reports the
        median with its interval, outside lists its 9 differences and reports no statistic.
    """

    flights = ['f%d' % i for i in range(19)]
    coverage = {flight: i < 10 for i, flight in enumerate(flights)}
    run, reference = _Group(np.arange(19.0) + 50.0, flights), _Group(np.full(19, 50.0), flights)
    rows = Comparison_Paired(_Table('Run', {('real', '', 10): run}), _Table('Reference', {('real', '', 10): reference}), coverage)
    assert [(r['support'], r['flights']) for r in rows] == [('all', 19), ('inside', 10), ('outside', 9)]
    assert rows[1]['median_mm'] == pytest.approx(4.5) and rows[1]['values'] == []
    assert np.isnan(rows[2]['median_mm']) and np.isnan(rows[2]['low_mm']) and rows[2]['values'] == list(np.arange(10.0, 19.0))


def test_sim_to_real_sets_the_real_error_against_the_first_datasets_test_partition():
    """
    Description:
        The sim-to-real row of the criterion: the real flights' median and p90 against the
        test partition of the run's first dataset (not of a further dataset, not the validation partition), their
        difference and ratio; the interval is that of the difference with the two sides resampled independently
        (the real flights from the seed, the simulated ones from the seed's stream 1), computed here on its own.
        An N without real flights gives no row. Two sides with the same values show the independence: their
        difference is 0 and its interval lies on both sides of 0, where one resample for both would give 0 alone.
    """

    real = np.array([30.0, 42.0, 35.0, 51.0, 28.0, 39.0, 44.0, 33.0, 47.0, 36.0, 40.0, 31.0])
    simulated = np.linspace(20.0, 60.0, 41)
    groups = {('real', '', 20): _Group(real), ('test', 'Default', 20): _Group(simulated), ('test', 'Other', 20): _Group(simulated + 100.0),
              ('validation', 'Default', 20): _Group(simulated + 200.0), ('test', 'Default', 40): _Group(simulated)}
    rows = Comparison_Sim_To_Real(_Table('Run', groups), 'test')
    assert len(rows) == 1
    row = rows[0]
    assert (row['run'], row['n'], row['dataset'], row['real_flights'], row['simulated_flights']) == ('Run', 20, 'Default', 12, 41)
    assert (row['real_median_mm'], row['simulated_median_mm']) == (37.5, 40.0)
    assert row['real_p90_mm'] == pytest.approx(np.percentile(real, 90)) and row['simulated_p90_mm'] == pytest.approx(56.0)
    assert row['difference_mm'] == -2.5 and row['ratio'] == pytest.approx(0.9375)
    index_real = np.random.default_rng(CONST_SEED).integers(0, 12, size=(CONST_RESAMPLES, 12))
    index_simulated = np.random.default_rng([CONST_SEED, 1]).integers(0, 41, size=(CONST_RESAMPLES, 41))
    expected = np.percentile(np.median(real[index_real], axis=1) - np.median(simulated[index_simulated], axis=1), CONST_PERCENTILES)
    assert (row['low_mm'], row['high_mm']) == pytest.approx(tuple(expected), abs=1e-12)
    twin = Comparison_Sim_To_Real(_Table('Twin', {('real', '', 10): _Group(real), ('test', 'Default', 10): _Group(real)}), 'test')[0]
    assert twin['difference_mm'] == 0.0 and twin['ratio'] == 1.0 and twin['low_mm'] < 0.0 < twin['high_mm']


def test_the_horizon_ranges_are_listed_in_horizon_order():
    """
    Description:
        The horizon ranges of the summary and of the printed report follow the horizon, whatever the order of the
        stored rows: a file whose first flight has no sample in an early range names a later range first (F14's
        real flights at N = 20 stored [0.3, 0.5) before [0.2, 0.3)), and the tables must not inherit that order.
    """

    stored = ['[1.2, 2]', '[0.3, 0.5)', '[0, 0.1)', '[0.2, 0.3)']
    flights = ['f%d' % i for i in range(12)]
    per_horizon = [_Range(flight, label, '%d.000000' % (10 * (k + 1))) for flight in flights for k, label in enumerate(stored)]
    rows = Comparison_Summary(Build_Table('Run', 'Default', [_Row(flight) for flight in flights], per_horizon))
    ranges = [r for r in rows if r['metric'].startswith('range_')]
    assert [r['metric'] for r in ranges] == ['range_[0, 0.1)_mm', 'range_[0.2, 0.3)_mm', 'range_[0.3, 0.5)_mm', 'range_[1.2, 2]_mm']
    assert [(r['value'], r['flights']) for r in ranges] == [(30.0, 12), (40.0, 12), (20.0, 12), (10.0, 12)]
    line = Report_Lines(rows, [], [])[0]
    assert 'ranges [0, 0.1) 30.0 mm (12), [0.2, 0.3) 40.0 mm (12), [0.3, 0.5) 20.0 mm (12), [1.2, 2] 10.0 mm (12)' in line


def test_the_report_shows_the_seeds_range_the_flight_counts_and_listed_values():
    """
    Description:
        The printed report: a statistic over several seeds with the range of the seeds' values, one seed without;
        a horizon range with its flight count; a metric too small for statistics with its values; the paired
        difference with its interval and the flights used and left out; the sim-to-real line.
    """

    values = np.array([np.arange(1.0, 13.0), np.arange(3.0, 15.0)])
    ranges = {'[0, 0.1)': values, '[1.2, 2]': np.where(np.arange(12) < 3, values, np.nan)}
    table = _Table('Learned', {('real', '', 20): _Group(values, ranges=ranges), ('test', 'Default', 20): _Group(np.arange(1.0, 13.0))})
    reference = _Table('Numerical', {('real', '', 20): _Group(np.arange(1.0, 13.0))})
    lines = Report_Lines(Comparison_Summary(table) + Comparison_Summary(reference), Comparison_Paired(table, reference),
                         Comparison_Sim_To_Real(table, 'test'))
    assert lines[0].startswith('Learned real (real) all N=20: 12 flights, 0 refused; remainder RMS median 7.5 [6.5–8.5] / p90 ')
    assert 'ranges [0, 0.1) 7.5 [6.5–8.5] mm (12), [1.2, 2] values 2.0 3.0 4.0 mm (3)' in lines[0]
    assert lines[1].startswith('Learned test Default N=20: 12 flights, 0 refused; remainder RMS median 6.5 / p90 ')
    assert lines[2].startswith('Numerical real (real) all N=20: 12 flights, 0 refused; remainder RMS median 6.5 / p90 ')
    assert lines[3].startswith('sim-to-real Learned N=20: real 7.5 / p90 ') and 'simulated Default 6.5 / p90 ' in lines[3]
    assert 'real - simulated +1.0 [' in lines[3] and lines[3].endswith('ratio 1.15')
    assert lines[4].startswith('paired Learned - Numerical, real (real) all N=20: +1.00 [+1.00, +1.00] mm (12 flights, 0 left out)')
    assert len(lines) == 5
