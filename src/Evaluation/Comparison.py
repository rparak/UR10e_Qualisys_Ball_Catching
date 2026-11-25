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
# File Name: ../Evaluation/Comparison.py
## =========================================================================== ##

# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Evaluation/Core - the smallest group that reports statistics, the outcome of a refused prediction
from Evaluation.Core import CONST_MINIMUM_FLIGHTS, CONST_REFUSED

"""
Description:
    Initialization of constants.

    The comparison of stored evaluations (SPEC §7.12, §9.7; F14): per run the
    summary of each partition, dataset and N, on real flights also inside and outside the support of the
    generated throws; the paired difference of two runs on the same flights with a bootstrap interval; and a
    run's real-flight error against its own simulated error. It works on the rows of per_flight.csv and
    per_horizon.csv as Evaluation.Report reads them and never on a recording, a dataset or a predictor.
    Pure computation: no file, no printing.

    One convention throughout: a learned predictor has several training seeds. A statistic of a run is the
    median over its seeds of the per-seed statistic, with the range of the per-seed values; a paired
    difference uses a flight's error averaged over the seeds, and a flight without a remainder RMS on any seed
    of either side (a refused prediction, or a prefix that reaches the flight's last frame) is left out and
    counted.
"""
# The bootstrap of every interval: flights resampled with replacement this many times, from
# this seed, the interval the 2.5th and 97.5th percentiles of the resampled statistic. Every interval starts its own
# generator from the seed, so an interval does not depend on the order of the calls.
#   Unit [int], [int], [Tuple<float>, %]
CONST_BOOTSTRAP_RESAMPLES = 10000
CONST_BOOTSTRAP_SEED = 20261118
CONST_BOOTSTRAP_INTERVAL = (2.5, 97.5)
# The resamples computed at once (memory: seeds x this x flights values).
#   Unit [int]
CONST_BOOTSTRAP_BLOCK = 500
# The supports a real flight is reported in (SPEC §2.13.6): every flight, and the flights whose release state lies
# inside or outside the support of the generated throws (coverage.csv).
#   Unit [Tuple<string>]
CONST_SUPPORTS = ('all', 'inside', 'outside')
# The partition of the real flights (SPEC §9.7).
#   Unit [string]
CONST_REAL = 'real'


class Comparison_Error_Cls(ValueError):
    """
    Description:
        Stored evaluations cannot be compared: rows that contradict each other or a statistic that is not known.
        The message names what was found and what to do.
    """


@dataclass
class Group_Str:
    """
    Description:
        The stored results of one run on one partition, dataset and N: a matrix per metric, one row per training
        seed (one row for a numerical predictor) and one column per flight. A value that does not exist is NaN: every
        metric of a refused prediction, the remainder RMS of a prefix that reaches the flight's last frame, a horizon
        range a flight has no sample in.
    """

    # The flights, in the order of the stored rows, and the training seeds, sorted ('' for a numerical predictor).
    #   Unit [List<string>]
    Flights: tp.List[str] = field(default_factory=list)
    Seeds: tp.List[str] = field(default_factory=list)
    # The remainder RMS, the end mismatch (predicted minus true end) and the inference time.
    #   Unit [Matrix<float> SxF, mm], [Matrix<float> SxF, ms], [Matrix<float> SxF, ms]
    Remainder_RMS: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    End_Mismatch: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Inference: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    # The predictions the predictor refused (SPEC §7.12).
    #   Unit [Matrix<bool> SxF]
    Refused: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    # Per horizon range, by its label as stored ('[0, 0.1)' ...), the flight's position error in the range.
    #   Unit [Dict<string, Matrix<float> SxF>, mm]
    Ranges: tp.Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Evaluation_Table_Str:
    """
    Description:
        One stored evaluation as the comparison uses it: the run's name, the dataset whose test partition is the
        simulated side of its sim-to-real rows (the run's first dataset: 'test' evaluates the real flights with
        that dataset's configuration, SPEC §9.7), and its groups.
    """

    # The run's name and its first dataset.
    #   Unit [string]
    Name: str = ''
    First_Dataset: str = ''
    # The groups by (partition, dataset, N).
    #   Unit [Dict<Tuple<string, string, int>, Group_Str(object)>]
    Groups: tp.Dict[tp.Tuple[str, str, int], Group_Str] = field(default_factory=dict)


def _Value(text: str) -> float:
    """
    Description:
        A stored field as a number; an empty field (a value that does not exist) is NaN.

    Args:
        (1) text [string]: The field.

    Returns:
        (1) parameter [float]: The value.
    """

    return float(text) if text != '' else float('nan')


def Build_Table(name: str, first_dataset: str, per_flight: tp.List[tp.Dict[str, str]], per_horizon: tp.List[tp.Dict[str, str]]
                ) -> Evaluation_Table_Str:
    """
    Description:
        The table of one stored evaluation, from the rows of its per_flight.csv and of its per_horizon.csv (any
        subset of the latter: the horizon ranges are read from the rows given, against the recording or the
        exact truth, never against the observed positions).

    Args:
        (1) name [string]: The run's name.
        (2) first_dataset [string]: The run's first dataset (its label).
        (3) per_flight [List<Dict<string, string>>]: The rows of per_flight.csv (Evaluation.Report.Read_Per_Flight).
        (4) per_horizon [List<Dict<string, string>>]: Rows of per_horizon.csv (Evaluation.Report.Read_Per_Horizon).

    Returns:
        (1) parameter [Evaluation_Table_Str(object)]: The table.
    """

    members: tp.Dict[tp.Tuple[str, str, int], tp.Tuple[tp.Dict[str, int], tp.Dict[str, int]]] = {}
    for row in per_flight:
        flights, seeds = members.setdefault((row['partition'], row['dataset'], int(row['n'])), ({}, {}))
        flights.setdefault(row['flight'], len(flights))
        seeds.setdefault(row['seed'], 0)
    table = Evaluation_Table_Str(Name=name, First_Dataset=first_dataset)
    index: tp.Dict[tp.Tuple[str, str, int], tp.Tuple[tp.Dict[str, int], tp.Dict[str, int]]] = {}
    for key, (flights, seeds) in members.items():
        order = {seed: i for i, seed in enumerate(sorted(seeds))}
        shape = (len(order), len(flights))
        table.Groups[key] = Group_Str(Flights=list(flights), Seeds=sorted(seeds), Remainder_RMS=np.full(shape, np.nan),
                                      End_Mismatch=np.full(shape, np.nan), Inference=np.full(shape, np.nan),
                                      Refused=np.zeros(shape, dtype=bool))
        index[key] = (flights, order)
    seen: tp.Set[tp.Tuple[tp.Tuple[str, str, int], str, str]] = set()
    for row in per_flight:
        key = (row['partition'], row['dataset'], int(row['n']))
        if (key, row['flight'], row['seed']) in seen:
            raise Comparison_Error_Cls('The evaluation of %s holds the flight %s of %s %s, N = %d, seed "%s" twice; a prediction is '
                                       'stored once. Evaluate the run again.' % (name, row['flight'], key[0], key[1], key[2], row['seed']))
        seen.add((key, row['flight'], row['seed']))
        group, (flights, order) = table.Groups[key], index[key]
        s, f = order[row['seed']], flights[row['flight']]
        group.Remainder_RMS[s, f] = _Value(row['remainder_rms_mm'])
        group.End_Mismatch[s, f] = _Value(row['end_mismatch_ms'])
        group.Inference[s, f] = _Value(row['inference_ms'])
        group.Refused[s, f] = row['crossing'] == CONST_REFUSED
    for row in per_horizon:
        key = (row['partition'], row['dataset'], int(row['n']))
        if not row['horizon'].startswith('[') or row['truth'] == 'observed' or key not in table.Groups:
            continue
        group, (flights, order) = table.Groups[key], index[key]
        matrix = group.Ranges.setdefault(row['horizon'], np.full((len(order), len(flights)), np.nan))
        matrix[order[row['seed']], flights[row['flight']]] = _Value(row['position_error_mm'])
    return table


def _Statistic(values: np.ndarray, statistic: str) -> np.ndarray:
    """
    Description:
        The median or the p90 (linear interpolation) along the last axis, over the finite values; NaN where there
        is none.

    Args:
        (1) values [Matrix<float>]: The values.
        (2) statistic [string]: 'median' or 'p90'.

    Returns:
        (1) parameter [Matrix<float>]: The statistic, the last axis removed.
    """

    if statistic not in ('median', 'p90'):
        raise Comparison_Error_Cls('The statistic "%s" is not known; use median or p90.' % statistic)
    out = np.full(values.shape[:-1], np.nan)
    present = np.isfinite(values).any(axis=-1)
    if not present.any():
        return out
    if np.isfinite(values).all():
        return np.median(values, axis=-1) if statistic == 'median' else np.percentile(values, 90, axis=-1)
    out[present] = (np.nanmedian(values[present], axis=-1) if statistic == 'median' else np.nanpercentile(values[present], 90, axis=-1))
    return out


def Seed_Statistic(values: np.ndarray, statistic: str) -> tp.Tuple[float, float, float]:
    """
    Description:
        A run's statistic of a metric: per training seed the statistic over its flights, then the median over
        the seeds, with the smallest and the largest per-seed value. Values that do not exist (NaN) are left out;
        every result is NaN when no value exists.

    Args:
        (1) values [Matrix<float> SxF]: The metric, one row per seed.
        (2) statistic [string]: 'median' or 'p90'.

    Returns:
        (1) parameter [Tuple<float, float, float>]: The median over the seeds, the smallest and the largest seed's.
    """

    per_seed = _Statistic(np.asarray(values, dtype=np.float64), statistic) if np.size(values) else np.zeros(0)
    per_seed = per_seed[np.isfinite(per_seed)]
    if not per_seed.size:
        return float('nan'), float('nan'), float('nan')
    return float(np.median(per_seed)), float(per_seed.min()), float(per_seed.max())


def Flight_Means(values: np.ndarray) -> np.ndarray:
    """
    Description:
        Per flight the metric averaged over the training seeds; NaN when any seed has none.

    Args:
        (1) values [Matrix<float> SxF]: The metric, one row per seed.

    Returns:
        (1) parameter [Vector<float> 1xF]: The means.
    """

    return np.asarray(values, dtype=np.float64).mean(axis=0) if np.size(values) else np.zeros(0)


def Resampled_Medians(values: np.ndarray, stream: int = 0) -> np.ndarray:
    """
    Description:
        The bootstrap of a run's median: the flights (columns) are resampled with replacement
        CONST_BOOTSTRAP_RESAMPLES times, the same resample for every seed, and each resample gives the median
        over the seeds of the per-seed median. One row gives the resampled medians of a vector. The generator
        starts from CONST_BOOTSTRAP_SEED (stream 0) or from the seed and the stream's number, so two sides that
        must be resampled independently take two streams.

    Args:
        (1) values [Matrix<float> SxF]: The metric, one row per seed.
        (2) stream [int]: The generator's stream.

    Returns:
        (1) parameter [Vector<float> 1xR]: The resampled statistic.
    """

    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    rng = np.random.default_rng(CONST_BOOTSTRAP_SEED if stream == 0 else [CONST_BOOTSTRAP_SEED, stream])
    index = rng.integers(0, values.shape[1], size=(CONST_BOOTSTRAP_RESAMPLES, values.shape[1]))
    out = np.empty(CONST_BOOTSTRAP_RESAMPLES)
    for a in range(0, CONST_BOOTSTRAP_RESAMPLES, CONST_BOOTSTRAP_BLOCK):
        per_seed = _Statistic(values[:, index[a:a + CONST_BOOTSTRAP_BLOCK]], 'median')
        out[a:a + CONST_BOOTSTRAP_BLOCK] = _Statistic(per_seed.T, 'median')
    return out


def _Interval(resampled: np.ndarray) -> tp.Tuple[float, float]:
    """
    Description:
        The bootstrap interval of a resampled statistic (CONST_BOOTSTRAP_INTERVAL).

    Args:
        (1) resampled [Vector<float> 1xR]: The resampled statistic.

    Returns:
        (1) parameter [Tuple<float, float>]: The lower and the upper end.
    """

    low, high = np.percentile(resampled, CONST_BOOTSTRAP_INTERVAL)
    return float(low), float(high)


def _Supports(partition: str, flights: tp.List[str], coverage: tp.Optional[tp.Dict[str, bool]]) -> tp.List[tp.Tuple[str, np.ndarray]]:
    """
    Description:
        The supports of a group with the flights of each: every flight, and for the real flights with a
        coverage the flights inside and outside the support of the generated throws.

    Args:
        (1) partition [string]: The group's partition.
        (2) flights [List<string>]: Its flights.
        (3) coverage [Dict<string, bool>]: Per real flight whether it is inside the support, or None.

    Returns:
        (1) parameter [List<Tuple<string, Vector<bool> 1xF>>]: Per support its name and the flights in it.
    """

    everything = np.ones(len(flights), dtype=bool)
    if partition != CONST_REAL or coverage is None:
        return [(CONST_SUPPORTS[0], everything)]
    missing = [flight for flight in flights if flight not in coverage]
    if missing:
        raise Comparison_Error_Cls('The coverage does not name %d real flight(s) of the evaluation (%s); both come from one run of '
                                   'scripts/catch.py test. Evaluate the run again.' % (len(missing), ', '.join(missing[:3])))
    inside = np.array([coverage[flight] for flight in flights], dtype=bool)
    return [(CONST_SUPPORTS[0], everything), (CONST_SUPPORTS[1], inside), (CONST_SUPPORTS[2], ~inside)]


def _Range_Start(label: str) -> float:
    """
    Description:
        The lower end of a horizon range from its stored label ('[0.2, 0.3)' -> 0.2), by which the ranges are
        ordered: the stored rows name them in the order in which the first flights reach them.

    Args:
        (1) label [string]: The range's label as stored (SPEC §9.7).

    Returns:
        (1) parameter [float]: The lower end [s].
    """

    return float(label[1:].split(',')[0])


def Comparison_Summary(table: Evaluation_Table_Str, coverage: tp.Optional[tp.Dict[str, bool]] = None) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Description:
        The summary rows of one evaluation: per partition, dataset, support and N the remainder RMS (median and
        p90), the end mismatch (median of the absolute value), the inference time (median and p90), each horizon
        range stored, in the order of the horizon, and the refused predictions (the metric 'refused': its value their number over every seed,
        its 'flights' the flights of the group). A row's 'flights' are the flights the metric exists on. A metric
        on fewer than CONST_MINIMUM_FLIGHTS flights reports no statistics and lists its per-flight values (the
        seeds averaged) instead (SPEC §7.12).

    Args:
        (1) table [Evaluation_Table_Str(object)]: The evaluation.
        (2) coverage [Dict<string, bool>]: Per real flight whether it is inside the support, or None.

    Returns:
        (1) parameter [List<Dict<string, Any>>]: The rows: run, partition, dataset, support, n, metric, flights,
                                                 value, seed_min, seed_max, values.
    """

    rows = []
    for (partition, dataset, n), group in sorted(table.Groups.items()):
        metrics = [('remainder_rms_median_mm', group.Remainder_RMS, 'median'), ('remainder_rms_p90_mm', group.Remainder_RMS, 'p90'),
                   ('end_mismatch_abs_median_ms', np.abs(group.End_Mismatch), 'median'), ('inference_median_ms', group.Inference, 'median'),
                   ('inference_p90_ms', group.Inference, 'p90')]
        metrics += [('range_%s_mm' % label, matrix, 'median')
                    for label, matrix in sorted(group.Ranges.items(), key=lambda item: _Range_Start(item[0]))]
        for support, members in _Supports(partition, group.Flights, coverage):
            head = {'run': table.Name, 'partition': partition, 'dataset': dataset, 'support': support, 'n': n}
            for metric, matrix, statistic in metrics:
                chosen = matrix[:, members & np.isfinite(matrix).any(axis=0)]
                row = dict(head, metric=metric, flights=chosen.shape[1], value=float('nan'), seed_min=float('nan'), seed_max=float('nan'),
                           values=[])
                if chosen.shape[1] < CONST_MINIMUM_FLIGHTS:
                    row['values'] = [float(v) for v in Flight_Means(chosen) if np.isfinite(v)]
                else:
                    row['value'], row['seed_min'], row['seed_max'] = Seed_Statistic(chosen, statistic)
                rows.append(row)
            rows.append(dict(head, metric='refused', flights=int(members.sum()), value=float(group.Refused[:, members].sum()),
                             seed_min=float('nan'), seed_max=float('nan'), values=[]))
    return rows


def Comparison_Paired(table: Evaluation_Table_Str, against: Evaluation_Table_Str,
                      coverage: tp.Optional[tp.Dict[str, bool]] = None) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Description:
        The paired difference of two evaluations in remainder RMS on the flights both hold: per flight the run's
        error averaged over its seeds minus the reference's, then the median with its bootstrap interval over the
        flights (Resampled_Medians), per partition, dataset, support and N that both evaluated. A flight without a
        remainder RMS on a seed of either side, or that only one side holds, is left out and counted. Fewer than
        CONST_MINIMUM_FLIGHTS flights: the differences are listed and no statistic is reported.

    Args:
        (1) table [Evaluation_Table_Str(object)]: The run.
        (2) against [Evaluation_Table_Str(object)]: The reference.
        (3) coverage [Dict<string, bool>]: Per real flight whether it is inside the support, or None.

    Returns:
        (1) parameter [List<Dict<string, Any>>]: The rows: run, against, partition, dataset, support, n, flights,
                                                 left_out, median_mm, low_mm, high_mm, values.
    """

    rows = []
    for key in sorted(set(table.Groups) & set(against.Groups)):
        partition, dataset, n = key
        a, b = table.Groups[key], against.Groups[key]
        other, own = {flight: i for i, flight in enumerate(b.Flights)}, set(a.Flights)
        flights = a.Flights + [flight for flight in b.Flights if flight not in own]
        difference = np.full(len(flights), np.nan)
        mean_a, mean_b = Flight_Means(a.Remainder_RMS), Flight_Means(b.Remainder_RMS)
        for i, flight in enumerate(a.Flights):
            if flight in other:
                difference[i] = mean_a[i] - mean_b[other[flight]]
        for support, members in _Supports(partition, flights, coverage):
            chosen = difference[members & np.isfinite(difference)]
            row = {'run': table.Name, 'against': against.Name, 'partition': partition, 'dataset': dataset, 'support': support, 'n': n,
                   'flights': int(chosen.size), 'left_out': int(members.sum() - chosen.size), 'median_mm': float('nan'),
                   'low_mm': float('nan'), 'high_mm': float('nan'), 'values': []}
            if chosen.size < CONST_MINIMUM_FLIGHTS:
                row['values'] = [float(v) for v in chosen]
            else:
                row['median_mm'] = float(np.median(chosen))
                row['low_mm'], row['high_mm'] = _Interval(Resampled_Medians(chosen))
            rows.append(row)
    return rows


def Comparison_Sim_To_Real(table: Evaluation_Table_Str, simulated_partition: str) -> tp.List[tp.Dict[str, tp.Any]]:
    """
    Description:
        A run's real-flight error against its own simulated error (the sim-to-real criterion), per N:
        the remainder RMS (median and p90, Seed_Statistic) on the real flights and on the simulated partition
        given (the held-out one) of the run's first dataset, the difference of the two medians with its bootstrap
        interval (the real and the simulated flights resampled independently) and their ratio. An N without both
        sides, or with fewer than CONST_MINIMUM_FLIGHTS flights on a side, gives no row.

    Args:
        (1) table [Evaluation_Table_Str(object)]: The evaluation that holds both sides.
        (2) simulated_partition [string]: The partition of the simulated side, as stored (SPEC §9.7).

    Returns:
        (1) parameter [List<Dict<string, Any>>]: The rows: run, n, dataset, real_flights, real_median_mm,
                                                 real_p90_mm, simulated_flights, simulated_median_mm,
                                                 simulated_p90_mm, difference_mm, low_mm, high_mm, ratio.
    """

    rows = []
    for n in sorted({key[2] for key in table.Groups}):
        real, simulated = table.Groups.get((CONST_REAL, '', n)), table.Groups.get((simulated_partition, table.First_Dataset, n))
        if real is None or simulated is None:
            continue
        sides = [group.Remainder_RMS[:, np.isfinite(group.Remainder_RMS).any(axis=0)] for group in (real, simulated)]
        if sides[0].shape[1] < CONST_MINIMUM_FLIGHTS or sides[1].shape[1] < CONST_MINIMUM_FLIGHTS:
            continue
        (real_median, _, _), (simulated_median, _, _) = Seed_Statistic(sides[0], 'median'), Seed_Statistic(sides[1], 'median')
        low, high = _Interval(Resampled_Medians(sides[0], 0) - Resampled_Medians(sides[1], 1))
        rows.append({'run': table.Name, 'n': n, 'dataset': table.First_Dataset, 'real_flights': sides[0].shape[1],
                     'real_median_mm': real_median, 'real_p90_mm': Seed_Statistic(sides[0], 'p90')[0],
                     'simulated_flights': sides[1].shape[1], 'simulated_median_mm': simulated_median,
                     'simulated_p90_mm': Seed_Statistic(sides[1], 'p90')[0], 'difference_mm': real_median - simulated_median,
                     'low_mm': low, 'high_mm': high, 'ratio': real_median / simulated_median})
    return rows


def Report_Lines(summary: tp.List[tp.Dict[str, tp.Any]], paired: tp.List[tp.Dict[str, tp.Any]],
                 sim_to_real: tp.List[tp.Dict[str, tp.Any]]) -> tp.List[str]:
    """
    Description:
        The printed report of a comparison: per run, partition, dataset, support and N one line with the flights,
        the refused predictions, the remainder RMS, the end mismatch, the inference time and the horizon ranges
        with their flight counts; then the sim-to-real lines and the paired differences. A statistic over the seeds
        shows the range of the per-seed values when they differ; a group below the minimum of flights shows its
        values.

    Args:
        (1) summary [List<Dict<string, Any>>]: The rows of Comparison_Summary.
        (2) paired [List<Dict<string, Any>>]: The rows of Comparison_Paired.
        (3) sim_to_real [List<Dict<string, Any>>]: The rows of Comparison_Sim_To_Real.

    Returns:
        (1) parameter [List<string>]: The lines.
    """

    def Cell(row: tp.Dict[str, tp.Any], form: str = '%.1f') -> str:
        """
        Description:
            A summary row's value: the statistic, with the seeds' range when they differ, or the listed values.

        Args:
            (1) row [Dict<string, Any>]: The row.
            (2) form [string]: The number format.

        Returns:
            (1) parameter [string]: The cell.
        """

        if not np.isfinite(row['value']):
            return 'values ' + ' '.join(form % v for v in row['values']) if row['values'] else '-'
        if row['seed_min'] == row['seed_max']:
            return form % row['value']
        return (form + ' [' + form + '–' + form + ']') % (row['value'], row['seed_min'], row['seed_max'])

    lines = []
    groups: tp.Dict[tp.Tuple, tp.Dict[str, tp.Dict[str, tp.Any]]] = {}
    for row in summary:
        groups.setdefault((row['run'], row['partition'], row['dataset'], row['support'], row['n']), {})[row['metric']] = row
    for (run, partition, dataset, support, n), metrics in groups.items():
        line = ('%s %s %s%s N=%d: %d flights, %d refused; remainder RMS median %s / p90 %s mm; end mismatch |median| %s ms; '
                'inference median %s / p90 %s ms') % (
            run, partition, dataset or '(real)', '' if partition != CONST_REAL else ' ' + support, n, metrics['refused']['flights'],
            int(metrics['refused']['value']), Cell(metrics['remainder_rms_median_mm']), Cell(metrics['remainder_rms_p90_mm']),
            Cell(metrics['end_mismatch_abs_median_ms']), Cell(metrics['inference_median_ms'], '%.2f'), Cell(metrics['inference_p90_ms'], '%.2f'))
        ranges = [(name[6:-3], row) for name, row in metrics.items() if name.startswith('range_')]
        if ranges:
            line += '; ranges ' + ', '.join('%s %s mm (%d)' % (label, Cell(row), row['flights']) for label, row in ranges)
        lines.append(line)
    for row in sim_to_real:
        lines.append('sim-to-real %s N=%d: real %.1f / p90 %.1f mm (%d flights); simulated %s %.1f / p90 %.1f mm (%d flights); '
                     'real - simulated %+.1f [%+.1f, %+.1f] mm; ratio %.2f'
                     % (row['run'], row['n'], row['real_median_mm'], row['real_p90_mm'], row['real_flights'], row['dataset'],
                        row['simulated_median_mm'], row['simulated_p90_mm'], row['simulated_flights'], row['difference_mm'], row['low_mm'],
                        row['high_mm'], row['ratio']))
    for row in paired:
        where = '%s %s%s' % (row['partition'], row['dataset'] or '(real)', '' if row['partition'] != CONST_REAL else ' ' + row['support'])
        if np.isfinite(row['median_mm']):
            result = '%+.2f [%+.2f, %+.2f] mm' % (row['median_mm'], row['low_mm'], row['high_mm'])
        else:
            result = 'values ' + ' '.join('%+.2f' % v for v in row['values']) + ' mm' if row['values'] else '-'
        lines.append('paired %s - %s, %s N=%d: %s (%d flights, %d left out)' % (row['run'], row['against'], where, row['n'], result,
                                                                             row['flights'], row['left_out']))
    return lines
