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
# File Name: ../test/Plot/test_figures.py
## =========================================================================== ##

# System (Default)
import os
# CSV (the epochs' files of the fixture)
import csv
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Learning/Core - the columns of epochs.csv
from Learning.Core import CONST_EPOCH_COLUMNS
#   ../Plot/Core - the figures' data
from Plot.Core import (Training_Curves, Load_Comparison, Load_Horizons, Load_Horizon_Ranges, Load_Restitution, Load_Classes, Load_Sweep,
                       Load_Decision,
                       Load_Interception, Throw_Str, Real_Str, Motion_Str, CONST_CLASS_COLUMNS)
#   ../Plot/Style - the look
from Plot.Style import Style_Context, Save_Figure, CONST_THEMES, CONST_BACKGROUND
#   ../Plot/Figures
from Plot import Figures
#   test/scripts/Script_Helpers - the constructed stored results
from Script_Helpers import _Plot_Results

"""
Description:
    Initialization of constants.

    The figures of scripts/plots.py (the visualisation, delivery 2; SPEC §7.14): each figure is built from small stored
    files written by their own writers (Script_Helpers._Plot_Results) through its loader, in both themes, and every data
    artist's plotted arrays equal the values written (the plotted data is compared, not the image
    bytes). The rows the loaders must leave out are in the files. Expectations come from the fixture's values.
"""
# Per comparison set its table, partition and dataset, as scripts/plots.py names them.
#   Unit [Dict<string, Tuple<string, string, string>>]
CONST_TABLES = {'validation': ('Final_Validation', 'validation', 'Default'), 'held_out': ('Final_Test', 'test', 'Default'),
                'real': ('Final_Test', 'real', '')}
# The restitution of each constructed contact (the figure's axis is computed by a function; here a table).
#   Unit [Dict<Tuple<float, float>, float>]
CONST_RESTITUTION = {(0.004, 0.7): 0.20, (0.004, 0.5): 0.29, (0.004, 0.3): 0.45}


@pytest.fixture(scope='module')
def stored(tmp_path_factory):
    """
    Description:
        The constructed stored results (Script_Helpers._Plot_Results) and the values written: (folder, values).
    """

    folder = tmp_path_factory.mktemp('stored')
    return str(folder), _Plot_Results(str(folder))


def _Data(figure, gid):
    """
    Description:
        The x and y data of the one artist with the gid; a Rectangle gives its x, bottom and height.
    """

    found = figure.findobj(lambda artist: artist.get_gid() == gid)
    assert len(found) == 1, (gid, len(found))
    artist = found[0]
    if hasattr(artist, 'get_height'):
        return np.array([artist.get_x()]), np.array([artist.get_y(), artist.get_height()])
    if hasattr(artist, 'get_data_3d'):
        return artist.get_data_3d()
    return np.asarray(artist.get_xdata(), dtype=float), np.asarray(artist.get_ydata(), dtype=float)


@pytest.mark.parametrize('theme', CONST_THEMES)
def test_the_comparison_plots_each_methods_accuracy_paired_difference_and_inference_time_of_compares_tables(stored, theme):
    """
    Description:
        The comparison's panels hold, per method and set, the median remainder RMS of the support 'all' (the 'inside' rows,
        5 mm higher, and the third run left out), the paired difference of the learned run against the numerical one, and
        the inference time's median, each against N.
    """

    folder, values = stored
    data = Load_Comparison(os.path.join(folder, 'Comparisons'), {'QT_BASIS': 'Learned', 'GLS_M4': 'Run'}, CONST_TABLES)
    with Style_Context(theme, False) as look:
        figure = Figures.Draw_Comparison(data, look)
    for (method, key), (value, low, high) in values['accuracy'].items():
        assert np.allclose(_Data(figure, 'accuracy/%s/%s' % (method, key))[1], value)
        assert np.allclose(data.Accuracy[(method, key)], [value, low, high])
        assert np.allclose(_Data(figure, 'inference/%s/%s' % (method, key))[1], values['inference'][(method, key)][0])
    for key, (median, low, high) in values['paired'].items():
        assert np.allclose(_Data(figure, 'paired/%s' % key)[1], median)
        assert np.allclose(data.Paired[key], [median, low, high], atol=1e-6)
    assert list(data.N) == values['n'] and data.Flights == {'validation': 1499, 'held_out': 1500, 'real': 45}


@pytest.mark.parametrize('kind', ['page', 'video'])
def test_the_comparison_bars_show_each_methods_median_rms_and_inference_time_and_the_speedup_of_the_stored_medians(stored, kind):
    """
    Description:
        Per method, set (the simulated test partition, the real flights) and N a bar of the median remainder RMS and one of
        the median inference time, at the values of compare's tables; the speed-up is the geometric mean, over N and both
        sets, of the numerical predictor's median inference time over the learned one's, written rounded.
    """

    folder, values = stored
    data = Load_Comparison(os.path.join(folder, 'Comparisons'), {'QT_BASIS': 'Learned', 'GLS_M4': 'Run'}, CONST_TABLES)
    ratios = [g / q for key in ('held_out', 'real') for g, q in zip(values['inference'][('GLS_M4', key)][0], values['inference'][('QT_BASIS',
                                                                                                                                  key)][0])]
    assert data.Speedup == pytest.approx(float(np.exp(np.mean(np.log(ratios)))))
    with Style_Context('dark', False, kind, key=True) as look:
        figure = Figures.Draw_Comparison_Bars(data, look)
    for method in ('QT_BASIS', 'GLS_M4'):
        for key in ('held_out', 'real'):
            for k, n in enumerate(values['n']):
                assert _Data(figure, 'bars/accuracy/%s/%s/%d' % (method, key, n))[1][1] == pytest.approx(values['accuracy'][(method, key)][0][k])
                assert _Data(figure, 'bars/inference/%s/%s/%d' % (method, key, n))[1][1] == pytest.approx(values['inference'][(method, key)][0][k])
    assert ('%.0f' % data.Speedup) in figure.findobj(lambda a: a.get_gid() == 'speedup')[0].get_text()
    assert figure.axes[0].get_ylabel() == {'page': 'prediction error (RMS) [mm]', 'video': 'prediction error [mm]'}[kind]
    values = [a for a in figure.axes[0].texts if a.get_text() and a.get_gid() is None]
    assert values and all(a.get_rotation() == (90.0 if kind == 'page' else 0.0) for a in values)


@pytest.mark.parametrize('theme', CONST_THEMES)
def test_the_prediction_error_plots_the_stored_ranges_median_and_90th_percentile_at_one_n_of_both_sets(stored, theme):
    """
    Description:
        From the final evaluations' summary.csv only, per set (the simulated test partition, the real flights) and method at
        N = 20: per horizon range at its middle the median over the training seeds of each seed's median, and the band's
        upper edge the median of their 90th percentiles; the observed-truth ranges, the spread rows, N = 10 and a range
        without a median left out.
    """

    folder, values = stored
    data = Load_Horizon_Ranges({'QT_BASIS': os.path.join(folder, 'Evaluations', 'Learned', 'test'),
                                'GLS_M4': os.path.join(folder, 'Evaluations', 'Run', 'test')}, {'held_out': ('test', 'Default'), 'real': ('real',
                                                                                                                                          '')}, 20)
    with Style_Context(theme, False) as look:
        figure = Figures.Draw_Prediction_Error(data, 20, look)
    for (key, method), (low, high, median, p90) in values['ranges'].items():
        assert np.allclose(data[(key, method)][:4], [low, high, median, p90])
        x, y = _Data(figure, 'horizon_range/%s/%s' % (key, method))
        assert np.allclose(x, 0.5 * (np.array(low) + np.array(high))) and np.allclose(y, median)
        assert np.allclose(_Data(figure, 'p90/%s/%s' % (key, method))[1], p90)


def test_the_horizon_figure_plots_the_median_over_the_seeds_of_each_seeds_median_without_the_spread_rows(stored):
    """
    Description:
        The fixed-horizon errors of the dataset asked for, per method and N: the median over the training seeds (the
        numerical predictor's one), the summary's spread rows, the end metric and the other dataset left out.
    """

    folder, values = stored
    data = Load_Horizons({'QT_BASIS': os.path.join(folder, 'Evaluations', 'Learned', 'validation'),
                          'GLS_M4': os.path.join(folder, 'Evaluations', 'Run', 'validation')}, 'Default')
    with Style_Context('dark', False) as look:
        figure = Figures.Draw_Horizon(data, look)
    assert list(data.Horizon) == [0.05, 0.1, 0.2]
    for (method, n), (median, low, high) in values['horizon'].items():
        x, y = _Data(figure, 'horizon/%s/%d' % (method, n))
        assert np.allclose(x, [0.05, 0.1, 0.2]) and np.allclose(y, median)
        assert np.allclose(data.Error[(method, n)], [median, low, high])


def test_the_restitution_figure_plots_each_modes_share_against_the_computed_restitution_and_marks_the_cliff(stored):
    """
    Description:
        The retained share of the goes of the first seed against the restitution the function gives each contact, per
        mode (the markers variant its own), dataset and acceleration; the cliff where the share crosses one half, linear
        between the two contacts around it: the static catch's between 0.20 (0.97) and 0.29 (0.30) at 800 deg/s^2.
    """

    folder, values = stored
    data = Load_Restitution({'static': os.path.join(folder, 'Catches', 'Learned', 'validation', 'summary.csv'),
                             'moving': os.path.join(folder, 'Catches', 'Learned', 'moving', 'validation', 'summary.csv')},
                            lambda t, d: CONST_RESTITUTION[(t, d)])
    with Style_Context('light', False) as look:
        figure = Figures.Draw_Restitution(data, look)
    with Style_Context('dark', False, key=True) as look:
        key_set = Figures.Draw_Restitution(data, look)
    assert not key_set.findobj(lambda a: str(a.get_gid()).startswith('share/markers/'))
    assert len(key_set.findobj(lambda a: str(a.get_gid()).startswith('share/'))) == 8
    box = key_set.findobj(lambda a: a.get_gid() == 'cliffs/Catch_In')[0].get_text()
    assert 'static %.2f / %.2f' % (data.Cliff[('static', 'Catch_In', 800)], data.Cliff[('static', 'Catch_In', 400)]) in box
    assert 'levels simulated: static 3, moving 3' in box
    assert [c[:2] for c in data.Contacts['static']] == [(0.004, 0.7), (0.004, 0.5), (0.004, 0.3)]
    assert set(data.Share) == set(values['share'])
    for (mode, dataset, acceleration), points in values['share'].items():
        x, y = _Data(figure, 'share/%s/%s/%d' % (mode, dataset, acceleration))
        order = sorted(points, key=lambda p: CONST_RESTITUTION[p[0]])
        assert np.allclose(x, [CONST_RESTITUTION[p[0]] for p in order]) and np.allclose(y, [100.0 * p[1] for p in order])
    assert data.Cliff[('static', 'Catch_In', 800)] == pytest.approx(0.20 + (0.97 - 0.5) * 0.09 / (0.97 - 0.30))
    assert _Data(figure, 'cliff/static/Catch_In/800')[0][0] == pytest.approx(data.Cliff[('static', 'Catch_In', 800)])
    assert np.isnan(data.Cliff[('markers', 'Catch_In', 800)])


def test_the_classes_figure_stacks_the_nominal_contacts_classes_summed_over_the_seeds_as_shares(stored):
    """
    Description:
        Each bar is a partition's, mode's, dataset's and acceleration's classes with the nominal contact, summed over the
        seeds, the levels and the markers variant left out, stacked as shares of the throws in CONST_CLASS_COLUMNS' order.
    """

    folder, values = stored
    data = Load_Classes({(p, mode): os.path.join(folder, 'Catches', 'Learned', *(('moving',) if mode == 'moving' else ()), p, 'summary.csv')
                         for p in ('validation', 'test') for mode in ('static', 'moving')}, (0.004, 0.7))
    with Style_Context('dark', False) as look:
        figure = Figures.Draw_Classes(data, look)
    names = ('caught', 'bounced_out', 'rim', 'no_go_catchable', 'no_go')
    for key, counts in values['classes'].items():
        assert [int(data[key][CONST_CLASS_COLUMNS.index(c)]) for c in names] == counts
        bottom = 0.0
        for c, count in zip(names, counts):
            _, (y, height) = _Data(figure, 'classes/%s/%s/%s/%d/%s' % (key + (c,)))
            assert y == pytest.approx(bottom) and height == pytest.approx(100.0 * count / sum(counts))
            bottom += height


def test_the_sweep_figure_plots_the_share_of_the_catchable_against_the_scale_and_the_runs_own_error_as_a_line(stored):
    """
    Description:
        Per error source, dataset and acceleration the retained share of the catchable against the injected error's scale;
        the run's own error (scale NaN) a horizontal line at its share.
    """

    folder, values = stored
    data = Load_Sweep(os.path.join(folder, 'Catches', 'Learned', 'sweep', 'summary.csv'))
    with Style_Context('dark', False) as look:
        figure = Figures.Draw_Sweep(data, look)
    for (source, dataset, acceleration), points in values['sweep'].items():
        x, y = _Data(figure, 'sweep/%s/%s/%d' % (source, dataset, acceleration))
        if source == 'run':
            assert np.allclose(y, 100.0 * points[0][1])
        else:
            assert np.allclose(x, [p[0] for p in points]) and np.allclose(y, [100.0 * p[1] for p in points])


def test_the_decision_figure_plots_the_mean_false_no_go_rate_and_the_largest_false_go_bound_of_both_rules(stored):
    """
    Description:
        Per rule (F9's stages, F10's every sample), partition, dataset and acceleration: the false-no-go rate's mean over
        the seeds and the false-go rate's upper bound's mean, in percent.
    """

    folder, values = stored
    data = Load_Decision({p: os.path.join(folder, 'Decisions', 'Learned', p, 'summary.csv') for p in ('validation', 'test')},
                         {p: os.path.join(folder, 'Interceptions', 'Learned', p, 'summary.csv') for p in ('validation', 'test')})
    with Style_Context('dark', False) as look:
        figure = Figures.Draw_Decision(data, look)
    for key, (rate, false_go, upper) in values['decision'].items():
        assert np.allclose(data[key], [rate, false_go, upper])
        assert _Data(figure, 'decision/false_no_go/%s/%s/%s/%d' % key)[1][0] == pytest.approx(100.0 * np.mean(rate))
        assert _Data(figure, 'decision/false_go_upper/%s/%s/%s/%d' % key)[1][0] == pytest.approx(100.0 * np.mean(upper))
        mark = figure.findobj(lambda a: a.get_gid() == 'decision/false_no_go/%s/%s/%s/%d' % key)[0]
        assert (mark.get_markerfacecolor() == CONST_BACKGROUND['dark']) == (key[1] == 'test')
    for dataset, acceleration in {(key[2], key[3]) for key in values['decision']}:
        total = sum(sum(v[1]) for key, v in values['decision'].items() if key[2:] == (dataset, acceleration))
        assert figure.findobj(lambda a: a.get_gid() == 'false_goes/%s/%d' % (dataset, acceleration))[0].get_text() == '%d' % total


def test_the_interception_figure_plots_the_lateral_misses_distribution_and_the_plans_of_one_datasets_goes(stored):
    """
    Description:
        Per mode and acceleration the sorted lateral misses [mm] of the dataset's goes with a crossing, their cumulative
        share, and the share of the goes per number of accepted plans; the other dataset's rows and the no-go rows left out.
    """

    folder, values = stored
    data = Load_Interception({'static': os.path.join(folder, 'Interceptions', 'Learned', 'validation', 'per_throw.csv'),
                              'moving': os.path.join(folder, 'Interceptions', 'Learned', 'moving', 'validation', 'per_throw.csv')}, 'Catch_In')
    with Style_Context('dark', False) as look:
        figure = Figures.Draw_Interception(data, look)
    for key, lateral in values['lateral'].items():
        x, y = _Data(figure, 'lateral/%s/%d' % key)
        assert np.allclose(x, lateral) and np.allclose(y, 100.0 * np.arange(1, len(lateral) + 1) / len(lateral))
        x, y = _Data(figure, 'plans/%s/%d' % key)
        counts = np.bincount(values['plans'][key])
        assert np.allclose(x, np.arange(len(counts))) and np.allclose(y, 100.0 * counts / len(values['plans'][key]))


def test_the_training_figure_plots_every_seeds_train_and_validation_loss_in_millimetres(tmp_path):
    """
    Description:
        Each training seed's epochs.csv (seed_<s>/) gives its train and validation loss [m], plotted in mm against the epoch.
    """

    losses = {seed: [(epoch, 0.1 / epoch + seed * 0.001, 0.12 / epoch, 0.12 / epoch) for epoch in (1, 2, 3)] for seed in (0, 1, 2, 3, 4)}
    for seed, rows in losses.items():
        os.makedirs(tmp_path / ('seed_%d' % seed))
        with open(tmp_path / ('seed_%d' % seed) / 'epochs.csv', 'w', newline='', encoding='ascii') as file:
            csv.writer(file, lineterminator='\n').writerows([CONST_EPOCH_COLUMNS] + [['%d' % r[0]] + ['%.9f' % v for v in r[1:]] for r in rows])
    with Style_Context('dark', False) as look:
        figure = Figures.Draw_Training(Training_Curves(str(tmp_path)), look)
    for seed, rows in losses.items():
        x, y = _Data(figure, 'train/%d' % seed)
        assert np.allclose(x, [1, 2, 3]) and np.allclose(y, [1e3 * r[1] for r in rows], atol=1e-6)
        assert np.allclose(_Data(figure, 'validation/%d' % seed)[1], [1e3 * r[2] for r in rows], atol=1e-6)
    styles = [str(figure.findobj(lambda a, s=seed: a.get_gid() == 'train/%d' % s)[0]._unscaled_dash_pattern) for seed in losses]
    assert len(set(styles)) == len(losses)


def _Throw():
    """
    Description:
        A constructed throw along +y with predictions of both methods at two N.
    """

    t = np.linspace(0.0, 1.0, 101)
    truth = np.column_stack([np.zeros_like(t), -4.0 + 4.0 * t, 1.0 + 5.0 * t - 4.9 * t ** 2])
    throw = Throw_Str(Dataset='D', Flight=7, Truth_Time=t, Truth_Position=truth, Observed_Time=t[:60:2], Observed_Position=truth[:60:2])
    for method, bias in (('QT_BASIS', 0.01), ('GLS_M4', -0.02)):
        for n in (10, 20):
            q = t[2 * n:60]
            throw.Predicted[(method, n)] = (q, truth[2 * n:60] + [0.0, 0.0, bias / n], truth[2 * n:60])
    return throw


def test_the_prediction_figure_plots_the_true_path_and_each_prediction_in_the_throws_plane_and_their_error():
    """
    Description:
        In the throw's vertical plane (the distance along it from its first true point; here y + 4) the true path and each
        method's prediction at each N, in the panel and its zoom; and the position error [mm] against the time.
    """

    throw = _Throw()
    with Style_Context('dark', False) as look:
        figure = Figures.Draw_Prediction(throw, look)
    x, y = _Data(figure, 'truth/GLS_M4/zoom')
    assert np.allclose(x, throw.Truth_Position[:, 1] + 4.0) and np.allclose(y, throw.Truth_Position[:, 2])
    for (method, n), (q, predicted, truth) in throw.Predicted.items():
        for suffix in ('', '/zoom'):
            x, y = _Data(figure, 'predicted/%s/%d%s' % (method, n, suffix))
            assert np.allclose(x, predicted[:, 1] + 4.0) and np.allclose(y, predicted[:, 2])
        x, y = _Data(figure, 'error/%s/%d' % (method, n))
        assert np.allclose(x, q) and np.allclose(y, 1e3 * abs({'QT_BASIS': 0.01, 'GLS_M4': -0.02}[method]) / n)
        assert np.allclose(_Data(figure, 'start/%s/%d' % (method, n)), [[q[0]], [y[0]]])
    right = max(float(np.max(line.get_xdata())) for line in figure.axes[2].get_lines() if len(line.get_xdata()))
    labels = figure.findobj(lambda a: str(a.get_gid()).startswith('end_label/'))
    assert len(labels) == 2 and all(label.xy[0] == pytest.approx(right) and label.get_horizontalalignment() == 'left' for label in labels)
    low, high = figure.axes[2].get_xlim()
    assert high - right > 0.2 * (right - low)


def test_the_real_figures_plot_the_recorded_samples_the_prefix_and_both_paths_in_3d_and_2d():
    """
    Description:
        The 3D figure: the samples after the prefix, the prefix and each method's path; the 2D figure: x-z, y-z, each axis
        against the time and the error against the horizon per N.
    """

    t = np.linspace(0.0, 1.0, 50)
    position = np.column_stack([0.1 * t, -4.0 + 4.0 * t, 1.0 + 4.0 * t - 4.9 * t ** 2])
    real = Real_Str(Name='F', Rate=50.0, Time=t, Position=position, Shown_N=20)
    for method in ('QT_BASIS', 'GLS_M4'):
        real.Path[method] = (t[19:], position[19:] + 0.001)
        for n in (10, 20):
            real.Error[(method, n)] = (t[n:] - t[n - 1], np.full(len(t) - n, float(n)))
    with Style_Context('dark', False) as look:
        three, two = Figures.Draw_Real_3D(real, look), Figures.Draw_Real_2D(real, look)
    assert np.allclose(np.array(_Data(three, 'recorded')), position[20:].T)
    assert np.allclose(np.array(_Data(three, 'prefix')), position[:20].T)
    assert np.allclose(np.array(_Data(three, 'path/GLS_M4')), real.Path['GLS_M4'][1].T)
    x, y = _Data(two, 'path/QT_BASIS/y')
    assert np.allclose(x, real.Path['QT_BASIS'][1][:, 1]) and np.allclose(y, real.Path['QT_BASIS'][1][:, 2])
    assert np.allclose(_Data(two, 'recorded/tz')[1], position[:, 2])
    assert np.allclose(_Data(two, 'error/GLS_M4/20')[1], 20.0)


def test_the_readme_variant_of_the_2d_real_figure_holds_the_y_z_view_and_the_error_only():
    """
    Description:
        The key set's variant of the 2D figure: two panels, (a) y-z with the recorded samples and both paths, (b) the error
        against the horizon per method and N; no x-z view and no axis against the time.
    """

    t = np.linspace(0.0, 1.0, 50)
    position = np.column_stack([0.1 * t, -4.0 + 4.0 * t, 1.0 + 4.0 * t - 4.9 * t ** 2])
    real = Real_Str(Name='F', Rate=50.0, Time=t, Position=position, Shown_N=20)
    for method in ('QT_BASIS', 'GLS_M4'):
        real.Path[method] = (t[19:], position[19:] + 0.001)
        for n in (10, 20):
            real.Error[(method, n)] = (t[n:] - t[n - 1], np.full(len(t) - n, float(n)))
    with Style_Context('light', False, key=True) as look:
        figure = Figures.Draw_Real_2D(real, look)
    assert len(figure.axes) == 2
    x, y = _Data(figure, 'recorded/y')
    assert np.allclose(x, position[:, 1]) and np.allclose(y, position[:, 2])
    for method in ('QT_BASIS', 'GLS_M4'):
        x, y = _Data(figure, 'path/%s/y' % method)
        assert np.allclose(x, real.Path[method][1][:, 1]) and np.allclose(y, real.Path[method][1][:, 2])
        for n in (10, 20):
            assert np.allclose(_Data(figure, 'error/%s/%d' % (method, n))[1], float(n))
    assert not figure.findobj(lambda a: str(a.get_gid()) in ('recorded/x', 'recorded/tz', 'path/QT_BASIS/x', 'path/GLS_M4/tz'))
    assert figure.axes[0].get_xlabel() == '$y$ [m]' and figure.axes[1].get_xlabel() == 'horizon [s]'


def _Motion():
    """
    Description:
        A constructed motion: the five phases, the contact at 1.0 s, a cup moving with the ball before it.
    """

    t = np.arange(0.0, 1.5, 0.002)
    motion = Motion_Str(Contact=1.0, Relative_Speed=5.5, Contact_Part='cup_bottom', Class='caught', Mouth_Pass=0.98,
                        At_Contact={'moving': (0.998, 6.0, 0.2, 5.8), 'static': (0.99, 6.0, 0.0, 6.0)})
    motion.Phases = [('observe', 0.0, 0.2), ('move', 0.2, 0.5), ('wait', 0.5, 0.8), ('stroke', 0.8, 1.0), ('braking', 1.0, 1.2), ('rest', 1.2, 1.5)]
    for mode, cup in (('moving', np.clip(t - 0.8, 0.0, None)), ('static', np.zeros_like(t))):
        motion.Time[mode], motion.Cup_Speed[mode], motion.Ball_Speed[mode] = t, cup, 7.0 - t
        motion.Distance[mode] = np.abs(1.0 - t) + 0.01
    motion.Joints, motion.Joint_Speeds = np.outer(t, np.arange(1, 7)), np.tile(np.arange(1.0, 7.0), (len(t), 1))
    return motion


def test_the_motion_figure_plots_both_catches_cup_speed_the_balls_speed_the_distance_and_the_phases():
    """
    Description:
        From the release to 0.4 s after the last phase's start, the ball's speed, each catch's cup speed and distance; each
        phase shaded over its span in both panels; the contact marked; both catches' relative speed at contact written (the
        moving one's from its catch row, the static one's from its record); and the joints' positions and speeds in degrees.
    """

    motion = _Motion()
    with Style_Context('dark', False) as look:
        figure, joints = Figures.Draw_Motion(motion, look), Figures.Draw_Joints(motion, look)
    t = motion.Time['moving']
    keep = (t >= 0.0) & (t <= 1.6)
    for mode in ('moving', 'static'):
        x, y = _Data(figure, 'cup_speed/%s' % mode)
        assert np.allclose(x, t[keep]) and np.allclose(y, motion.Cup_Speed[mode][keep])
        assert np.allclose(_Data(figure, 'distance/%s' % mode)[1], motion.Distance[mode][keep])
    assert np.allclose(_Data(figure, 'ball_speed/moving')[1], motion.Ball_Speed['moving'][keep])
    spans = figure.findobj(lambda artist: artist.get_gid() == 'phase/stroke')
    assert len(spans) == 2 and all(np.allclose([s.get_x(), s.get_x() + s.get_width()], [0.8, 1.0]) for s in spans)
    assert len(figure.findobj(lambda artist: artist.get_gid() == 'contact')) == 2
    assert figure.axes[0].get_ylim()[1] == pytest.approx(1.22 * float(np.max(motion.Ball_Speed['moving'][keep])))
    texts = {mode: figure.findobj(lambda a, m=mode: a.get_gid() == 'relative/%s' % m)[0].get_text() for mode in ('moving', 'static')}
    assert texts == {'moving': 'moving 5.50 m/s', 'static': 'static 6.00 m/s'}
    assert np.allclose(_Data(joints, 'speed/4')[1], np.degrees(4.0))


def test_a_video_figure_is_written_at_3840_by_2160_pixels(tmp_path):
    """
    Description:
        The video's kind gives a PNG of 3840 x 2160 px (12.8 x 7.2 in at 300 dpi, no tight bounding box), whatever it draws.
    """

    from PIL import Image
    with Style_Context('dark', False, 'video') as look:
        written = Save_Figure(Figures.Draw_Motion(_Motion(), look), str(tmp_path / 'video'), ('png',))
    with Image.open(written[0]) as image:
        assert image.size == (3840, 2160)
