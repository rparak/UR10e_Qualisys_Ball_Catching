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
# File Name: ../Plot/Figures.py
## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Matplotlib (Figures) [pip3 install matplotlib]: the legend's handles, the 3D axes
import matplotlib
import matplotlib.figure
import matplotlib.patches
import matplotlib.ticker
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Plot/Core - the figures' data
from Plot.Core import (Comparison_Str, Horizon_Str, Throw_Str, Real_Str, Share_Str, Motion_Str, CONST_COMPARISON_SETS, CONST_SPEEDUP_SETS,
                       CONST_CLASS_COLUMNS)
#   ../Plot/Style - the look
from Plot.Style import (Look_Str, CONST_MARKERS, CONST_ACCELERATION_STYLES, CONST_SET_STYLES, CONST_N_STYLES, CONST_JOINT_STYLES, CONST_SEED_STYLES,
                        CONST_DEGREES_S2, CONST_BACKGROUND)

"""
Description:
    Initialization of constants.

    The figures of scripts/plots.py (the visualisation, delivery 2; SPEC §7.14): each function draws one figure from its
    data (Plot.Core) in a look (Plot.Style.Style_Context) and returns it; nothing here reads a file or computes a
    plotted number. Every data artist carries a gid naming what it shows ('<quantity>/<method>/<set>/...'), so that a
    test compares the plotted arrays with the source. No title inside an image (a caption goes with it, captions.csv).
"""
# The sets' names in a legend.
#   Unit [Dict<string, string>]
CONST_SET_NAMES = {'validation': 'simulated validation', 'held_out': 'simulated test', 'real': 'real validation flights'}
# The phases' names in the motion figure.
#   Unit [Dict<string, string>]
CONST_PHASE_NAMES = {'observe': 'observe and decide', 'move': 'move to the stroke', 'wait': 'wait', 'stroke': 'stroke', 'braking': 'braking',
                     'rest': 'rest'}
# The part of the cup a ball meets first, in words (the catch row's first_contact).
#   Unit [Dict<string, string>]
CONST_PART_NAMES = {'cup_bottom': 'the cup\'s bottom', 'cup_wall': 'the cup\'s wall', 'cup_lip': 'the cup\'s lip'}
# The bars' sets in the comparison's bars: the simulated test plain, the real flights hatched.
#   Unit [Dict<string, string or None>]
CONST_SET_HATCHES = {'held_out': None, 'real': '///'}
# The catch's classes' names in a legend.
#   Unit [Dict<string, string>]
CONST_CLASS_NAMES = {'caught': 'caught', 'bounced_out': 'bounced out', 'rim': 'rim', 'hit_robot': 'hit the robot', 'missed': 'missed',
                     'hit_stand': 'hit the stand', 'late': 'late', 'no_go_catchable': 'no go, catchable', 'no_go': 'no go'}
# The horizontal offset of the sets at one N in the comparison, as a fraction of N's step.
#   Unit [float]
CONST_OFFSET = 0.12


def _Legend(axes: tp.Any, look: Look_Str, handles: tp.List[tp.Any], **options: tp.Any) -> None:
    """
    Description:
        A legend of handles drawn in the theme's colours, outside the data where the options say so.

    Args:
        (1) axes [matplotlib.axes.Axes(object)]: The panel.
        (2) look [Look_Str(object)]: The look.
        (3) handles [List<Artist>]: The handles, each with its label.
        (4) options [Dict<string, Any>]: matplotlib's legend options.
    """

    axes.legend(handles=handles, **options)


def _Log_Axis(axis: tp.Any, subs: tp.Sequence[float] = (1.0, 2.0, 5.0)) -> None:
    """
    Description:
        A logarithmic axis labelled with plain numbers (60, not 6 x 10^1), its minor ticks unlabelled.

    Args:
        (1) axis [matplotlib.axis.Axis(object)]: The axis (axes.yaxis).
        (2) subs [Sequence<float>]: The labelled multiples of each decade.
    """

    axis.axes.set_yscale('log') if axis is axis.axes.yaxis else axis.axes.set_xscale('log')
    axis.set_major_locator(matplotlib.ticker.LogLocator(subs=subs))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: '%g' % value))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _Handle(look: Look_Str, label: str, colour: str, linestyle: str = '-', marker: tp.Optional[str] = None) -> Line2D:
    """
    Description:
        A legend handle: a line of a colour, style and marker, with its label set for the text engine.

    Args:
        (1) look [Look_Str(object)]: The look.
        (2) label [string]: The label.
        (3) colour [string]: Its colour.
        (4) linestyle [string]: Its line style.
        (5) marker [string or None]: Its marker.

    Returns:
        (1) parameter [Line2D(object)]: The handle.
    """

    return Line2D([], [], color=colour, linestyle=linestyle, marker=marker, label=look.Text(label))


def Draw_Training(curves: tp.Dict[int, np.ndarray], look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The learning curves of the learned predictor: per training seed (its own line style) the train loss and the
        validation loss [mm] against the epoch, on a logarithmic axis labelled down to its smallest decade's multiples.

    Args:
        (1) curves [Dict<int, Matrix<float> Ex4>]: Per seed its epochs (Plot.Core.Training_Curves).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, 2, sharey=True)
    colour = look.Colour('QT_BASIS')
    for k, (column, name) in enumerate(((1, 'train'), (2, 'validation'))):
        for n, (seed, epochs) in enumerate(sorted(curves.items())):
            line, = axes[k].plot(epochs[:, 0], 1e3 * epochs[:, column], color=colour, linestyle=CONST_SEED_STYLES[n % len(CONST_SEED_STYLES)],
                                 label=look.Text('seed %d' % seed))
            line.set_gid('%s/%d' % (name, seed))
        axes[k].set_xlabel(look.Text('epoch [-]'))
        axes[k].set_ylabel(look.Text('%s loss [mm]' % name))
        look.Panel(axes[k], 'ab'[k])
    _Log_Axis(axes[0].yaxis, (1.0, 2.0, 3.0, 4.0, 5.0))
    axes[1].tick_params(axis='y', labelleft=True)
    axes[1].legend()
    return figure


def Draw_Comparison(c: Comparison_Str, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The two predictors compared (the full output): (a) the median remainder RMS per N on each set (the range over the
        training seeds as bars), (b) the learned predictor's median paired difference against the numerical one with its
        95 % bootstrap interval, (c) the inference time per prediction, its median with the 90th percentile as the bar's
        top (log scale).

    Args:
        (1) c [Comparison_Str(object)]: The comparison (Plot.Core.Load_Comparison).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, 3)
    methods = list(dict.fromkeys(method for method, _ in c.Accuracy))
    sets = list(CONST_COMPARISON_SETS)
    x = np.arange(len(c.N), dtype=np.float64)
    for m, method in enumerate(methods):
        for s, key in enumerate(sets):
            offset = CONST_OFFSET * (s - 1) + 0.04 * (m - 0.5)
            value, low, high = c.Accuracy[(method, key)]
            bars = axes[0].errorbar(x + offset, value, yerr=[value - low, high - value], color=look.Colour(method), linestyle=CONST_SET_STYLES[key],
                                    marker=CONST_MARKERS[method])
            bars.lines[0].set_gid('accuracy/%s/%s' % (method, key))
            median, p90 = c.Inference[(method, key)]
            bars = axes[2].errorbar(x + offset, median, yerr=[np.zeros_like(median), p90 - median], color=look.Colour(method),
                                    linestyle=CONST_SET_STYLES[key], marker=CONST_MARKERS[method])
            bars.lines[0].set_gid('inference/%s/%s' % (method, key))
    for s, key in enumerate(sets):
        median, low, high = c.Paired[key]
        bars = axes[1].errorbar(x + CONST_OFFSET * (s - 1), median, yerr=[median - low, high - median], color=look.Colour('truth'),
                                linestyle=CONST_SET_STYLES[key], marker='o')
        bars.lines[0].set_gid('paired/%s' % key)
    axes[1].axhline(0.0, color=look.Colour('observed'), linewidth=0.5)
    _Log_Axis(axes[2].yaxis)
    for k, label in enumerate(('median remainder RMS [mm]', '%s $-$ %s [mm]' % tuple(methods[:2]), 'inference time [ms]')):
        axes[k].set_xticks(x, ['%d' % n for n in c.N])
        axes[k].xaxis.set_tick_params(which='minor', bottom=False, top=False)
        axes[k].set_xlabel(look.Text('observed samples $N$ [-]'))
        axes[k].set_ylabel(look.Text(label))
        look.Panel(axes[k], 'abc'[k])
    handles = [_Handle(look, method, look.Colour(method), marker=CONST_MARKERS[method]) for method in methods]
    handles += [_Handle(look, '%s (%d)' % (CONST_SET_NAMES[key], c.Flights[key]), look.Colour('truth'), CONST_SET_STYLES[key]) for key in sets]
    figure.legend(handles=handles, loc='outside lower center', ncols=len(handles) if look.Kind == 'page' else 3)
    return figure


def _Bar_Value(axes: tp.Any, bar: tp.Any, value: float, text: str, look: Look_Str) -> None:
    """
    Description:
        A bar's value written above its top (or its whisker's), small, in the theme's text colour.

    Args:
        (1) axes [matplotlib.axes.Axes(object)]: The panel.
        (2) bar [Rectangle(object)]: The bar.
        (3) value [float]: The height the text sits above.
        (4) text [string]: The text.
        (5) look [Look_Str(object)]: The look.
    """

    axes.annotate(look.Text(text), (bar.get_x() + 0.5 * bar.get_width(), value), xytext=(0, 1.5), textcoords='offset points', ha='center',
                  va='bottom', fontsize=(0.62 if look.Kind == 'page' else 0.8) * matplotlib.rcParams['font.size'],
                  rotation=90 if look.Kind == 'page' else 0)


def Draw_Comparison_Bars(c: Comparison_Str, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The two predictors compared as bars (the README's and the video's): (a) the median remainder RMS [mm] per N, the
        simulated test partition plain and the real validation flights hatched, colour by method, the range over the
        learned predictor's training seeds as the error bar (the stored tables hold no 95 % interval of a median), the value
        above each bar; (b) the inference time per prediction [ms] on a logarithmic axis, the median as the bar and the 90th
        percentile as the whisker (the stored percentile), the value above each bar, and how many times faster the learned
        predictor is (Comparison_Str.Speedup).

    Args:
        (1) c [Comparison_Str(object)]: The comparison (Plot.Core.Load_Comparison).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, 2)
    methods = list(dict.fromkeys(method for method, _ in c.Accuracy))
    sets = list(CONST_SPEEDUP_SETS)
    x = np.arange(len(c.N), dtype=np.float64)
    width = 0.8 / (len(methods) * len(sets))
    for s, key in enumerate(sets):
        for m, method in enumerate(methods):
            position = x + width * (s * len(methods) + m - 0.5 * (len(methods) * len(sets) - 1))
            value, low, high = c.Accuracy[(method, key)]
            median, p90 = c.Inference[(method, key)]
            for k, (panel, height, error, text) in enumerate(((axes[0], value, [value - low, high - value],
                                                               '%.1f' if look.Kind == 'page' else '%.0f'),
                                                              (axes[1], median, [np.zeros_like(median), p90 - median], '%.2g'))):
                bars = panel.bar(position, height, width=0.92 * width, color=look.Colour(method), hatch=CONST_SET_HATCHES[key],
                                 edgecolor=CONST_BACKGROUND[look.Theme], linewidth=0.0, yerr=error, ecolor=look.Colour('truth'),
                                 error_kw={'elinewidth': 0.6 * matplotlib.rcParams['lines.linewidth']})
                for n, bar in zip(c.N, bars.patches):
                    bar.set_gid('bars/%s/%s/%s/%d' % (('accuracy', 'inference')[k], method, key, n))
                for bar, top, number in zip(bars.patches, height + error[1], height):
                    _Bar_Value(panel, bar, top, text % number, look)
    _Log_Axis(axes[1].yaxis)
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 4.0)
    axes[1].annotate(look.Text('$\\approx$ %.0f$\\times$ faster' % c.Speedup), (0.5, 0.97), xycoords='axes fraction', ha='center', va='top',
                     color=look.Colour(methods[0]), gid='speedup')
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.12)
    for k, (page, video) in enumerate((('prediction error (RMS) [mm]', 'prediction error [mm]'),
                                       ('inference time per prediction [ms]', 'time per prediction [ms]'))):
        axes[k].set_xticks(x, ['$N = %d$' % n for n in c.N])
        axes[k].xaxis.set_tick_params(which='both', bottom=False, top=False)
        axes[k].set_ylabel(look.Words(page, video))
        if look.Kind == 'page':
            look.Panel(axes[k], 'ab'[k])
    handles = [matplotlib.patches.Patch(facecolor=look.Colour(method), label=look.Text(method)) for method in methods]
    handles += [matplotlib.patches.Patch(facecolor='none', edgecolor=look.Colour('truth'), hatch=CONST_SET_HATCHES[key],
                                         label=look.Words('%s (%d)' % (CONST_SET_NAMES[key], c.Flights[key]),
                                                          {'held_out': 'simulated', 'real': 'real flights'}[key])) for key in sets]
    figure.legend(handles=handles, loc='outside lower center', ncols=len(handles))
    return figure


def Draw_Horizon(h: Horizon_Str, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The position error at the fixed horizons on the simulated validation, one panel per N, both predictors (the
        learned one's range over its training seeds shaded).

    Args:
        (1) h [Horizon_Str(object)]: The errors (Plot.Core.Load_Horizons).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, len(h.N), sharey=True, squeeze=False)[0]
    methods = list(dict.fromkeys(method for method, _ in h.Error))
    for k, n in enumerate(h.N):
        for method in methods:
            median, low, high = h.Error[(method, int(n))]
            line, = axes[k].plot(h.Horizon, median, color=look.Colour(method), marker=CONST_MARKERS[method], label=look.Text(method))
            line.set_gid('horizon/%s/%d' % (method, n))
            axes[k].fill_between(h.Horizon, low, high, color=look.Colour(method), alpha=0.2, linewidth=0)
        axes[k].set_xlabel(look.Text('horizon [s]'))
        axes[k].set_title(look.Text('$N = %d$' % n), fontsize=axes[k].xaxis.label.get_size())
        look.Panel(axes[k], 'abcdef'[k])
    axes[0].set_ylabel(look.Text('median position error [mm]'))
    axes[0].legend()
    return figure


def _Along(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Description:
        The horizontal distance of points along a throw's direction from its origin.

    Args:
        (1) points [Matrix<float> Kx3]: The points [m].
        (2) origin [Vector<float> 1x3]: The origin [m].
        (3) direction [Vector<float> 1x2]: The horizontal unit direction.

    Returns:
        (1) parameter [Vector<float> 1xK]: The distances [m].
    """

    return (points[:, :2] - origin[:2]) @ direction


def Draw_Prediction(throw: Throw_Str, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The prediction converging on one simulated validation throw: (a), (b) each predictor's stored predictions from the
        first N observed samples (the smallest N dotted, the largest solid) against the true path, in the vertical plane of
        the throw (the horizontal distance along it from its first true point), with a zoom on the last 0.5 m observed;
        (c) their position error against the time from the release, each curve starting at the first sample after its
        prefix (marked) and named at its end (for the learned predictor).

    Args:
        (1) throw [Throw_Str(object)]: The throw (Plot.Core.Load_Throw).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, 3)
    methods = list(dict.fromkeys(method for method, _ in throw.Predicted))
    n_values = sorted({n for _, n in throw.Predicted})
    origin = throw.Truth_Position[0]
    span = throw.Truth_Position[-1, :2] - origin[:2]
    direction = span / np.linalg.norm(span)
    end = _Along(throw.Observed_Position, origin, direction)[-1]
    for m, method in enumerate(methods):
        ax = axes[m]
        zoom = ax.inset_axes([0.06, 0.06, 0.42, 0.40])
        for target, suffix in ((ax, ''), (zoom, '/zoom')):
            line, = target.plot(_Along(throw.Truth_Position, origin, direction), throw.Truth_Position[:, 2], color=look.Colour('truth'),
                                label=look.Text('true path'))
            line.set_gid('truth/%s%s' % (method, suffix))
            dots, = target.plot(_Along(throw.Observed_Position, origin, direction), throw.Observed_Position[:, 2], linestyle='none', marker='.',
                                color=look.Colour('observed'), markersize=2.0, label=look.Text('observed samples'))
            dots.set_gid('observed/%s%s' % (method, suffix))
            for k, n in enumerate(n_values):
                if (method, n) not in throw.Predicted:
                    continue
                _, predicted, _ = throw.Predicted[(method, n)]
                line, = target.plot(_Along(predicted, origin, direction), predicted[:, 2], color=look.Colour(method), linestyle=CONST_N_STYLES[k % 3])
                line.set_gid('predicted/%s/%d%s' % (method, n, suffix))
        inside = np.abs(_Along(throw.Truth_Position, origin, direction) - (end - 0.25)) <= 0.25
        z = throw.Truth_Position[inside, 2]
        zoom.set_xlim(end - 0.5, end)
        zoom.set_ylim(z.min() - 0.05, z.max() + 0.05)
        zoom.tick_params(which='both', labelleft=False, labelbottom=False)
        ax.indicate_inset_zoom(zoom, edgecolor=look.Colour('observed'))
        ax.set_xlabel(look.Text('distance along the throw [m]'))
        ax.set_ylabel(look.Text('height $z$ [m]'))
        ax.set_title(look.Text(method), fontsize=ax.xaxis.label.get_size())
        look.Panel(ax, 'ab'[m])
    for m, method in enumerate(methods):
        for k, n in enumerate(n_values):
            if (method, n) not in throw.Predicted:
                continue
            t, predicted, truth = throw.Predicted[(method, n)]
            error = 1e3 * np.linalg.norm(predicted - truth, axis=1)
            line, = axes[2].plot(t, error, color=look.Colour(method), linestyle=CONST_N_STYLES[k % 3])
            line.set_gid('error/%s/%d' % (method, n))
            start, = axes[2].plot(t[:1], error[:1], marker='o', markersize=3.0, color=look.Colour(method), linestyle='none')
            start.set_gid('start/%s/%d' % (method, n))
            if m == 0:
                ends = [1e3 * np.linalg.norm(throw.Predicted[(other, n)][1][-1] - throw.Predicted[(other, n)][2][-1]) for other in methods
                        if (other, n) in throw.Predicted]
                label = axes[2].annotate(look.Text('$N = %d$' % n), (t[-1], float(np.sqrt(np.min(ends) * np.max(ends)))), xytext=(5, 0),
                                         textcoords='offset points', ha='left', va='center', fontsize=0.75 * matplotlib.rcParams['font.size'])
                label.set_gid('end_label/%d' % n)
    _Log_Axis(axes[2].yaxis)
    low, high = axes[2].get_xlim()
    axes[2].set_xlim(low, high + 0.25 * (high - low))
    axes[2].set_xlabel(look.Text('time from the release [s]'))
    axes[2].set_ylabel(look.Text('position error [mm]'))
    look.Panel(axes[2], 'c')
    handles = [_Handle(look, 'true path', look.Colour('truth')), Line2D([], [], color=look.Colour('observed'), marker='.', linestyle='none',
                                                                        label=look.Text('observed samples'))]
    handles += [_Handle(look, method, look.Colour(method)) for method in methods]
    handles += [_Handle(look, 'from $N = %d$' % n, look.Colour('truth'), CONST_N_STYLES[k % 3]) for k, n in enumerate(n_values)]
    figure.legend(handles=handles, loc='outside lower center', ncols=len(handles) if look.Kind == 'page' else 4)
    return figure


def Draw_Prediction_Error(ranges: tp.Dict[tp.Tuple[str, str], np.ndarray], n: int, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The position error against the prediction horizon at one N, both predictors: (a) the simulated test partition,
        (b) the real validation flights, both from the stored summaries of F14's final evaluations; per horizon range (at
        its middle) and method, in its colour, the median as a solid line and the 90th percentile as a dotted one (the
        summaries hold no quartiles).

    Args:
        (1) ranges [Dict<Tuple<string, string>, Matrix<float> 5xR>]: The errors (Plot.Core.Load_Horizon_Ranges).
        (2) n [int]: The N, for the label.
        (3) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    keys = [key for key in CONST_SPEEDUP_SETS if any(k == key for k, _ in ranges)]
    axes = figure.subplots(1, len(keys), sharey=True, squeeze=False)[0]
    methods = list(dict.fromkeys(method for _, method in ranges))
    for k, key in enumerate(keys):
        for method in methods:
            low, high, median, p90, _ = ranges[(key, method)]
            middle = 0.5 * (low + high)
            line, = axes[k].plot(middle, median, color=look.Colour(method), marker=CONST_MARKERS[method], label=look.Text(method))
            line.set_gid('horizon_range/%s/%s' % (key, method))
            edge, = axes[k].plot(middle, p90, color=look.Colour(method), linestyle=':', marker=CONST_MARKERS[method], markerfacecolor='none')
            edge.set_gid('p90/%s/%s' % (key, method))
        axes[k].set_xlabel(look.Text('prediction horizon [s]'))
        axes[k].set_title(look.Text('%s, $N = %d$' % (CONST_SET_NAMES[key], n)), fontsize=axes[k].xaxis.label.get_size())
        look.Panel(axes[k], 'ab'[k])
    axes[0].set_ylabel(look.Text('position error [mm]'))
    handles = [_Handle(look, method, look.Colour(method), marker=CONST_MARKERS[method]) for method in methods]
    handles += [_Handle(look, 'solid: median', look.Colour('truth'), '-'), _Handle(look, 'dotted: 90th percentile', look.Colour('truth'), ':')]
    axes[0].legend(handles=handles, loc='upper left')
    return figure


def _Theme_3D(axes: tp.Any, look: Look_Str) -> None:
    """
    Description:
        A 3D panel in the theme: its panes the background, its grid muted.

    Args:
        (1) axes [Axes3D(object)]: The panel.
        (2) look [Look_Str(object)]: The look.
    """

    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        axis._axinfo['grid']['color'] = look.Colour('observed')
        axis._axinfo['grid']['linewidth'] = 0.2
        axis._axinfo['grid']['linestyle'] = ':'


def Draw_Real_3D(real: Real_Str, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        One real identification flight in 3D (the robot frame by the assumed transform): the recorded samples, the first
        N of them (the prefix) and both predictors' paths from that prefix; the box's aspect is the data's ranges (true to
        scale), at most three ticks per axis and no minor ticks.

    Args:
        (1) real [Real_Str(object)]: The flight (Plot.Core.Real_Predictions).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = matplotlib.figure.Figure(figsize=look.Figure().get_size_inches())
    axes = figure.add_subplot(projection='3d')
    _Theme_3D(axes, look)
    n = real.Shown_N
    line, = axes.plot(*real.Position[n:].T, linestyle='none', marker='.', markersize=2.5, color=look.Colour('observed'),
                      label=look.Text('recorded samples'))
    line.set_gid('recorded')
    line, = axes.plot(*real.Position[:n].T, linestyle='none', marker='o', markersize=1.2, color=look.Colour('truth'),
                      label=look.Text('the first $N = %d$ samples' % n))
    line.set_gid('prefix')
    for method, (_, path) in real.Path.items():
        line, = axes.plot(*path.T, color=look.Colour(method), linestyle='-' if method == 'QT_BASIS' else '--', label=look.Text(method))
        line.set_gid('path/%s' % method)
    points = np.vstack([real.Position] + [p for _, p in real.Path.values()])
    low, high = points.min(axis=0), points.max(axis=0)
    for k, set_limits in enumerate((axes.set_xlim, axes.set_ylim, axes.set_zlim)):
        set_limits(low[k], high[k])
    axes.set_box_aspect(high - low, zoom=1.3)
    for k, (axis, name) in enumerate(((axes.xaxis, 'x'), (axes.yaxis, 'y'), (axes.zaxis, 'z'))):
        narrow = high[k] - low[k] < 0.1 * float(np.max(high - low))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
        if narrow:
            middle = round(0.5 * (low[k] + high[k]), 2)
            axis.set_major_locator(matplotlib.ticker.FixedLocator([middle]))
            axis.set_major_formatter(matplotlib.ticker.FixedFormatter([look.Text('$%s = %g$ m' % (name, middle))]))
            continue
        axis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        axis.set_label_text(look.Text('$%s$ [m]' % name))
    axes.view_init(elev=32, azim=-30)
    axes.set_position([0.0, -0.1, 1.0, 1.2])
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    return figure


def Draw_Real_2D(real: Real_Str, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        One real identification flight in 2D: (a) x-z and (b) y-z with both paths from the prefix, (c) each axis against
        the time with both paths, (d) the position error against the horizon for each N. The key set's variant (the
        README's) holds (b) and (d) only, as (a) and (b): a real flight is nearly planar.

    Args:
        (1) real [Real_Str(object)]: The flight (Plot.Core.Real_Predictions).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    if look.Key:
        figure = look.Figure()
        plane, error = figure.subplots(1, 2)
        _Real_Plane(plane, real, look, 1, 'y')
        _Real_Error(error, real, look)
        for k, axis in enumerate((plane, error)):
            look.Panel(axis, 'ab'[k])
        return figure
    figure = look.Figure('tall')
    axes = figure.subplots(2, 2).ravel()
    for k, (i, name) in enumerate(((0, 'x'), (1, 'y'))):
        _Real_Plane(axes[k], real, look, i, name)
    for i, name in enumerate('xyz'):
        line, = axes[2].plot(real.Time, real.Position[:, i], linestyle='none', marker='.', markersize=1.5, color=look.Colour('observed'))
        line.set_gid('recorded/t%s' % name)
        for method, (t, path) in real.Path.items():
            line, = axes[2].plot(t, path[:, i], color=look.Colour(method), linestyle='-' if method == 'QT_BASIS' else '--')
            line.set_gid('path/%s/t%s' % (method, name))
        middle = len(real.Time) // 2
        axes[2].annotate(look.Text('$%s$' % name), (real.Time[middle], real.Position[middle, i]), xytext=(0, 4), textcoords='offset points',
                         ha='center', va='bottom')
    axes[2].set_xlabel(look.Text('time from the first kept sample [s]'))
    axes[2].set_ylabel(look.Text('position [m]'))
    _Real_Error(axes[3], real, look)
    for k in range(4):
        look.Panel(axes[k], 'abcd'[k])
    return figure


def _Real_Plane(axis: tp.Any, real: Real_Str, look: Look_Str, i: int, name: str) -> None:
    """
    Description:
        One plane of a real flight against z: the recorded samples and both paths from the prefix.

    Args:
        (1) axis [matplotlib.axes.Axes(object)]: The panel.
        (2) real [Real_Str(object)]: The flight.
        (3) look [Look_Str(object)]: The look.
        (4) i [int]: The horizontal axis's index (0 x, 1 y).
        (5) name [string]: Its name.
    """

    line, = axis.plot(real.Position[:, i], real.Position[:, 2], linestyle='none', marker='.', markersize=2.0, color=look.Colour('observed'))
    line.set_gid('recorded/%s' % name)
    for method, (_, path) in real.Path.items():
        line, = axis.plot(path[:, i], path[:, 2], color=look.Colour(method), linestyle='-' if method == 'QT_BASIS' else '--')
        line.set_gid('path/%s/%s' % (method, name))
    axis.set_xlabel(look.Text('$%s$ [m]' % name))
    axis.set_ylabel(look.Text('$z$ [m]'))


def _Real_Error(axis: tp.Any, real: Real_Str, look: Look_Str) -> None:
    """
    Description:
        A real flight's position error against the horizon, per method and N, with the legend.

    Args:
        (1) axis [matplotlib.axes.Axes(object)]: The panel.
        (2) real [Real_Str(object)]: The flight.
        (3) look [Look_Str(object)]: The look.
    """

    n_values = sorted({n for _, n in real.Error})
    for method in dict.fromkeys(method for method, _ in real.Error):
        for k, n in enumerate(n_values):
            horizon, error = real.Error[(method, n)]
            line, = axis.plot(horizon, error, color=look.Colour(method), linestyle=CONST_N_STYLES[k % 3])
            line.set_gid('error/%s/%d' % (method, n))
    _Log_Axis(axis.yaxis)
    axis.set_xlabel(look.Text('horizon [s]'))
    axis.set_ylabel(look.Text('position error [mm]'))
    _Legend(axis, look, [_Handle(look, method, look.Colour(method)) for method in real.Path]
            + [_Handle(look, '$N = %d$' % n, look.Colour('truth'), CONST_N_STYLES[k % 3]) for k, n in enumerate(n_values)], loc='lower right')


def Draw_Restitution(share: Share_Str, look: Look_Str, datasets: tp.Sequence[str] = ('Catch_In', 'Catch_Near')) -> matplotlib.figure.Figure:
    """
    Description:
        The retained share of the goes against the restitution (the mean over the phases of a contact within a step), one
        panel per dataset: the static catch and the moving catch, and in the full output also the moving catch with the
        ball's markers as collision shapes (left out of the key set's variant); the first seed, each point with its 95 %
        interval, 800 °/s² solid and 400 dashed; the cliff (where the share crosses one half) marked on the half line and
        its restitution listed in a box off the lines.

    Args:
        (1) share [Share_Str(object)]: The shares (Plot.Core.Load_Restitution).
        (2) look [Look_Str(object)]: The look.
        (3) datasets [Sequence<string>]: The datasets, one panel each.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, len(datasets), sharey=True, squeeze=False)[0]
    modes = [mode for mode in ('static', 'moving', 'markers') if any(key[0] == mode for key in share.Share) and not (look.Key and mode == 'markers')]
    accelerations = sorted({key[2] for key in share.Share}, reverse=True)
    for k, dataset in enumerate(datasets):
        lines = []
        for mode in modes:
            cliffs = []
            for acceleration in accelerations:
                key = (mode, dataset, acceleration)
                if key not in share.Share:
                    continue
                e, s, low, high = share.Share[key]
                bars = axes[k].errorbar(e, 100 * s, yerr=[100 * (s - low), 100 * (high - s)], color=look.Colour(mode), marker=CONST_MARKERS[mode],
                                        linestyle=CONST_ACCELERATION_STYLES.get(acceleration, '-'))
                bars.lines[0].set_gid('share/%s/%s/%d' % (mode, dataset, acceleration))
                cliff = share.Cliff.get(key, float('nan'))
                if mode != 'markers' and np.isfinite(cliff):
                    mark, = axes[k].plot([cliff], [50.0], marker='x', color=look.Colour(mode),
                                         markersize=2.5 * matplotlib.rcParams['lines.linewidth'] + 3,
                                         linestyle='none')
                    mark.set_gid('cliff/%s/%s/%d' % (mode, dataset, acceleration))
                cliffs.append('%.2f' % cliff if np.isfinite(cliff) else '-')
            if mode != 'markers':
                lines.append('%s %s' % (mode, ' / '.join(cliffs)))
        levels = ', '.join('%s %d' % (mode, len(share.Contacts.get(mode, []))) for mode in modes)
        speeds = ' / '.join('%d' % a for a in accelerations)
        box = axes[k].text(0.0, -0.2, look.Text('cliff (half retained), %s %s:\n%s\nlevels simulated: %s' % (speeds, CONST_DEGREES_S2,
                                                                                                              '\n'.join(lines),
                                                                                                         levels)),
                           transform=axes[k].transAxes, ha='left', va='top', fontsize=0.75 * matplotlib.rcParams['font.size'])
        box.set_gid('cliffs/%s' % dataset)
        axes[k].axhline(100 * 0.5, color=look.Colour('observed'), linewidth=0.4, linestyle=':')
        axes[k].set_xlabel(look.Text('restitution of the ball\'s contact [-]'))
        axes[k].set_title(look.Text(dataset), fontsize=axes[k].xaxis.label.get_size())
        look.Panel(axes[k], 'ab'[k])
    axes[0].set_ylabel(look.Text('retained of the goes [%]'))
    names = {'static': 'static catch', 'moving': 'moving catch', 'markers': 'moving, markers as shapes'}
    handles = [_Handle(look, names[mode], look.Colour(mode), marker=CONST_MARKERS[mode]) for mode in modes]
    handles += [_Handle(look, '%d %s' % (a, CONST_DEGREES_S2), look.Colour('truth'), CONST_ACCELERATION_STYLES.get(a, '-')) for a in accelerations]
    handles.append(Line2D([], [], color=look.Colour('truth'), marker='x', linestyle='none', label=look.Text('cliff: half retained')))
    figure.legend(handles=handles, loc='outside lower center', ncols=len(handles) if look.Kind == 'page' else 3)
    return figure


def Draw_Classes(classes: tp.Dict[tp.Tuple[str, str, str, int], np.ndarray], look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The catch's classes with the nominal contact, as shares of the throws, one panel per partition: per dataset and
        acceleration a bar of the static catch and one of the moving catch (hatched), stacked by class.

    Args:
        (1) classes [Dict<Tuple<string, string, string, int>, Vector<int>>]: The counts (Plot.Core.Load_Classes).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    partitions = list(dict.fromkeys(key[0] for key in classes))
    figure = look.Figure()
    axes = figure.subplots(1, len(partitions), sharey=True, squeeze=False)[0]
    shown = [c for c in CONST_CLASS_COLUMNS if any(counts[CONST_CLASS_COLUMNS.index(c)] for counts in classes.values())]
    for k, partition in enumerate(partitions):
        groups = sorted({(key[2], key[3]) for key in classes if key[0] == partition}, key=lambda g: (g[0], -g[1]))
        ticks = []
        for g, (dataset, acceleration) in enumerate(groups):
            for m, mode in enumerate(('static', 'moving')):
                key = (partition, mode, dataset, acceleration)
                if key not in classes:
                    continue
                share = 100.0 * classes[key] / max(1, int(classes[key].sum()))
                bottom = 0.0
                for c in shown:
                    value = share[CONST_CLASS_COLUMNS.index(c)]
                    bar = axes[k].bar(g + 0.4 * (m - 0.5), value, width=0.38, bottom=bottom, color=look.Colour(c),
                                      hatch='///' if mode == 'moving' else None,
                                      edgecolor=look.Colour('truth'), linewidth=0.3)
                    bar.patches[0].set_gid('classes/%s/%s/%s/%d/%s' % (partition, mode, dataset, acceleration, c))
                    bottom += value
            ticks.append(look.Text('%s %d' % (dataset, acceleration)))
        axes[k].set_xticks(np.arange(len(groups)), ticks, rotation=35, ha='right', rotation_mode='anchor')
        axes[k].tick_params(axis='x', which='both', top=False, bottom=False)
        axes[k].set_title(look.Text(partition), fontsize=axes[k].xaxis.label.get_size())
        look.Panel(axes[k], 'ab'[k])
    axes[0].set_ylabel(look.Text('share of the throws [%]'))
    handles = [matplotlib.patches.Patch(facecolor=look.Colour(c), edgecolor=look.Colour('truth'),
                                        label=look.Text(CONST_CLASS_NAMES[c])) for c in shown]
    handles.append(matplotlib.patches.Patch(facecolor='none', hatch='///', edgecolor=look.Colour('truth'), label=look.Text('hatched: moving')))
    handles.append(matplotlib.patches.Patch(facecolor='none', edgecolor=look.Colour('truth'), label=look.Text('plain: static')))
    figure.legend(handles=handles, loc='outside right center')
    return figure


def Draw_Sweep(sweep: tp.Dict[tp.Tuple[str, str, int], np.ndarray], look: Look_Str, datasets: tp.Sequence[str] = ('Catch_In', 'Catch_Near')
               ) -> matplotlib.figure.Figure:
    """
    Description:
        The robustness sweep of the static catch (F11, as recorded): the retained share of the catchable throws against the
        injected error's scale (the real error model, and the simulated one at scale 1), the run's own error as a
        horizontal line, one panel per dataset; 800 °/s² solid, 400 dashed.

    Args:
        (1) sweep [Dict<Tuple<string, string, int>, Matrix<float> 4xS>]: The sweep (Plot.Core.Load_Sweep).
        (2) look [Look_Str(object)]: The look.
        (3) datasets [Sequence<string>]: The datasets, one panel each.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, len(datasets), sharey=True, squeeze=False)[0]
    colours = {'real': 'static', 'simulated': 'GLS_M4', 'run': 'QT_BASIS'}
    for k, dataset in enumerate(datasets):
        for (source, label, acceleration), values in sorted(sweep.items()):
            if label != dataset:
                continue
            scale, share, low, high = values
            style = CONST_ACCELERATION_STYLES.get(acceleration, '-')
            if source == 'run':
                line = axes[k].axhline(100 * share[0], color=look.Colour(colours[source]), linestyle=style, linewidth=0.8)
                line.set_gid('sweep/run/%s/%d' % (dataset, acceleration))
                continue
            bars = axes[k].errorbar(scale, 100 * share, yerr=[100 * (share - low), 100 * (high - share)], color=look.Colour(colours.get(source,
                                                                                                                                        'static')),
                                    marker='o' if source == 'real' else 's', linestyle=style)
            bars.lines[0].set_gid('sweep/%s/%s/%d' % (source, dataset, acceleration))
        axes[k].set_xlabel(look.Text('injected error, scale of the real error [-]'))
        axes[k].set_title(look.Text(dataset), fontsize=axes[k].xaxis.label.get_size())
        look.Panel(axes[k], 'ab'[k])
    axes[0].set_ylabel(look.Text('retained of the catchable [%]'))
    accelerations = sorted({key[2] for key in sweep}, reverse=True)
    handles = [_Handle(look, 'real error model', look.Colour('static'), marker='o'), _Handle(look, 'simulated error model', look.Colour('GLS_M4'),
                                                                                             marker='s'),
               _Handle(look, 'QT_BASIS\'s own error', look.Colour('QT_BASIS'))]
    handles += [_Handle(look, '%d %s' % (a, CONST_DEGREES_S2), look.Colour('truth'), CONST_ACCELERATION_STYLES.get(a, '-')) for a in accelerations]
    figure.legend(handles=handles, loc='outside lower center', ncols=len(handles))
    return figure


def Draw_Decision(decision: tp.Dict[tp.Tuple[str, str, str, int], np.ndarray], look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The decision's errors: (a) the false-no-go rate and (b) the false-go rate's 95 % upper bound, per dataset and
        acceleration, F9's stage rule (circles) and F10's decision at every sample (squares), validation filled and test
        hollow; the mean over the seeds with their range; above each group in (b) the number of false goes of all its rows.

    Args:
        (1) decision [Dict<Tuple<string, string, string, int>, Matrix<float> 3xS>]: The errors (Plot.Core.Load_Decision).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, 2)
    groups = sorted({(key[2], key[3]) for key in decision}, key=lambda g: (g[0], -g[1]))
    colour = look.Colour('QT_BASIS')
    totals: tp.Dict[tp.Tuple[str, int], tp.List[float]] = {}
    for (stage, partition, dataset, acceleration), values in sorted(decision.items()):
        g = groups.index((dataset, acceleration)) + (-0.15 if stage == 'stages' else 0.15) + (-0.05 if partition == 'validation' else 0.05)
        for k, row in enumerate((0, 2)):
            v = 100.0 * values[row]
            mean = float(np.mean(v))
            axes[k].errorbar([g], [mean], yerr=[[mean - v.min()], [v.max() - mean]], color=colour, linestyle='none', marker='none')
            mark, = axes[k].plot([g], [mean], linestyle='none', marker='o' if stage == 'stages' else 's', color=colour, zorder=3,
                                 markerfacecolor=colour if partition == 'validation' else CONST_BACKGROUND[look.Theme], markeredgecolor=colour)
            mark.set_gid('decision/%s/%s/%s/%s/%d' % (('false_no_go', 'false_go_upper')[k], stage, partition, dataset, acceleration))
        totals.setdefault((dataset, acceleration), [0.0, 0.0])
        totals[(dataset, acceleration)][0] += float(values[1].sum())
        totals[(dataset, acceleration)][1] = max(totals[(dataset, acceleration)][1], float(100.0 * values[2].max()))
    for g, group in enumerate(groups):
        text = axes[1].annotate(look.Text('%d' % totals[group][0]), (g, totals[group][1]), xytext=(0, 4), textcoords='offset points', ha='center',
                                fontsize=0.7 * matplotlib.rcParams['font.size'])
        text.set_gid('false_goes/%s/%d' % group)
    for k, label in enumerate(('false-no-go rate [%]', 'false-go rate, 95 % upper bound [%]')):
        axes[k].set_xticks(np.arange(len(groups)), [look.Text('%s %d' % g) for g in groups], rotation=35, ha='right', rotation_mode='anchor')
        axes[k].tick_params(axis='x', which='both', top=False, bottom=False)
        axes[k].set_ylabel(look.Text(label))
        look.Panel(axes[k], 'ab'[k])
    background = CONST_BACKGROUND[look.Theme]
    _Legend(axes[0], look, [_Handle(look, 'stage rule (F9)', colour, 'none', 'o'), _Handle(look, 'every sample (F10)', colour, 'none', 's'),
                            Line2D([], [], color=colour, marker='o', linestyle='none', markerfacecolor=background, label=look.Text('test (hollow)'))],
            loc='upper left')
    return figure


def Draw_Interception(interception: tp.Dict[tp.Tuple[str, int], tp.Tuple[np.ndarray, np.ndarray]], look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The interception of the goes: (a) the share of the goes within a lateral miss at the cup's mouth (its cumulative
        distribution), (b) the accepted plans per go; the static and the moving catch, 800 °/s² solid and 400 dashed.

    Args:
        (1) interception [Dict<Tuple<string, int>, Tuple<Vector, Vector>>]: The misses and plans (Plot.Core.Load_Interception).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure()
    axes = figure.subplots(1, 2)
    for (mode, acceleration), (lateral, plans) in sorted(interception.items()):
        style = CONST_ACCELERATION_STYLES.get(acceleration, '-')
        line, = axes[0].plot(lateral, 100.0 * np.arange(1, len(lateral) + 1) / max(1, len(lateral)), color=look.Colour(mode), linestyle=style,
                             label=look.Text('%s, %d %s' % (mode, acceleration, CONST_DEGREES_S2)))
        line.set_gid('lateral/%s/%d' % (mode, acceleration))
        counts = np.bincount(plans.astype(np.int64))
        line, = axes[1].step(np.arange(len(counts)), 100.0 * counts / max(1, len(plans)), where='mid', color=look.Colour(mode), linestyle=style)
        line.set_gid('plans/%s/%d' % (mode, acceleration))
    axes[0].set_xlabel(look.Text('lateral miss at the mouth [mm]'))
    axes[0].set_ylabel(look.Text('goes within it [%]'))
    axes[1].set_xlabel(look.Text('accepted plans per go [-]'))
    axes[1].set_ylabel(look.Text('share of the goes [%]'))
    axes[0].legend(loc='lower right')
    for k in range(2):
        look.Panel(axes[k], 'ab'[k])
    return figure


def Draw_Motion(motion: Motion_Str, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        One throw caught in motion against the time from the release, the moving catch's phases shaded and named: (top) the
        cup's speed along the ball's direction beside the ball's speed, the relative speed at the first cup contact marked
        for the moving catch (its catch row) and beside it for the static catch of the same throw (dashed, its cup at rest at
        the catch; from its record), the contact line naming the part of the cup the ball meets; (bottom, smaller) the
        distance from the cup's mouth to the ball, its dip marked as the ball passing the mouth.

    Args:
        (1) motion [Motion_Str(object)]: The motion (Plot.Core.Motion_From_Records).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure('tall')
    top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': (3, 1)})
    start, end = 0.0, motion.Phases[-1][1] + 0.4
    small = 0.8 * matplotlib.rcParams['font.size']
    for k, (name, a, b) in enumerate(motion.Phases):
        for ax in (top, bottom):
            span = ax.axvspan(a, b, color=look.Colour('observed'), alpha=0.08 + 0.10 * (k % 2), linewidth=0)
            span.set_gid('phase/%s' % name)
        top.annotate(look.Text(CONST_PHASE_NAMES.get(name, name)), (a if k == 0 else 0.5 * (a + min(b, end)), 1.0), xycoords=('data',
                                                                                                                              'axes fraction'),
                     xytext=(3 if k == 0 else 0, -4), textcoords='offset points', ha='left' if k == 0 else 'center', va='top', fontsize=small)
    keep = {mode: (t >= start) & (t <= end) for mode, t in motion.Time.items()}
    line, = top.plot(motion.Time['moving'][keep['moving']], motion.Ball_Speed['moving'][keep['moving']], color=look.Colour('truth'),
                     label=look.Text('ball\'s speed'))
    line.set_gid('ball_speed/moving')
    for mode in motion.Time:
        style = '-' if mode == 'moving' else '--'
        line, = top.plot(motion.Time[mode][keep[mode]], motion.Cup_Speed[mode][keep[mode]], color=look.Colour(mode), linestyle=style,
                         label=look.Text('cup along the ball, %s catch' % mode))
        line.set_gid('cup_speed/%s' % mode)
        line, = bottom.plot(motion.Time[mode][keep[mode]], motion.Distance[mode][keep[mode]], color=look.Colour(mode), linestyle=style)
        line.set_gid('distance/%s' % mode)
    for ax in (top, bottom):
        line = ax.axvline(motion.Contact, color=look.Colour('truth'), linewidth=0.6, linestyle='-.')
        line.set_gid('contact')
    top.annotate(look.Text('$\\leftarrow$ ball meets %s' % CONST_PART_NAMES.get(motion.Contact_Part, motion.Contact_Part)), (motion.Contact, 0.84),
                 xycoords=('data', 'axes fraction'), xytext=(3, 0), textcoords='offset points', ha='left', va='center', fontsize=small)
    top.annotate(look.Text('relative speed at contact:'), (motion.Contact, 0.74), xycoords=('data', 'axes fraction'), xytext=(6, 0),
                 textcoords='offset points', ha='left', va='center', fontsize=small)
    for row, (mode, text) in enumerate((('moving', 'moving %.2f m/s' % motion.Relative_Speed), ('static', None))):
        if mode not in motion.At_Contact:
            continue
        t, ball, cup, relative = motion.At_Contact[mode]
        top.annotate('', (t, ball), xytext=(t, cup), arrowprops={'arrowstyle': '<->', 'color': look.Colour(mode), 'linewidth': 0.6,
                                                                 'linestyle': '-' if mode == 'moving' else '--'})
        label = top.annotate(look.Text(text or 'static %.2f m/s' % relative), (motion.Contact, 0.66 - 0.08 * row), xycoords=('data', 'axes fraction'),
                             xytext=(12, 0), textcoords='offset points', va='center', ha='left', color=look.Colour(mode), fontsize=small)
        label.set_gid('relative/%s' % mode)
    bottom.annotate(look.Text('ball passes the mouth'), (motion.Mouth_Pass, float(np.min(motion.Distance['moving']))), xytext=(-30, 2),
                    textcoords='offset points', ha='right', va='bottom', fontsize=small,
                    arrowprops={'arrowstyle': '->', 'color': look.Colour('truth'), 'linewidth': 0.5})
    top.set_ylim(top=1.22 * float(np.max(motion.Ball_Speed['moving'][keep['moving']])))
    top.set_ylabel(look.Words('speed [m/s]', 'speed [m/s]'))
    top.legend(loc='center left')
    _Log_Axis(bottom.yaxis, (1.0,))
    bottom.set_ylabel(look.Text('mouth to ball [m]'))
    bottom.set_xlabel(look.Text('time from the release [s]'))
    bottom.set_xlim(start, end)
    look.Panel(top, 'a')
    look.Panel(bottom, 'b')
    return figure


def Draw_Joints(motion: Motion_Str, look: Look_Str) -> matplotlib.figure.Figure:
    """
    Description:
        The moving catch's six joints against the time (for the documentation): (a) their positions, (b) their speeds; the
        first cup contact marked.

    Args:
        (1) motion [Motion_Str(object)]: The motion (Plot.Core.Motion_From_Records).
        (2) look [Look_Str(object)]: The look.

    Returns:
        (1) parameter [matplotlib.figure.Figure(object)]: The figure.
    """

    figure = look.Figure('tall')
    axes = figure.subplots(2, 1, sharex=True)
    t = motion.Time['moving']
    start, end = motion.Phases[0][1] - 0.1, motion.Phases[-1][1] + 0.4
    keep = (t >= start) & (t <= end)
    for j in range(6):
        for k, values in enumerate((motion.Joints, motion.Joint_Speeds)):
            line, = axes[k].plot(t[keep], np.degrees(values[keep, j]), color=look.Colour('joint_%d' % (j + 1)), linestyle=CONST_JOINT_STYLES[j],
                                 label=look.Text('joint %d' % (j + 1)))
            line.set_gid('%s/%d' % (('position', 'speed')[k], j + 1))
    for k, label in enumerate(('joint position [$^{\\circ}$]', 'joint speed [$^{\\circ}$/s]')):
        axes[k].axvline(motion.Contact, color=look.Colour('truth'), linewidth=0.6, linestyle='-.')
        axes[k].set_ylabel(look.Text(label))
        look.Panel(axes[k], 'ab'[k])
    axes[1].set_xlabel(look.Text('time from the release [s]'))
    axes[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    return figure
