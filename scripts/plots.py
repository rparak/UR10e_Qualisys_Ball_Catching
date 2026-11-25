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
# File Name: ../scripts/plots.py
## =========================================================================== ##

# System (Default)
import sys
# OS (Operating system interfaces)
import os
# The repository root, built from this file's location, so the script runs from any working
# directory.
#   Unit [string]
CONST_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
#   Add access if it is not in the system path.
if os.path.join(CONST_REPO_ROOT, 'src') not in sys.path:
    sys.path.append(os.path.join(CONST_REPO_ROOT, 'src'))
# Typing (Support for type hints)
import typing as tp
# Argparse (Command-line parsing)
import argparse
# Dataclasses (a configuration with another catch mode)
import dataclasses
# Importlib (scripts/catch.py's helpers, loaded from its file)
import importlib.util
# Time (the time a command took)
import time
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Lib.:
#   ../Configuration/Core - the system configuration
from Configuration.Core import Load_Configuration, Configuration_Error_Cls, CONST_LEARNED_MODEL_NAMES, CONST_CATCH_MODES
#   ../Recording/Core, Identification - the robot frame, a flight's name, the recordings' errors
from Recording.Core import Transform_To_Robot, Recording_Error_Cls
from Recording.Identification import Flight_Name
#   ../Dataset/Core, Manifest - the identification flights (only their files are opened), a dataset's errors, the models' hash
from Dataset.Core import Identification_Flights, Read_Dataset, Dataset_Error_Cls
from Dataset.Manifest import Model_Hash, Read_Manifest
#   ../Prediction/Core, ../Learning/Core, Predictor - a predictor of a run, in NumPy
from Prediction.Core import Prediction_Error_Cls
from Learning.Core import Features_Error_Cls
from Learning.Predictor import Run_Predictor
#   ../Evaluation/Report - the stored files' error
from Evaluation.Report import Report_Error_Cls
#   ../Plot/Core, Style, Figures - the figures' data, their look, their drawing
from Plot.Core import (Training_Curves, Load_Comparison, Load_Horizons, Load_Horizon_Ranges, Throw_Errors, Choose_Throw, Load_Throw, Real_Predictions,
                       Load_Restitution, Load_Classes, Load_Sweep, CONST_SPEEDUP_SETS, CONST_MAX_GAP,
                       Load_Decision, Load_Interception, Motion_From_Records, Read_Captions, Write_Captions, Plot_Error_Cls)
from Plot.Style import Style_Context, Save_Figure, Latex_Available, Style_Error_Cls, CONST_THEMES
from Plot import Figures
#   ../Scene/Core, Ball, ../Robot/Core, Motion, ../Catch/Core, ../View/Core - the errors of the scene, the robot, the catch and a record
from Scene.Core import Scene_Error_Cls
from Scene.Ball import Ball_Error_Cls
from Robot.Core import Robot_Error_Cls
from Robot.Motion import Motion_Error_Cls
from Catch.Core import Catch_Error_Cls
from View.Core import View_Error_Cls


def _Load_Catch_Script() -> tp.Any:
    """
    Description:
        The module of scripts/catch.py, loaded from its file, as scripts/Maintenance/derive.py loads it: the figures share
        its helpers (a run's copy, the models from the configuration, the decision's datasets, the interception's setup,
        the stored check of a recorded throw). One direction only: catch.py never loads this script.

    Returns:
        (1) parameter [module]: The module.
    """

    if 'catch_script' in sys.modules:
        return sys.modules['catch_script']
    specification = importlib.util.spec_from_file_location('catch_script', os.path.join(CONST_REPO_ROOT, 'scripts', 'catch.py'))
    module = importlib.util.module_from_spec(specification)
    sys.modules['catch_script'] = module
    specification.loader.exec_module(module)
    return module


"""
Description:
    Initialization of constants.

    The figures of the stored results (the visualisation, delivery 2;
    SPEC §7.14, §9.16): one sub-command per figure group, each reading only the stored files under
    Data/Generated/ that scripts/catch.py writes, the shipped configuration and, for 'real', the 45 identification
    recordings (never a validation recording); 'all' runs them all. A test evaluation's per-flight files are never
    opened, no test partition of a dataset is read, nothing is counted and no stored file is written. Each figure is
    written into the full output (--out, its theme), and the key set besides it: README/ (both themes, for GitHub's
    <picture>) and Video/ (the dark theme at 3840 x 2160 px).
"""
# The user script's module and what is taken from it: one source each.
CONST_CATCH_SCRIPT = _Load_Catch_Script()
CONST_CONFIG_FILE = CONST_CATCH_SCRIPT.CONST_CONFIG_FILE
CONST_RECORDINGS_DIR = CONST_CATCH_SCRIPT.CONST_RECORDINGS_DIR
CONST_SAFETY = CONST_CATCH_SCRIPT.CONST_SAFETY
CONST_RUN_ERROR = CONST_CATCH_SCRIPT.Run_Error_Cls
# The stored results' folder and the figures' folder in it, with the key set's two folders.
#   Unit [string]
CONST_GENERATED_DIR = os.path.join(CONST_REPO_ROOT, 'Data', 'Generated')
CONST_FIGURES = 'Figures'
CONST_README = 'README'
CONST_VIDEO = 'Video'
# The key set: the full output's figure that plays each role in the
# README (both themes) and in the video (the dark theme, CONST_VIDEO_THEME). The README's real flight is the 2D figure's
# y-z view and error (the flight is nearly planar); the 3D figure stays in the full output.
#   Unit [Dict<string, string>], [Dict<string, string>], [string]
CONST_README_ROLES = {'comparison_bars': 'comparison_bars', 'prediction_error': 'prediction_error', 'prediction': 'prediction', 'real_2d': 'real',
                      'catch_restitution': 'catch', 'motion': 'motion'}
CONST_VIDEO_ROLES = {'comparison_bars': 'comparison_bars', 'motion': 'motion'}
CONST_VIDEO_THEME = 'dark'
# The runs of the two predictors the figures compare (QT_BASIS and GLS_M4), by Run.Name.
#   Unit [string]
CONST_LEARNED_RUN = 'QT_BASIS'
CONST_NUMERICAL_RUN = 'GLS_M4_F14'
# The arrival speed of the restitution's drop: the speed F11's addendum measured the levels at (SPEC §4.14).
#   Unit [float, m/s]
CONST_REBOUND_SPEED = 7.0
# The comparisons whose tables the comparison figure reads (F14's two; SPEC §4.9), and per set of the figure the table,
# the partition and the dataset of its rows (the simulated sets on Default, the real flights without a dataset).
#   Unit [string], [Dict<string, Tuple<string, string, string>>]
CONST_VALIDATION_COMPARISON = 'Final_Validation'
CONST_TEST_COMPARISON = 'Final_Test'
CONST_COMPARISON_TABLES = {'validation': (CONST_VALIDATION_COMPARISON, 'validation', 'Default'), 'held_out': (CONST_TEST_COMPARISON, 'test',
                                                                                                              'Default'),
                           'real': (CONST_TEST_COMPARISON, 'real', '')}
# The rows of the final (test) evaluations' summary.csv per set of the prediction error's figure: the partition and the
# dataset (the stored outputs of F14's counted read; never a per-flight file).
#   Unit [Dict<string, Tuple<string, string>>]
CONST_HORIZON_SETS = {'held_out': ('test', 'Default'), 'real': ('real', '')}
# The errors a sub-command reports as [ERROR] lines with exit 1.
#   Unit [Tuple<type>]
CONST_ERRORS = (Plot_Error_Cls, Style_Error_Cls, Configuration_Error_Cls, Recording_Error_Cls, Dataset_Error_Cls, Prediction_Error_Cls,
                Features_Error_Cls, Report_Error_Cls, Scene_Error_Cls, Ball_Error_Cls, Robot_Error_Cls, Motion_Error_Cls, Catch_Error_Cls,
                View_Error_Cls, OSError)
# The sub-commands in the order 'all' runs them.
#   Unit [Tuple<string>]
CONST_COMMANDS = ('training', 'comparison', 'prediction', 'decision', 'interception', 'catch', 'real', 'motion')


class Output_Cls(object):
    """
    Description:
        Where one call writes its figures: the full output in one theme (PDF and PNG), the key set's README variants (both
        themes, PNG) and video variants (the dark theme at the video's size, PNG), and the captions of all of them.

    Initialization of the Class:
        Args:
            (1) folder [string]: The full output's folder.
            (2) theme [string]: Its theme (one of CONST_THEMES).
            (3) usetex [bool]: LaTeX sets the text.

        Example:
            Initialization:
                Cls = Output_Cls('Data/Generated/Figures', 'dark', True)

            Features:
                Cls.Write('comparison', lambda look: Figures.Draw_Comparison(data, look), caption)
                Cls.Close()
    """

    def __init__(self, folder: str, theme: str, usetex: bool) -> None:
        self.Folder, self.Theme, self.Usetex = folder, theme, usetex
        self.Captions: tp.List[tp.Tuple[str, str]] = []

    def Write(self, name: str, draw: tp.Callable[[tp.Any], tp.Any], caption: str, key_caption: tp.Optional[str] = None) -> None:
        """
        Description:
            Draw and save one figure: <folder>/<name>_<theme>.pdf and .png; for a figure of the key set also
            README/readme_<role>_<theme>.png for both themes and Video/video_<role>_dark.png (drawn as the key set's variants,
            Look_Str.Key); each with its caption.

        Args:
            (1) name [string]: The figure's name in the full output.
            (2) draw [Callable[[Look_Str], Figure]]: Draws it in a look.
            (3) caption [string]: Its caption.
            (4) key_caption [string or None]: The key set's variants' caption, when they leave panels out; None: caption.
        """

        variants = [(os.path.join(self.Folder, '%s_%s' % (name, self.Theme)), self.Theme, 'page', ('pdf', 'png'))]
        if name in CONST_README_ROLES:
            variants += [(os.path.join(self.Folder, CONST_README, 'readme_%s_%s' % (CONST_README_ROLES[name], theme)), theme, 'page', ('png',))
                         for theme in CONST_THEMES]
        if name in CONST_VIDEO_ROLES:
            variants.append((os.path.join(self.Folder, CONST_VIDEO, 'video_%s_%s' % (CONST_VIDEO_ROLES[name], CONST_VIDEO_THEME)), CONST_VIDEO_THEME,
                             'video', ('png',)))
        for k, (path, theme, kind, formats) in enumerate(variants):
            with Style_Context(theme, self.Usetex, kind, key=k > 0) as look:
                figure = draw(look)
                for written in Save_Figure(figure, path, formats):
                    relative = os.path.relpath(written, self.Folder)
                    self.Captions.append((relative, caption if k == 0 or key_caption is None else key_caption))
                    print('[INFO] Wrote %s' % written, file=sys.stderr)

    def Close(self) -> None:
        """
        Description:
            captions.csv of the full output's folder: the captions of this call's files, after those of the files an
            earlier call wrote and this one did not.
        """

        path = os.path.join(self.Folder, 'captions.csv')
        written = {name for name, _ in self.Captions}
        earlier = [row for row in (Read_Captions(path) if os.path.isfile(path) else []) if row[0] not in written]
        Write_Captions(path, earlier + self.Captions)


def _Runs(generated: str, args: argparse.Namespace) -> tp.Dict[str, tp.Tuple[str, tp.Any]]:
    """
    Description:
        The two predictors' runs (--learned, --numerical) under <generated>/Models/, each checked as scripts/catch.py
        checks a run's copy: the learned one a learned predictor, the numerical one not.

    Args:
        (1) generated [string]: The stored results' folder.
        (2) args [argparse.Namespace]: --learned, --numerical.

    Returns:
        (1) parameter [Dict<string, Tuple<string, Training_Configuration_Str>>]: Per method (Model.Name) its run folder and
                                                                                configuration, the learned one first.
    """

    out = {}
    for name, learned in ((args.learned, True), (args.numerical, False)):
        folder = os.path.join(generated, 'Models', name)
        training, _, _ = CONST_CATCH_SCRIPT._Load_Run_Copy(folder)
        if (training.Model_Name in CONST_LEARNED_MODEL_NAMES) != learned:
            raise CONST_RUN_ERROR('The run %s is a %s predictor (%s); --%s names the %s one.' % (folder, 'learned' if not learned else 'numerical',
                                                                                             training.Model_Name,
                                                                                             'learned' if learned else 'numerical',
                                                                                             'learned' if learned else 'numerical'))
        out[training.Model_Name] = (folder, training)
    return out


def _Training(args: argparse.Namespace, output: Output_Cls) -> None:
    """
    Description:
        'training': the final learned predictor's learning curves (every training seed's epochs.csv).

    Args:
        (1) args [argparse.Namespace]: The options.
        (2) output [Output_Cls(object)]: The output.
    """

    runs = _Runs(args.generated, args)
    method, (folder, training) = next(iter(runs.items()))
    curves = Training_Curves(folder)
    output.Write('training', lambda look: Figures.Draw_Training(curves, look),
                 '%s (%s): the train and the validation loss per epoch of its %d training seeds, on the simulated train and validation '
                 'partitions. %s' % (method, training.Name, len(curves), CONST_SAFETY))


def _Test_Rows(args: argparse.Namespace, runs: tp.Dict[str, tp.Tuple[str, tp.Any]]) -> str:
    """
    Description:
        The test ledger's rows of the two runs' final (test) evaluations, from their manifests (test_evaluation), for a
        caption that names the stored outputs of the counted read.

    Args:
        (1) args [argparse.Namespace]: --generated.
        (2) runs [Dict<string, Tuple<string, Training_Configuration_Str>>]: The runs (_Runs).

    Returns:
        (1) parameter [string]: The rows, by method.
    """

    rows = []
    for method, (_, training) in runs.items():
        manifest = Read_Manifest(os.path.join(args.generated, 'Evaluations', training.Name, 'test', 'manifest.json'))
        rows.append('%s row %s' % (method, manifest.get('test_evaluation', '?')))
    return ', '.join(rows)


def _Comparison(args: argparse.Namespace, output: Output_Cls) -> None:
    """
    Description:
        'comparison': the two predictors' accuracy and inference time from compare's tables (Final_Validation, Final_Test)
        only: the three-panel comparison (the full output) and the bars (the key set).

    Args:
        (1) args [argparse.Namespace]: The options.
        (2) output [Output_Cls(object)]: The output.
    """

    runs = _Runs(args.generated, args)
    data = Load_Comparison(os.path.join(args.generated, 'Comparisons'), {method: training.Name for method, (_, training) in runs.items()},
                           CONST_COMPARISON_TABLES)
    methods = list(runs)
    sources = ('compare\'s tables %s and %s; the 45 real validation flights only through %s, the stored outputs of F14\'s counted final '
               'evaluations (the test ledger: %s); no recording opened; real flights rest on the assumed transform'
               % (CONST_VALIDATION_COMPARISON, CONST_TEST_COMPARISON, CONST_TEST_COMPARISON, _Test_Rows(args, runs)))
    output.Write('comparison', lambda look: Figures.Draw_Comparison(data, look),
                 '%s against %s: (a) the median remainder RMS of the predicted trajectory after N observed samples on the simulated '
                 'validation and test partitions (Default) and on the 45 real validation flights (the bars: the range over the training '
                 'seeds), (b) the paired difference with its 95 %% bootstrap interval, (c) the inference time per prediction, median and 90th '
                 'percentile (one process, one thread, CPU). From %s. %s' % (methods[0], methods[1], sources, CONST_SAFETY))
    output.Write('comparison_bars', lambda look: Figures.Draw_Comparison_Bars(data, look),
                 '%s against %s at N = %s: (a) the prediction error, the median over the flights of its RMS over the rest of the flight, '
                 'on the simulated test partition (Default, plain) and on the 45 real validation flights (hatched), the error bar the range '
                 'over %s\'s %d training seeds as stored (compare\'s tables hold no 95 %% interval of a median; %s is deterministic and has '
                 'no seed range); (b) the inference time per prediction, the median as the bar and the 90th percentile as the whisker as '
                 'stored (one process, one thread, CPU); %s is about %.0f times faster (the geometric mean over N and both sets of %s\'s '
                 'median over %s\'s). From %s. %s'
                 % (methods[0], methods[1], ', '.join('%d' % n for n in data.N), methods[0], len(runs[methods[0]][1].Seeds), methods[1], methods[0],
                    data.Speedup, methods[1], methods[0], sources, CONST_SAFETY))


def _Prediction(args: argparse.Namespace, output: Output_Cls) -> None:
    """
    Description:
        'prediction': one simulated validation throw with both predictors' stored predictions (--index, or chosen by
        Plot.Core.Choose_Throw's rule), the error at the fixed horizons of both validation evaluations, and the error
        against the prediction horizon at the decision's first N on the simulated test partition and the real validation
        flights from the stored summaries of F14's final evaluations.

    Args:
        (1) args [argparse.Namespace]: The options.
        (2) output [Output_Cls(object)]: The output.
    """

    configuration = Load_Configuration(args.config)
    runs = _Runs(args.generated, args)
    evaluations = {method: os.path.join(args.generated, 'Evaluations', training.Name, 'validation') for method, (_, training) in runs.items()}
    first = next(iter(evaluations.values()))
    manifest = Read_Manifest(os.path.join(first, 'manifest.json'))
    label = os.path.basename(os.path.normpath(manifest['datasets'][0]['path']))
    dataset = os.path.join(args.generated, 'Datasets', label)
    seeds = {method: ('%d' % training.Seeds[0]) if method in CONST_LEARNED_MODEL_NAMES else '' for method, (_, training) in runs.items()}
    comparison = Load_Comparison(os.path.join(args.generated, 'Comparisons'), {method: training.Name for method, (_, training) in runs.items()},
                                 CONST_COMPARISON_TABLES)
    data = None
    if args.index is None:
        data = Read_Dataset(os.path.join(dataset, 'flights.npz'))
        rms, end, gaps = Throw_Errors({method: (evaluations[method], seeds[method]) for method in runs}, label, data)
        medians = {method: dict(zip(comparison.N.tolist(), comparison.Accuracy[(method, 'held_out')][0])) for method in runs}
        index, score = Choose_Throw(rms, end, gaps, medians, list(next(iter(runs.values()))[1].Observed_Samples), label)
        rule = ('chosen by the rule: no gap between its observed samples over %g s, its remainder RMS and its error at the end of the curve '
                'falling from each N to the next for both methods, and of those the nearest the simulated test medians (the smallest sum of '
                '|ln(RMS / median)|, %.2f)' % (CONST_MAX_GAP, score))
        print('[INFO] The throw %s %d, %s.' % (label, index, rule), file=sys.stderr)
    else:
        index, rule = args.index, 'named with --index'
    throw = Load_Throw(dataset, index, {method: (evaluations[method], seeds[method]) for method in runs}, data)
    horizons = Load_Horizons(evaluations, label)
    n = configuration.Decision.Stages[0]
    ranges = Load_Horizon_Ranges({method: os.path.join(args.generated, 'Evaluations', training.Name, 'test') for method, (_,
                                                                                                                          training) in runs.items()},
                                 CONST_HORIZON_SETS, n)
    names = ' and '.join('%s%s' % (method, ' (seed %s)' % seeds[method] if seeds[method] else '') for method in runs)
    learned_name = next(iter(runs))
    first = ranges[('real', learned_name)]
    output.Write('prediction', lambda look: Figures.Draw_Prediction(throw, look),
                 'The prediction converging on the simulated validation throw %d of %s (%s): %s from the first N observed samples against '
                 'the true path, in the throw\'s vertical plane, and their position error against the time, each curve from the first sample '
                 'after its prefix; the stored predictions of the validation evaluations. %s' % (index, label, rule, names, CONST_SAFETY))
    output.Write('prediction_horizon', lambda look: Figures.Draw_Horizon(horizons, look),
                 'The median position error at the fixed horizons on the simulated validation partition of %s, per N; the shaded band is '
                 'the range over the training seeds. %s' % (label, CONST_SAFETY))
    output.Write('prediction_error', lambda look: Figures.Draw_Prediction_Error(ranges, n, look),
                 'The position error against the prediction horizon at N = %d (the decision\'s first sample count, Decision.Stages), %s: '
                 '(a) the simulated test partition of Default (against the exact truth), (b) the 45 real validation flights (against the '
                 'recorded samples, in the robot frame by the assumed transform); each point is one horizon range at its middle (%s), per '
                 'method in its colour the median as a solid line and the 90th percentile as a dotted one (the stored summaries hold no '
                 'quartiles; a range with fewer than 10 flights is left out); the ranges hold %s throws in (a) and %s of the 45 flights in '
                 '(b). From the summary.csv of F14\'s counted final evaluations (the test ledger: %s), never a per-flight file; no recording '
                 'opened. %s' % (n, ' and '.join(runs), ', '.join('[%g, %g) s' % (a, b) for a, b in zip(*first[:2])),
                                 ' / '.join('%d' % c for c in ranges[('held_out', learned_name)][4]),
                                 ' / '.join('%d' % c for c in ranges[('real', learned_name)][4]), _Test_Rows(args, runs), CONST_SAFETY))


def _Stage_Folder(args: argparse.Namespace, kind: str, run: str, partition: str, moving: bool = False) -> str:
    """
    Description:
        A stage's stored results of a run: <generated>/<kind>/<Run.Name>[/moving]/<partition>.

    Args:
        (1) args [argparse.Namespace]: --generated.
        (2) kind [string]: Decisions, Interceptions or Catches.
        (3) run [string]: Run.Name.
        (4) partition [string]: validation, test or sweep.
        (5) moving [bool]: The moving catch's.

    Returns:
        (1) parameter [string]: The folder.
    """

    return os.path.join(args.generated, kind, run, *((CONST_CATCH_MODES[1],) if moving else ()), partition)


def _Decision(args: argparse.Namespace, output: Output_Cls) -> None:
    """
    Description:
        'decision': the false-no-go and false-go rates of F9's stage rule and F10's decision at every sample, on validation
        and test (their stored summaries).

    Args:
        (1) args [argparse.Namespace]: The options.
        (2) output [Output_Cls(object)]: The output.
    """

    run = args.learned
    data = Load_Decision({p: os.path.join(_Stage_Folder(args, 'Decisions', run, p), 'summary.csv') for p in ('validation', 'test')},
                         {p: os.path.join(_Stage_Folder(args, 'Interceptions', run, p), 'summary.csv') for p in ('validation', 'test')})
    output.Write('decision', lambda look: Figures.Draw_Decision(data, look),
                 'The catch decision of %s: the false-no-go rate and the false-go rate\'s 95 %% Clopper-Pearson upper bound (the number of '
                 'false goes written above), F9\'s stage rule and F10\'s decision at every sample, on the validation and the test partitions of '
                 'the catch datasets; the mean over the seeds with their range. %s' % (run, CONST_SAFETY))


def _Interception(args: argparse.Namespace, output: Output_Cls) -> None:
    """
    Description:
        'interception': the lateral miss at the mouth and the accepted plans of the goes of the first decision dataset, the
        static and the moving catch (their stored validation rows).

    Args:
        (1) args [argparse.Namespace]: The options.
        (2) output [Output_Cls(object)]: The output.
    """

    configuration = Load_Configuration(args.config)
    label = os.path.basename(os.path.normpath(configuration.Decision.Datasets[0].Path))
    run = args.learned
    data = Load_Interception({'static': os.path.join(_Stage_Folder(args, 'Interceptions', run, 'validation'), 'per_throw.csv'),
                              'moving': os.path.join(_Stage_Folder(args, 'Interceptions', run, 'validation', moving=True), 'per_throw.csv')}, label)
    output.Write('interception', lambda look: Figures.Draw_Interception(data, look),
                 'The interception of %s\'s goes on the validation partition of %s, every seed: the lateral miss at the cup\'s mouth and the '
                 'accepted plans per go, the static and the moving catch. %s' % (run, label, CONST_SAFETY))


def _Catch(args: argparse.Namespace, output: Output_Cls) -> None:
    """
    Description:
        'catch': the retained share against the restitution (static beside moving, and the markers variant), the classes
        and the robustness sweep (static, as recorded), from the stored catches; each contact's restitution computed in the
        catch's scene (Catch.Physical.Rebound at CONST_REBOUND_SPEED).

    Args:
        (1) args [argparse.Namespace]: The options.
        (2) output [Output_Cls(object)]: The output.
    """

    #   ../Catch/Physical - the catch's scene, its contact and the restitution
    from Catch.Physical import Catch_Scene, Contact_Str, Set_Contact, Mean_Rebound, CONST_REBOUND_PHASES

    configuration = Load_Configuration(args.config)
    contact = configuration.Contact
    scene = Catch_Scene(configuration.Areas, Contact_Str(Step=contact.Step, Friction=list(contact.Friction), Settle_Time=contact.Settle_Time,
                                                          Rest_Speed=contact.Rest_Speed, Rest_Time=contact.Rest_Time,
                                                          Levels=[(contact.Time_Constant, contact.Damping_Ratio)]),
                        Flight_Model_Str=CONST_CATCH_SCRIPT._Flight_Model(configuration))

    computed: tp.Dict[tp.Tuple[float, float], float] = {}

    def Restitution(time_constant: float, damping_ratio: float) -> float:
        """
        Description:
            The restitution of a contact in the catch's scene.

        Args:
            (1) time_constant [float]: The contact's time constant [s].
            (2) damping_ratio [float]: Its damping ratio [-].

        Returns:
            (1) parameter [float]: The mean rebound ratio at CONST_REBOUND_SPEED over CONST_REBOUND_PHASES phases [-].
        """

        Set_Contact(scene, time_constant, damping_ratio)
        computed[(time_constant, damping_ratio)] = Mean_Rebound(scene, CONST_REBOUND_SPEED)
        return computed[(time_constant, damping_ratio)]

    run = args.learned
    share = Load_Restitution({'static': os.path.join(_Stage_Folder(args, 'Catches', run, 'validation'), 'summary.csv'),
                              'moving': os.path.join(_Stage_Folder(args, 'Catches', run, 'validation', moving=True), 'summary.csv')}, Restitution)
    cliffs = ', '.join('%s %d °/s² %.2f' % (mode, acceleration, value) for (mode, dataset, acceleration), value in sorted(share.Cliff.items())
                       if dataset == 'Catch_In' and mode != 'markers' and np.isfinite(value))
    levels = ', '.join('%.3f' % value for value in sorted(computed.values()))
    static = share.Contacts.get('static', [])
    added = [c for c in share.Contacts.get('moving', []) if (c[0], c[1]) not in {(a, b) for a, b, _ in static}]
    common = ('the restitution of each level is the mean of the ball\'s rebound on the floor at %g m/s over %d phases of its first contact '
              'within a step, computed in the catch\'s scene (SPEC §4.14\'s measure): %s. The static catch was run at %d levels only, %s '
              '(its stored validation catch, F11\'s Contact.Levels and the nominal contact, SPEC §4.15); %s were added with F15 and run for '
              'the moving catch only (SPEC §4.16), so the static catch\'s missing points there are not missing data; the '
              'validation partitions, the first seed, 95 %% intervals; the cliff where the share crosses one half (Catch_In: %s). The '
              'restitution of the real ball is assumed (0.20)'
              % (CONST_REBOUND_SPEED, CONST_REBOUND_PHASES, levels, len(static), ', '.join('%.3f (%g s, %g)' % (c[2], c[0], c[1]) for c in static),
                 ' and '.join('%.3f (%g s, %g)' % (c[2], c[0], c[1]) for c in added) or 'no further levels', cliffs or 'none'))
    output.Write('catch_restitution', lambda look: Figures.Draw_Restitution(share, look),
                 'The retained share of the goes of %s against the restitution of the ball\'s contact, the static and the moving catch; in the '
                 'full output also the moving catch with the ball\'s markers as rigid collision shapes (a sensitivity variant: the real ball\'s '
                 'markers held as rigid bodies, an assumption; the README\'s variant leaves it out). %s. %s' % (run, common, CONST_SAFETY))
    classes = Load_Classes({(p, mode): os.path.join(_Stage_Folder(args, 'Catches', run, p, moving=mode == 'moving'), 'summary.csv')
                            for p in ('validation', 'test') for mode in ('static', 'moving')}, (contact.Time_Constant, contact.Damping_Ratio))
    output.Write('catch_classes', lambda look: Figures.Draw_Classes(classes, look),
                 'The classes of the catch of %s with the nominal contact, summed over the seeds, as shares of the throws of each dataset and '
                 'acceleration: the static catch and the moving catch (hatched), on the validation and the test partitions. %s' % (run, CONST_SAFETY))
    sweep = Load_Sweep(os.path.join(_Stage_Folder(args, 'Catches', run, 'sweep'), 'summary.csv'))
    output.Write('catch_sweep', lambda look: Figures.Draw_Sweep(sweep, look),
                 'The robustness sweep of the static catch (F11, as recorded): the retained share of the catchable throws against the scale of '
                 'an injected prediction error, the first seed, the validation partitions; the horizontal lines are %s\'s own error. %s'
                 % (run, CONST_SAFETY))


def _Real(args: argparse.Namespace, output: Output_Cls) -> None:
    """
    Description:
        'real': both predictors on one real identification flight (--flight NAME, or --pick-seed S, drawn and printed when
        neither is given), in the robot frame by the assumed transform; only the identification recordings are opened.

    Args:
        (1) args [argparse.Namespace]: The options.
        (2) output [Output_Cls(object)]: The output.
    """

    configuration = Load_Configuration(args.config)
    runs = _Runs(args.generated, args)
    flight_model, observation_model = CONST_CATCH_SCRIPT._Flight_Model(configuration), CONST_CATCH_SCRIPT._Observation_Model(configuration)
    model_sha256 = Model_Hash(Flight_Model_Str=flight_model, Observation_Model_Str=observation_model)
    predictors = {}
    for method, (folder, training) in runs.items():
        stored = Read_Manifest(os.path.join(folder, 'manifest.json')).get('model_sha256')
        if stored != model_sha256:
            raise CONST_RUN_ERROR('The run %s was made with the models %s, not this configuration\'s %s; one flight model drives the predictor '
                                '(SPEC §2.5).' % (folder, stored, model_sha256))
        predictors[method] = Run_Predictor(method, folder, training.Seeds[0] if method in CONST_LEARNED_MODEL_NAMES else None,
                                           Flight_Model_Str=flight_model, Observation_Model_Str=observation_model)
    flights = Identification_Flights(args.recordings)
    names = [Flight_Name(flight) for flight in flights]
    if args.flight is not None:
        if args.flight not in names:
            raise Plot_Error_Cls('%s is not one of the 45 identification flights (a validation recording is never shown); they are named as '
                                 '%s.' % (args.flight, names[0]))
        chosen = names.index(args.flight)
    else:
        pick_seed = int(np.random.SeedSequence().generate_state(1)[0] % 1000000) if args.pick_seed is None else args.pick_seed
        chosen = int(np.random.default_rng(pick_seed).integers(len(flights)))
        print('[INFO] The identification flight %s, picked with --pick-seed %d.' % (names[chosen], pick_seed), file=sys.stderr)
    flight = flights[chosen]
    real_name = names[chosen]
    learned = next(iter(runs.values()))[1]
    shown = configuration.Decision.Stages[0] if args.n is None else args.n
    real = Real_Predictions(names[chosen], float(flight.Rate), flight.Time, Transform_To_Robot(flight.Position,
                                                                                               Transform_Str=configuration.Transform),
                            predictors, list(learned.Observed_Samples), shown)
    note = ('An identification flight, the recording %s (its first 40 ms trimmed), opened with the other 44 identification recordings and '
            'no validation recording: these 45 flights shaped the flight model, the observation model and the generator, so this is a '
            'demonstration, not an independent evaluation (the 45 validation flights are in the comparison). In the robot frame by the assumed '
            'transform. %s' % (os.path.join('Data', 'Recordings', real_name), CONST_SAFETY))
    output.Write('real_3d', lambda look: Figures.Draw_Real_3D(real, look),
                 'The real identification flight %s: the recorded samples, the first N = %d of them and the paths %s predict from them; the '
                 'box true to scale. %s' % (real.Name, shown, ' and '.join(runs), note))
    output.Write('real_2d', lambda look: Figures.Draw_Real_2D(real, look),
                 'The real identification flight %s: x-z and y-z with the paths from the first N = %d samples, each axis against the time, '
                 'and the position error against the horizon for N = %s. %s'
                 % (real.Name, shown, ', '.join('%d' % n for n in learned.Observed_Samples), note),
                 key_caption='The real identification flight %s: (a) the side view y-z with the paths from the first N = %d samples, (b) '
                             'the position error against the horizon for N = %s (the flight is nearly planar: its recorded x spans %.0f mm). '
                             '%s' % (real.Name, shown, ', '.join('%d' % n for n in learned.Observed_Samples),
                                     1e3 * float(np.ptp(real.Position[:, 0])), note))


def _Motion(args: argparse.Namespace, output: Output_Cls) -> None:
    """
    Description:
        'motion': one stored validation throw of the first decision dataset (--index, by default the first go with a stroke
        at the learned run's first seed and the first acceleration) recorded by the catch's own code (View.Core.Record_Throw)
        in the moving and in the static mode, each with the margins of its own stored validation interception, and each
        compared with its stored rows as 'view' compares them; then the motion against the time and the joints.

    Args:
        (1) args [argparse.Namespace]: The options.
        (2) output [Output_Cls(object)]: The output.
    """

    #   ../Scene/Core, ../Catch/Core, Decision, Interception, Physical, ../View/Core - the scene, the context, the margins, the record
    from Scene.Core import Build_Scene, Scene_Robot
    from Catch.Core import Catch_Context
    from Catch.Decision import Read_Margins
    from Catch.Interception import Read_Per_Throw
    from Catch.Physical import Catch_Scene, Contact_Str
    from View.Core import Record_Throw, Partition_Code

    catch_script = CONST_CATCH_SCRIPT
    configuration = Load_Configuration(args.config)
    modes = {mode: dataclasses.replace(configuration, Interception=dataclasses.replace(configuration.Interception, Catch_Mode=mode))
             for mode in (CONST_CATCH_MODES[1], CONST_CATCH_MODES[0])}
    run = os.path.join(args.generated, 'Models', args.learned)
    training, _, config_bytes = catch_script._Load_Run_Copy(run)
    seed = ('%d' % training.Seeds[0]) if training.Model_Name in CONST_LEARNED_MODEL_NAMES else ''
    folder = os.path.join(args.generated, 'Datasets', os.path.basename(os.path.normpath(configuration.Decision.Datasets[0].Path)))
    label = os.path.basename(folder)
    acceleration = float(np.degrees(configuration.Robot.Joint_Accelerations[0]))
    interceptions = os.path.join(args.generated, 'Interceptions')
    stored = {mode: os.path.join(catch_script._Run_Folder(c, interceptions, training.Name), 'validation') for mode, c in modes.items()}
    index = args.index
    if index is None:
        for row in Read_Per_Throw(os.path.join(stored[CONST_CATCH_MODES[1]], 'per_throw.csv')):
            if row['dataset'] == label and row['seed'] == seed and float(row['acceleration_deg_s2']) == acceleration and row['decision'] == 'go' \
                    and row['plan_mode'] == CONST_CATCH_MODES[1]:
                index = int(row['flight'])
                break
        if index is None:
            raise Plot_Error_Cls('%s holds no go with a stroke of %s; the motion figure shows one.' % (stored[CONST_CATCH_MODES[1]], label))
    sizes = [int(value) for value in Read_Manifest(os.path.join(folder, 'manifest.json'))['partition_sizes']]
    if Partition_Code(index, sizes) != 1:
        raise Plot_Error_Cls('The throw %d of %s is not in its validation partition; the motion figure shows a stored validation throw (a test '
                             'throw is never read).' % (index, label))
    datasets = catch_script._Decision_Datasets(configuration)
    model = Build_Scene(Areas_Str=configuration.Areas)
    context = Catch_Context(model, Scene_Robot(model), Criteria_Str=catch_script._Catch_Criteria(configuration))
    setup, _, _ = catch_script._Interception_Setup(configuration, training, run, context)
    contact = configuration.Contact
    settings = Contact_Str(Step=contact.Step, Friction=list(contact.Friction), Settle_Time=contact.Settle_Time, Rest_Speed=contact.Rest_Speed,
                           Rest_Time=contact.Rest_Time, Levels=[(contact.Time_Constant, contact.Damping_Ratio)])
    scene = Catch_Scene(configuration.Areas, settings, Flight_Model_Str=setup.Flight)
    records = {}
    for mode, c in modes.items():
        manifest = Read_Manifest(os.path.join(stored[mode], 'manifest.json'))
        differing = [key for key, value in catch_script._Planner_Fields(c, config_bytes, datasets).items() if manifest.get(key) != value]
        if differing:
            raise CONST_RUN_ERROR('The stored interception %s differs from this call in %s; run scripts/catch.py catch again with this '
                                'configuration and run.' % (stored[mode], ', '.join(differing)))
        mode_setup = dataclasses.replace(setup, Catch_Mode=mode, Margins=Read_Margins(os.path.join(stored[mode], 'margins.csv')))
        print('[INFO] Recording the throw %s %d in the %s catch.' % (label, index, mode), file=sys.stderr)
        records[mode] = Record_Throw(mode_setup, label, os.path.join(folder, 'flights.npz'), index, seed, 0, settings, every=10 ** 6, scene=scene)
        catches = os.path.join(catch_script._Run_Folder(c, os.path.join(args.generated, 'Catches'), training.Name), 'validation')
        equal, found = catch_script._Stored_Check(records[mode], stored[mode], catches)
        if equal is False:
            raise Plot_Error_Cls('The %s catch of the throw %d: %s; the stored results and this code no longer agree.' % (mode, index, found))
        print('[INFO] Stored check, %s catch: %s.' % (mode, found if equal else 'not made, %s' % found), file=sys.stderr)
    motion = Motion_From_Records(records, scene.Model, scene.Data, scene.Robot, moving=CONST_CATCH_MODES[1])
    what = 'the validation throw %d of %s with %s%s at %g °/s²' % (index, label, training.Model_Name, ' (seed %s)' % seed if seed else '',
                                                                  acceleration)
    print('At the first cup contact (from the records; the moving catch row\'s contact_speed_m_s %.3f m/s):' % motion.Relative_Speed)
    for mode, (t, ball, cup, relative) in motion.At_Contact.items():
        print('  %-7s t %.4f s, ball %.3f m/s, cup along the ball %.3f m/s, relative %.3f m/s' % (mode, t, ball, cup, relative))
    output.Write('motion', lambda look: Figures.Draw_Motion(motion, look),
                 'The moving catch of %s (%s): the cup\'s speed along the ball\'s direction beside the ball\'s speed from the release; the '
                 'relative speed at the first cup contact, the moving catch\'s from its catch row and the static catch\'s (the same throw, '
                 'dashed, its cup at rest at the catch) from its record; the contact line where the ball meets the cup; below, the distance '
                 'from the cup\'s mouth to the ball, its dip the ball passing the mouth. The shaded phases are the moving catch\'s. Recorded '
                 'by the catch\'s own code, both equal to the stored rows. The ball\'s contact and the compute budget are assumed. %s'
                 % (what, motion.Class, CONST_SAFETY))
    output.Write('motion_joints', lambda look: Figures.Draw_Joints(motion, look),
                 'The joints of the moving catch of %s: their positions and speeds against the time, the first cup contact marked. %s'
                 % (what, CONST_SAFETY))


CONST_HANDLERS = {'training': _Training, 'comparison': _Comparison, 'prediction': _Prediction, 'decision': _Decision,
                  'interception': _Interception, 'catch': _Catch, 'real': _Real, 'motion': _Motion}


def _Parser() -> argparse.ArgumentParser:
    """
    Description:
        The command line: one sub-command per figure group and 'all', each with the common options; an abbreviated
        option is a usage error.

    Returns:
        (1) parameter [argparse.ArgumentParser(object)]: The parser.
    """

    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument('--generated', default=CONST_GENERATED_DIR, help='the stored results\' folder (Data/Generated)')
    common.add_argument('--out', default=None, help='the figures\' folder (<generated>/Figures); README/ and Video/ are written in it')
    common.add_argument('--theme', default=CONST_THEMES[0], choices=CONST_THEMES, help='the full output\'s theme (dark)')
    common.add_argument('--no-latex', action='store_true', help='matplotlib\'s mathtext even when LaTeX is installed')
    common.add_argument('--learned', default=CONST_LEARNED_RUN, help='the learned predictor\'s run name (QT_BASIS)')
    common.add_argument('--numerical', default=CONST_NUMERICAL_RUN, help='the numerical predictor\'s run name (GLS_M4_F14)')
    common.add_argument('--config', default=CONST_CONFIG_FILE, help='the system configuration (Config/ball_catching.yaml)')
    common.add_argument('--recordings', default=CONST_RECORDINGS_DIR,
                        help='the recordings (Data/Recordings); only the identification files are opened')
    common.add_argument('--index', type=int, default=None, help='prediction, motion: the stored validation throw')
    common.add_argument('--flight', default=None, help='real: the identification flight, by its name')
    common.add_argument('--pick-seed', type=int, default=None, help='real: the seed of the flight\'s draw (printed when not given)')
    common.add_argument('--n', type=int, default=None, help='real: the samples the paths are predicted from (the first of Decision.Stages)')
    parser = argparse.ArgumentParser(prog='plots.py', description='The figures of the stored results (SPEC §9.16). %s' % CONST_SAFETY,
                                     allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in CONST_COMMANDS + ('all',):
        commands.add_parser(name, parents=[common], allow_abbrev=False)
    return parser


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    """
    Description:
        Run one sub-command, or all of them in CONST_COMMANDS' order ('all' runs every one and reports each failure), with
        the LaTeX mode printed; captions.csv is written after the figures.

    Args:
        (1) argv [Sequence<string> or None]: The arguments (sys.argv[1:] when None).

    Returns:
        (1) parameter [int]: 0 written; 1 a figure could not be drawn; 2 a usage error.
    """

    args = _Parser().parse_args(argv)
    args.generated = os.path.abspath(args.generated)
    usetex = Latex_Available() and not args.no_latex
    print('[INFO] LaTeX mode: %s.' % ('usetex (latex and dvipng found)' if usetex else 'no-latex (matplotlib\'s mathtext; %s)'
                                       % ('--no-latex given' if args.no_latex else 'latex or dvipng not found')), file=sys.stderr)
    output = Output_Cls(os.path.join(args.generated, CONST_FIGURES) if args.out is None else os.path.abspath(args.out), args.theme, usetex)
    started, failed = time.perf_counter(), []
    for name in (CONST_COMMANDS if args.command == 'all' else (args.command,)):
        try:
            CONST_HANDLERS[name](args, output)
        except CONST_ERRORS + (CONST_RUN_ERROR,) as error:
            print('[ERROR] %s: %s' % (name, error), file=sys.stderr)
            failed.append(name)
    if output.Captions:
        output.Close()
    for path, _ in output.Captions:
        print(os.path.join(output.Folder, path))
    print('[INFO] %d files written into %s in %.0f s%s. %s' % (len(output.Captions), output.Folder, time.perf_counter() - started,
                                                              '; failed: %s' % ', '.join(failed) if failed else '', CONST_SAFETY), file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
