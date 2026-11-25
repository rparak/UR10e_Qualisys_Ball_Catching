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
# File Name: ../test/scripts/test_plots.py
## =========================================================================== ##

# System (Default)
import os
import sys
# Shutil (the stored tree assembled from the fixtures' folders)
import shutil
# JSON (JSON encoder and decoder)
import json
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Dataset/Core - the small dataset's flights
from Dataset.Core import Read_Dataset, Partition_Indices, CONST_PARTITION_VALIDATION
#   ../Recording/Core - the split of the recordings by their names
from Recording.Core import Assign_Sets, CONST_SET_IDENTIFICATION
#   ../Evaluation/Report - a stored evaluation's predictions
from Evaluation.Report import Prediction_Record_Str, Write_Predictions
#   ../Plot/Core - the captions read back
from Plot.Core import Read_Captions
#   ./Script_Helpers - the scripts' shared test helpers
from Script_Helpers import _Load_Script, _Trained_Run, _Plot_Results, _Throw_Errors, _Stored_Folders

"""
Description:
    Initialization of constants.

    scripts/plots.py, the figures of the stored results (the visualisation, delivery 2),
    run in process on a small stored tree (the 30-flight dataset and the two quickly trained
    runs of test/scripts/conftest.py with the constructed results of Script_Helpers._Plot_Results) while an audit hook
    records every file opened: each figure and the key set are written, no test evaluation's per-flight file, no test
    partition's dataset and no validation recording is opened. Expectations come from the owner's decisions (the key
    set's names and folders) and the fixture.
"""
# The files each sub-command writes in the dark theme: the full output and the key set.
#   Unit [Dict<string, Set<string>>]
CONST_WRITTEN = {'training': {'training_dark.pdf', 'training_dark.png'},
                 'comparison': {'comparison_dark.pdf', 'comparison_dark.png', 'comparison_bars_dark.pdf', 'comparison_bars_dark.png',
                                'README/readme_comparison_bars_dark.png', 'README/readme_comparison_bars_light.png',
                                'Video/video_comparison_bars_dark.png'},
                 'prediction': {'prediction_dark.pdf', 'prediction_dark.png', 'prediction_horizon_dark.pdf', 'prediction_horizon_dark.png',
                                'prediction_error_dark.pdf', 'prediction_error_dark.png', 'README/readme_prediction_dark.png',
                                'README/readme_prediction_light.png', 'README/readme_prediction_error_dark.png',
                                'README/readme_prediction_error_light.png'},
                 'decision': {'decision_dark.pdf', 'decision_dark.png'}, 'interception': {'interception_dark.pdf', 'interception_dark.png'},
                 'catch': {'catch_restitution_dark.pdf', 'catch_restitution_dark.png', 'catch_classes_dark.pdf', 'catch_classes_dark.png',
                           'catch_sweep_dark.pdf', 'catch_sweep_dark.png', 'README/readme_catch_dark.png', 'README/readme_catch_light.png'},
                 'real': {'real_3d_dark.pdf', 'real_3d_dark.png', 'real_2d_dark.pdf', 'real_2d_dark.png', 'README/readme_real_dark.png',
                          'README/readme_real_light.png'}}
# The files of a stored test evaluation that no figure opens.
#   Unit [Tuple<string>]
CONST_FORBIDDEN = ('test/per_flight.csv', 'test/per_horizon.csv', 'test/predictions.npz')
# The files opened while an audit is on (sys.addaudithook: a hook cannot be removed, so it records only when asked).
#   Unit [Dict<string, Any>]
_AUDIT = {'on': False, 'opened': []}


def _Hook(event, args):
    if _AUDIT['on'] and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        _AUDIT['opened'].append(os.path.abspath(os.fsdecode(args[0])))


sys.addaudithook(_Hook)


@pytest.fixture(scope='module')
def plots(root):
    """
    Description:
        scripts/plots.py loaded as a module (it loads scripts/catch.py's helpers).
    """

    return _Load_Script(root, os.path.join('scripts', 'plots.py'))


@pytest.fixture(scope='module')
def stored(root, tmp_path_factory, small_dataset, learned_run):
    """
    Description:
        A stored tree as Data/Generated/ holds it: Models/Learned (the module's learned run) and Models/Run (a GLS_M4 run),
        Datasets/Small, the two validation evaluations with the stored predictions of the first validation throw (from its
        observations, the offset falling from 10 mm at N = 10 to 5 mm at N = 20) and every validation throw's remainder RMS
        falling with N (only the first has stored predictions, so the prediction figure's rule chooses it), and the
        constructed results: (folder, the throw's index).
    """

    tmp = tmp_path_factory.mktemp('plots')
    folder = tmp / 'Generated'
    shutil.copytree(learned_run[0], folder / 'Models' / 'Learned')
    shutil.copytree(_Trained_Run(root, tmp, [small_dataset], samples='[10, 20]'), folder / 'Models' / 'Run')
    shutil.copytree(small_dataset[0], folder / 'Datasets' / 'Small')
    _Plot_Results(str(folder), dataset_label='Small')
    data = Read_Dataset(str(folder / 'Datasets' / 'Small' / 'flights.npz'))
    index = int(Partition_Indices(data, CONST_PARTITION_VALIDATION)[0])
    t, x = data.Observations(index)
    for run, seed in (('Learned', '0'), ('Run', '')):
        records = [Prediction_Record_Str(Flight='%d' % index, Partition='validation', Dataset='Small', Seed=seed, N=n, Grid_Start=0.0, T_0=t[n - 1],
                                         T_End=t[-1], X_First=np.full(6, np.nan), Query_Time=t[n:],
                                         Query_State=np.column_stack([x[n:] + 0.1 / n, np.zeros((len(t) - n, 3))])) for n in (10, 20)]
        Write_Predictions(str(folder / 'Evaluations' / run / 'validation' / 'predictions.npz'), records)
        errors = {int(i): {n: 100.0 - n for n in (10, 20)} for i in Partition_Indices(data, CONST_PARTITION_VALIDATION)}
        _Throw_Errors(str(folder / 'Evaluations' / run / 'validation'), 'Small', errors, seed)
        manifest = {'datasets': [{'path': 'Data/Generated/Datasets/Small'}]}
        (folder / 'Evaluations' / run / 'validation' / 'manifest.json').write_text(json.dumps(manifest),
                                                                                    encoding='utf-8')
    return folder, index


def _Plots(plots, stored, out, *args):
    """
    Description:
        plots.py's main() with the stored tree, the numerical and learned runs of the fixture and no LaTeX, under the
        audit: (its exit code, the files opened).
    """

    _AUDIT['opened'], _AUDIT['on'] = [], True
    try:
        code = plots.main(list(args) + ['--generated', str(stored[0]), '--out', str(out), '--learned', 'Learned', '--numerical', 'Run', '--no-latex'])
    finally:
        _AUDIT['on'] = False
    return code, list(_AUDIT['opened'])


def _Identification(root):
    """
    Description:
        The identification recordings' paths, from the split of their names (no file opened).
    """

    folder = os.path.join(root, 'Data', 'Recordings')
    names = [(sub, name) for sub in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, sub))
             for name in sorted(os.listdir(os.path.join(folder, sub))) if name.endswith('.txt')]
    return {os.path.abspath(os.path.join(folder, *key)) for key, value in Assign_Sets(names).items() if value == CONST_SET_IDENTIFICATION}


@pytest.mark.parametrize('command', sorted(CONST_WRITTEN))
def test_each_sub_command_writes_its_figures_and_the_key_set_and_opens_no_test_evaluation_or_validation_recording(plots, stored, root, tmp_path,
                                                                                                                    command, capsys, monkeypatch):
    """
    Description:
        Each sub-command (but motion, whose scene and area's map are the slow test's) exits 0 and writes exactly its
        figures: the full output in the dark theme (PDF and PNG), the README's both themes and the video's dark one for a
        figure of the key set, and captions.csv with one row per file; it prints the LaTeX mode. No stored test
        evaluation's per-flight file is opened, no recording but an identification one, and a dataset's flights.npz only
        for the prediction figure's validation throw (chosen by its rule; the 30-flight dataset's validation throws all have a
        gap over the rule's 0.1 s, so the limit is raised to 1 s here, its own test is test/Plot/test_core.py's).
    """

    monkeypatch.setattr('Plot.Core.CONST_MAX_GAP', 1.0)
    code, opened = _Plots(plots, stored, tmp_path, command, '--pick-seed', '0')
    err = capsys.readouterr().err
    assert code == 0, err
    assert '[INFO] LaTeX mode: no-latex' in err
    written = {os.path.relpath(os.path.join(directory, name), tmp_path) for directory, _, names in os.walk(tmp_path) for name in names}
    assert written == CONST_WRITTEN[command] | {'captions.csv'}
    assert sorted(name for name, _ in Read_Captions(str(tmp_path / 'captions.csv'))) == sorted(CONST_WRITTEN[command])
    assert all(caption.endswith(plots.CONST_SAFETY) for _, caption in Read_Captions(str(tmp_path / 'captions.csv')))
    captions = dict(Read_Captions(str(tmp_path / 'captions.csv')))
    if command == 'catch':
        assert 'The static catch was run at 3 levels only' in captions['catch_restitution_dark.png']
    if command == 'comparison':
        assert 'GLS_M4 is deterministic and has no seed range' in captions['comparison_bars_dark.png']
    if command == 'prediction':
        assert 'the ranges hold' in captions['prediction_error_dark.png'] and '[0, 0.1) s' in captions['prediction_error_dark.png']
    assert not [path for path in opened if path.replace(os.sep, '/').endswith(CONST_FORBIDDEN)]
    recordings = [path for path in opened if os.path.join('Data', 'Recordings') in path and path.endswith('.txt')]
    assert set(recordings) <= _Identification(root) and (len(recordings) == 45) == (command == 'real')
    assert [path for path in opened if path.endswith('flights.npz')] == ([str(stored[0] / 'Datasets' / 'Small' / 'flights.npz')]
                                                                         if command == 'prediction' else [])


def test_the_real_figure_states_that_the_identification_flights_shaped_the_models_and_refuses_a_validation_recording(plots, stored, root, tmp_path,
                                                                                                                       capsys):
    """
    Description:
        The real figures' captions say the flight is an identification flight that shaped the models, a demonstration and
        not an independent evaluation, in the robot frame by the assumed transform; a validation recording's name is
        refused before any file is opened, and a later call keeps the earlier call's captions.
    """

    assert _Plots(plots, stored, tmp_path, 'real', '--pick-seed', '0')[0] == 0
    captions = dict(Read_Captions(str(tmp_path / 'captions.csv')))
    assert all('demonstration, not an independent evaluation' in captions[name] and 'assumed transform' in captions[name]
               for name in CONST_WRITTEN['real'])
    assert all('(a) the side view y-z' in captions[name] and 'x-z' not in captions[name] for name in CONST_WRITTEN['real'] if 'readme_' in name)
    assert all('x-z and y-z' in captions['real_2d_dark.%s' % extension] for extension in ('pdf', 'png'))
    validation = sorted(set(os.path.join(sub, name) for sub in os.listdir(os.path.join(root, 'Data', 'Recordings'))
                            if os.path.isdir(os.path.join(root, 'Data', 'Recordings', sub))
                            for name in os.listdir(os.path.join(root, 'Data', 'Recordings', sub)) if name.endswith('.txt'))
                        - set(os.path.relpath(path, os.path.join(root, 'Data', 'Recordings')) for path in _Identification(root)))[0]
    code, opened = _Plots(plots, stored, tmp_path, 'real', '--flight', validation)
    assert code == 1 and 'is not one of the 45 identification flights' in capsys.readouterr().err
    assert not set(opened) - _Identification(root) & {p for p in opened if p.endswith('.txt') and 'Recordings' in p}
    assert _Plots(plots, stored, tmp_path, 'decision')[0] == 0
    assert set(dict(Read_Captions(str(tmp_path / 'captions.csv')))) == CONST_WRITTEN['real'] | CONST_WRITTEN['decision']


@pytest.mark.parametrize('position', ['train', 'test'])
def test_a_prediction_throw_outside_the_validation_partition_is_refused_before_the_datasets_file_is_opened(plots, stored, tmp_path, position, capsys):
    """
    Description:
        --index of a train or a test throw (the 30-flight dataset: 21 / 5 / 4) is refused with exit 1, naming the validation
        flights; the dataset's flights.npz is never opened.
    """

    code, opened = _Plots(plots, stored, tmp_path, 'prediction', '--index', '0' if position == 'train' else '27')
    assert code == 1 and 'is not in its validation partition (the flights 21 to 25)' in capsys.readouterr().err
    assert not [path for path in opened if path.endswith('flights.npz')]


def test_an_abbreviated_option_is_a_usage_error(plots, stored, tmp_path):
    """
    Description:
        Every parser refuses an abbreviated option (as scripts/catch.py's): exit 2.
    """

    with pytest.raises(SystemExit) as error:
        plots.main(['comparison', '--generated', str(stored[0]), '--no-lat'])
    assert error.value.code == 2


@pytest.mark.slow
def test_all_on_the_stored_results_writes_the_full_output_and_the_key_set_with_the_motion_equal_to_the_stored_rows(root, tmp_path):
    """
    Description:
        plots.py all on Data/Generated/ (the final runs QT_BASIS and GLS_M4_F14; skipped in a clean clone), in a fresh
        interpreter without LaTeX: exit 0; every figure in the dark theme, the README's five in both themes and the video's
        two; the motion's two records equal the stored interception and catch rows byte for byte (printed).
    """

    _Stored_Folders(root, 'Catches')
    import subprocess
    result = subprocess.run([sys.executable, os.path.join(root, 'scripts', 'plots.py'), 'all', '--no-latex', '--pick-seed', '0', '--out',
                             str(tmp_path)],
                            cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stderr.count('equal the stored validation rows byte for byte') == 2
    readme = {'README/readme_%s_%s.png' % (role, theme) for role in ('comparison_bars', 'prediction_error', 'prediction', 'real', 'catch', 'motion')
              for theme in ('dark', 'light')}
    written = {os.path.relpath(os.path.join(directory, name), tmp_path) for directory, _, names in os.walk(tmp_path) for name in names}
    full = {'%s_dark.%s' % (name, extension) for name in ('training', 'comparison', 'comparison_bars', 'prediction', 'prediction_horizon',
                                                          'prediction_error', 'decision', 'interception',
                                                          'catch_restitution', 'catch_classes', 'catch_sweep', 'real_3d', 'real_2d', 'motion',
                                                          'motion_joints') for extension in ('pdf', 'png')}
    assert written == full | readme | {'Video/video_comparison_bars_dark.png', 'Video/video_motion_dark.png', 'captions.csv'}


def test_the_throws_errors_are_each_runs_rms_and_the_error_at_its_curves_last_query(stored):
    """
    Description:
        Throw_Errors gives, per method and (flight, N), the remainder RMS of per_flight.csv and the error of the stored
        prediction's last query against the dataset's truth at that time (here the last observation, offset by 0.1 / N m
        on each axis, against the truth: recomputed here from the dataset); per flight with a prediction the largest gap
        between its observed samples.
    """

    from Plot.Core import Throw_Errors
    folder, index = stored
    data = Read_Dataset(str(folder / 'Datasets' / 'Small' / 'flights.npz'))
    rms, end, gaps = Throw_Errors({'QT_BASIS': (str(folder / 'Evaluations' / 'Learned' / 'validation'), '0'),
                             'GLS_M4': (str(folder / 'Evaluations' / 'Run' / 'validation'), '')}, 'Small', data)
    t, x = data.Observations(index)
    frame_time, frame_position = data.Truth(index)
    truth = np.array([np.interp(t[-1], frame_time, frame_position[:, k]) for k in range(3)])
    for method in ('QT_BASIS', 'GLS_M4'):
        for n in (10, 20):
            assert rms[method][(index, n)] == pytest.approx(100.0 - n)
            assert end[method][(index, n)] == pytest.approx(1e3 * np.linalg.norm(x[-1] + 0.1 / n - truth))
        assert sorted(end[method]) == [(index, 10), (index, 20)]
    assert gaps == {index: pytest.approx(float(np.max(np.diff(t))))}
