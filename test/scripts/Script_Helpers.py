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
# File Name: ../test/scripts/Script_Helpers.py
## =========================================================================== ##

# System (Default)
import os
import sys
# Regular expressions (Default; a configuration line replaced in a copy)
import re
# Subprocess (Subprocess management)
import subprocess
# CSV (CSV file reading and writing)
import csv
# JSON (JSON encoder and decoder)
import json
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest

"""
Description:
    Initialization of constants.

    What the tests of the two scripts share (test/scripts/test_catch.py, test/scripts/Maintenance/test_derive.py and the
    fixtures of test/scripts/conftest.py): a script run the way a person runs it (a fresh interpreter, from an
    unrelated working directory), a script loaded as a module with its
    guards, and the configurations and runs the tests build. test/conftest.py puts this folder on sys.path.
"""
# The scripts under test, relative to the repository root: the user's and the derivations'.
#   Unit [string]
CONST_SCRIPT = os.path.join('scripts', 'catch.py')
CONST_DERIVE_SCRIPT = os.path.join('scripts', 'Maintenance', 'derive.py')
# The readers of the recordings each script names, which a leakage guard replaces (_Forbid_Leaks): every name must
# exist in its script, so a reader that is renamed or moved fails the guard instead of leaving it.
#   Unit [Dict<string, Tuple<string>>]
CONST_RECORDING_READERS = {'catch.py': ('Load_Recordings', 'Load_Donors', 'Segment_Flights'),
                           'derive.py': ('Load_Recordings', 'Identification_Flights', 'Segment_Flights')}


def _Run_Derive(root, cwd, *args):
    """
    Description:
        scripts/Maintenance/derive.py in a fresh interpreter, as _Run runs scripts/catch.py.
    """

    return _Run(root, cwd, *args, script=CONST_DERIVE_SCRIPT)


def _Load_Derive(root):
    """
    Description:
        scripts/Maintenance/derive.py imported as a module (in-process: for the leakage tests); its CONST_CATCH_SCRIPT is
        the user script it loaded.
    """

    return _Load_Script(root, CONST_DERIVE_SCRIPT)


def _Run(root, cwd, *args, script=None):
    return subprocess.run([sys.executable, os.path.join(root, script or CONST_SCRIPT), *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=120)


def _Dataset_Config(root, tmp_path, flights, rates=None, target_radius=None):
    """
    Description:
        A copy of the shipped dataset configuration with another flight count (and rates, and radius of the target
        area), as a path.
    """

    text = open(os.path.join(root, 'Config', 'dataset.yaml'), encoding='utf-8').read()
    text = text.replace('  Flights: 10000\n', '  Flights: %d\n' % flights).replace('  Name: Default\n', '  Name: Small\n')
    if rates is not None:
        text = text.replace('  Rates: [100, 200, 300]\n', '  Rates: %s\n' % rates)
    if target_radius is not None:
        assert text.count('    Radius: 1.5\n') == 1
        text = text.replace('    Radius: 1.5\n', '    Radius: %s\n' % target_radius)
    path = tmp_path / ('dataset_%d.yaml' % flights)
    path.write_text(text, encoding='utf-8')
    return path


def _Stored_Folders(root, kind):
    """
    Description:
        The folders of Data/Generated/<kind>/ (hidden ones aside); the test is skipped with the reason when
        Data/Generated/Datasets/ is absent (a clean clone: Data/Generated/ is never committed).
    """

    if not os.path.isdir(os.path.join(root, 'Data', 'Generated', 'Datasets')):
        pytest.skip('Data/Generated/Datasets/ is absent: a clean clone holds no stored dataset (Data/Generated/ is never committed)')
    parent = os.path.join(root, 'Data', 'Generated', kind)
    return sorted(name for name in os.listdir(parent) if os.path.isdir(os.path.join(parent, name)) and not name.startswith('.'))


def _Training_Config(root, tmp_path, datasets, name='Run', model='GLS_M4', samples='[10, 20]'):
    """
    Description:
        A training configuration naming the given (folder, hash) datasets by repository-relative paths, as a path.
    """

    entries = ''.join('    - Path: %s\n      Hash: %s\n' % (os.path.relpath(str(folder), root), digest) for folder, digest in datasets)
    path = tmp_path / ('training_%s.yaml' % name)
    path.write_text('Run:\n  Name: %s\n  Datasets:\n%sInput:\n  Observed_Samples: %s\nModel:\n  Name: %s\nOutput:\n  Folder: %s\n'
                    % (name, entries, samples, model, os.path.relpath(str(tmp_path / 'models'), root)), encoding='utf-8')
    return path


def _Load_Script(root, script=None):
    """
    Description:
        scripts/catch.py imported as a module (in-process: for the leakage tests, whose readers are replaced, and for
        the check of the stored datasets and runs, which no sub-command makes without reading their data).
    """

    import importlib.util
    spec = importlib.util.spec_from_file_location('script_under_test', os.path.join(root, script or CONST_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _Forbid_Leaks(module, monkeypatch):
    """
    Description:
        Replace the test-partition selector and every reader of the recordings in the script by functions that
        raise (SPEC §2.13.7-13.8, §10 Leakage). For scripts/Maintenance/derive.py also in the user script it loaded,
        whose helpers it calls (the guards intact).
    """

    selector = module.Partition_Indices

    def Guarded(data, code):
        if code == 2:
            raise AssertionError('the test partition (code 2, SPEC §9.5) was selected')
        return selector(data, code)

    def Refuse(*args, **kwargs):
        raise AssertionError('a recording was read')

    monkeypatch.setattr(module, 'Partition_Indices', Guarded)
    for name in CONST_RECORDING_READERS[os.path.basename(module.__file__)]:
        monkeypatch.setattr(module, name, Refuse)
    if os.path.basename(module.__file__) == 'derive.py':
        _Forbid_Leaks(module.CONST_CATCH_SCRIPT, monkeypatch)


def _Run_Guarded(root, cwd, monkeypatch, capsys, *args, derive=False):
    """
    Description:
        A script's main() in process (scripts/catch.py, or with derive scripts/Maintenance/derive.py), from `cwd`, with
        the arguments, the test partition and every recording reader guarded (_Forbid_Leaks) and the real validation
        flights replaced by a failure: a refusal before a counted read that a mutation removes ends at a guard, never at
        a recording (the subprocess of _Run has no guard). Its result has _Run's returncode, stdout
        and stderr.
    """

    import types
    module = _Load_Derive(root) if derive else _Load_Script(root)
    _Forbid_Leaks(module, monkeypatch)
    monkeypatch.setattr(module.CONST_CATCH_SCRIPT if derive else module, '_Real_Flights', lambda: pytest.fail('a real flight was read'))
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, 'argv', [os.path.join(root, CONST_DERIVE_SCRIPT if derive else CONST_SCRIPT), *args])
    capsys.readouterr()
    code = module.main()
    captured = capsys.readouterr()
    return types.SimpleNamespace(returncode=code, stdout=captured.out, stderr=captured.err)


def _Trained_Run(root, tmp_path, datasets, dirty=False, name='Run', model='GLS_M4', samples='[10, 20]'):
    """
    Description:
        A run written by 'train' whose manifest's git_dirty is then set as given: the tests run on whatever tree
        state the developer has, so the flag that 'test' checks is set explicitly here (its refusal is tested on
        its own, below).
    """

    config = _Training_Config(root, tmp_path, datasets, name=name, model=model, samples=samples)
    assert _Run(root, tmp_path, 'train', '--config', str(config)).returncode == 0
    run = tmp_path / 'models' / name
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    manifest['git_dirty'] = dirty
    (run / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return run


def _Learned_Config(root, tmp_path, dataset, name='Learned', model='QT_BASIS', flights=21, epochs=2, hidden=8, samples='[10, 20]',
                    trained_on=None):
    """
    Description:
        A training configuration of a learned predictor (SPEC §8.3; F8) scored on `dataset` and trained on
        `trained_on` (the same dataset by default), 5 seeds on the CPU, as a path.
    """

    entry = lambda item: '    - Path: %s\n      Hash: %s\n' % (os.path.relpath(str(item[0]), root), item[1])
    path = tmp_path / ('training_%s.yaml' % name)
    path.write_text('Run:\n  Name: %s\n  Datasets:\n%s  Seeds: [0, 1, 2, 3, 4]\n  Device: cpu\nInput:\n  Observed_Samples: %s\n'
                    'Model:\n  Name: %s\n  Hidden: %d\nTraining:\n  Datasets:\n%s  Flights: %d\nLoss:\n  Velocity_Weight: 0.0\n'
                    'Optimiser:\n  Learning_Rate: 0.003\n  Batch_Size: 16\n  Max_Epochs: %d\n  Weight_Decay: 0.0\n'
                    'Early_Stopping:\n  Patience: 5\n  Min_Delta: 0.0\nOutput:\n  Folder: %s\n'
                    % (name, entry(dataset), samples, model, hidden, ''.join(entry(d) for d in (trained_on or [dataset])), flights, epochs,
                       os.path.relpath(str(tmp_path / 'models'), root)), encoding='utf-8')
    return path


def _Stored_Rows(path):
    with open(path, encoding='ascii', newline='') as file:
        return list(csv.DictReader(file))


def _Decision_Config(root, tmp_path, dataset, digest=None, drag=None, centre=None, quantile=None):
    """
    Description:
        A copy of the shipped system configuration whose Decision evaluates and calibrates on one dataset (a folder, with
        its hash or another), with a 0.1 m cube of Areas.Catch at its centre (27 lattice points: the checks stay quick; or
        another centre), the static catch whatever mode is shipped (_Moving_Config makes it a moving one), and optionally
        another drag or quantile; as a path.
    """

    text = open(os.path.join(root, 'Config', 'ball_catching.yaml'), encoding='utf-8').read()
    start, end = text.index('  Datasets:\n    - Path: Data/Generated/Datasets/Catch_In'), text.index('  # The datasets, among Datasets')
    relative = os.path.relpath(str(dataset[0]), root)
    text = text[:start] + '  Datasets:\n    - Path: %s\n      Hash: %s\n' % (relative, digest or dataset[1]) + text[end:]
    text = re.sub(r'  Calibration: \[.*\]\n', '  Calibration: [%s]\n' % relative, text)
    text = text.replace('    Size: [0.3, 0.75, 0.5]\n', '    Size: [0.1, 0.1, 0.1]\n')
    text, count = re.subn(r'\n  Catch_Mode: (static|moving)\n', '\n  Catch_Mode: static\n', text)
    assert count == 1
    if centre is not None:
        text = text.replace('    Centre: [0.7, -0.175, 0.9]\n', '    Centre: %s\n' % centre)
    if drag is not None:
        text = text.replace('  Drag: 0.00930496257\n', '  Drag: %s\n' % drag)
    if quantile is not None:
        text = text.replace('  Quantile: 0.95\n', '  Quantile: %s\n' % quantile)
    path = tmp_path / ('decision_%d.yaml' % len(list(tmp_path.glob('decision_*.yaml'))))
    path.write_text(text, encoding='utf-8')
    return path


def _Committed_Copy(run, tmp_path):
    """
    Description:
        A copy of a run (the module's learned run: QT_BASIS, 5 seeds, about 1 ms per prediction, so that a decision at every
        sample stays quick; GLS_M4 takes about 150 ms at N 60) whose manifest's git_dirty is False, as _Trained_Run sets it.
    """

    import shutil
    copy = tmp_path / 'models' / 'Learned'
    shutil.copytree(str(run), str(copy))
    manifest = json.loads((copy / 'manifest.json').read_text(encoding='utf-8'))
    (copy / 'manifest.json').write_text(json.dumps(dict(manifest, git_dirty=False), sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return copy


def _Catch_Args(run, config, out, partition='validation', workers=1):
    """
    Description:
        The arguments of 'catch' as argparse gives them.
    """

    import argparse
    return argparse.Namespace(run=str(run), config=str(config), out=str(out), partition=partition, workers=workers)


def _Sweep_Args(run, config, out, interceptions, evaluations, workers=1):
    """
    Description:
        The arguments of the sweep of scripts/Maintenance/derive.py as argparse gives them.
    """

    import argparse
    return argparse.Namespace(run=str(run), config=str(config), out=str(out), interceptions=str(interceptions), evaluations=str(evaluations),
                              workers=workers)


def _Catch_Config(root, tmp_path, dataset, scales='[0.0, 2.0]', throws=6):
    """
    Description:
        _Decision_Config with a quick Contact section: a settle window of 0.6 s, one sensitivity level (0.004 s, 0.3) and,
        for the sweep, two scales on the first 6 validation throws; as a path.
    """

    path = _Decision_Config(root, tmp_path, dataset)
    text = path.read_text(encoding='utf-8')
    text, count = re.subn(r'\n  Levels: \[\[.*\]\]\n', '\n  Levels: [[0.004, 0.3]]\n', text)
    assert count == 1
    for old, new in (('  Settle_Time: 2.5\n', '  Settle_Time: 0.6\n'),
                     ('  Sweep_Scales: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]\n', '  Sweep_Scales: %s\n' % scales),
                     ('  Sweep_Throws: 1500\n', '  Sweep_Throws: %d\n' % throws)):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8')
    return path


def _Moving_Config(path, tmp_path, name='moving.yaml', mode='moving'):
    """
    Description:
        A copy of a system configuration file with Interception.Catch_Mode moving (F15), or the mode given, whatever mode
        it has; as a path.
    """

    text, count = re.subn(r'\n  Catch_Mode: (static|moving)\n', '\n  Catch_Mode: %s\n' % mode, open(str(path), encoding='utf-8').read())
    assert count == 1
    copy = tmp_path / name
    copy.write_text(text, encoding='utf-8')
    return copy


def _Forbid_Stored_Errors(module, monkeypatch):
    """
    Description:
        Replace the reader of a stored evaluation's per-horizon errors in the user script by a function that raises
        (F11's guard: the stored real-flight errors reach 'compare' and the sweep of scripts/Maintenance/derive.py only;
        'catch' reads none).
    """

    def Refuse(*args, **kwargs):
        raise AssertionError('the stored per-horizon errors were read')

    monkeypatch.setattr(module, 'Read_Per_Horizon', Refuse)


def _Stored_Errors(folder, partition, flights, error):
    """
    Description:
        A constructed per_horizon.csv of a stored evaluation: for every flight, N = 20 and 40 and each of the seven
        horizon ranges one row with the position error error(flight, n, range) [mm].
    """

    from Evaluation.Report import CONST_PER_HORIZON_COLUMNS
    labels = ['[0, 0.1)', '[0.1, 0.2)', '[0.2, 0.3)', '[0.3, 0.5)', '[0.5, 0.8)', '[0.8, 1.2)', '[1.2, 2]']
    os.makedirs(str(folder), exist_ok=True)
    with open(os.path.join(str(folder), 'per_horizon.csv'), 'w', encoding='ascii') as file:
        file.write(','.join(CONST_PER_HORIZON_COLUMNS) + '\n')
        for flight in range(flights):
            for n in (20, 40):
                for index, label in enumerate(labels):
                    row = dict(flight='%d' % flight, partition=partition, dataset='' if partition == 'real' else 'Default', n='%d' % n,
                               horizon=label, truth='observed' if partition == 'real' else 'exact',
                               position_error_mm=repr(float(error(flight, n, index))), samples='5', direction_error_deg='', speed_error_m_s='',
                               seed='0')
                    file.write(','.join('"%s"' % row[name] if ',' in row[name] else row[name] for name in CONST_PER_HORIZON_COLUMNS) + '\n')


def _Table_Row(columns, values):
    """
    Description:
        A row in the order of `columns`: the values given by name, 0 for every other column.
    """

    return tuple(values.get(column, 0) for column in columns)


def _Plot_Results(folder, learned='Learned', numerical='Run', dataset_label='Default'):
    """
    Description:
        The stored results the figures of scripts/plots.py read, small and constructed, each file written by its own writer
        under `folder` as Data/Generated/ holds them (the visualisation, delivery 2): compare's two comparisons, the two
        validation evaluations' summary.csv, the decision's, the interception's (static and moving, with per_throw.csv)
        and the catch's (static, moving with the markers variant, the sweep) files of the learned run. Rows the figures
        must leave out are among them (another run, the support 'inside', the summary's spread rows, another dataset,
        a level of another seed, the markers variant in the classes). The evaluations' summaries are of `dataset_label` (and of
        Noise_0, left out). Returns the values written, by what they are.
    """

    from Evaluation.Report import Write_Comparison_Table, Write_Summary, CONST_COMPARISON_SUMMARY_COLUMNS, CONST_COMPARISON_PAIRED_COLUMNS
    from Catch.Decision import Write_Summary as Write_Decision_Summary, CONST_SUMMARY_COLUMNS as CONST_DECISION_SUMMARY
    from Catch.Interception import (Write_Summary as Write_Interception_Summary, Write_Per_Throw as Write_Interception_Rows,
                                    CONST_SUMMARY_COLUMNS as CONST_INTERCEPTION_SUMMARY,
                                    CONST_MOVING_SUMMARY_COLUMNS as CONST_INTERCEPTION_MOVING_SUMMARY,
                                    CONST_PER_THROW_COLUMNS as CONST_INTERCEPTION_ROWS, CONST_MOVING_COLUMNS as CONST_INTERCEPTION_MOVING_ROWS)
    from Catch.Physical import (Write_Summary as Write_Catch_Summary, CONST_SUMMARY_COLUMNS as CONST_CATCH_SUMMARY,
                                CONST_MOVING_SUMMARY_COLUMNS as CONST_CATCH_MOVING_SUMMARY, CONST_SWEEP_COLUMNS)

    expected = {'n': [10, 20, 40], 'accuracy': {}, 'inference': {}, 'paired': {}, 'horizon': {}, 'decision': {}, 'lateral': {}, 'plans': {},
                'share': {}, 'classes': {}, 'sweep': {}}
    sets = {'validation': ('Final_Validation', 'validation', 'Default', 1499), 'held_out': ('Final_Test', 'test', 'Default', 1500),
            'real': ('Final_Test', 'real', '', 45)}
    tables = {}
    for s, (key, (name, partition, dataset, flights)) in enumerate(sets.items()):
        summary, paired = tables.setdefault(name, ([], []))
        for run, method, base, inference in ((learned, 'QT_BASIS', (60.0, 37.0, 22.0), (0.4, 0.5, 0.8)),
                                             (numerical, 'GLS_M4', (58.0, 36.0, 21.0), (30.0, 35.0, 50.0)), ('Other', '', (90.0, 90.0, 90.0), (9.0,
                                                                                                                                               9.0,
                                                                                                                                               9.0))):
            values, medians = [b + 2.0 * s for b in base], list(inference)
            spread = 1.0 if method == 'QT_BASIS' else 0.0
            for k, n in enumerate(expected['n']):
                for support, shift in (('all', 0.0), ('inside', 5.0)):
                    common = {'run': run, 'partition': partition, 'dataset': dataset, 'support': support, 'n': n, 'flights': flights, 'values': ''}
                    summary.append(dict(common, metric='remainder_rms_median_mm', value=values[k] + shift, seed_min=values[k] + shift - spread,
                                        seed_max=values[k] + shift + spread))
                    summary.append(dict(common, metric='inference_median_ms', value=medians[k], seed_min=medians[k], seed_max=medians[k]))
                    summary.append(dict(common, metric='inference_p90_ms', value=1.5 * medians[k], seed_min=1.5 * medians[k],
                                        seed_max=1.5 * medians[k]))
            if method:
                expected['accuracy'][(method, key)] = [values, [v - spread for v in values], [v + spread for v in values]]
                expected['inference'][(method, key)] = [medians, [1.5 * m for m in medians]]
        medians = [-0.3 + 0.1 * s, 0.4 + 0.1 * s, -0.2 + 0.1 * s]
        for k, n in enumerate(expected['n']):
            for support, shift in (('all', 0.0), ('inside', 3.0)):
                paired.append({'run': learned, 'against': numerical, 'partition': partition, 'dataset': dataset, 'support': support, 'n': n,
                               'flights': flights, 'left_out': 0, 'median_mm': medians[k] + shift, 'low_mm': medians[k] + shift - 0.5,
                               'high_mm': medians[k] + shift + 0.6, 'values': ''})
        expected['paired'][key] = [medians, [m - 0.5 for m in medians], [m + 0.6 for m in medians]]
    for name, (summary, paired) in tables.items():
        os.makedirs(os.path.join(folder, 'Comparisons', name), exist_ok=True)
        Write_Comparison_Table(os.path.join(folder, 'Comparisons', name, 'summary.csv'), CONST_COMPARISON_SUMMARY_COLUMNS, summary)
        Write_Comparison_Table(os.path.join(folder, 'Comparisons', name, 'paired.csv'), CONST_COMPARISON_PAIRED_COLUMNS, paired)
    for run, method, seeds in ((learned, 'QT_BASIS', ('0', '1')), (numerical, 'GLS_M4', ('',))):
        rows = []
        for seed in seeds + ('spread_sd',):
            for dataset in (dataset_label, 'Noise_0'):
                for n in expected['n']:
                    for h in (0.05, 0.1, 0.2, 'end'):
                        median = 0.01 if seed == 'spread_sd' else (100.0 * h + n / 10.0 + (int(seed) if seed else 0.5) + (dataset == 'Noise_0') * 50.0
                                                                   if h != 'end' else 999.0)
                        rows.append({'partition': 'validation', 'dataset': dataset, 'seed': seed, 'n': n, 'group': 'all', 'value': '',
                                     'metric': 'fixed_%s_mm' % h, 'count': 10, 'mean': median, 'median': median, 'p90': median, 'max': median})
        os.makedirs(os.path.join(folder, 'Evaluations', run, 'validation'), exist_ok=True)
        Write_Summary(os.path.join(folder, 'Evaluations', run, 'validation', 'summary.csv'), rows)
        for n in expected['n']:
            per_seed = [[100.0 * h + n / 10.0 + (int(seed) if seed else 0.5) for h in (0.05, 0.1, 0.2)] for seed in seeds]
            expected['horizon'][(method, n)] = [list(np.median(per_seed, axis=0)), list(np.min(per_seed, axis=0)), list(np.max(per_seed, axis=0))]
    groups = [(dataset, acceleration) for dataset in ('Catch_In', 'Catch_Near') for acceleration in (800, 400)]
    for kind, columns, writer in (('Decisions', CONST_DECISION_SUMMARY, Write_Decision_Summary),
                                  ('Interceptions', CONST_INTERCEPTION_SUMMARY, Write_Interception_Summary)):
        stage = 'stages' if kind == 'Decisions' else 'samples'
        for p, partition in enumerate(('validation', 'test')):
            rows = []
            for g, (dataset, acceleration) in enumerate(groups):
                values = [(0.30 + 0.01 * g + 0.02 * seed + 0.1 * (stage == 'samples') + 0.003 * p, seed, 0.001 + 0.0005 * seed) for seed in (0, 1)]
                rows += [_Table_Row(columns, {'dataset': dataset, 'partition': partition, 'seed': seed, 'acceleration_deg_s2': acceleration,
                                              'false_no_go_rate': v[0], 'false_go': v[1], 'false_go_upper': v[2]}) for seed, v in zip((0, 1), values)]
                expected['decision'][(stage, partition, dataset, acceleration)] = [list(c) for c in zip(*values)]
            os.makedirs(os.path.join(folder, kind, learned, partition), exist_ok=True)
            writer(os.path.join(folder, kind, learned, partition, 'summary.csv'), rows)
    for mode, extra in (('static', ()), ('moving', CONST_INTERCEPTION_MOVING_ROWS)):
        rows = []
        for acceleration in (800, 400):
            lateral, plans = [], []
            for flight in range(8):
                go = flight % 4 != 3
                value = 0.002 * (flight + 1) + 0.001 * (acceleration == 400) + 0.0005 * (mode == 'moving')
                values = {'dataset': 'Catch_In', 'flight': flight, 'seed': 0, 'acceleration_deg_s2': acceleration,
                          'decision': 'go' if go else 'no-go',
                          'plans': (flight % 3) + 1 if go else '', 'lateral_m': value if go and flight != 5 else ''}
                rows.append(_Table_Row(CONST_INTERCEPTION_ROWS + extra, values))
                rows.append(_Table_Row(CONST_INTERCEPTION_ROWS + extra, dict(values, dataset='Catch_Near', lateral_m=0.5)))
                if go:
                    plans.append((flight % 3) + 1)
                    if flight != 5:
                        lateral.append(1e3 * value)
            expected['lateral'][(mode, acceleration)], expected['plans'][(mode, acceleration)] = sorted(lateral), plans
        target = os.path.join(folder, 'Interceptions', learned, *(('moving',) if mode == 'moving' else ()), 'validation')
        os.makedirs(target, exist_ok=True)
        Write_Interception_Rows(os.path.join(target, 'per_throw.csv'), rows, moving=mode == 'moving')
    os.makedirs(os.path.join(folder, 'Interceptions', learned, 'moving', 'test'), exist_ok=True)
    levels = ((0.004, 0.7), (0.004, 0.5), (0.004, 0.3))
    shares = {'static': (0.97, 0.30, 0.02), 'moving': (0.99, 0.95, 0.40), 'markers': (0.99, 0.98, 0.90)}
    classes = ('caught', 'bounced_out', 'rim', 'no_go_catchable', 'no_go')
    for partition in ('validation', 'test'):
        for mode in ('static', 'moving'):
            moving = mode == 'moving'
            columns = CONST_CATCH_SUMMARY + (CONST_CATCH_MOVING_SUMMARY if moving else ())
            rows = []
            for dataset, acceleration in groups:
                for seed in (0, 1):
                    variants = [(mode, levels[k]) for k in range(3)] if partition == 'validation' and seed == 0 else [(mode, levels[0])]
                    if moving and partition == 'validation' and seed == 0:
                        variants += [('markers', level) for level in levels]
                    for variant, level in variants:
                        share = shares[variant][levels.index(level)] - 0.01 * (acceleration == 400)
                        counts = [40 + seed + (variant == 'markers') * 7, 2 + moving, 1, 5 + seed, 50]
                        rows.append(_Table_Row(columns, dict({'dataset': dataset, 'partition': partition, 'seed': seed,
                                                              'acceleration_deg_s2': acceleration,
                                                              'time_constant_s': level[0], 'damping_ratio': level[1], 'retained_of_go': share,
                                                              'retained_of_go_lower': share - 0.01, 'retained_of_go_upper': share + 0.005,
                                                              'markers': 'true' if variant == 'markers' else 'false'}, **dict(zip(classes, counts)))))
                        if partition == 'validation' and seed == 0:
                            expected['share'].setdefault((variant, dataset, acceleration), []).append((level, share, share - 0.01, share + 0.005))
                        if level == levels[0] and variant != 'markers':
                            key = (partition, mode, dataset, acceleration)
                            expected['classes'][key] = [a + b for a, b in zip(expected['classes'].get(key, [0] * 5), counts)]
            target = os.path.join(folder, 'Catches', learned, *(('moving',) if moving else ()), partition)
            os.makedirs(target, exist_ok=True)
            Write_Catch_Summary(os.path.join(target, 'summary.csv'), rows, moving=moving)
    rows = []
    for dataset, acceleration in groups:
        for source, scales in (('real', (0.0, 1.0, 2.0)), ('simulated', (1.0,)), ('run', (float('nan'),))):
            values = []
            for scale in scales:
                share = 0.84 - 0.1 * (0.0 if scale != scale else scale) - 0.07 * (acceleration == 400) - 0.03 * (source == 'simulated')
                rows.append(_Table_Row(CONST_SWEEP_COLUMNS + CONST_CATCH_SUMMARY, {'error_source': source, 'error_scale': scale, 'dataset': dataset,
                                                                                    'partition': 'validation', 'seed': 0,
                                                                                    'acceleration_deg_s2': acceleration,
                                                                                    'time_constant_s': 0.004, 'damping_ratio': 0.7,
                                                                                    'retained_of_catchable': share,
                                                                                    'retained_of_catchable_lower': share - 0.02,
                                                                                    'retained_of_catchable_upper': share + 0.02}))
                values.append((scale, share, share - 0.02, share + 0.02))
            expected['sweep'][(source, dataset, acceleration)] = values
    os.makedirs(os.path.join(folder, 'Catches', learned, 'sweep'), exist_ok=True)
    Write_Catch_Summary(os.path.join(folder, 'Catches', learned, 'sweep', 'summary.csv'), rows, sweep=True)
    expected['ranges'] = {}
    for run, method, seeds in ((learned, 'QT_BASIS', ('0', '1')), (numerical, 'GLS_M4', ('',))):
        rows = []
        for key, partition, dataset in (('held_out', 'test', 'Default'), ('real', 'real', '')):
            values = {}
            for seed in seeds + ('spread_sd',):
                for n in (10, 20):
                    for (a, b), counted in (((0.0, 0.1), True), ((0.1, 0.2), True), ((0.2, 0.3), True), ((1.2, 2.0), False)):
                        median = 10.0 * (a + 1.0) + (int(seed) if seed.isdigit() else 0.5) + 5.0 * (key == 'real') + (n == 10) * 50.0
                        if seed == 'spread_sd':
                            median = 0.01
                        for metric, shift in (('range_%g_%g_mm' % (a, b), 0.0), ('range_observed_%g_%g_mm' % (a, b), 7.0)):
                            rows.append({'partition': partition, 'dataset': dataset, 'seed': seed, 'n': n, 'group': 'all', 'value': '',
                                         'metric': metric,
                                         'count': 40 + len(seed), 'mean': median, 'median': median + shift if counted else float('nan'),
                                         'p90': 2.0 * median + shift if counted else float('nan'), 'max': 3.0 * median})
                        if counted and n == 20 and seed != 'spread_sd' and metric:
                            values.setdefault((a, b), []).append(median)
            expected['ranges'][(key, method)] = [[a for a, _ in sorted(values)], [b for _, b in sorted(values)],
                                                 [float(np.median(values[k])) for k in sorted(values)],
                                                 [float(np.median([2.0 * v for v in values[k]])) for k in sorted(values)]]
        os.makedirs(os.path.join(folder, 'Evaluations', run, 'test'), exist_ok=True)
        Write_Summary(os.path.join(folder, 'Evaluations', run, 'test', 'summary.csv'), rows)
        with open(os.path.join(folder, 'Evaluations', run, 'test', 'manifest.json'), 'w', encoding='utf-8') as file:
            json.dump({'test_evaluation': 5 if method == 'QT_BASIS' else 3}, file)
    return expected


def _Throw_Errors(folder, dataset, errors, seed):
    """
    Description:
        A validation evaluation's per_flight.csv (its writer's columns) holding per flight and N a remainder RMS [mm]
        (`errors`: flight -> N -> RMS): what the prediction figure's rule reads besides the stored predictions.
    """

    from Evaluation.Report import CONST_PER_FLIGHT_COLUMNS
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'per_flight.csv'), 'w', newline='', encoding='ascii') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(CONST_PER_FLIGHT_COLUMNS)
        for flight, per_n in errors.items():
            for n, rms in per_n.items():
                row = dict.fromkeys(CONST_PER_FLIGHT_COLUMNS, '')
                row.update(flight='%d' % flight, partition='validation', dataset=dataset, n='%d' % n, remainder_rms_mm='%.6f' % rms, seed=seed)
                writer.writerow([row[c] for c in CONST_PER_FLIGHT_COLUMNS])
