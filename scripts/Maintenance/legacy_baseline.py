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
# File Name: ../scripts/Maintenance/legacy_baseline.py
## =========================================================================== ##

# System (Default)
import sys
# OS (Operating system interfaces)
import os
# Argparse (Command-line parsing)
import argparse
# Hashlib (Secure hashes)
import hashlib
# Shutil (High-level file operations)
import shutil
# Subprocess (Subprocess management)
import subprocess
# Time (Time access and conversions)
import time
# Typing (Support for type hints)
import typing as tp

"""
Description:
    Initialization of constants.

    Regenerates the 2022 per-throw baseline (SPEC §4.1, §9.3) by running the unmodified benchmark of
    commit 0e59db2 in an external Python 3.10 / NumPy 1.22.4 interpreter, and compares it byte for
    byte with Data/Legacy/Baseline/. It does not go through src/: running the 2022 code itself is its
    purpose (a harness takes the project's path). It
    writes only into --out (default Data/Generated/Legacy_Baseline/, git-ignored), including the
    temporary git worktree of 0e59db2, which it removes; Data/Legacy/ is only read.
"""
# The repository root, built from this file's location.
#   Unit [string]
CONST_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# The commit of the 2022 code and its benchmark script (SPEC §4.1).
#   Unit [string]
CONST_COMMIT = '0e59db2'
CONST_BENCHMARK = 'src/Prediction/Helpers/benchmark_prediction.py'
# The inputs, the committed baseline and the default output folder (SPEC §6, §9.3).
#   Unit [string]
CONST_INPUT_DIR = os.path.join(CONST_REPO_ROOT, 'Data', 'Legacy')
CONST_REFERENCE_DIR = os.path.join(CONST_REPO_ROOT, 'Data', 'Legacy', 'Baseline')
CONST_OUT_DIR = os.path.join(CONST_REPO_ROOT, 'Data', 'Generated', 'Legacy_Baseline')
# The methods (as the input files name them) and the key states: 12 settings.
#   Unit [Tuple<string>]
CONST_METHODS = ('Midpoint', 'Heun', 'Second-Runge-Kutta', 'Fourth-Runge-Kutta')
CONST_KINDS = ('Last', 'Mean', 'Mean_Opt')
# The three lines of the benchmark that change, each found exactly once: its two in-script settings
# and the folder of its inputs (SPEC §4.1). Everything else runs unmodified.
#   Unit [Tuple<string>]
CONST_METHOD_LINE = "    PREDICTOR_CORRECTOR_METHOD_NAME = 'Midpoint'"
CONST_KIND_LINE = "    PREDICTION_TYPE = 'Mean_Opt'"
CONST_INPUT_LINE = "pd.read_csv(f'../Data/{CONST_FILE_NAME}.txt')"
# The interpreter the baseline was made with, and how to get it (SPEC §9.3).
#   Unit [string]
CONST_INTERPRETER_HELP = ('a Python 3.10 with NumPy 1.22.4, pandas 1.4.4 and Matplotlib 3.5.3, e.g. with Miniforge: '
                          'conda create -y -p <dir> -c conda-forge python=3.10 numpy=1.22.4 pandas=1.4.4 matplotlib=3.5.3, '
                          'then --python <dir>/bin/python')
# Run inside the 2022 interpreter: replaces plt.show by writing the per-throw errors (the y data of
# the first line of the figure the benchmark plots) in %.10e, and runs the substituted copy from the
# benchmark's folder, whose sys.path and input paths are relative to it.
#   Unit [string]
CONST_DRIVER = '''
import os, sys, runpy
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
script, helpers, out = sys.argv[1:4]
def _Dump(*args, **kwargs):
    np.savetxt(out, plt.gcf().axes[0].lines[0].get_ydata(), fmt='%.10e')
plt.show = _Dump
os.chdir(helpers)
sys.argv = [script]
try:
    runpy.run_path(script, run_name='__main__')
except SystemExit:
    pass
'''
# The probe of the interpreter: the packages the benchmark imports, and np.asscalar, which it calls
# and NumPy removed in 1.23.
#   Unit [string]
CONST_PROBE = ('import sys, numpy, pandas, matplotlib; numpy.asscalar; '
               'print(sys.version.split()[0], numpy.__version__, pandas.__version__, matplotlib.__version__)')


def _Log(level: str, message: str) -> None:
    """
    Description:
        Prints one [INFO] / [ERROR] line on stderr, so stdout holds only the comparison report.

    Args:
        (1 - 2) level, message [string]: The level and the sentence.
    """

    print('[%s] %s' % (level, message), file=sys.stderr, flush=True)


def Substitute_Settings(text: str, method: str, kind: str, input_dir: str) -> str:
    """
    Description:
        The benchmark's text with its method, its key state and its input folder set.

    Args:
        (1) text [string]: The benchmark script of 0e59db2.
        (2 - 3) method, kind [string]: One of CONST_METHODS, one of CONST_KINDS.
        (4) input_dir [string]: The folder of the input files.

    Returns:
        (1) parameter [string]: The substituted text.
    """

    for line, new in ((CONST_METHOD_LINE, "    PREDICTOR_CORRECTOR_METHOD_NAME = '%s'" % method),
                      (CONST_KIND_LINE, "    PREDICTION_TYPE = '%s'" % kind),
                      (CONST_INPUT_LINE, "pd.read_csv(f'%s/{CONST_FILE_NAME}.txt')" % input_dir)):
        if text.count(line) != 1:
            raise ValueError('The line %r occurs %d times in %s of %s, not once.' % (line, text.count(line), CONST_BENCHMARK, CONST_COMMIT))
        text = text.replace(line, new)
    return text


def _Remove_Worktree(path: str) -> None:
    """
    Description:
        Removes the temporary worktree (also one left by an interrupted run) and prunes git's record of it.

    Args:
        (1) path [string]: The worktree's folder.
    """

    if os.path.isdir(path):
        subprocess.run(['git', 'worktree', 'remove', '--force', path], cwd=CONST_REPO_ROOT, capture_output=True, timeout=120)
    subprocess.run(['git', 'worktree', 'prune'], cwd=CONST_REPO_ROOT, capture_output=True, timeout=120)
    shutil.rmtree(path, ignore_errors=True)


def _Sha256(path: str) -> str:
    """
    Description:
        The SHA-256 of a file's bytes.

    Args:
        (1) path [string]: The file.

    Returns:
        (1) parameter [string]: The hexadecimal digest.
    """

    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _Run(python: str, out_dir: str) -> tp.Optional[str]:
    """
    Description:
        Runs the 12 settings in a temporary worktree of CONST_COMMIT under out_dir, writing one file
        per setting; the worktree is removed whatever happens.

    Args:
        (1) python [string]: The 2022 interpreter.
        (2) out_dir [string]: The output folder.

    Returns:
        (1) parameter [string]: None when every setting ran, else the reason it could not.
    """

    work = os.path.join(out_dir, '.work')
    tree = os.path.join(work, CONST_COMMIT)
    _Remove_Worktree(tree)
    os.makedirs(work, exist_ok=True)
    result = subprocess.run(['git', 'worktree', 'add', '--detach', tree, CONST_COMMIT], cwd=CONST_REPO_ROOT,
                            capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return 'git worktree add failed: %s' % result.stderr.strip()
    try:
        helpers = os.path.join(tree, os.path.dirname(CONST_BENCHMARK))
        with open(os.path.join(tree, CONST_BENCHMARK)) as handle:
            text = handle.read()
        for method in CONST_METHODS:
            for kind in CONST_KINDS:
                name = '%s_%s.txt' % (method, kind)
                script = os.path.join(work, 'benchmark_%s_%s.py' % (method, kind))
                with open(script, 'w') as handle:
                    handle.write(Substitute_Settings(text, method, kind, CONST_INPUT_DIR))
                t_0 = time.time()
                run = subprocess.run([python, '-c', CONST_DRIVER, script, helpers, os.path.join(out_dir, name)],
                                     capture_output=True, text=True, timeout=1800)
                if run.returncode != 0 or not os.path.isfile(os.path.join(out_dir, name)):
                    return '%s failed (exit %d): %s' % (name, run.returncode, run.stderr.strip()[-500:])
                _Log('INFO', '%s written in %.1f s.' % (name, time.time() - t_0))
    finally:
        _Remove_Worktree(tree)
        shutil.rmtree(work, ignore_errors=True)
    return None


def main():
    """
    Description:
        Regenerates the 12 baseline files with the 2022 code and compares them byte for byte with
        Data/Legacy/Baseline/ (SPEC §9.3). Exit codes: 0 every file identical, 1 a file differs (each
        named), 2 it could not check (no usable interpreter, no git, a failed run).

            .venv/bin/python scripts/Maintenance/legacy_baseline.py --python <dir>/bin/python [--out DIR]
    """

    parser = argparse.ArgumentParser(description='Regenerate the 2022 baseline with the 0e59db2 code and compare it.')
    parser.add_argument('--python', required=True, help=CONST_INTERPRETER_HELP)
    parser.add_argument('--out', default=CONST_OUT_DIR, help='write into this directory (default: Data/Generated/Legacy_Baseline/)')
    args = parser.parse_args()

    if not (os.path.isfile(args.python) and os.access(args.python, os.X_OK)):
        _Log('ERROR', 'The 2022 interpreter %s does not exist or is not executable. Install %s.' % (args.python, CONST_INTERPRETER_HELP))
        return 2
    probe = subprocess.run([args.python, '-c', CONST_PROBE], capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        _Log('ERROR', '%s cannot run the 2022 code (%s). Install %s.'
             % (args.python, (probe.stderr.strip().splitlines() or ['no output'])[-1], CONST_INTERPRETER_HELP))
        return 2
    _Log('INFO', 'Python, NumPy, pandas, Matplotlib: %s.' % probe.stdout.strip())
    if subprocess.run(['git', 'cat-file', '-e', CONST_COMMIT + '^{commit}'], cwd=CONST_REPO_ROOT, capture_output=True, timeout=60).returncode != 0:
        _Log('ERROR', 'Commit %s is not in this repository; fetch it and run again.' % CONST_COMMIT)
        return 2

    os.makedirs(args.out, exist_ok=True)
    reason = _Run(args.python, args.out)
    if reason is not None:
        _Log('ERROR', 'The baseline could not be regenerated: %s' % reason)
        return 2

    names = sorted('%s_%s.txt' % (m, k) for m in CONST_METHODS for k in CONST_KINDS)
    with open(os.path.join(args.out, 'SHA256SUMS'), 'w') as handle:
        handle.write(''.join('%s  %s\n' % (_Sha256(os.path.join(args.out, name)), name) for name in names))
    differs = 0
    for name in names:
        same = _Sha256(os.path.join(args.out, name)) == _Sha256(os.path.join(CONST_REFERENCE_DIR, name))
        differs += not same
        print('%-34s %s' % (name, 'identical' if same else 'DIFFERS from Data/Legacy/Baseline/'))
    print('%d of %d files byte-identical to Data/Legacy/Baseline/.' % (len(names) - differs, len(names)))
    return 1 if differs else 0


if __name__ == '__main__':
    sys.exit(main())
