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
# File Name: ../test/scripts/Maintenance/test_legacy_baseline.py
## =========================================================================== ##

# System (Default)
import os
import sys
# Importlib (The implementation of import)
import importlib.util
# Subprocess (Subprocess management)
import subprocess

"""
Description:
    Initialization of constants.

    The static tests of scripts/Maintenance/legacy_baseline.py. A full run needs the external 2022
    interpreter (Python 3.10, NumPy 1.22.4) and about 2 minutes, so the suite does not run it
    (SPEC §9.3 gives the command).
"""
# The script under test, relative to the repository root.
#   Unit [string]
CONST_SCRIPT = os.path.join('scripts', 'Maintenance', 'legacy_baseline.py')


def _Load(root):
    spec = importlib.util.spec_from_file_location('legacy_baseline', os.path.join(root, CONST_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_setting_of_the_0e59db2_benchmark_is_found_once_and_set(root):
    """
    Description:
        The three lines the harness changes occur exactly once in the benchmark of 0e59db2, and every
        one of the 12 settings comes out set: a harness that missed a line would run the default
        Midpoint / Mean_Opt for every file.
    """

    text = subprocess.run(['git', 'show', '0e59db2:src/Prediction/Helpers/benchmark_prediction.py'], cwd=root,
                          capture_output=True, text=True, timeout=60, check=True).stdout
    module = _Load(root)
    for method in ('Midpoint', 'Heun', 'Second-Runge-Kutta', 'Fourth-Runge-Kutta'):
        for kind in ('Last', 'Mean', 'Mean_Opt'):
            new = module.Substitute_Settings(text, method, kind, '/inputs')
            assert "    PREDICTOR_CORRECTOR_METHOD_NAME = '%s'\n" % method in new
            assert "    PREDICTION_TYPE = '%s'\n" % kind in new
            assert "pd.read_csv(f'/inputs/{CONST_FILE_NAME}.txt')" in new
            assert len(new.splitlines()) == len(text.splitlines())


def test_a_missing_interpreter_is_refused_with_exit_2_and_the_install_command(root, tmp_path):
    """
    Description:
        Run the way a person does (a fresh interpreter, an unrelated working directory): an
        interpreter that does not exist is an [ERROR] naming it and how to install the right one,
        exit 2 (could not check), and nothing is written.
    """

    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, os.path.join(root, CONST_SCRIPT), '--python', '/nonexistent/bin/python',
                             '--out', str(out)], cwd=str(tmp_path), capture_output=True, text=True, timeout=120)
    assert result.returncode == 2, result.stderr
    assert '[ERROR] The 2022 interpreter /nonexistent/bin/python does not exist' in result.stderr
    assert 'conda create' in result.stderr and 'numpy=1.22.4' in result.stderr
    assert not out.exists()
