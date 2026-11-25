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
# File Name: ../test/Docs/test_layout.py
## =========================================================================== ##

# Abstract syntax trees (Default)
import ast
# System (Default)
import os
# Regular expressions (Default)
import re
# Subprocess (Subprocess management)
import subprocess

"""
Description:
    Initialization of constants.

    The sets are written out here, never read from the README or the filesystem they check; a new
    root entry changes this set and SPEC §6 in the same commit.
"""
# The entries git tracks at the root (SPEC §6): pytest.ini (pytest reads it at the root only), CITATION.cff
# (GitHub reads it at the root only) and CHANGELOG.md beside the code, data and documentation folders.
# scripts/ joined with F1, models/ with F4, Media/ (the README's images) and docs/ with F13.
#   Unit [Set<string>]
CONST_ROOT_ENTRIES = {'LICENSE', 'README.md', 'CHANGELOG.md', 'CITATION.cff', '.gitignore', 'install.sh',
                      'requirements.txt', 'src', 'test', 'Config', 'Data', 'CAD', 'pytest.ini', 'scripts', 'models', 'Media', 'docs'}
# The documents, in docs/.
#   Unit [Tuple<string>]
CONST_PROJECT_DOCS = ('SPEC.md', 'KNOWN_ISSUES.md')


def test_the_root_holds_only_the_declared_entries(root):
    """
    Description:
        The files git tracks at the root are the declared entries and nothing else; the documents
        live in docs/, not at the root.
    """

    result = subprocess.run(['git', 'ls-files'], cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    entries = {path.split('/', 1)[0] for path in result.stdout.splitlines()}
    assert entries == CONST_ROOT_ENTRIES
    # Tracked, not only on disk: a .gitignore rule would otherwise leave them out of every clone.
    tracked = set(result.stdout.splitlines())
    for name in CONST_PROJECT_DOCS:
        assert 'docs/' + name in tracked, name


def test_no_path_differs_from_another_only_in_case(root):
    """
    Description:
        No two tracked paths differ only in letter case: a case-insensitive file system (macOS by
        default) would check out only one of them. A case-only rename goes through a temporary name.
    """

    result = subprocess.run(['git', 'ls-files'], cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    seen = {}
    for path in result.stdout.splitlines():
        assert path.lower() not in seen, (path, seen.get(path.lower()))
        seen[path.lower()] = path


# The declared modules of each src/ domain (a Core.py and its topic modules), written
# out, never read from src/; a domain joins with its feature (SPEC §7).
#   Unit [Dict<string, Set<string>>]
CONST_DOMAIN_MODULES = {'Configuration': {'Core.py'}, 'Geometry': {'Core.py'}, 'Flight': {'Core.py'},
                        'Recording': {'Core.py', 'Segmentation.py', 'Report.py', 'Identification.py'},
                        'Observation': {'Core.py', 'Estimation.py'}, 'Dataset': {'Core.py', 'Manifest.py', 'Calibration.py'},
                        'Prediction': {'Core.py'}, 'Learning': {'Core.py', 'Models.py', 'Predictor.py', 'Training.py'},
                        'Evaluation': {'Core.py', 'Report.py', 'Comparison.py'}, 'Robot': {'Core.py', 'Motion.py'},
                        'Scene': {'Core.py', 'Ball.py'}, 'View': {'Core.py', 'Theme.py'}, 'Plot': {'Core.py', 'Style.py', 'Figures.py'},
                        'Catch': {'Core.py', 'Workspace.py', 'Decision.py', 'Interception.py', 'Physical.py', 'Robustness.py'}}


def test_each_domain_holds_its_declared_modules(root):
    """
    Description:
        Each src/ domain holds exactly its declared modules, and each module except a Const.py has
        its mirrored test file (test/<Domain>/test_<module>.py).
    """

    src = os.path.join(root, 'src')
    domains = {name for name in os.listdir(src) if os.path.isdir(os.path.join(src, name)) and not name.startswith(('.', '_'))}
    assert domains == set(CONST_DOMAIN_MODULES)
    for domain, modules in CONST_DOMAIN_MODULES.items():
        found = {name for name in os.listdir(os.path.join(src, domain)) if name.endswith('.py')}
        assert found == modules, domain
        for module in sorted(modules - {'Const.py'}):
            test = os.path.join(root, 'test', domain, 'test_' + module.lower())
            assert os.path.isfile(test), test


def _Python_Files(root, folders):
    for folder in folders:
        for directory, names, files in os.walk(os.path.join(root, folder)):
            names[:] = [name for name in names if name != '__pycache__']
            for name in sorted(files):
                if name.endswith('.py'):
                    yield os.path.relpath(os.path.join(directory, name), root)


def test_nothing_under_src_imports_from_test(root):
    """
    Description:
        src/ never reaches into test/: no import of a module that lives only there (the 2022 test
        reference, a conftest) and no path built from 'test' (SPEC §2.1: the legacy pipeline
        is a test reference, not project code).
    """

    test_modules = {os.path.splitext(os.path.basename(path))[0] for path in _Python_Files(root, ['test'])
                    if not os.path.basename(path).startswith('test_')} | {'test', 'Legacy'}
    hits = []
    for path in _Python_Files(root, ['src']):
        with open(os.path.join(root, path)) as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or '').split('.')[0]]
            elif isinstance(node, ast.Constant) and node.value == 'test':
                names = ['test']
            else:
                continue
            hits += ['%s:%d %s' % (path, node.lineno, name) for name in names if name in test_modules]
    assert hits == []


def test_the_legacy_constants_are_defined_and_read_only_in_the_test_reference(root):
    """
    Description:
        The 2022 parameters (370 Hz, g 9.81 m/s^2, 20 states, the flight cuboid, the 1.01 m
        hemisphere, 1e-5 m) are defined in test/Legacy/Reference.py and read only there and in its
        tests, so they never become the project's values by an import.
    """

    prefix = 'CONST_' + 'LEGACY_'
    allowed = {os.path.join('test', 'Legacy', 'Reference.py'), os.path.join('test', 'Legacy', 'test_reference.py')}
    readers, definers = set(), set()
    for path in _Python_Files(root, ['src', 'test', 'scripts']):
        with open(os.path.join(root, path)) as handle:
            text = handle.read()
        if prefix in text and path != os.path.join('test', 'Docs', 'test_layout.py'):
            readers.add(path)
        if re.search(r'^' + prefix + r'\w+\s*=', text, re.M):
            definers.add(path)
    assert readers <= allowed, sorted(readers - allowed)
    assert definers == {os.path.join('test', 'Legacy', 'Reference.py')}
