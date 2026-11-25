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
# File Name: ../test/Dataset/test_manifest.py
## =========================================================================== ##

# Hashlib (Secure hashes)
import hashlib
# JSON (JSON encoder and decoder)
import json
# Dataclasses (Data Classes)
from dataclasses import replace
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Lib.: the manifest under test and the models it hashes
#   ../Dataset/Manifest, ../Flight/Core, ../Observation/Core
from Dataset.Manifest import Build_Manifest, Model_Hash, Read_Manifest, Sha256_Bytes, Sha256_File, Write_Manifest
from Flight.Core import Flight_Model_Str
from Observation.Core import Observation_Model_Str

"""
Description:
    Initialization of constants.

    The manifest's fields are those of SPEC §2.13.10, written out here.
"""
# The fields every manifest carries (SPEC §2.13.10, §9.5).
#   Unit [Set<string>]
CONST_FIELDS = {'manifest_version', 'kind', 'sha256', 'git_commit', 'git_dirty', 'platform', 'machine', 'python', 'numpy'}


def test_the_manifest_has_its_fields_sorted_and_no_wall_clock(tmp_path):
    """
    Description:
        A written manifest holds the fields of SPEC §2.13.10 and the caller's (the seed, the model hash),
        reads back equal, is written with sorted keys, and has no field of a date or a time.
    """

    manifest = Build_Manifest('dataset', {'configuration': 'a' * 64, 'data': 'b' * 64}, {'seed': 20261111, 'model_sha256': 'c' * 64},
                              'f' * 40, False)
    path = str(tmp_path / 'manifest.json')
    Write_Manifest(path, manifest)
    back = Read_Manifest(path)
    assert back == manifest and CONST_FIELDS | {'seed', 'model_sha256'} == set(back)
    text = open(path, encoding='utf-8').read()
    assert text == json.dumps(back, sort_keys=True, indent=2) + '\n'
    assert not any(word in key for key in back for word in ('time', 'date', 'clock'))


def test_an_edited_configuration_changes_its_hash(tmp_path):
    """
    Description:
        The hash of a configuration's bytes changes with one added space; a file's hash is the SHA-256 of its
        bytes.
    """

    text = b'Dataset:\n  Seed: 20261111\n'
    assert Sha256_Bytes(text) != Sha256_Bytes(text.replace(b'Seed:', b'Seed: '))
    path = tmp_path / 'config.yaml'
    path.write_bytes(text)
    assert Sha256_File(str(path)) == hashlib.sha256(text).hexdigest()


def test_the_model_hash_changes_with_a_bound():
    """
    Description:
        The error model's bounds are part of the model a dataset was generated with: its hash
        changes with Offset_Bound and with White_Bound, and a model without bounds differs from a bounded one,
        so datasets of different bounds cannot share a run (SPEC §9.6).
    """

    flight = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.0093, Drag_Std=0.0002)
    observation = Observation_Model_Str(White_Std=np.full(3, 4e-4), White_Dof=2.1, Drift_Std=np.full(3, 3e-3), Drift_Time=0.08,
                                        Offset_Std=np.full(3, 9e-3), Offset_Time=0.1, Offset_Dof=2.5, Offset_Bound=0.070,
                                        White_Bound=0.040)
    base = Model_Hash(Flight_Model_Str=flight, Observation_Model_Str=observation)
    assert base != Model_Hash(Flight_Model_Str=flight, Observation_Model_Str=replace(observation, Offset_Bound=0.060))
    assert base != Model_Hash(Flight_Model_Str=flight, Observation_Model_Str=replace(observation, White_Bound=0.030))
    assert base != Model_Hash(Flight_Model_Str=flight, Observation_Model_Str=replace(observation, Offset_Bound=np.inf, White_Bound=np.inf))


def test_the_model_hash_changes_with_any_parameter():
    """
    Description:
        The identified model's hash (SPEC §2.13.10) is the same for equal parameters and changes when the drag
        moves by 1e-12 1/m or an observation parameter moves.
    """

    flight = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -9.81]), Drag=0.0093, Drag_Std=0.0002)
    observation = Observation_Model_Str(White_Std=np.array([1e-4, 2e-4, 3e-4]), White_Dof=3.0, Drift_Std=np.array([1e-3, 2e-3, 3e-3]),
                                        Drift_Time=0.08, Offset_Std=np.array([9e-3, 1e-2, 6e-3]), Offset_Time=0.1, Offset_Dof=2.5)
    base = Model_Hash(Flight_Model_Str=flight, Observation_Model_Str=observation)
    assert base == Model_Hash(Flight_Model_Str=replace(flight), Observation_Model_Str=replace(observation))
    assert base != Model_Hash(Flight_Model_Str=replace(flight, Drag=0.0093 + 1e-12), Observation_Model_Str=observation)
    assert base != Model_Hash(Flight_Model_Str=flight, Observation_Model_Str=replace(observation, Offset_Time=0.15))
