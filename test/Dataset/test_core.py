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
# File Name: ../test/Dataset/test_core.py
## =========================================================================== ##

# System (Default)
import os
# Hashlib (Secure hashes)
import hashlib
# Dataclasses (Data Classes)
from dataclasses import replace
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.: the dataset generator under test, the configuration, the models, the split and the areas
#   ../Dataset/Core, ../Configuration/Core, ../Flight/Core, ../Observation/Core, ../Recording/Core, ../Geometry/Core
import Dataset.Core as Dataset_Core
from Dataset.Core import (Dataset_Error_Cls, Generate_Dataset, Load_Donors, Partition_Codes, Partition_Indices, Partition_Sizes, Read_Dataset,
                          Target_Area, Write_Dataset)
from Configuration.Core import Load_Configuration, Load_Dataset_Configuration
from Flight.Core import Flight_Model_Str, Simulate, Simulate_At
from Observation.Core import Observation_Model_Str
from Recording.Core import Assign_Sets, CONST_SET_VALIDATION
from Geometry.Core import Cuboid_Cls, Spherical_Area_Cls

"""
Description:
    Initialization of constants.

    The datasets here are small (20-80 flights) from the shipped configurations; the 1 000-flight
    acceptance of SPEC §2.13.2-13.3 is in the slow tier. Expectations come from the rules of SPEC §7.6 and
    §9.5, the flight model's state function or the owner's decisions, never from the generator.
"""
# The floor height of SPEC §14.6 and the shooting tolerance of SPEC §7.6.
#   Unit [float, m]
CONST_FLOOR = 0.030
CONST_MISS = 1e-9


@pytest.fixture(scope='module')
def setup(root):
    configuration = Load_Configuration(os.path.join(root, 'Config', 'ball_catching.yaml'))
    dataset = Load_Dataset_Configuration(os.path.join(root, 'Config', 'dataset.yaml'))
    flight = Flight_Model_Str(Gravity=np.array([0.0, 0.0, -configuration.Flight.Gravity]), Drag=configuration.Flight.Drag,
                              Drag_Std=configuration.Flight.Drag_Std)
    o = configuration.Observation
    observation = Observation_Model_Str(White_Std=np.array(o.White_Std), White_Dof=o.White_Dof, Drift_Std=np.array(o.Drift_Std),
                                        Drift_Time=o.Drift_Time, Offset_Std=np.array(o.Offset_Std), Offset_Time=o.Offset_Time,
                                        Offset_Dof=o.Offset_Dof, Offset_Bound=o.Offset_Bound, White_Bound=o.White_Bound)
    donors = Load_Donors(os.path.join(root, 'Data', 'Recordings'))
    return configuration, dataset, flight, observation, donors


def _Generate(setup, model=None, **changes):
    configuration, dataset, flight, observation, donors = setup
    observation = observation if model is None else model
    observation_choice = replace(dataset.Observation, **{k: v for k, v in changes.items() if k in ('Rates', 'Noise_Scale', 'Pattern', 'Donors')})
    dataset = replace(dataset, Observation=observation_choice, **{k: v for k, v in changes.items() if k in ('Flights', 'Seed')})
    return Generate_Dataset(dataset, configuration, donors, Flight_Model_Str=flight, Observation_Model_Str=observation)


@pytest.fixture(scope='module')
def small(setup):
    return _Generate(setup, Flights=40)


def test_the_partition_sizes_follow_the_largest_remainder_rule():
    """
    Description:
        The largest-remainder rule (SPEC §2.13.1): 10 000 flights at 0.70 / 0.15 / 0.15 give 7000 / 1500 /
        1500; 14 286 give 10000 / 2143 / 2143 (quotas 10000.2, 2142.9, 2142.9); 10 give
        7 / 2 / 1 (quotas 7, 1.5, 1.5: the tie goes to the earlier partition); 7 give 5 / 1 / 1.
    """

    ratios = (0.70, 0.15, 0.15)
    assert Partition_Sizes(10000, ratios) == [7000, 1500, 1500]
    assert Partition_Sizes(14286, ratios) == [10000, 2143, 2143]
    assert Partition_Sizes(10, ratios) == [7, 2, 1]
    assert Partition_Sizes(7, ratios) == [5, 1, 1]


def test_the_partitions_are_disjoint_index_blocks(small):
    """
    Description:
        The partitions are index blocks, disjoint by flight: of 40 flights the
        first 28 are train, the next 6 validation, the last 6 test.
    """

    assert Partition_Codes(20, (0.70, 0.15, 0.15)).tolist() == [0] * 14 + [1] * 3 + [2] * 3
    assert small.Partition.tolist() == [0] * 28 + [1] * 6 + [2] * 6


def test_only_the_identification_files_are_opened(root, monkeypatch):
    """
    Description:
        Leakage (SPEC §2.13.5, §2.14): with the recording reader replaced by one that raises for any file of
        the validation set, the donors still load: 45 files are opened, 10 / 30 / 5 donors at 100 / 200 /
        300 Hz.
    """

    folder = os.path.join(root, 'Data', 'Recordings')
    names = [(sub, name) for sub in sorted(os.listdir(folder)) for name in sorted(os.listdir(os.path.join(folder, sub)))]
    sets = Assign_Sets(names)
    opened = []
    original = Dataset_Core.Load_Recording

    def Guarded(path):
        key = (os.path.basename(os.path.dirname(path)), os.path.basename(path))
        if sets[key] == CONST_SET_VALIDATION:
            raise AssertionError('a validation flight was opened: %s' % (key,))
        opened.append(key)
        return original(path)

    monkeypatch.setattr(Dataset_Core, 'Load_Recording', Guarded)
    donors = Load_Donors(folder)
    assert len(opened) == 45 and len(donors) == 45
    assert [sum(d.Rate == rate for d in donors) for rate in (100.0, 200.0, 300.0)] == [10, 30, 5]


def test_every_throw_lies_in_its_areas_and_ranges(setup, small):
    """
    Description:
        Each release lies in Areas.Throw and each target in Throw.Target_Area, the half facing the thrower of
        a sphere's upper hemisphere (within its radius of its centre, not below it, y not beyond it); the release speed
        is in Throw.Release_Speed and the flight time in Throw.Flight_Time;
        the drawn apex is above the release.
    """

    configuration, dataset = setup[0], setup[1]
    throw = Cuboid_Cls(configuration.Areas.Throw.Centre, configuration.Areas.Throw.Size)
    centre, radius = np.array(dataset.Throw.Target_Area.Centre), dataset.Throw.Target_Area.Radius
    for i in range(small.Flights):
        assert throw.Is_Point_Inside(small.Release_State[i, :3]), i
        assert np.linalg.norm(small.Target[i] - centre) <= radius + 1e-12, i
        assert small.Target[i, 2] >= centre[2] - 1e-12 and small.Target[i, 1] <= centre[1] + 1e-12, i
        assert dataset.Throw.Release_Speed[0] <= np.linalg.norm(small.Release_State[i, 3:]) <= dataset.Throw.Release_Speed[1], i
        assert dataset.Throw.Flight_Time[0] <= small.Flight_Time[i] <= dataset.Throw.Flight_Time[1], i
        assert small.Apex_Drawn[i] > small.Release_State[i, 2], i


def test_the_targets_fill_the_facing_half_of_a_1_5_m_sphere_beyond_the_robot_area(setup, small):
    """
    Description:
        The targets have their own area, Throw.Target_Area, the facing half of a sphere of
        r 1.5 m centred like the robot hemisphere (0, 0, 0.3607 m), not Areas.Robot (r 1.0 m). Of 40 targets uniform
        in it, the share beyond 1.0 m of the centre is 1 - (1.0 / 1.5)^3 = 70 % (here between 50 and 90 %), none
        beyond 1.5 m: the datasets hold throws the robot cannot catch. With the radius set to 1.0 m (F7's
        targets) every target lies within 1.0 m.
    """

    configuration, dataset = setup[0], setup[1]
    area = dataset.Throw.Target_Area
    assert (area.Centre, area.Radius) == ([0.0, 0.0, 0.3607], 1.5) and area.Centre == configuration.Areas.Robot.Centre
    assert Target_Area(dataset, configuration).Shape == 'quarter_sphere' and Target_Area(dataset, configuration).Radius == 1.5
    distance = np.linalg.norm(small.Target - np.array(area.Centre), axis=1)
    assert distance.max() <= 1.5 and 0.5 <= np.mean(distance > configuration.Areas.Robot.Radius) <= 0.9
    narrow = replace(dataset, Flights=40, Throw=replace(dataset.Throw, Target_Area=replace(area, Radius=1.0)))
    data = Generate_Dataset(narrow, configuration, setup[4], Flight_Model_Str=setup[2], Observation_Model_Str=setup[3])
    assert np.linalg.norm(data.Target - np.array(area.Centre), axis=1).max() <= 1.0


def _Catch_Dataset(setup, margin, flights, seed=7):
    configuration, dataset, flight, observation, donors = setup
    catch = replace(dataset, Flights=flights, Seed=seed, Throw=replace(dataset.Throw, Target_Shape='catch_area', Target_Margin=margin))
    return catch, Generate_Dataset(catch, configuration, donors, Flight_Model_Str=flight, Observation_Model_Str=observation)


def test_the_catch_targets_fill_the_catch_area_grown_by_the_margin(setup):
    """
    Description:
        A catch dataset aims its throws uniformly into Areas.Catch grown by
        Throw.Target_Catch_Area.Margin on every face. With the margin 0, all 40 targets lie in Areas.Catch (its faces
        included) and cover each axis over more than half of its size; with 0.25 m all 40 lie in the grown box and some
        outside Areas.Catch. Each throw passes through its target at its flight time (the shooting, SPEC §7.6).
    """

    configuration, flight = setup[0], setup[2]
    centre, size = np.array(configuration.Areas.Catch.Centre), np.array(configuration.Areas.Catch.Size)
    _, data = _Catch_Dataset(setup, 0.0, 40)
    offset = np.abs(data.Target - centre)
    assert np.all(offset <= size / 2.0 + 1e-12)
    assert np.all(np.ptp(data.Target, axis=0) > size / 2.0)
    for i in range(data.Flights):
        reached = Simulate(data.Release_State[i:i + 1], np.array([data.Flight_Time[i]]), Flight_Model_Str=flight)[0, 0, :3]
        assert np.linalg.norm(reached - data.Target[i]) < CONST_MISS
    _, near = _Catch_Dataset(setup, 0.25, 40)
    offset = np.abs(near.Target - centre)
    assert np.all(offset <= size / 2.0 + 0.25 + 1e-12) and np.any(np.any(offset > size / 2.0, axis=1))


def test_a_catch_dataset_is_reproduced_and_keeps_its_stable_prefix(setup):
    """
    Description:
        A catch dataset follows the generator's rules (SPEC §2.13.2-13.3): the same configuration and seed give the same
        arrays, the first 20 of 40 flights equal a 20-flight dataset, and Dataset.Core.Target_Area is Areas.Catch grown by
        the margin on every face (0.1 m: each edge 0.2 m longer, the same centre).
    """

    catch, first = _Catch_Dataset(setup, 0.1, 40)
    _, again = _Catch_Dataset(setup, 0.1, 40)
    _, prefix = _Catch_Dataset(setup, 0.1, 20)
    for name in ('Release_State', 'Target', 'Observed_Time', 'Observed_Position', 'Truth_Position'):
        assert np.array_equal(getattr(first, name), getattr(again, name))
    assert np.array_equal(first.Release_State[:20], prefix.Release_State) and np.array_equal(first.Target[:20], prefix.Target)
    area = Target_Area(catch, setup[0])
    assert isinstance(area, Cuboid_Cls)
    assert area.Centre.tolist() == setup[0].Areas.Catch.Centre
    assert area.Size.tolist() == pytest.approx([v + 0.2 for v in setup[0].Areas.Catch.Size], abs=1e-12)


def test_the_shooting_reaches_each_target_at_its_flight_time(setup, small):
    """
    Description:
        The flight model's state function from each release state passes through its target at its flight
        time within 1e-9 m (SPEC §7.6); Simulate_At equals Simulate row by row (test/Flight/test_core.py).
    """

    p = Simulate_At(small.Release_State, small.Flight_Time, Flight_Model_Str=setup[2])[:, :3]
    assert np.all(np.linalg.norm(p - small.Target, axis=1) < CONST_MISS)


def test_the_truth_is_every_frame_from_the_release_to_the_floor(setup, small):
    """
    Description:
        The truth (SPEC §2.13.4, §7.6): at the floor time the ball centre is at z = 0.030 m within 1e-9 m
        (every flight); the truth holds every frame j with j / rate <= the floor time, each equal to the state
        function at j / rate (the first 10 flights).
    """

    z = Simulate_At(small.Release_State, small.Floor_Time, Flight_Model_Str=setup[2])[:, 2]
    assert np.all(np.abs(z - CONST_FLOOR) < 1e-9)
    for i in range(10):
        t, p = small.Truth(i)
        assert len(t) == int(np.floor(small.Floor_Time[i] * small.Rate[i] + 1e-9)) + 1 and t[-1] <= small.Floor_Time[i] + 1e-12
        assert np.array_equal(p, Simulate(small.Release_State[i], t, Flight_Model_Str=setup[2])[:, :3]), i


def test_the_samples_are_stored_as_observed(small):
    """
    Description:
        Samples as observed (SPEC §2.13.4): every observed time is a whole frame of its flight's rate (its
        frame / rate, exactly), the release frame is observed, and the steps are non-uniform (dropouts) in at
        least 90 % of the flights.
    """

    nonuniform = 0
    for i in range(small.Flights):
        t, _ = small.Observations(i)
        a, b = small.Observed_Offset[i], small.Observed_Offset[i + 1]
        frame = small.Observed_Frame[a:b]
        assert np.array_equal(t, frame / small.Rate[i]) and frame[0] == 0
        nonuniform += len(np.unique(np.diff(frame))) > 1
    assert nonuniform >= 0.9 * small.Flights


def test_without_dropouts_and_error_the_observations_are_the_truth(setup):
    """
    Description:
        With Observation.Dropout.Pattern none and Observation.Noise.Scale 0, every frame is observed and
        equals the truth exactly.
    """

    data = _Generate(setup, Flights=10, Pattern='none', Noise_Scale=0.0)
    for i in range(data.Flights):
        assert np.array_equal(data.Observations(i)[1], data.Truth(i)[1]), i


def test_the_first_flights_do_not_depend_on_the_dataset_size(setup, small):
    """
    Description:
        The stable prefix (SPEC §2.13.3, small): the first 40 flights of an 80-flight dataset equal the
        40-flight dataset in every per-flight array (the partitions excepted: index blocks, F6 Q10).
    """

    large = _Generate(setup, Flights=80)
    for name in ('Rate', 'Release_State', 'Target', 'Apex_Drawn', 'Flight_Time', 'Floor_Time', 'Draws'):
        assert np.array_equal(getattr(large, name)[:40], getattr(small, name)), name
    for name in ('Truth', 'Observations'):
        for i in range(40):
            for x, y in zip(getattr(large, name)(i), getattr(small, name)(i)):
                assert np.array_equal(x, y), (name, i)


def test_a_regenerated_dataset_is_byte_identical(setup, small, tmp_path):
    """
    Description:
        Byte-for-byte reproducibility (SPEC §2.13.2, small): the same configuration and seed give a
        flights.npz with the same SHA-256; read back, it holds the same arrays.
    """

    Write_Dataset(str(tmp_path / 'a.npz'), small)
    Write_Dataset(str(tmp_path / 'b.npz'), _Generate(setup, Flights=40))
    digest = [hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in ('a.npz', 'b.npz')]
    assert digest[0] == digest[1]
    back = Read_Dataset(str(tmp_path / 'a.npz'))
    assert np.array_equal(back.Observed_Position, small.Observed_Position) and back.Donor_Name.tolist() == small.Donor_Name.tolist()


def test_another_format_version_is_refused(small, tmp_path):
    """
    Description:
        A flights.npz of another format version is refused, naming both versions (SPEC §9.5).
    """

    path = tmp_path / 'other.npz'
    Write_Dataset(str(path), small)
    with np.load(str(path)) as file:
        arrays = {key: file[key] for key in file.files}
    arrays['format_version'] = np.array(2, dtype=np.int64)
    np.savez(str(path), **arrays)
    with pytest.raises(Dataset_Error_Cls, match='format version 2; this code reads version 1'):
        Read_Dataset(str(path))


def test_a_rate_without_donors_is_refused_before_generating(setup):
    """
    Description:
        A rate without donors (150 Hz) is refused before anything is generated, naming it.
    """

    with pytest.raises(Dataset_Error_Cls, match='150 Hz has no donors'):
        _Generate(setup, Flights=5, Rates=[200.0, 150.0])


def test_370_hz_is_an_option_on_the_300_hz_donors(setup):
    """
    Description:
        370 Hz is an optional rate: its flights are on the 370 Hz frame grid and
        replay only 300 Hz donors.
    """

    data = _Generate(setup, Flights=10, Rates=[370.0])
    assert np.all(data.Rate == 370.0) and data.Donor_Name.size > 0
    assert all(name.startswith('Frequency_300_Hz_') for name in data.Donor_Name)
    for i in range(data.Flights):
        t, _ = data.Observations(i)
        assert np.allclose(t * 370.0, np.round(t * 370.0), atol=1e-9)


def test_a_donor_pool_replays_its_own_donors_only(setup):
    """
    Description:
        Observation.Dropout.Donors A (F6 Q11): every replayed donor is in pool A.
    """

    pool_a = {d.Name for d in setup[4] if d.Pool == 'A'}
    data = _Generate(setup, Flights=10, Donors='A')
    assert data.Donor_Name.size > 0 and set(data.Donor_Name.tolist()) <= pool_a


def test_a_target_area_not_facing_the_thrower_is_refused(setup):
    """
    Description:
        The targets lie in the half of Throw.Target_Area's sphere facing the thrower at -y: turned by +90 deg about z
        instead of -90 deg it keeps the half at +y and is refused,
        naming the field.
    """

    configuration, dataset = setup[0], setup[1]
    turned = replace(dataset, Throw=replace(dataset.Throw, Target_Area=replace(dataset.Throw.Target_Area,
                                                                              Orientation=[0.0, 0.0, np.pi / 2.0])))
    assert Target_Area(dataset, configuration).Shape == 'quarter_sphere'
    with pytest.raises(Dataset_Error_Cls, match='Throw.Target_Area does not face the thrower'):
        Target_Area(turned, configuration)


def test_throws_that_cannot_be_drawn_are_refused(setup):
    """
    Description:
        A flight-time range no apex can give (0.10-0.11 s) is refused after its draws, never generated with
        another distribution.
    """

    configuration, dataset, flight, observation, donors = setup
    narrow = replace(dataset, Flights=2, Throw=replace(dataset.Throw, Flight_Time=[0.10, 0.11]))
    with pytest.raises(Dataset_Error_Cls, match='No throw candidate met Throw.Flight_Time'):
        Generate_Dataset(narrow, configuration, donors, Flight_Model_Str=flight, Observation_Model_Str=observation)


@pytest.mark.slow
def test_the_first_1000_of_2000_flights_equal_a_1000_flight_dataset(setup):
    """
    Description:
        The stable prefix, SPEC §2.13.3 as written: the first 1 000 flights of a 2 000-flight dataset equal a
        1 000-flight dataset in every per-flight array and every flight's truth and observations.
    """

    one, two = _Generate(setup, Flights=1000), _Generate(setup, Flights=2000)
    for name in ('Rate', 'Release_State', 'Target', 'Apex_Drawn', 'Flight_Time', 'Floor_Time', 'Draws'):
        assert np.array_equal(getattr(two, name)[:1000], getattr(one, name)), name
    for name, offset in (('Truth_Position', 'Truth_Offset'), ('Observed_Position', 'Observed_Offset'), ('Observed_Frame', 'Observed_Offset')):
        assert np.array_equal(getattr(two, name)[:getattr(two, offset)[1000]], getattr(one, name)), name


def _Largest_Error(data):
    frames = np.concatenate([data.Truth_Offset[i] + data.Observed_Frame[data.Observed_Offset[i]:data.Observed_Offset[i + 1]]
                             for i in range(data.Flights)])
    return float(np.linalg.norm(data.Observed_Position - data.Truth_Position[frames], axis=1).max())


@pytest.mark.slow
def test_no_flight_of_1000_carries_an_observation_error_above_100_mm(setup):
    """
    Description:
        The acceptance of the bounds: in 1 000 flights of the shipped
        configuration no observed position is more than 0.1 m from the truth of its frame (the offset is bounded
        at 0.070 m, the white noise at 0.040 m; the largest real residual is 55.9 mm), at Observation.Noise.Scale
        1. The same flights from the model without the bounds carry errors above 0.1 m: the case the bounds
        exist for.
    """

    observation = setup[3]
    assert (observation.Offset_Bound, observation.White_Bound) == (0.070, 0.040)
    assert _Largest_Error(_Generate(setup, Flights=1000)) <= 0.1
    assert _Largest_Error(_Generate(setup, model=replace(observation, Offset_Bound=np.inf, White_Bound=np.inf), Flights=1000)) > 0.1


@pytest.mark.slow
def test_a_1000_flight_dataset_is_reproduced_byte_for_byte(setup, tmp_path):
    """
    Description:
        Byte-for-byte reproducibility, SPEC §2.13.2: two 1 000-flight datasets from the same configuration and
        seed have the same SHA-256.
    """

    for name in ('a.npz', 'b.npz'):
        Write_Dataset(str(tmp_path / name), _Generate(setup, Flights=1000))
    assert hashlib.sha256((tmp_path / 'a.npz').read_bytes()).hexdigest() == hashlib.sha256((tmp_path / 'b.npz').read_bytes()).hexdigest()


def test_a_partition_is_selected_by_its_code_and_an_unknown_one_is_refused():
    """
    Description:
        Partition_Indices (the one selector of a partition, SPEC §2.13.7-13.8): of the codes 0 0 1 2 0 1, code 0
        (train) is flights 0, 1, 4, code 1 (validation) 2, 5 and code 2 (test) 3; code 3 is refused.
    """

    from Dataset.Core import Dataset_Str
    data = Dataset_Str(Partition=np.array([0, 0, 1, 2, 0, 1], dtype=np.int8), Rate=np.full(6, 200.0))
    assert [list(Partition_Indices(data, code)) for code in (0, 1, 2)] == [[0, 1, 4], [2, 5], [3]]
    with pytest.raises(Dataset_Error_Cls, match='0 \\(train\\), 1 \\(validation\\), 2'):
        Partition_Indices(data, 3)
