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
# File Name: ../test/View/test_core.py
## =========================================================================== ##

# Dataclasses (A record changed in a copy)
import dataclasses
# System (Default)
import os
# Struct (The PNG's chunks, read back)
import struct
# Zlib (The PNG's image data and checksums, read back)
import zlib
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# MuJoCo (Physics simulation) [pip3 install mujoco]
import mujoco
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Dataset/Core - a small dataset's file
from Dataset.Core import Dataset_Str, Write_Dataset
#   ../Catch/Interception - the setup a record is computed with
from Catch.Interception import Interception_Setup_Str
#   ../Catch/Physical - the catch's settings
from Catch.Physical import Contact_Str
#   ../View/Core
import View.Core as View_Core
from View.Core import (Partition_Code, Pick_Throw, Method_Colour, Record_Throw, Frame_At, Frame_Times, Overlay_Lines, Legend, Decorations, Draw,
                       Zone_Box, Add_Ghost, Ghost_Positions, Set_Camera, Legend_Figure, Offscreen_Cls, Write_Png, Frame_Row, Write_Frames,
                       Throw_Record_Str, Camera_Str, Decoration_Str, View_Error_Cls, CONST_THEMES, CONST_DEFAULT_THEME, CONST_CAMERAS,
                       CONST_DEFAULT_CAMERA, CONST_ZONES, CONST_DIAGRAM, CONST_FRAME_COLUMNS, CONST_SAFETY)
#   ../View/Theme - the themed scene shown
from View.Theme import Themed_Scene

"""
Description:
    Initialization of constants.

    The view of one throw (SPEC §7.13): the choice of a throw and the refusal of a test partition,
    what a record shows at a time, the decorations in the diagram's language, the overlay, the camera presets, the
    offscreen frame and the frames' two files. The records are constructed here (a ball on a known arc, 30 samples, a
    go at the 24th), so the expectations come from the construction and from the owner's diagram (its colours as
    sampled in Part A: the throw zone #FF0000, the flight zone #4576C4, the catch zone #6CB145, the prediction's
    marker #70AD47), never from the code under test. The record computed by the catch's own code is tested, in the
    slow tier, in test/scripts/test_catch.py.
"""
# The diagram's colours (sRGB, 0-255), as sampled from the owner's image, its file's name and its hash.
#   Unit [Dict<string, Tuple<int, int, int>>], [Tuple<int, int, int>], [Tuple<string, string>]
CONST_DIAGRAM_ZONES = {'Throw': (255, 0, 0), 'Flight': (69, 118, 196), 'Catch': (108, 177, 69)}
CONST_DIAGRAM_GREEN = (112, 173, 71)
CONST_DIAGRAM_FILE = ('1795373317963_image.png', 'd44d0f66d5ce2fbcff732a177d846a47894911b49305f86052736732c084f3d3')
# The constructed throw: 30 samples every 0.01 s from 0.01 s, the first decision sample the 20th, the go at the 24th,
# the catch time 0.40 s, the record's end 0.50 s; three predictions shown, at N 20, 24 and 28.
#   Unit [int], [int], [float, s], [float, s]
CONST_FIRST_N = 20
CONST_GO_N = 24
CONST_CATCH_TIME = 0.40
CONST_END = 0.50


def _Arc(t):
    """
    Description:
        The constructed ball's position at the times t: along +y at 5 m/s from (0.7, -2, 1), rising and falling.
    """

    t = np.asarray(t, dtype=np.float64)
    return np.stack([np.full(t.shape, 0.7), -2.0 + 5.0 * t, 1.0 + 2.0 * t - 4.905 * t ** 2], axis=-1)


def _Record(go=True, methods=('QT_BASIS', 'GLS_M4')):
    """
    Description:
        A record constructed by hand: the truth on _Arc every 0.002 s to 0.5 s, the samples on it, the predictions at
        N 20, 24 and 28 (each path from its sample to the catch time once the go is made, 5 mm beside the truth for a
        comparison; the comparison refused at N 24), two plans starting at 0.26 and 0.31 s, and 251 states of 13
        positions (the robot's six joints at the addresses 7 to 12, the ball at 0 to 2); the go's ball is inside the cup
        from 0.42 to 0.44 s and from 0.46 s on, above its mouth between.
    """

    truth_time = np.arange(0.0, CONST_END + 1e-9, 0.002)
    observed_time = 0.01 * np.arange(1, 31)
    record = Throw_Record_Str(Dataset='Catch_In', Flight=7, Seed='0', Acceleration=np.radians(800.0), Method=methods[0],
                              Truth_Time=truth_time, Truth_Position=_Arc(truth_time), Observed_Time=observed_time,
                              Observed_Position=_Arc(observed_time), First_N=CONST_FIRST_N, Methods=list(methods),
                              Sample_N=np.array([20, 24, 28]), Used=np.array([True, True, False]),
                              Decision='go' if go else 'no-go', Catchable=True, Decided_N=26, Robot_Address=np.arange(7, 13), Ball_Address=0)
    if go:
        record.Go_N, record.Go_Time, record.Catch_Time = CONST_GO_N, 0.24, CONST_CATCH_TIME
        record.Plan_Time, record.Plan_Goal = np.array([0.26, 0.31]), np.array([np.full(6, 0.1), np.full(6, 0.2)])
        record.Class, record.Outcome = 'caught', 'caught'
    else:
        record.Class, record.Outcome = 'no_go_catchable', 'missed'
    for k, name in enumerate(methods):
        paths, marked = np.full((3, 40, 3), np.nan), np.zeros(3, dtype=bool)
        for row, n in enumerate((20, 24, 28)):
            if k == 1 and n == 24:
                continue
            marked[row] = go and n >= CONST_GO_N
            end = CONST_CATCH_TIME if marked[row] else CONST_END
            paths[row] = _Arc(np.linspace(observed_time[n - 1], end, 40)) + (0.0, 0.0, 0.005 * k)
        record.Paths[name], record.Catch_Point[name] = paths, marked
        record.Inference[name] = np.array([0.0008, 0.0012, 0.0010]) * (50.0 if k else 1.0)
    record.State_Time = truth_time
    record.State_Position = np.zeros((len(truth_time), 13))
    record.State_Position[:, 0:3], record.State_Position[:, 3] = _Arc(truth_time), 1.0
    if go:
        record.State_In_Cup = ((truth_time >= 0.42 - 1e-9) & (truth_time < 0.44 - 1e-9)) | (truth_time >= 0.46 - 1e-9)
    return record


@pytest.fixture(scope='module')
def themed(configuration):
    """
    Description:
        The scene with the default theme and an offscreen buffer of 320 x 180 px, built once for this module.
    """

    return Themed_Scene(Areas_Str=configuration.Areas, Theme_Str=CONST_THEMES[CONST_DEFAULT_THEME], width=320, height=180)


def test_a_throws_partition_comes_from_the_sizes_and_a_test_throw_is_refused():
    """
    Description:
        The partitions are index blocks (train, validation, test), so a throw's partition follows from the three sizes
        alone, before the dataset's file is opened: with the sizes 10 / 45 / 45 the throw 9 is a train throw, 10 and 54
        validation throws, 55 a test throw. A test throw is refused naming the allowed range (0 to 54), as are an index
        outside the dataset and sizes that are not three counts; the last allowed throw is given back.
    """

    sizes = [10, 45, 45]
    assert [Partition_Code(index, sizes) for index in (0, 9, 10, 54, 55, 99)] == [0, 0, 1, 1, 2, 2]
    assert Pick_Throw(sizes, index=54) == 54 and Pick_Throw(sizes, index=0) == 0
    with pytest.raises(View_Error_Cls) as error:
        Pick_Throw(sizes, index=55)
    assert 'test partition' in str(error.value) and '0 to 54' in str(error.value) and 'counted commands' in str(error.value)
    for index in (-1, 100):
        with pytest.raises(View_Error_Cls, match='is not in the dataset'):
            Pick_Throw(sizes, index=index)
    with pytest.raises(View_Error_Cls, match='three counts'):
        Partition_Code(0, [10, 90])
    with pytest.raises(View_Error_Cls, match='No throw is named'):
        Pick_Throw(sizes)


def test_a_random_throw_is_a_train_or_validation_throw_and_the_same_for_the_same_seed():
    """
    Description:
        A throw picked with a seed is never a test throw (the sizes 10 / 45 / 45: every pick of 300 seeds is below 55),
        covers the train and the validation partition, and is the same for the same seed; a dataset without a train or
        validation throw has none to pick.
    """

    sizes = [10, 45, 45]
    picks = [Pick_Throw(sizes, pick_seed=seed) for seed in range(300)]
    assert max(picks) <= 54 and min(picks) >= 0 and min(picks) < 10 and max(picks) >= 10 and len(set(picks)) > 40
    assert picks == [Pick_Throw(sizes, pick_seed=seed) for seed in range(300)]
    with pytest.raises(View_Error_Cls, match='no train or validation throw'):
        Pick_Throw([0, 0, 5], pick_seed=1)


def test_a_test_throw_is_refused_before_anything_is_computed(tmp_path, monkeypatch):
    """
    Description:
        Record_Throw itself refuses a throw whose stored partition is the test partition, before the predictor is
        loaded and the interception computed (both replaced by functions that fail), naming the throw and the rule; a
        throw outside the dataset, an acceleration that is not configured and a display step below 1 are refused too.
    """

    data = Dataset_Str(Partition=np.array([0, 1, 2], dtype=np.int8), Rate=np.full(3, 100.0), Release_State=np.zeros((3, 6)), Target=np.zeros((3, 3)),
                       Apex_Drawn=np.zeros(3), Flight_Time=np.ones(3), Floor_Time=np.ones(3), Draws=np.ones(3, dtype=np.int64),
                       Truth_Offset=np.arange(4), Truth_Position=np.zeros((3, 3)), Observed_Offset=np.arange(4),
                       Observed_Frame=np.zeros(3, dtype=np.int64), Observed_Time=np.zeros(3), Observed_Position=np.zeros((3, 3)),
                       Donor_Offset=np.arange(4), Donor_Name=np.array(['a', 'b', 'c']))
    path = str(tmp_path / 'flights.npz')
    Write_Dataset(path, data)

    def Fail(*args, **kwargs):
        raise AssertionError('the throw was computed')

    monkeypatch.setattr(View_Core, 'Intercept_Chunk', Fail)
    monkeypatch.setattr(View_Core, 'Run_Predictor', Fail)
    setup = Interception_Setup_Str(Accelerations=[1.0, 2.0], Stages=[20])
    contact = Contact_Str(Levels=[(0.004, 0.7)])
    with pytest.raises(View_Error_Cls) as error:
        Record_Throw(setup, 'Small', path, 2, '', 0, contact)
    assert 'The throw 2 of Small is in its test partition' in str(error.value) and 'counted commands' in str(error.value)
    with pytest.raises(View_Error_Cls, match='The throw 3 is not in the dataset Small'):
        Record_Throw(setup, 'Small', path, 3, '', 0, contact)
    with pytest.raises(View_Error_Cls, match='The acceleration 2 is not one of the 2'):
        Record_Throw(setup, 'Small', path, 0, '', 2, contact)
    with pytest.raises(View_Error_Cls, match='K >= 1, found 0'):
        Record_Throw(setup, 'Small', path, 0, '', 0, contact, every=0)
    with pytest.raises(AssertionError, match='the throw was computed'):
        Record_Throw(setup, 'Small', path, 1, '', 0, contact)


def test_a_frame_shows_the_samples_seen_the_newest_prediction_and_the_decisions_phase():
    """
    Description:
        What the constructed go shows at a time: at 0.195 s 19 samples are seen, the phase is 'waiting' and no
        prediction is shown; at 0.20 s the 20th sample (the first decision sample) is seen, the phase is 'deciding'
        and the prediction of N 20 is the newest; at 0.24 s (the go's sample) 'GO' with the prediction of N 24, at
        0.27 s the first plan is executed, at 0.35 s the second, the newest prediction that of N 28; the class is
        empty until the record's last time, 0.50 s. The state shown is the last at or before the time.
    """

    record = _Record()
    frames = {t: Frame_At(record, t) for t in (0.0, 0.195, 0.20, 0.239, 0.24, 0.27, 0.35, 0.499, 0.50, 9.0)}
    assert [(frames[t].Seen, frames[t].Phase, frames[t].Prediction) for t in (0.0, 0.195, 0.20, 0.239, 0.24)] == \
        [(0, 'waiting', -1), (19, 'waiting', -1), (20, 'deciding', 0), (23, 'deciding', 0), (24, 'GO', 1)]
    assert [(frames[t].Plan, frames[t].Prediction) for t in (0.24, 0.27, 0.35)] == [(-1, 1), (0, 1), (1, 2)]
    assert [frames[t].Outcome for t in (0.35, 0.499, 0.50, 9.0)] == ['', '', 'caught', 'caught']
    assert [frames[t].State for t in (0.0, 0.195, 0.20, 0.50, 9.0)] == [0, 97, 100, 250, 250]


def test_a_no_go_is_final_when_the_decisions_end_and_shows_its_class_then():
    """
    Description:
        The constructed no-go: 'waiting' before the 20th sample, 'deciding' until the last sample the pipeline decided
        at is seen (the 26th, at 0.26 s), then 'NO-GO' with its class, no plan ever.
    """

    record = _Record(go=False)
    shown = [(Frame_At(record, t).Phase, Frame_At(record, t).Outcome, Frame_At(record, t).Plan) for t in (0.1, 0.20, 0.255, 0.26, 0.5)]
    assert shown == [('waiting', '', -1), ('deciding', '', -1), ('deciding', '', -1), ('NO-GO', 'no_go_catchable', -1),
                     ('NO-GO', 'no_go_catchable', -1)]


def test_the_decorations_follow_the_diagram(configuration):
    """
    Description:
        The diagram's language on the constructed go, in the side preset. At 0.15 s (15 samples seen): 15 red spheres on
        the samples joined by 14 red lines, no marker, no prediction, the true path ahead as dots 0.05 m apart, none
        behind the ball. At 0.30 s (30 samples, the go made): the 20 samples of the detection phase stay red with 19
        lines and the red marker "Throw Detection" at the 20th sample; each of the 10 later samples is drawn, in the
        theme's colour for them (a gap in them is then a dropout, never the drawing); the driving method's prediction
        of N 28 is 39 lines in the diagram's green
        ending in the green marker "Catch Prediction" at its path's last point, the comparison's in its own colour
        with an unlabelled, smaller marker. The zones are the three areas of the configuration as a box and its edges
        each, in the diagram's colours, labelled (the flight zone from the throw area's near face, y -4.04 to -1.00 m);
        without zones or labels they are left out.
    """

    theme, camera = CONST_THEMES['dark'], CONST_CAMERAS['side']
    record = _Record()
    early = Decorations(record, Frame_At(record, 0.15), Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=camera, zones=False)
    red = (1.0, 0.0, 0.0, 1.0)
    spheres = [d for d in early if d.Kind == 'sphere' and tuple(d.Colour) == red]
    lines = [d for d in early if d.Kind == 'line']
    assert len(spheres) == 15 and len(lines) == 14 and all(tuple(d.Colour) == red for d in lines) and not any(d.Label for d in early)
    assert np.allclose([d.Position for d in spheres], _Arc(0.01 * np.arange(1, 16))) and np.allclose(lines[3].Extent, _Arc(0.05))
    dots = [d for d in early if d.Kind == 'sphere' and tuple(d.Colour) != red]
    assert len(dots) > 20 and all(d.Position[1] >= _Arc(0.15)[1] for d in dots)
    gaps = np.linalg.norm(np.diff([d.Position for d in dots], axis=0), axis=1)
    assert np.all(np.abs(gaps - 0.05) < 0.012)

    late = Decorations(record, Frame_At(record, 0.30), Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=camera)
    labelled = {d.Label: d for d in late if d.Label}
    assert set(labelled) == {'Throw Detection', 'Catch Prediction', 'Throw Area', 'Flight Area', 'Catch Area'}
    detection, prediction = labelled['Throw Detection'], labelled['Catch Prediction']
    assert tuple(detection.Colour) == red and np.allclose(detection.Position, _Arc(0.20)) and detection.Radius == 0.030
    green = tuple(value / 255.0 for value in CONST_DIAGRAM_GREEN) + (1.0,)
    assert np.allclose(prediction.Colour, green) and np.allclose(prediction.Position, _Arc(CONST_CATCH_TIME)) and prediction.Radius == 0.030
    assert sum(1 for d in late if d.Kind == 'sphere' and tuple(d.Colour) == red and d.Radius < 0.02) == 20
    assert sum(1 for d in late if d.Kind == 'line' and tuple(d.Colour) == red) == 19
    neutral = [d for d in late if d.Kind == 'sphere' and np.allclose(d.Colour[:3], theme.Samples) and d.Radius == 0.006]
    assert np.allclose([d.Position for d in neutral], _Arc(0.01 * np.arange(21, 31)))
    driving = [d for d in late if d.Kind == 'line' and np.allclose(d.Colour, green)]
    other = [d for d in late if d.Kind == 'line' and tuple(d.Colour) not in (red,) and not np.allclose(d.Colour, green)]
    assert len(driving) == 39 and len(other) == 39 and driving[0].Radius > other[0].Radius
    assert np.allclose(driving[0].Position, _Arc(0.28)) and np.allclose(driving[-1].Extent, _Arc(CONST_CATCH_TIME))
    comparison = [d for d in late if d.Kind == 'sphere' and d.Radius == 0.018]
    assert len(comparison) == 1 and comparison[0].Label == '' and np.allclose(comparison[0].Position, _Arc(CONST_CATCH_TIME) + (0, 0, 0.005))
    assert late.index(other[0]) < late.index(driving[0])
    for name, colour in CONST_DIAGRAM_ZONES.items():
        area = getattr(configuration.Areas, name)
        box = [d for d in late if d.Kind == 'box' and np.allclose(d.Colour[:3], np.array(colour) / 255.0)]
        frame = [d for d in late if d.Kind == 'frame' and np.allclose(d.Colour[:3], np.array(colour) / 255.0)]
        assert len(box) == 1 and len(frame) == 1 and box[0].Colour[3] == theme.Zone_Alpha and frame[0].Colour[3] == theme.Edge_Alpha
        centre, half = np.array(area.Centre, dtype=np.float64), 0.5 * np.array(area.Size)
        if name == 'Flight':
            centre[1], half[1] = 0.5 * (-4.04 - 1.00), 0.5 * (-1.00 + 4.04)
        assert np.allclose(box[0].Position, centre) and np.allclose(box[0].Extent, half)
        assert np.allclose(frame[0].Position, centre) and np.allclose(frame[0].Extent, half)
        assert np.allclose(labelled[name + ' Area'].Position, centre + (0.0, 0.0, half[2]))
    bare = Decorations(record, Frame_At(record, 0.30), Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=camera, zones=False,
                       labels=False)
    assert not any(d.Kind in ('box', 'frame') for d in bare) and not any(d.Label for d in bare) and len(bare) == len(late) - 9


def test_a_prediction_without_a_catch_time_or_refused_has_no_marker(configuration):
    """
    Description:
        Before the go (0.22 s, the prediction of N 20) both methods' paths are drawn to their own ends without a marker;
        at 0.25 s (N 24) the comparison's prediction was refused and is not drawn, the driving method's ends in its
        marker; a no-go never has a marker. A method without a colour of its own takes the first spare colour.
    """

    theme, camera, record = CONST_THEMES['light'], CONST_CAMERAS['overview'], _Record()
    draw = (lambda rec, t: Decorations(rec, Frame_At(rec, t), Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=camera, zones=False))
    before = draw(record, 0.22)
    assert sum(1 for d in before if d.Kind == 'line' and d.Colour[0] != 1.0) == 78 and not any(d.Label == 'Catch Prediction' for d in before)
    assert not any(d.Kind == 'sphere' and d.Radius in (0.030, 0.018) and d.Colour[0] != 1.0 for d in before)
    refused = draw(record, 0.25)
    assert sum(1 for d in refused if d.Kind == 'line' and d.Colour[0] != 1.0) == 39 and any(d.Label == 'Catch Prediction' for d in refused)
    no_go = _Record(go=False)
    assert not any(d.Label == 'Catch Prediction' for t in (0.22, 0.25, 0.3) for d in draw(no_go, t))
    assert Method_Colour('QT_BASIS', ['QT_BASIS', 'GLS_M4']) == CONST_DIAGRAM_GREEN
    assert Method_Colour('GLS_M4', ['QT_BASIS', 'GLS_M4']) != CONST_DIAGRAM_GREEN
    spare = Method_Colour('Other_Run', ['QT_BASIS', 'Other_Run', 'Third'])
    assert spare not in (CONST_DIAGRAM_GREEN, Method_Colour('GLS_M4', [])) and spare != Method_Colour('Third', ['QT_BASIS', 'Other_Run', 'Third'])


def test_the_legend_names_each_method_the_observed_samples_and_the_true_path(configuration):
    """
    Description:
        The legend: each method with its colour, the driving one first, then "observed samples" and
        "true path", so that a viewer knows what the dots are. Their colours are the ones drawn: the later samples'
        spheres and the true path's dots of the decorations have exactly the legend's two colours; the two differ
        clearly in each theme (by at least 60 of 255 in every channel), the samples the stronger tone on the theme's
        background.
    """

    record = _Record()
    for name, strongest in (('dark', max), ('light', min)):
        theme = CONST_THEMES[name]
        entries = Legend(record, Theme_Str=theme)
        assert [entry[0] for entry in entries] == ['QT_BASIS', 'GLS_M4', 'observed samples', 'true path']
        assert entries[0][1] == CONST_DIAGRAM_GREEN and entries[1][1] == Method_Colour('GLS_M4', ['QT_BASIS', 'GLS_M4'])
        samples, truth = np.array(entries[2][1]), np.array(entries[3][1])
        assert np.all(np.abs(samples - truth) >= 60) and strongest(samples[0], truth[0]) == samples[0]
        shown = Decorations(record, Frame_At(record, 0.30), Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=CONST_CAMERAS['side'],
                            zones=False)
        later = [d for d in shown if d.Kind == 'sphere' and d.Radius == 0.006]
        dots = [d for d in shown if d.Kind == 'sphere' and d.Radius == 0.0045]
        assert len(later) == 10 and len(dots) > 3
        assert all(np.allclose(np.array(d.Colour[:3]) * 255.0, samples, atol=0.51) for d in later)
        assert all(np.allclose(np.array(d.Colour[:3]) * 255.0, truth, atol=0.51) for d in dots)


def test_after_the_catch_time_the_overlay_says_whether_the_ball_is_in_the_cup():
    """
    Description:
        Between a go's catch time (0.40 s) and its class at the record's end (0.50 s) the outcome's row is no longer
        "-": the constructed ball is not yet in the cup at 0.41 s ("-"), inside from 0.42 s ("in
        cup (settling)"), above the mouth from 0.44 s ("out of cup": it has been inside), inside again from 0.46 s, and
        at 0.50 s the class is shown. Before the catch time the row stays "-" whatever the ball does; a no-go and a
        record without the per-state flag show "-" until their class.
    """

    record = _Record()
    rows = {t: (Frame_At(record, t).Settling, Overlay_Lines(record, Frame_At(record, t))[1].split('\n')[6])
            for t in (0.39, 0.40, 0.41, 0.42, 0.439, 0.44, 0.459, 0.46, 0.499, 0.50)}
    assert [rows[t] for t in (0.39, 0.40, 0.41)] == [('', '-')] * 3
    assert rows[0.42] == rows[0.439] == rows[0.46] == rows[0.499] == ('in cup (settling)', 'in cup (settling)')
    assert rows[0.44] == rows[0.459] == ('out of cup', 'out of cup') and rows[0.50] == ('', 'caught')
    early = _Record()
    early.State_In_Cup[:] = True
    assert Frame_At(early, 0.39).Settling == '' and Overlay_Lines(early, Frame_At(early, 0.39))[1].split('\n')[6] == '-'
    no_go = _Record(go=False)
    assert [Frame_At(no_go, t).Settling for t in (0.2, 0.45)] == ['', '']
    bare = _Record()
    bare.State_In_Cup = np.zeros(0, dtype=bool)
    assert Frame_At(bare, 0.45).Settling == '' and Overlay_Lines(bare, Frame_At(bare, 0.45))[1].split('\n')[6] == '-'


def test_the_flight_zone_is_drawn_from_the_throw_areas_near_face_and_the_configuration_is_unchanged(configuration):
    """
    Description:
        The configured Areas.Flight (y -5.00 to -1.00 m, the 2022 lab's) reaches 0.96 m into Areas.Throw (y -6.46 to
        -4.04 m, the measured releases' support). The view draws the flight zone from the throw area's face towards
        the robot, y -4.04 to -1.00 m (3.04 m deep), with its x and z as configured, as in the owner's diagram; the
        throw and the catch zone are drawn as configured. It is for the display only: the
        configuration's areas keep their values. A flight area that ends before the throw area's near face, or begins
        beyond it, is drawn as configured.
    """

    import copy
    areas = configuration.Areas
    assert (areas.Flight.Centre, areas.Flight.Size) == ([0.0, -3.0, 2.25], [4.0, 4.0, 3.5])
    assert (areas.Throw.Centre, areas.Throw.Size) == ([-0.125, -5.25, 1.275], [3.67, 2.42, 0.91])
    centre, half = Zone_Box('Flight', areas)
    assert np.allclose(centre, (0.0, -2.52, 2.25)) and np.allclose(half, (2.0, 1.52, 1.75))
    assert np.allclose(centre[1] - half[1], -4.04) and np.allclose(centre[1] + half[1], -1.00)
    for name in ('Throw', 'Catch'):
        area = getattr(areas, name)
        box = Zone_Box(name, areas)
        assert np.allclose(box[0], area.Centre) and np.allclose(box[1], 0.5 * np.array(area.Size))
    assert (areas.Flight.Centre, areas.Flight.Size) == ([0.0, -3.0, 2.25], [4.0, 4.0, 3.5])
    apart = copy.deepcopy(areas)
    apart.Flight.Centre, apart.Flight.Size = [0.0, -2.0, 2.25], [4.0, 2.0, 3.5]
    assert np.allclose(Zone_Box('Flight', apart)[0], (0.0, -2.0, 2.25)) and np.allclose(Zone_Box('Flight', apart)[1], (2.0, 1.0, 1.75))
    behind = copy.deepcopy(areas)
    behind.Flight.Centre, behind.Flight.Size = [0.0, -6.0, 2.25], [4.0, 2.0, 3.5]
    assert np.allclose(Zone_Box('Flight', behind)[0], (0.0, -6.0, 2.25)) and np.allclose(Zone_Box('Flight', behind)[1], (2.0, 1.0, 1.75))


def test_no_prediction_is_drawn_after_the_catch_time(configuration):
    """
    Description:
        The constructed go's catch time is 0.40 s: at 0.40 s the driving method's and the
        comparison's newest prediction are still drawn, each with its marker; from the next instant no prediction's line
        and no prediction's marker is drawn: the ball is at the cup. The samples, the detection's marker and the true
        path ahead stay. A no-go has no catch time: its predictions are drawn to the record's end.
    """

    theme, camera = CONST_THEMES['dark'], CONST_CAMERAS['side']
    draw = (lambda rec, t: Decorations(rec, Frame_At(rec, t), Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=camera, zones=False))
    predicted = (lambda shown: [d for d in shown if (d.Kind == 'line' and d.Colour[0] != 1.0) or d.Radius in (0.030, 0.018) and d.Colour[0] != 1.0])
    go = _Record()
    at = draw(go, CONST_CATCH_TIME)
    assert len(predicted(at)) == 2 * 39 + 2 and any(d.Label == 'Catch Prediction' for d in at)
    for t in (CONST_CATCH_TIME + 0.002, 0.45, CONST_END):
        after = draw(go, t)
        assert predicted(after) == [] and not any(d.Label == 'Catch Prediction' for d in after)
        assert any(d.Label == 'Throw Detection' for d in after) and sum(1 for d in after if d.Kind == 'line') == 19
    assert len([d for d in draw(go, 0.45) if d.Kind == 'sphere' and d.Radius == 0.0045]) > 3
    no_go = _Record(go=False)
    assert len(predicted(draw(no_go, 0.45))) == 2 * 39 and len(predicted(draw(no_go, CONST_END))) == 2 * 39


def test_a_camera_that_opens_or_hides_a_zone_leaves_its_faces_or_all_of_it_out(configuration):
    """
    Description:
        The thrower's preset stands inside the throw zone: that zone is not drawn at all (no box, no edges, no label)
        and the flight zone keeps its edges and its label without its faces, so that nothing tints the view; the catch
        zone is drawn whole. The side preset draws every zone whole. The presets are the four of the owner's list, the
        side one orthographic and the default; the diagram's file and colours are the ones sampled.
    """

    record, theme = _Record(), CONST_THEMES['dark']
    kinds = {}
    for name in ('thrower', 'side'):
        shown = Decorations(record, Frame_At(record, 0.1), Theme_Str=theme, Areas_Str=configuration.Areas, Camera_Str=CONST_CAMERAS[name])
        kinds[name] = (sum(d.Kind == 'box' for d in shown), sum(d.Kind == 'frame' for d in shown), sorted(d.Label for d in shown if d.Label))
    assert kinds['thrower'] == (1, 2, ['Catch Area', 'Flight Area']) and kinds['side'] == (3, 3, ['Catch Area', 'Flight Area', 'Throw Area'])
    assert set(CONST_CAMERAS) == {'side', 'overview', 'cup', 'thrower'} and CONST_DEFAULT_CAMERA == 'side'
    assert CONST_CAMERAS['side'].Extent > 0.0 and all(CONST_CAMERAS[name].Extent == 0.0 for name in ('overview', 'cup', 'thrower'))
    assert CONST_DIAGRAM == CONST_DIAGRAM_FILE and {name: colour for name, _, colour in CONST_ZONES} == CONST_DIAGRAM_ZONES
    assert [label for _, label, _ in CONST_ZONES] == ['Throw Area', 'Flight Area', 'Catch Area']
    assert CONST_DEFAULT_THEME == 'dark' and set(CONST_THEMES) == {'dark', 'light'}


def test_the_overlay_names_the_method_the_decision_and_this_runs_inference_time():
    """
    Description:
        The overlay's two columns at 0.35 s of the constructed go: eight rows; the method with its seed, the throw with
        its acceleration in deg/s2, 30 samples, the newest prediction's wall time 1.0 ms marked "(this run)", "GO at
        N 24", plan 2 of 2, no outcome yet, the time. Before the first decision sample: waiting, no inference time, no
        plan; at the end the class in words. Only ASCII, which MuJoCo's font holds.
    """

    record = _Record()
    left, right = Overlay_Lines(record, Frame_At(record, 0.35))
    assert left.split('\n') == ['method', 'throw', 'samples N', 'inference', 'decision', 'plan', 'outcome', 'time']
    assert right.split('\n') == ['QT_BASIS (seed 0)', 'Catch_In 7, 800 deg/s2', '30', '1.0 ms (this run)', 'GO at N 24', '2 of 2', '-', '0.350 s']
    early = Overlay_Lines(record, Frame_At(record, 0.05))[1].split('\n')
    assert early[2:7] == ['5', '-', 'waiting', '-', '-']
    assert Overlay_Lines(record, Frame_At(record, 0.5))[1].split('\n')[6] == 'caught'
    no_go = _Record(go=False)
    assert Overlay_Lines(no_go, Frame_At(no_go, 0.3))[1].split('\n')[4:7] == ['NO-GO', '-', 'no go catchable']
    assert (left + right + CONST_SAFETY).isascii() and 'nothing is run on, or validated on, the real UR10e' in CONST_SAFETY


def test_a_moving_catchs_overlay_names_the_stroke_its_scale_and_the_two_speeds():
    """
    Description:
        F15: a moving catch's record has three more rows before the outcome: the catch (the
        mode with the stroke's scale), the cup's speed along the ball and the relative speed at the crossing, both shown
        from the stroke's start on and "-" before it; while the last plan, the stroke, runs the plan's row says "the
        stroke". A go of a moving catch that fell back to a static plan says so, keeps "2 of 2" and shows no speed; a
        no-go says "moving". A static record has none of the three rows.
    """

    record = dataclasses.replace(_Record(), Moving=True, Stroke=True, Stroke_Scale=0.75, Stroke_Time=0.31, Cup_Speed=1.34, Relative_Speed=5.67)
    left, right = Overlay_Lines(record, Frame_At(record, 0.35))
    assert left.split('\n') == ['method', 'throw', 'samples N', 'inference', 'decision', 'plan', 'catch', 'cup speed', 'relative speed', 'outcome',
                                'time']
    assert right.split('\n')[5:] == ['the stroke', 'moving, stroke x 0.75', '1.34 m/s', '5.67 m/s', '-', '0.350 s']
    assert Overlay_Lines(record, Frame_At(record, 0.28))[1].split('\n')[5:9] == ['1 of 2', 'moving, stroke x 0.75', '-', '-']
    fallback = dataclasses.replace(_Record(), Moving=True)
    assert Overlay_Lines(fallback, Frame_At(fallback, 0.35))[1].split('\n')[5:9] == ['2 of 2', 'moving, a static plan', '-', '-']
    no_go = dataclasses.replace(_Record(go=False), Moving=True)
    assert Overlay_Lines(no_go, Frame_At(no_go, 0.3))[1].split('\n')[5:9] == ['-', 'moving', '-', '-']
    assert 'catch' not in Overlay_Lines(_Record(), Frame_At(_Record(), 0.35))[0].split('\n') and (left + right).isascii()


def test_the_frames_times_follow_the_video_rate_and_the_speed():
    """
    Description:
        Frames for a 60 Hz video at a tenth of the real speed are 1/600 s of the throw apart: from 0.30 to 0.31 s that
        is 7 frames; at the real speed over the whole record (0 to 0.5 s) 31 frames, the last at 0.5 s. A rate or a
        speed of 0 and a first time after the last are refused.
    """

    record = _Record()
    slow = Frame_Times(record, 60.0, 0.1, 0.30, 0.31)
    assert len(slow) == 7 and np.allclose(slow, 0.30 + np.arange(7) / 600.0)
    whole = Frame_Times(record, 60.0, 1.0)
    assert len(whole) == 31 and whole[0] == 0.0 and abs(whole[-1] - 0.5) < 1e-12
    for fps, speed, start, end in ((0.0, 1.0, None, None), (60.0, 0.0, None, None), (60.0, 1.0, 0.4, 0.3)):
        with pytest.raises(View_Error_Cls):
            Frame_Times(record, fps, speed, start, end)


def _Read_Png(path):
    """
    Description:
        A PNG file read back by the specification (not by the writer under test): the signature, each chunk's length,
        type, data and CRC-32, the header's fields, and the pixels from the inflated, unfiltered rows.
    """

    with open(path, 'rb') as file:
        raw = file.read()
    assert raw[:8] == bytes([137, 80, 78, 71, 13, 10, 26, 10])
    chunks, at = [], 8
    while at < len(raw):
        length, kind = struct.unpack('>I', raw[at:at + 4])[0], raw[at + 4:at + 8]
        payload = raw[at + 8:at + 8 + length]
        assert struct.unpack('>I', raw[at + 8 + length:at + 12 + length])[0] == zlib.crc32(kind + payload) & 0xffffffff
        chunks.append((kind, payload))
        at += 12 + length
    assert [kind for kind, _ in chunks] == [b'IHDR', b'IDAT', b'IEND'] and chunks[2][1] == b''
    width, height, depth, colour, compression, filtering, interlace = struct.unpack('>IIBBBBB', chunks[0][1])
    assert (depth, colour, compression, filtering, interlace) == (8, 2, 0, 0, 0)
    rows = np.frombuffer(zlib.decompress(chunks[1][1]), dtype=np.uint8).reshape(height, 1 + 3 * width)
    assert np.all(rows[:, 0] == 0)
    return rows[:, 1:].reshape(height, width, 3)


def test_a_png_frame_reads_back_as_the_same_pixels(tmp_path):
    """
    Description:
        A frame written as PNG with the standard library reads back, by the PNG specification, as the same 5 x 7 RGB
        pixels: the signature, the three chunks with their checksums, 8 bits per channel, RGB, no interlace, every row
        unfiltered. An image that is not rows of RGB bytes is refused naming its shape.
    """

    image = np.random.default_rng(20261122).integers(0, 256, (5, 7, 3)).astype(np.uint8)
    path = str(tmp_path / 'frame.png')
    Write_Png(path, image)
    assert np.array_equal(_Read_Png(path), image)
    for wrong in (image[:, :, :2], image.astype(np.float64), image[0]):
        with pytest.raises(View_Error_Cls, match='rows of RGB bytes'):
            Write_Png(str(tmp_path / 'wrong.png'), wrong)


def test_frames_csv_holds_what_each_frame_shows(tmp_path):
    """
    Description:
        frames.csv: a header of its 11 columns and one row per frame: before the first prediction the prediction's N,
        the inference time and the plan are empty; at 0.35 s of the constructed go the phase GO, 30 samples, the
        prediction of N 28 with 1.000 ms, plan 2 of 2; at the end the class.
    """

    record = _Record()
    rows = [Frame_Row(record, Frame_At(record, t), number, 'frame_%06d.png' % number) for number, t in enumerate((0.05, 0.35, 0.5))]
    path = str(tmp_path / 'frames.csv')
    Write_Frames(path, rows)
    lines = open(path, encoding='ascii').read().split('\n')
    assert lines[0] == ','.join(CONST_FRAME_COLUMNS) and len(CONST_FRAME_COLUMNS) == 11 and lines[-1] == '' and len(lines) == 5
    assert lines[1] == '0,frame_000000.png,0.050000,waiting,5,QT_BASIS,,,,2,'
    assert lines[2] == '1,frame_000001.png,0.350000,GO,30,QT_BASIS,28,1.000,2,2,'
    assert lines[3] == '2,frame_000002.png,0.500000,GO,30,QT_BASIS,28,1.000,2,2,caught'


def test_a_camera_preset_sets_the_free_camera_and_what_is_shown(themed):
    """
    Description:
        The side preset makes MuJoCo's free camera orthographic with the preset's height (4.9 m) and looks along +x
        (azimuth 0) level; the overview is a perspective view of 32 degrees. Either shows the floor (group 0) and the
        visual geoms (2), never the model's own area boxes (1), the collision shapes (3, 4) or a site; the rail and its
        cameras (5) only where the preset has them.
    """

    model = themed
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    Set_Camera(model, camera, option, Camera_Str=CONST_CAMERAS['side'])
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE and camera.orthographic == 1 and model.vis.global_.orthographic == 1
    assert model.vis.global_.fovy == pytest.approx(4.9) and (camera.azimuth, camera.elevation) == (0.0, 0.0) and camera.distance == pytest.approx(8.6)
    assert list(option.geomgroup) == [1, 0, 1, 0, 0, 0] and not any(option.sitegroup)
    Set_Camera(model, camera, option, Camera_Str=CONST_CAMERAS['overview'])
    assert camera.orthographic == 0 and model.vis.global_.orthographic == 0 and model.vis.global_.fovy == pytest.approx(32.0)
    assert list(option.geomgroup) == [1, 0, 1, 0, 0, 1] and np.allclose(camera.lookat, CONST_CAMERAS['overview'].Lookat)


def test_decorations_are_drawn_as_decor_geoms_and_a_full_scene_is_refused(themed):
    """
    Description:
        Draw adds one geom per decoration to MuJoCo's scene: a sphere, a capsule between a line's two points, a box and
        a box's edges, each of the decoration category (no shadow) with its colour and its label; a scene without room
        is refused naming the counts, nothing added.
    """

    scene = mujoco.MjvScene(themed, maxgeom=4)
    decorations = [Decoration_Str(Kind='sphere', Position=(1.0, 2.0, 3.0), Radius=0.03, Colour=(1.0, 0.0, 0.0, 1.0), Label='Throw Detection'),
                   Decoration_Str(Kind='line', Position=(0.0, 0.0, 1.0), Extent=(0.0, 0.0, 3.0), Radius=0.006, Colour=(0.0, 1.0, 0.0, 1.0)),
                   Decoration_Str(Kind='box', Position=(0.5, 0.0, 1.0), Extent=(0.1, 0.2, 0.3), Colour=(0.0, 0.0, 1.0, 0.1)),
                   Decoration_Str(Kind='frame', Position=(0.5, 0.0, 1.0), Extent=(0.1, 0.2, 0.3), Colour=(0.0, 0.0, 1.0, 0.5))]
    Draw(scene, decorations)
    geoms = [scene.geoms[k] for k in range(4)]
    kinds = mujoco.mjtGeom
    assert scene.ngeom == 4 and [g.type for g in geoms] == [kinds.mjGEOM_SPHERE, kinds.mjGEOM_CAPSULE, kinds.mjGEOM_BOX, kinds.mjGEOM_LINEBOX]
    assert all(g.category == mujoco.mjtCatBit.mjCAT_DECOR for g in geoms)
    assert geoms[0].label == 'Throw Detection' and geoms[1].label == '' and np.allclose(geoms[0].pos, (1.0, 2.0, 3.0)) and geoms[0].size[0] == 0.03
    assert np.allclose(geoms[0].rgba, (1.0, 0.0, 0.0, 1.0)) and np.allclose(geoms[2].rgba, (0.0, 0.0, 1.0, 0.1), atol=1e-6)
    assert np.allclose(geoms[1].pos, (0.0, 0.0, 2.0)) and geoms[1].size[2] == pytest.approx(1.0) and np.allclose(geoms[2].size, (0.1, 0.2, 0.3))
    with pytest.raises(View_Error_Cls, match='room for 4 geoms and holds 4; 1 decorations do not fit'):
        Draw(scene, decorations[:1])
    assert scene.ngeom == 4


def test_the_ghost_is_the_robot_alone_at_the_plans_goal_and_faded(themed):
    """
    Description:
        The ghost's positions are the frame's with the robot's joints at the goal of the plan being executed (none
        before the first plan). Added to a scene, the ghost is the moving bodies' visible geoms, the robot's faded to
        0.28 of their opacity, the ball's and every site invisible, all decorations; the model's static geoms (the
        floor, the stand) are not added again.
    """

    record = _Record()
    assert Ghost_Positions(record, Frame_At(record, 0.25)) is None
    positions = Ghost_Positions(record, Frame_At(record, 0.32))
    assert np.allclose(positions[7:13], 0.2) and np.allclose(positions[0:3], _Arc(0.32)) and positions is not record.State_Position[160]
    assert np.allclose(Ghost_Positions(record, Frame_At(record, 0.27))[7:13], 0.1)
    model = themed
    robot = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'ur10e/' + name) for name in
             ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')]
    pose = np.array(model.qpos0)
    pose[model.jnt_qposadr[robot]] = (0.5, -1.5, -1.5, -1.0, -1.0, 0.0)
    option, scene = mujoco.MjvOption(), mujoco.MjvScene(model, maxgeom=500)
    option.geomgroup[:] = (1, 0, 1, 0, 0, 0)
    Add_Ghost(model, pose, option, scene)
    added = [scene.geoms[k] for k in range(scene.ngeom)]
    assert all(g.category == mujoco.mjtCatBit.mjCAT_DECOR and g.transparent == 1 for g in added)
    assert all(g.rgba[3] == 0.0 for g in added if g.objtype != mujoco.mjtObj.mjOBJ_GEOM)
    geoms = [g for g in added if g.objtype == mujoco.mjtObj.mjOBJ_GEOM]
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
    bodies = {int(model.geom_bodyid[g.objid]) for g in geoms}
    assert len(geoms) >= 18 and 0 not in bodies and ball in bodies
    for g in geoms:
        original = float(model.geom_rgba[g.objid, 3]) if model.geom_matid[g.objid] < 0 else float(model.mat_rgba[model.geom_matid[g.objid], 3])
        assert g.rgba[3] == pytest.approx(0.0 if model.geom_bodyid[g.objid] == ball else 0.28 * original, abs=1e-6)


def test_the_legends_figure_names_each_method_in_its_colour():
    """
    Description:
        The window's legend is one of MuJoCo's figures with its legend switched on: each entry's name with the entry's
        colour as its line colour, each line with two points (MuJoCo lists only lines that have points), no tick labels.
    """

    figure = Legend_Figure([('QT_BASIS', (112, 173, 71)), ('GLS_M4', (196, 160, 40))])
    names = [bytes(figure.linename[k]).split(b'\0')[0].decode() for k in range(3)]
    assert figure.flg_legend == 1 and names == ['QT_BASIS', 'GLS_M4', ''] and list(figure.linepnt[:3]) == [2, 2, 0]
    assert np.allclose(figure.linergb[0], np.array((112, 173, 71)) / 255.0) and np.allclose(figure.linergb[1], np.array((196, 160, 40)) / 255.0)
    assert figure.flg_ticklabel[0] == 0 and figure.flg_ticklabel[1] == 0


def test_an_offscreen_frame_shows_the_decorations_and_with_the_overlay_the_legends_swatches(themed):
    """
    Description:
        One offscreen frame of 320 x 180 px from the side preset: a red sphere of 0.6 m at the point the camera looks at
        is red at the image's centre, and without it the centre is not; with the overlay's text the legend's swatch in
        the method's colour appears in the top right corner and the top left corner turns into the overlay's dark
        ground, without it neither is there (a clean frame); between the texts the two frames are the same. The image
        is rows of RGB bytes, the top row first: the sphere set 1.2 m higher is red in the upper half only. A frame
        larger than the model's offscreen buffer is refused.
    """

    model = themed
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = CONST_CAMERAS['side']
    red = Decoration_Str(Kind='sphere', Position=tuple(camera.Lookat), Radius=0.6, Colour=(1.0, 0.0, 0.0, 1.0))
    offscreen = Offscreen_Cls(model, 320, 180)
    try:
        clean = offscreen.Render(data, camera, [])
        marked = offscreen.Render(data, camera, [red])
        high = offscreen.Render(data, camera, [Decoration_Str(Kind='sphere', Position=(0.0, -2.65, 3.4), Radius=0.3, Colour=(1.0, 0.0, 0.0, 1.0))])
        swatch = (40, 200, 230)
        burnt = offscreen.Render(data, camera, [], texts=('method\ntime', 'QT_BASIS\n0.350 s'), legend=[('QT_BASIS', swatch)])
    finally:
        offscreen.Close()
    # Red, and not the ball's orange (its green channel is 0.45 of its red one).
    is_red = (lambda image: (image[..., 0] > 150) & (image[..., 1] < 0.3 * image[..., 0]) & (image[..., 2] < 0.3 * image[..., 0]))
    assert clean.shape == (180, 320, 3) and clean.dtype == np.uint8
    assert is_red(marked)[90, 160] and not is_red(clean).any() and is_red(marked).sum() > 1000
    assert is_red(high)[:90].sum() > 100 and is_red(high)[90:].sum() == 0
    is_swatch = (lambda image: np.all(np.abs(image.astype(np.int64) - swatch) < 12, axis=-1))
    assert is_swatch(burnt)[:40, 200:].sum() >= 20 and not is_swatch(clean).any()
    assert np.abs(burnt[:30, :60].astype(np.int64) - clean[:30, :60]).sum() > 30000 and np.array_equal(burnt[85:125, 100:220], clean[85:125, 100:220])
    with pytest.raises(View_Error_Cls, match='does not fit the model\'s offscreen buffer of 320 x 180 px'):
        Offscreen_Cls(model, 640, 360)
