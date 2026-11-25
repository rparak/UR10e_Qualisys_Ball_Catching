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
# File Name: ../test/Plot/test_style.py
## =========================================================================== ##

# System (Default)
import os
# Matplotlib (Figures) [pip3 install matplotlib]
import matplotlib
# Pytest (Testing framework) [pip3 install pytest]
import pytest
# Custom Lib.:
#   ../Plot/Style
from Plot import Style
from Plot.Style import (Contrast_Ratio, Latex_Available, Style_Parameters, Style_Context, Save_Figure, Look_Str, Style_Error_Cls,
                        CONST_COLOURS, CONST_BACKGROUND, CONST_FOREGROUND, CONST_THEMES)

"""
Description:
    Initialization of constants.

    The figures' one style (the visualisation, delivery 2, with the dark theme by default):
    every colour that carries meaning readable on both backgrounds (WCAG 2.1, 4.5:1),
    LaTeX detected, the dark theme first, PDF and PNG at 300 dpi. Expectations come from the WCAG
    definition, the owner's decisions and matplotlib's documented parameters.
"""
# The contrast floor of a colour that carries meaning (WCAG 2.1), written out here.
#   Unit [float]
CONST_FLOOR = 4.5


def test_white_on_black_has_the_wcag_ratio_of_21_and_a_colour_on_itself_1():
    """
    Description:
        The WCAG ratio's two ends, and #777777 on white (4.48, just under the floor, the reference value of WCAG's own
        examples), either way round.
    """

    assert Contrast_Ratio('#FFFFFF', '#000000') == pytest.approx(21.0)
    assert Contrast_Ratio('#56B4E9', '#56B4E9') == pytest.approx(1.0)
    assert Contrast_Ratio('#777777', '#FFFFFF') == pytest.approx(4.48, abs=0.01) == Contrast_Ratio('#FFFFFF', '#777777')


@pytest.mark.parametrize('theme', CONST_THEMES)
def test_every_entitys_colour_and_the_text_reach_4_5_to_1_on_their_themes_background(theme):
    """
    Description:
        Each entity's colour in a theme (both predictors, the catch's modes and classes, the joints, the true path and the
        observed samples) and the text against the theme's background: at least 4.5:1.
    """

    k = CONST_THEMES.index(theme)
    low = {entity: Contrast_Ratio(colours[k], CONST_BACKGROUND[theme]) for entity, colours in CONST_COLOURS.items()}
    assert min(low.values()) >= CONST_FLOOR, low
    assert Contrast_Ratio(CONST_FOREGROUND[theme], CONST_BACKGROUND[theme]) >= CONST_FLOOR


@pytest.mark.parametrize('theme', CONST_THEMES)
def test_no_go_is_the_lighter_grey_and_no_go_catchable_the_darker_in_each_theme(theme):
    """
    Description:
        In both themes 'no go' is lighter than 'no go, catchable' (a lighter colour is nearer white), so that the outcomes
        that matter carry the visual weight in the catch's classes.
    """

    k = CONST_THEMES.index(theme)
    lighter, darker = CONST_COLOURS['no_go'][k], CONST_COLOURS['no_go_catchable'][k]
    assert Contrast_Ratio(lighter, '#FFFFFF') < Contrast_Ratio(darker, '#FFFFFF'), (lighter, darker)


def test_the_dark_theme_is_the_default_and_both_predictors_have_their_own_colour_in_each_theme():
    """
    Description:
        Dark first; the two predictors differ in each theme; an unknown entity is refused by name.
    """

    assert CONST_THEMES == ('dark', 'light') and Look_Str().Theme == 'dark'
    for theme in CONST_THEMES:
        look = Look_Str(Theme=theme)
        assert look.Colour('QT_BASIS') != look.Colour('GLS_M4')
    with pytest.raises(Style_Error_Cls, match='No colour is defined for "LS"'):
        Look_Str().Colour('LS')


def test_latex_is_used_when_latex_and_dvipng_are_found(monkeypatch):
    """
    Description:
        usetex needs both programs on the PATH; either missing gives matplotlib's mathtext.
    """

    found = {'latex': '/bin/latex', 'dvipng': '/bin/dvipng'}
    monkeypatch.setattr(Style.shutil, 'which', lambda name: found.get(name))
    assert Latex_Available()
    found.pop('dvipng')
    assert not Latex_Available()


@pytest.mark.parametrize('usetex', [True, False])
def test_the_style_is_scienceplots_science_with_the_themes_colours_and_restored_after_its_context(usetex):
    """
    Description:
        SciencePlots' 'science' (its serif type and inward ticks) with usetex as asked, the theme's background and text,
        300 dpi; the global parameters are the same after the context as before it.
    """

    before = dict(matplotlib.rcParams)
    parameters = Style_Parameters('dark', usetex)
    assert parameters['text.usetex'] is usetex and parameters['xtick.direction'] == 'in' and parameters['font.family'] == ['serif']
    assert parameters['figure.facecolor'] == CONST_BACKGROUND['dark'] and parameters['text.color'] == CONST_FOREGROUND['dark']
    assert parameters['savefig.dpi'] == 300 and parameters['savefig.bbox'] == 'tight' and Style_Parameters('light', usetex,
                                                                                                           'video')['savefig.bbox'] is None
    with Style_Context('light', usetex) as look:
        assert matplotlib.rcParams['axes.facecolor'] == CONST_BACKGROUND['light'] and look.Usetex is usetex
    assert dict(matplotlib.rcParams) == before
    with pytest.raises(Style_Error_Cls, match="found 'grey'"):
        Style_Parameters('grey', False)


def test_a_label_is_escaped_for_latex_outside_its_math_only():
    """
    Description:
        With usetex an underscore, a percent sign, an ampersand and a hash are escaped outside $...$ and kept inside;
        mathtext takes the label as it is.
    """

    assert Look_Str(Usetex=True).Text('QT_BASIS 5 % & #1, $x_0$') == r'QT\_BASIS 5 \% \& \#1, $x_0$'
    assert Look_Str(Usetex=False).Text('QT_BASIS 5 %') == 'QT_BASIS 5 %'


def test_a_figure_is_saved_as_pdf_and_png_at_300_dpi_without_a_date_and_replaces_an_earlier_file(tmp_path):
    """
    Description:
        Both formats are written beside each other, the PNG at 300 dpi, the PDF without a creation date (so the same figure
        gives the same PDF twice); an existing file is replaced, no temporary file is left, another format is refused.
    """

    from PIL import Image
    paths = []
    for _ in range(2):
        with Style_Context('dark', False) as look:
            figure = look.Figure()
            figure.subplots().plot([0, 1], [0, 1], color=look.Colour('QT_BASIS'))
            paths.append(Save_Figure(figure, str(tmp_path / 'figure'), ('pdf', 'png')))
        paths[-1] = [open(path, 'rb').read() for path in paths[-1]]
    assert paths[0][0] == paths[1][0] and b'CreationDate' not in paths[0][0]
    assert sorted(os.listdir(tmp_path)) == ['figure.pdf', 'figure.png']
    with Image.open(tmp_path / 'figure.png') as image:
        assert tuple(round(v) for v in image.info['dpi']) == (300, 300)
    with pytest.raises(Style_Error_Cls, match="found 'svg'"):
        Save_Figure(figure, str(tmp_path / 'other'), ('svg',))
