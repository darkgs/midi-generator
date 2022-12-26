"""Pytest for midis dataset"""
from typing import List

from pretty_midi import PrettyMIDI


def test_midis_import(midis_datas: List[PrettyMIDI]):
    """
    Test whether GiantMIDI dataset is loaded propery.

    Parameters:
        midis_datas: List[PrettyMIDI] - a list of PrettyMIDI objects

    Returns:
        None
    """
    for midis_data in midis_datas:
        print(midis_data.instruments[0].program)
        print(midis_data.get_piano_roll().shape)
