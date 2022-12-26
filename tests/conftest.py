"""conftest.py for Pytest"""
from pathlib import Path
from typing import List

import pytest
from pretty_midi import PrettyMIDI


@pytest.fixture(name="midis_datas")
def fxt_midis_datas() -> List[PrettyMIDI]:
    """
    Pytest fixture of GiantMIDI dataset

    Parameters:
        None

    Returns:
        List[PrettyMIDI] - a list of PrettyMIDI objects
    """
    midis_data_home = "tests/data/midis"
    return [PrettyMIDI(str(midi_path)) for midi_path in Path(midis_data_home).rglob("*.mid")]
