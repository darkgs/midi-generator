"""conftest.py for Pytest"""
from pathlib import Path
from typing import List

import pytest


@pytest.fixture(name="fxt_midi_files", params=["midi_generator/tests/data/midis"])
def fxt_midi_files(request: str) -> List[str]:
    """
    Pytest fixture of GiantMIDI dataset

    Parameters:
        request: path to root dir of MIDI files

    Returns:
        List[str] - list of path to MIDI files
    """
    midi_dir_path = request.param
    return [str(midi_path) for midi_path in Path(midi_dir_path).rglob("*.mid")]
