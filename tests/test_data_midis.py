"""Pytest for midis dataset"""
from typing import List

import numpy as np

from midi_generator.preprocess.midi import MIDIData


def test_midis_import(fxt_midi_files: List[str]):
    """
    Test whether MIDI files are loaded to MIDIData properly

    Parameters:
        midi_files: List[str] - list of path to MIDI file

    Returns:
        None
    """
    pianoroll: np.array
    for midi_path in fxt_midi_files:
        midi = MIDIData(midi_path=midi_path)
        assert len(midi.get_instruments()) > 0
        for intsrument in midi.get_instruments():
            pianoroll = midi.get_pianoroll(instrument=intsrument)
            assert pianoroll.shape[-1] == 128
