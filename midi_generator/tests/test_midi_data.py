"""Pytest for midis dataset"""
from tempfile import TemporaryDirectory
from typing import List

import numpy as np

from midi_generator.dataset.midi import MIDIData, MIDIDataset


def test_fxt_midi_files(fxt_midi_files: List[str]):
    """
    fxt_midi_files has to provide some MIDI files

    Parameters:
        midi_files: List[str] - list of path to MIDI file

    Returns:
        None
    """
    assert len(fxt_midi_files) > 0


def test_mididata(fxt_midi_files: List[str]):
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
        # Each test MIDI has at least 1 instrument.
        assert len(midi.get_instruments()) > 0

        for instrument in midi.get_instruments():
            pianoroll = midi.get_pianoroll(instrument=instrument)
            # The shape of pianoroll is [time, pitch], and the range of pitch is fixed to 128
            assert pianoroll.shape[-1] == 128


# pylint: disable=protected-access
def test_mididata_pianorolls_cache(fxt_midi_files: List[str]):
    """
    Test pianorolls caching of MIDIData

    Parameters:
        midi_files: List[str] - list of path to MIDI file

    Returns:
        None
    """

    with TemporaryDirectory() as cache_home:
        for midi_path in fxt_midi_files:
            midi = MIDIData(midi_path=midi_path, cache_home=cache_home)

            # pianorolls for midi has to be cached from the previous MIDIData __init__ function
            assert midi._try_pianorolls_from_cache(midi_path=midi_path, cache_home=cache_home) is not None

            # load MIDI from the cache
            midi2 = MIDIData(midi_path=midi_path, cache_home=cache_home)

            # pianorolls loaded from the cache should be the same with the original one
            assert len(midi.get_instruments()) > 0
            assert len(midi.get_instruments()) == len(midi2.get_instruments())

            for instrument in midi.get_instruments():
                assert (midi._pianorolls[instrument] == midi2._pianorolls[instrument]).all()


def test_midi_dataset(fxt_midi_files: List[str]):
    """
    Test MIDIDataset which is a pytorch dataset object to handle MIDI

    Parameters:
        midi_files: List[str] - list of path to MIDI file
    """
    instruments = [MIDIData.INSTRUMENT.PIANO]
    dataset = MIDIDataset(midi_files=fxt_midi_files, instruments=instruments)

    assert len(dataset) == len(fxt_midi_files)

    for midi in dataset:
        assert midi.shape[1] == 128
        assert midi.shape[2] == len(instruments)


def test_midi_dataset_missing_instrument(fxt_midi_files: List[str]):
    """
    Test if MIDIDataset raises an exception when some of MIDI doesn't a required pianoroll

    Parameters:
        midi_files: List[str] - list of path to MIDI file
    """
    exception_detected = False
    try:
        MIDIDataset(
            midi_files=fxt_midi_files,
            instruments=[MIDIData.INSTRUMENT.REED, MIDIData.INSTRUMENT.PIANO],
        )
    except ValueError:
        exception_detected = True

    assert exception_detected
