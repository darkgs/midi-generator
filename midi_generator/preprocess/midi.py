"""Pytorch data loader for MIDI files"""
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np
import pypianoroll

from midi_generator.utils.log import Log


class MIDIData:
    """
    MIDIData class to manage single MIDI file.

    It utilizes 3rd libraries such that pretty_midi, and pypianoroll.
    We don't expose 3rd-party implementation to outside.

    This class converts MIDI to pianoroll np.array for each instruments.
    """

    class INSTRUMENT(Enum):
        """
        Musical instruments palyed in MIDI
        """

        PIANO = auto()
        CHROMATIC_PERCUSSION = auto()
        ORGAN = auto()
        GUITAR = auto()
        BASS = auto()
        STRINGS = auto()
        ENSEMBLE = auto()
        BRASS = auto()
        REED = auto()
        PIPE = auto()
        SYNTH_LEAD = auto()
        SYNTH_PAD = auto()
        SYNTH_EFFECTS = auto()
        ETHNIC = auto()
        PERCUSSIVE = auto()
        SOUND_EFFECTS = auto()

    _multi_track: pypianoroll.Multitrack
    _pianorolls: Dict[INSTRUMENT, np.array]

    def __init__(
        self,
        midi_path: Optional[str] = None,
        multi_track: Optional[pypianoroll.Multitrack] = None,
    ):
        """
        Parameters:
            midi_path: str - path to a MIDI file
            multi_track: pypianoroll.Multitrack - multi_track object storing entire song
        Return:
            None
        """
        if midi_path is None and multi_track is None:
            raise ValueError("One of midi_path nor multi_track should be passed to MIDIData")

        if midi_path is not None and multi_track is not None:
            Log.warning("MIDIData - both midi_path and multi_track are passed, ignore multi_track")

        # Assign _multi_track from one of midi_path or multi_track
        if midi_path:
            multi_track = pypianoroll.read(midi_path)

        self._pianorolls = self._extract_pianoroll(multi_track=multi_track)

    def _extract_pianoroll(self, multi_track: pypianoroll.Multitrack) -> Dict[INSTRUMENT, np.array]:
        """
        Extract the pianorolls of MIDI from the multi_track.
        Each instrument has thier own pianoroll np.array of shape (-1, 128) representing (time, pitch).

        Parameters:
            multi_track: pypianorooll.Multitrack - A multi_track to be parsed to pianoroll

        Returns:
            Dict[INSRUMENT, np.array] - A dictionary of pianoroll mapping INSTRUMENT to pianoroll np.array
        """
        pianorolls: Dict[MIDIData.INSTRUMENT, np.array] = {}
        track: pypianoroll.Track
        for track in multi_track.tracks:
            # program: int - [0, 127] that represents MIDI instruments.
            # we briefly uses 128 / 8 instruments by dividing program by 8
            instrument: MIDIData.INSTRUMENT = track.program / 8 + 1
            if instrument in pianorolls:
                Log.warning("Duplicated instruments found! we ignore the latter")
                continue
            pianorolls[instrument] = track.pianoroll

        return pianorolls

    def get_instruments(self) -> List[INSTRUMENT]:
        """
        Get a list of instruments in an MIDI

        Parameters:
            None

        Returns:
            List[MIDIData.INSTRUMENT] - A list of instruments sorted in the accending order of Enum
        """
        return sorted(list(self._pianorolls.keys()))

    def get_pianoroll(self, instrument: INSTRUMENT) -> np.array:
        """
        Get a pianoroll np.array for an instrument played in the track

        Parameters:
            instrument: INSTRUMENT - Enum value representing an instrument

        Returns:
            np.array - A pianoroll with shape of [time, pitch]
        """
        return self._pianorolls[instrument]
