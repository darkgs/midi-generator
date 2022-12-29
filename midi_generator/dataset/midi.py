"""Pytorch data loader for MIDI files"""
import hashlib
import os
import pickle
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np
import pypianoroll
import tqdm
from torch.utils.data import Dataset

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
        cache_home: Optional[str] = None,
    ):
        """
        Parameters:
            midi_path: str - path to a MIDI file
            multi_track: pypianoroll.Multitrack - multi_track object storing entire song
            cache_home: Optional[str] - path to cache home
        Return:
            None
        """
        if midi_path is None and multi_track is None:
            raise ValueError("One of midi_path nor multi_track should be passed to MIDIData")

        if midi_path is not None and multi_track is not None:
            Log.warning("MIDIData - both midi_path and multi_track are passed, ignore multi_track")

        # try to import the pianoroll of MIDI from the cache
        self._pianorolls = self._try_pianorolls_from_cache(midi_path=midi_path, cache_home=cache_home)

        # cache miss
        if self._pianorolls is None:
            # Assign _multi_track from one of midi_path or multi_track
            if midi_path:
                # raises ValueError if MIDI is unable to be parsed
                multi_track = pypianoroll.read(midi_path)

            pianorolls = self._extract_pianorolls(multi_track=multi_track)
            self._store_pianorolls_to_cache(midi_path=midi_path, cache_home=cache_home, pianorolls=pianorolls)
            self._pianorolls = pianorolls

    def _get_cache_path(self, midi_path: str, cache_home: str):
        """
        We store the pianoroll of MIDI by MD5 of MIDI path.

        Parameters:
            midi_path: str - path to a MIDI file
            cache_home: str - path to home of cache

        Returns:
            str - {cache_home}/{md5 of MIDI path}.pkl
        """
        cache_path = f"{cache_home}/{hashlib.md5(midi_path.encode()).hexdigest()}.pkl"

        return cache_path

    def _try_pianorolls_from_cache(self, midi_path: str, cache_home: str) -> Dict[INSTRUMENT, np.array]:
        """
        Try to load the pianorolls of MIDI from cache.
        We store the pianoroll of MIDI by MD5 of MIDI path.
        the pianoroll object will be stored as .pkl

        Parameters:
            midi_path: str - path to a MIDI file
            cache_home: str - path to home of cache

        Returns:
            Dict[INSTRUMENT, np.array] - if cache hit, pianoroll of the MIDI
            None - if cache miss
        """
        if cache_home is None:
            return None

        cache_path = self._get_cache_path(midi_path=midi_path, cache_home=cache_home)

        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f_cache:
                pianorolls = pickle.load(f_cache)
            return pianorolls

        return None

    def _store_pianorolls_to_cache(
        self, midi_path: str, cache_home: str, pianorolls: Dict[INSTRUMENT, np.array]
    ) -> None:
        """
        Store the pianorolls of MIDI to cache.

        Parameters:
            midi_path: str - path to a MIDI file
            cache_home: str - path to home of cache
            pianorolls: Dict[INSRUMENT, np.array] - A dictionary of pianoroll mapping INSTRUMENT to pianoroll np.array

        Returns:
            None
        """
        if cache_home is None:
            return

        cache_path = self._get_cache_path(midi_path=midi_path, cache_home=cache_home)

        # makedir if necessary
        if not os.path.isdir(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))

        with open(cache_path, "wb+") as f_cache:
            pickle.dump(pianorolls, f_cache)

    def _extract_pianorolls(self, multi_track: pypianoroll.Multitrack) -> Dict[INSTRUMENT, np.array]:
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
            # note that the index of Enum starts from 1
            instrument = MIDIData.INSTRUMENT(track.program / 8 + 1)
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


class MIDIDataset(Dataset):
    """
    Pytorch Dataset implementation for MIDI data
    """

    _midis: List[MIDIData]
    _instruments: List[MIDIData.INSTRUMENT]

    def __init__(
        self,
        midi_files: List[str],
        instruments: List[MIDIData.INSTRUMENT],
        cache_home: Optional[str] = None,
    ):
        """
        Parameters:
            midif_files: List[str] - A list of midi file paths
            instruments: List[MIDIData.INSTRUMENT]) - A list of instruments to be utilized on the futher processes
            cache_home: Optional[str] - path to cache home
        """
        Dataset.__init__(self)

        self._midis = []
        for midi_path in tqdm.tqdm(midi_files, desc="Loading MIDI files"):
            try:
                midi_data = MIDIData(midi_path=midi_path, cache_home=cache_home)
            except ValueError as exception:
                midi_data = None
                Log.warning(f"Error while loading {midi_path} - skip this file: {exception}")

            if midi_data is None:
                continue

            self._midis.append(midi_data)

        self._instruments = instruments

        self.validate_data()

    def validate_data(self):
        """
        Check if all MIDI data has necessary pianorolls for all instruments.

        Parameters:
            None - we check with this object's private variable, _midis and _instruments

        Returns:
            None - we trigger an exception imediately when found an error
        """

        for midi in self._midis:
            for instrument in self._instruments:
                if instrument in midi.get_instruments():
                    continue
                raise ValueError(f"Instrument {instrument.name} is missing in the MIDI of the dataset!")

    def __len__(self):
        """
        Returns:
            int - The length of this dataset
        """
        return len(self._midis)

    def __getitem__(self, index: int) -> np.array:
        """
        Get an MIDIData item.

        Parameters:
            index: int - An index of item to get.

        Returns:
            np.array - A pianoroll np.array with the shape of [time, pitch, # instruments]
        """
        pianorolls = [self._midis[index].get_pianoroll(instrument=instrument) for instrument in self._instruments]

        return np.stack(pianorolls, axis=2)
