"""
Train the MIDI Generator
"""
from pathlib import Path

from torch.utils.data import DataLoader

from midi_generator.dataset.midi import MIDIData, MIDIDataset


def main():
    """
    Temporal test function
    """
    midi_files = [str(midi_path) for midi_path in Path("data/midis").rglob("*.mid")]
    instruments = [MIDIData.INSTRUMENT.PIANO]
    num_epoch = 1
    cache_home = "cache/midis"

    dataset = MIDIDataset(midi_files=midi_files, instruments=instruments, cache_home=cache_home)
    train_dataloader = DataLoader(dataset)

    for _ in range(num_epoch):
        for _, _ in enumerate(train_dataloader):
            pass
