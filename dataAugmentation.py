import os
from tqdm import tqdm

import numpy as np
from scipy.io import wavfile

from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    AddImpulseResponse,
    FrequencyMask,
    TimeMask,
    AddGaussianSNR,
    ClippingDistortion,
    AddBackgroundNoise,
    AddShortNoises)

SAMPLE_RATE = 16000
CHANNELS = 1


def load_wav_file(sound_file_path):
    sample_rate, sound_np = wavfile.read(sound_file_path)
    if sample_rate != SAMPLE_RATE:
        raise Exception(
            "Unexpected sample rate {} (expected {})".format(sample_rate, SAMPLE_RATE)
        )

    if sound_np.dtype != np.float32:
        assert sound_np.dtype == np.int16
        sound_np = (sound_np / 32767).astype(np.float32)  # ends up roughly between -1 and 1

    return sound_np

def applyTransformations(fileName, output_dir, auxiliarSoundsDir):
    name = fileName.split(".")[0].split("/")[-1]
    samples = load_wav_file(fileName)

    # AddImpulseResponse
    augmenter = Compose(
        [AddImpulseResponse(p=1.0, ir_path=os.path.join(auxiliarSoundsDir, "helperSounds/ir"))]
    )
    output_file_path = os.path.join(
        output_dir, "{}_AddImpulseResponse_{:03d}.wav".format(name, 0)
    )

    augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    
    wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)
    # FrequencyMask
    augmenter = Compose([FrequencyMask(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "{}_FrequencyMask_{:03d}.wav".format(name, i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # TimeMask
    augmenter = Compose([TimeMask(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "{}_TimeMask_{:03d}.wav".format(name, i))
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddGaussianSNR
    augmenter = Compose([AddGaussianSNR(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "{}_AddGaussianSNR_{:03d}.wav".format(name, i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddGaussianNoise
    augmenter = Compose(
        [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "{}_AddGaussianNoise_{:03d}.wav".format(name, i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # TimeStretch
    augmenter = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "{}_TimeStretch_{:03d}.wav".format(name, i))
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # PitchShift
    augmenter = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "{}_itchShift_{:03d}.wav".format(name, i))
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Shift
    augmenter = Compose([Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, "{}_Shift_{:03d}.wav".format(name, i))
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Shift without rollover
    augmenter = Compose(
        [Shift(min_fraction=-0.5, max_fraction=0.5, rollover=False, p=1.0)]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "{}_ShiftWithoutRollover_{:03d}.wav".format(name, i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Normalize
    augmenter = Compose([Normalize(p=1.0)])
    output_file_path = os.path.join(output_dir, "{}_Normalize_{:03d}.wav".format(name, 0))
    augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # ClippingDistortion
    augmenter = Compose([ClippingDistortion(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "{}_ClippingDistortion_{:03d}.wav".format(name, i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddBackgroundNoise
    augmenter = Compose(
        [
            AddBackgroundNoise(
                sounds_path=os.path.join(auxiliarSoundsDir, "helperSounds/background_noises"), p=1.0
            )
        ]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "{}_AddBackgroundNoise_{:03d}.wav".format(name, i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddShortNoises
    augmenter = Compose(
        [
            AddShortNoises(
                sounds_path=os.path.join(auxiliarSoundsDir, "helperSounds/short_noises"),
                min_snr_in_db=0,
                max_snr_in_db=8,
                min_time_between_sounds=2.0,
                max_time_between_sounds=4.0,
                burst_probability=0.4,
                min_pause_factor_during_burst=0.01,
                max_pause_factor_during_burst=0.95,
                min_fade_in_time=0.005,
                max_fade_in_time=0.08,
                min_fade_out_time=0.01,
                max_fade_out_time=0.1,
                p=1.0,
            )
        ]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, "{}_AddShortNoises_{:03d}.wav".format(name, i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)


if __name__ == "__main__":
    OBJECTIVES = ["onion", "tomato", "pepper"]
    DIR = os.path.dirname(__file__)
    print(DIR)
    output_dir = os.path.join(DIR, "speechDatasetAugmented")
    rawFiles = os.path.join(DIR, 'speech_dataset')
    print(rawFiles)
    for objective in OBJECTIVES:
        currentDir = os.path.join(rawFiles, objective)
        currentOutputDir = os.path.join(output_dir, objective)
        os.makedirs(currentOutputDir, exist_ok=True)
        inputFiles = os.listdir(currentDir)
        print("Output DIR: " + currentOutputDir)
        for file in tqdm(inputFiles):
            applyTransformations(os.path.join(currentDir, file), currentOutputDir, DIR)
    
