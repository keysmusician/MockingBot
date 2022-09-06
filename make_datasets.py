'''
Prepares datasets.

This file is incomplete, but when complete it should export several TensorFlow
Datasets.
'''


'''
Helper functions and other objects to keep things clean and readable.
'''
class DatasetType:
    '''
    A dataset source type.

    May be either STATIC or DYNAMIC
    '''

    STATIC = 1
    DYNAMIC = 2


DYNAMIC_DATASETS = {
    'Sines': []
}


def print_available_datasets(static_datasets_path):
    '''
    Prints the names of all datasets available for use in this project.

    static_datasets_path: A path to WAV file datasets that can be loaded by
        TensorFlow.
    '''
    print('Datasets:')

    print('- Static (stored on drive):')

    for dataset_name in sorted(
            os.listdir(static_datasets_path), key=str.lower):
        if not dataset_name.startswith('.'):
            print(f'\t\t{dataset_name}')

    print('- Dynamic (generated in memory):')

    for dataset_name in DYNAMIC_DATASETS:
        print(f'\t\t{dataset_name}')


'''
Makes datasets available.
'''
import os
import pathlib


DATASETS_PATH = pathlib.Path('/Volumes/T7/Code/Datasets/')


# Show available datasets:
print_available_datasets(DATASETS_PATH)


'''
Sets the project's dataset.
'''
# Choose the desired dataset here.
# `DATASET_NAME` can be the name of any folder inside `DATASETS_PATH` or
# any key of 'DYNAMIC_DATASETS'. If using a dataset from `DATASETS_PATH`,
# set `DATASET_TYPE` to `DatasetType.STATIC`, otherwise set it to
# `DatasetType.Dynamic`.

# NOTE: "Kicks" actually does not work because the WAV files can be saved in
# various structures, not all of which are supported by TensorFlow. The files
# in "Kicks" are not all in a consistent compatible structure.

DATASET_TYPE = DatasetType.DYNAMIC

DATASET_NAME = 'Meows'

DATASET_PATH = DATASETS_PATH / DATASET_NAME


'''
Loads and displays a random WAV file from the dataset.
'''
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import wave


filenames = [
    filename for filename in os.listdir(DATASET_PATH)
    if not filename.startswith('.') and filename.endswith('.wav')
]

test_filename = random.choice(filenames)

print('File name:', test_filename)

test_file_path = DATASET_PATH / test_filename

with wave.open(test_file_path.as_posix()) as wav:
    bit_depth = wav.getsampwidth() * 8

    sample_rate = wav.getframerate()

# print('Bit depth:', bit_depth)
#
# print('Sample rate:', sample_rate)

# test_audio_tensor = tf.audio.decode_wav(
#     contents=tf.io.read_file(test_file_path.as_posix()),
#     desired_channels=1
# ).audio
#
# print('Tensor dimensions:', test_audio_tensor.shape)
#
# plt.plot(test_audio_tensor)
#
# For Google Colab:
# IPython.display.display(
#     IPython.display.Audio(test_file_path, rate=sample_rate))


'''
Builds and preprocesses the dataset.
'''
FT_frame_length, FT_frame_step = 513, 125

def load_wav(file_path):
    '''
    Loads a WAV file as a tensor.

    Stereo files will be flattened to be mono by taking channel 0.

    file_path: The path of a WAV file.

    Returns: Variable length `tf.Tensor`.
    '''
    # TFIO decoding
    # audio = tfio.audio.decode_wav(
    #   input=tf.io.read_file(file_path),
    #   dtype=tf.int16 if bit_depth == 16 else tf.int32
    # )
    #
    # Flatten to mono if necessary and remove the channel axis
    # return audio[:, 0]

    # TF decoding
    audio, _ = tf.audio.decode_wav(
        contents=tf.io.read_file(file_path),
        desired_channels=1,
        desired_samples=12_000
    )

    return tf.squeeze(audio)[2_000:]


def normalize(audio_tensor):
    '''
    Normalizes an audio signal.

    Scales an audio signal to entirely fill the range -1 to 1.

    audio_tensor: A tensor of audio data.

    Returns: `tf.float32` Normalized audio tensor.
    '''
    data_type_max = audio_tensor.dtype.max

    tensor_max = tf.reduce_max(tf.abs(audio_tensor))

    # Ensure `tensor_max` is non-zero to avoid arithmetic error
    scaling_factor = tf.cast(data_type_max / tensor_max, tf.float32)\
        if tensor_max != 0 else 1.0

    return tf.cast(audio_tensor, tf.float32) * scaling_factor / data_type_max


def input_pipeline(file_path):
    '''
    Performs dataset processing.

    file_path: A string tensor containing the name of a WAV file in the
        dataset.

    Returns: A 2D tensor of audio features (see STFT).
    '''
    if file_path.dtype == tf.string:
        signal = load_wav(file_path)
    else:
        signal = file_path

    spectrogram = tf.abs(
        tf.signal.stft(
            signal,
            frame_length=FT_frame_length,
            frame_step=FT_frame_step
        )
    )[0:-1, 0:-1]

    max = tf.reduce_max(spectrogram)

    return spectrogram / max


batch_size = 44

training_dataset = tf.data.Dataset.list_files(
    file_pattern=DATASET_PATH.as_posix() + '/[!.]*.wav',
    shuffle=True,
    seed=0
).map(
    map_func=input_pipeline,
    num_parallel_calls=tf.data.AUTOTUNE
).filter(
    lambda training_example:
      (not tf.math.reduce_any(tf.experimental.numpy.isnan(training_example)))
      and
      (not tf.math.reduce_any(tf.experimental.numpy.isinf(training_example)))
).batch(batch_size)

# Test the dataset:
for training_example in training_dataset:
    # Confirm there are no NaN's or inf's in the dataset:
    tf.debugging.assert_all_finite(training_example, str(training_example))

    # Confirm the values are normalized
    if not tf.experimental.numpy.isclose(
          tf.reduce_max(training_example).numpy(), 1):
      raise ValueError(f'Tensor is not normalized: {training_example}')

