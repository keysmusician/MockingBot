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
# set `DATASET_TYPE` to `DatsetType.STATIC`, otherwise set it to
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
# IPython.display.display(IPython.display.Audio(test_file_path, rate=sample_rate))


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

'''
# Create a mock dataset of sine waves
from make_sine_dataset import make_sine_dataset

training_dataset = make_sine_dataset(
    time_step_count=10_000,
    training_example_count=1_100,
    sample_rate=44_100,
    minimum_frequency=21,
    maximum_frequency=300) \
    .map(input_pipeline, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(batch_size)

print('Dataset element shape:', training_dataset.element_spec.shape)
'''


'''
Defines the model compilation settings.
'''
TensorBoard_directory = pathlib.Path('./TensorBoard')

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(targets, generator_output):
    '''
    Discriminator loss function.

    targets: The training examples the model is learning to mimic.
    generator_output: The simulation(s) the generator produced.

    Returns: Scalar loss value.
    '''
    target_loss = cross_entropy(tf.ones_like(targets), targets)

    simulation_loss = cross_entropy(
        tf.zeros_like(generator_output), generator_output)

    jitter = 0 # tf.random.uniform([1], 0, 0.001)

    return target_loss + simulation_loss + jitter


def generator_loss(generator_output):
    '''
    Generator loss function.

    generator_output: The simulation(s) the generator produced.

    Returns: Scalar loss value.
    '''
    jitter = 0 # tf.random.uniform([1], 0, 0.001)

    return cross_entropy(
        tf.ones_like(generator_output), generator_output) + jitter


class GAN_Monitor(tf.keras.callbacks.Callback):
    '''
    Generates audio and spectrogram logs for TensorBoard.
    '''

    def __init__(self, write_frequency=5, simulation_count=1):
        '''
        Initializes a GAN_Monitor.

        simulation_count: The number of "fake" examples to generate.
        '''
        self.simulation_count = simulation_count

        self.write_frequency = write_frequency

        try:
            self.file_number = len(
                os.listdir(TensorBoard_directory / 'GAN' / 'audio'))
        except FileNotFoundError:
            self.file_number = 0


    def on_epoch_end(self, epoch, logs=None):
        '''
        Writes a spectrogram and audio file after each epoch as a TensorBoard
        event.

        epoch: The just-completed epoch.
        logs: Unused.
        '''
        # Only write every `write_frequency` epochs
        if epoch % self.write_frequency != 0:
            return

        random_latent_vectors = tf.random.normal(
            shape=(self.simulation_count, self.model.latent_dimensions),
            seed=0
        )

        simulations = self.model.generator(random_latent_vectors).numpy()

        # figure, axes = plt.subplots(2, 1)

        # axes[1].imshow(simulations[0, :, :], cmap='gray')

        reconstructed_signal = tf.signal.inverse_stft(
            stfts=tf.cast(simulations, tf.complex64),
            frame_length=FT_frame_length,
            frame_step=FT_frame_step,
            window_fn=tf.signal.inverse_stft_window_fn(FT_frame_step),
        )

        # plt.plot(reconstructed_signal[0])

        # plt.show()

        # Normalize amplitude
        reconstructed_signal *= 1 / tf.reduce_max(reconstructed_signal)

        file_number = str(self.file_number)

        file_writer = tf.summary.create_file_writer(
            str(TensorBoard_directory / 'GAN' / 'audio' / file_number))

        with file_writer.as_default(step=epoch + 1):
            tf.summary.audio(
                name='Audio',
                data=reconstructed_signal[:, :, None],
                sample_rate=sample_rate
            )

        file_writer = tf.summary.create_file_writer(str(
            TensorBoard_directory / 'GAN' / 'spectrograms' / file_number))

        with file_writer.as_default(step=epoch + 1):
            tf.summary.image(
                name='Spectrogram',
                data=simulations[:, :, :, None] * 255
            )


def learning_rate_schedule(epoch, learning_rate):
    '''
    The learning rate schedule.

    epoch: The current epoch.
    learning_rate: The initial learning rate at the current epoch.

    Returns: The learning rate.
    '''
    if epoch < 100:
        return learning_rate
    else:
        return learning_rate * 0.99


# Set up training checkpoints in case training is interrupted:
# checkpoint_dir = './training_checkpoints'
#
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#
# checkpoint = tf.train.Checkpoint(
#     generator_optimizer=generator_optimizer,
#     discriminator_optimizer=discriminator_optimizer,
#     generator=generator,
#     discriminator=discriminator
# )


'''
Trains the model.
'''
from build_GAN import build_GAN


tf.keras.backend.clear_session()

latent_dimensions = 100

generator, discriminator, gan = build_GAN(latent_dimensions)

gan.compile(
    discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    generator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    generator_loss_function=generator_loss,
    discriminator_loss_function=discriminator_loss
)

gan.fit(
    training_dataset,
    epochs=800,
    shuffle=True,
    callbacks=[
        GAN_Monitor(),
        tf.keras.callbacks.TensorBoard(
            log_dir=TensorBoard_directory / 'logs',
            histogram_freq=1
        ),
        tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)
    ]
)

# Save the model weights
MODEL_PATH = pathlib.Path('./saved_models') / gan.name

try:
    model_version = len(os.listdir(MODEL_PATH))
except (FileNotFoundError):
    model_version = 0

MODEL_PATH = MODEL_PATH / f'v{model_version}'

gan.generator.save(MODEL_PATH / 'generator')

gan.discriminator.save(MODEL_PATH / 'discriminator')
