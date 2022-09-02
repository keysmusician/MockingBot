'''
Makes datasets available.
'''
import os
import pathlib


DATASETS_PATH = pathlib.Path('/Volumes/T7/Code/Datasets/')

# Show available datasets:
print('Datasets:')

for dataset_name in os.listdir(DATASETS_PATH):
    print(dataset_name)


'''
Sets the project's dataset.
'''
# Choose the desired dataset here.
# `DATASET_NAME` can be the name of any folder inside `DATASETS_PATH`:
#
# NOTE: "Kicks" actually does not work because the WAV files can be saved in
# various structures, not all of which are supported by TensorFlow. The files
# in "Kicks" are not all in a consistent compatible structure.
DATASET_NAME = 'Meows'

DATASET_PATH = DATASETS_PATH / DATASET_NAME


'''
Loads and plots a random WAV file from the dataset.
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

print('Bit depth:', bit_depth)

print('Sample rate:', sample_rate)

# test_audio_tensor = tf.audio.decode_wav(
#     contents=tf.io.read_file(test_file_path.as_posix()),
#     desired_channels=1
# ).audio
#
# print('Tensor dimensions:', test_audio_tensor.shape)
#
# plt.plot(test_audio_tensor)
#
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

    file_path: A string tensor containing the name of a WAV file in the dataset.

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

from build_GAN import build_GAN
import tensorflow.keras as keras

# Test the forward pass through the generator:
latent_dimensions = 10

generator, discriminator, gan = build_GAN(latent_dimensions)

noise = tf.random.normal([1, latent_dimensions])

test_simulation = generator(noise, training=False)

# plt.imshow(test_simulation[0, :, :], cmap='gray')

reconstructed_signal = tf.signal.inverse_stft(
    stfts=tf.cast(test_simulation, tf.complex64),
    frame_length=FT_frame_length,
    frame_step=FT_frame_step,
    window_fn=tf.signal.inverse_stft_window_fn(FT_frame_step),
)

# plt.plot(reconstructed_signal[0])

# plt.show()

# Test the forward pass through the discriminator:
evaluation = discriminator(test_simulation)

print('Test simulation shape:', test_simulation.shape)

print('Evaluation score:', evaluation.numpy().item())


'''
Defines the model compilation parameters.
'''
TensorBoard_log_directory = "./TensorBoard_logs"

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    target_loss = cross_entropy(tf.ones_like(real_output), real_output)

    simulation_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    unstuckage_noise = 0 # tf.random.uniform([1], 0, 0.001)

    return target_loss + simulation_loss + unstuckage_noise


def generator_loss(fake_output):
    unstuckage_noise = 0 # tf.random.uniform([1], 0, 0.001)

    return cross_entropy(
        tf.ones_like(fake_output), fake_output) + unstuckage_noise


class GANMonitor(keras.callbacks.Callback):

    def __init__(self, simulation_count=3, latent_dimensions=128):
        self.simulation_count = simulation_count
        self.latent_dimensions = latent_dimensions
        try:
            self.file_number = len(
                os.listdir(TensorBoard_log_directory + '/GAN/audio'))
        except:
            self.file_number = 0

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.simulation_count, self.latent_dimensions),
            seed=0
        )

        simulations = self.model.generator(random_latent_vectors).numpy()

        #figure, axes = plt.subplots(2, 1)

        #axes[1].imshow(simulations[0, :, :], cmap='gray')

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

        file_writer = tf.summary.create_file_writer(
            TensorBoard_log_directory + f'/GAN/audio/{self.file_number}')

        with file_writer.as_default(step=epoch + 1):
            tf.summary.audio(
                name='Audio',
                data=reconstructed_signal[:, :, None],
                sample_rate=sample_rate
            )

        file_writer = tf.summary.create_file_writer(
            TensorBoard_log_directory + f'/GAN/spectrograms/{self.file_number}'
        )

        with file_writer.as_default(step=epoch + 1):
            tf.summary.image(
                name='Spectrogram',
                data=simulations[:, :, :, None] * 255
            )

'''This didn't help
class DiscriminatorWarmup(keras.callbacks.Callback):
    def __init__(self, epochs_to_await):
        super().__init__()
        self.epochs_to_await = epochs_to_await

    def on_train_begin(self, logs):
        self.model.train_generator = False

    def on_epoch_end(self, epoch, logs):
        if epoch > self.epochs_to_await:
            self.model.train_generator = True
'''

# The discriminator and the generator optimizers are different since I will
# train two networks separately. Will set this up when I figure our Keras
# checkpointing:
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
#
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


'''
Trains the model.
'''
latent_dimensions = 100

examples_to_generate_count = 1

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

keras.backend.clear_session()

generator, discriminator, gan = build_GAN(latent_dimensions)

gan.compile(
    discriminator_optimizer=keras.optimizers.Adam(learning_rate=0.00002),
    generator_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    generator_loss_function=generator_loss,
    discriminator_loss_function=discriminator_loss
)

training_history = gan.fit(
    training_dataset,
    epochs=10,
    shuffle=True,
    callbacks=[
        GANMonitor(simulation_count=1, latent_dimensions=latent_dimensions),
        tf.keras.callbacks.TensorBoard(
            log_dir=TensorBoard_log_directory,
            histogram_freq=1
        ),
    ]
)

# Save the model weights
MODEL_PATH = pathlib.Path('./saved_models')

gan.generator.save(MODEL_PATH / 'generator')

gan.discriminator.save(MODEL_PATH / 'discriminator')
