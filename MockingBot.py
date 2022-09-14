'''
Builds and preprocesses the dataset.
'''
from make_sine_dataset import make_sine_dataset
from MockingBot_datasets import DatasetManager
import tensorflow as tf


# Register dataset sources
dataset_manager = DatasetManager(dataset_sources={
    'Sines': lambda _: make_sine_dataset(
                time_step_count=10_000,
                training_example_count=1_100,
                sample_rate=44_100,
                minimum_frequency=500,
                maximum_frequency=15000
            ), # Mock dataset of random sine waves
})

dataset_manager.discover('/Volumes/T7/Code/Datasets')

dataset_manager.print_datasets()

# Choose the desired dataset here:
# NOTE: "Kicks" actually does not work because the WAV files can be saved in
# various structures, not all of which are supported by TensorFlow. The files
# in "Kicks" are not all in a consistent compatible structure.
training_dataset = dataset_manager['Meows']

sample_rate = training_dataset.sample_rate

FT_frame_length, FT_frame_step = 513, 74

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

    return spectrogram


batch_size = 64

desired_dataset_size = 60_000

dataset_size = len(training_dataset)

training_dataset = training_dataset.map(
    map_func=input_pipeline,
    num_parallel_calls=tf.data.AUTOTUNE
).filter(
    lambda training_example:
      (not tf.math.reduce_any(tf.experimental.numpy.isnan(training_example)))
      and
      (not tf.math.reduce_any(tf.experimental.numpy.isinf(training_example)))
).batch(batch_size) # .repeat(int(desired_dataset_size / dataset_size))

spectrogram_shape = training_dataset.element_spec.shape[1:]

# print('Dataset element shape:', training_dataset.element_spec.shape)


'''
Defines the model compilation settings.
'''
import pathlib
import os


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

    def __init__(self, write_frequency=1, simulation_count=1):
        '''
        Initializes a GAN_Monitor.

        simulation_count: The number of "fake" examples to generate.
        '''
        self.simulation_count = simulation_count

        self.write_frequency = write_frequency


    def __write_logs(self):
        '''
        Writes a spectrogram and audio file after each epoch as a TensorBoard
        event.

        Must only be called after training begins.

        epoch: The last completed epoch.
        '''
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

        trial_number = str(self.file_number)

        file_writer = tf.summary.create_file_writer(
            str(TensorBoard_directory / self.model.name / trial_number /
                'audio')
        )

        optimizer_details = self.model.generator_optimizer.get_config()

        with file_writer.as_default(step=self.epoch + 1):
            tf.summary.audio(
                name='Audio',
                data=reconstructed_signal[:, :, tf.newaxis],
                sample_rate=sample_rate,
                description=f'{optimizer_details}'
            )

        file_writer = tf.summary.create_file_writer(
            str(TensorBoard_directory / self.model.name / trial_number /
                'spectrograms')
        )

        with file_writer.as_default(step=self.epoch + 1):
            tf.summary.image(
                name='Spectrogram',
                data=simulations[:, :, :, tf.newaxis] * 255
            )


    def on_epoch_end(self, epoch, logs=None):
        '''
        Writes a spectrogram and audio file after each epoch as a TensorBoard
        event.

        epoch: The just-completed epoch.
        logs: Unused.
        '''
        self.epoch = epoch
        # Only write every `write_frequency` epochs
        if epoch % self.write_frequency == 0:
            self.__write_logs()


    def on_train_begin(self, logs=None):
        '''
        Determines the file number of the folder for logging spectrograms.

        logs: Unused.
        '''
        directory_numbers = [-1]

        try:
            trial_files = os.listdir(TensorBoard_directory / self.model.name)
        except FileNotFoundError:
            trial_files = []

        for directory in trial_files:
            try:
                directory_numbers.append(int(directory))
            except ValueError:
                continue

        self.file_number = max(directory_numbers) + 1


    def on_train_end(self, logs=None):
        '''
        Writes a spectrogram and audio file after the final epoch.
        '''
        self.__write_logs()


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
        return learning_rate * 0.95


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

generator, discriminator, gan = build_GAN(
    latent_dimensions, *spectrogram_shape)

gan.compile(
    discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    generator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    generator_loss_function=generator_loss,
    discriminator_loss_function=discriminator_loss
)

print('\nIMPORTANT:\nDid you document any hyperparameter changes?\n')

try:
    gan.fit(
        training_dataset,
        epochs=1000,
        shuffle=True,
        use_multiprocessing=True,
        callbacks=[
            GAN_Monitor(),
            tf.keras.callbacks.TensorBoard(
                log_dir=TensorBoard_directory / 'logs',
                histogram_freq=1
            )
        ]
    )
except KeyboardInterrupt:
    # Wrap this in a function:
    print()
    save_models = input('Save models before quitting? ').strip().lower()

    if save_models in ('n', 'no', 'nope', 'nah'):
        print("Quitting without saving.")

        exit()
    elif save_models in ('y', 'yes', 'yeah', 'please'):
        print('Saving...')
    else:
        print("I didn't understand, saving just in case...")

# Save the model weights
SAVED_MODELS_PATH = pathlib.Path('./saved_models')

# I have yet to implement `GAN.save()`
gan.save_networks(SAVED_MODELS_PATH)

