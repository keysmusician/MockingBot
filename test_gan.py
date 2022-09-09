"""
Builds and tests a forward pass through a model.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model


def test_generator(generator:Model, example_count=1):
    """
    Generates and displays a simulated spectrogram from a generator model.

    generator: The generator of a GAN, taking as input some latent vector.
    example_count: The number of simulated spectrograms to generate.
    """

    # Test the forward pass through the generator:
    _batch_size, latent_dimensions = generator.input_shape

    noise = tf.random.normal([example_count, latent_dimensions])

    simulated_spectrogram = generator(noise, training=False)

    print('Simulated spectrogram shape:', simulated_spectrogram.shape)

    plt.imshow(simulated_spectrogram[0], cmap='gray')

    # reconstructed_signal = tf.signal.inverse_stft(
    #     stfts=tf.cast(test_simulation, tf.complex64),
    #     frame_length=FT_frame_length,
    #     frame_step=FT_frame_step,
    #     window_fn=tf.signal.inverse_stft_window_fn(FT_frame_step),
    # )

    # plt.plot(reconstructed_signal[0])

    plt.show()


def test_discriminator(discriminator):
    raise NotImplementedError
    # Test the forward pass through the discriminator:
    # evaluation = discriminator(test_simulation)

    # print('Evaluation score:', evaluation.numpy().item())

def test_gan(gan):
    raise NotImplementedError
    test_generator(gan.generator)
    test_discriminator(gan.discriminator)


if __name__ == '__main__':
    # This is a workaround while I migrate the data pipeline section
    # # Ideally you could `from MockingBot import SAVED_MODELS_PATH`
    import pathlib
    SAVED_MODELS_PATH = pathlib.Path('./saved_models')

    # Choose a saved model folder here:
    model_name = 'Dense4CentNet-tanh'

    generator = tf.keras.models.load_model(
        SAVED_MODELS_PATH / model_name / 'generator')

    discriminator = tf.keras.models.load_model(
        SAVED_MODELS_PATH / model_name / 'discriminator')

    test_generator(generator)
