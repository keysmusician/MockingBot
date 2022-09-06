import tensorflow as tf
import matplotlib.pyplot as plt

generator_spec = ' (Dense, 500 epochs with LRS)'

generator = tf.keras.models.load_model(
    './saved_models/generator' + generator_spec)

# These should ideally be imported from the main file
FT_frame_length, FT_frame_step = 513, 125

latent_dimensions = 100

noise = tf.random.normal([1, latent_dimensions])

test_simulation = generator(noise, training=False)

_, axes = plt.subplots(1, 2)

axes[0].imshow(tf.reverse(test_simulation[0], axis=[1]).numpy().T, cmap='magma')

reconstructed_signal = tf.signal.inverse_stft(
    stfts=tf.cast(test_simulation, tf.complex64),
    frame_length=FT_frame_length,
    frame_step=FT_frame_step,
    window_fn=tf.signal.inverse_stft_window_fn(FT_frame_step),
)

axes[1].plot(reconstructed_signal[0])

plt.show()
