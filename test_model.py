"""
Builds and tests a forward pass through a model.
"""
import matplotlib.pyplot as plt
from build_GAN import build_GAN
import tensorflow as tf


# Test the forward pass through the generator:
latent_dimensions = 10

generator, discriminator, gan = build_GAN(latent_dimensions)

noise = tf.random.normal([1, latent_dimensions])

test_simulation = generator(noise, training=False)

print('Test simulation shape:', test_simulation.shape)

# plt.imshow(test_simulation[0, :, :], cmap='gray')

# reconstructed_signal = tf.signal.inverse_stft(
#     stfts=tf.cast(test_simulation, tf.complex64),
#     frame_length=FT_frame_length,
#     frame_step=FT_frame_step,
#     window_fn=tf.signal.inverse_stft_window_fn(FT_frame_step),
# )

# plt.plot(reconstructed_signal[0])

# plt.show()

# Test the forward pass through the discriminator:
evaluation = discriminator(test_simulation)

print('Evaluation score:', evaluation.numpy().item())
