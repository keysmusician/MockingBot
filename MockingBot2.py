'''
An unsupervised additive generative general audio model.
'''
# Create a dataset of sine waves
from make_sine_dataset import make_sine_dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


sample_rate = 8_000

training_data = make_sine_dataset(
    time_step_count=10_000,
    training_example_count=7_000,
    sample_rate=sample_rate,
    minimum_frequency=21,
    maximum_frequency=sample_rate / 2,
    include_labels=True
)

FT_frame_length, FT_frame_step = 513, 100

histogram = []

for training_example in training_data:
    signal, frequency = training_example.values()

    frequency = frequency.numpy().item()

    stft = tf.signal.stft(
        signal,
        frame_length=FT_frame_length,
        frame_step=FT_frame_step
    )

    spectrogram = tf.abs(stft)

    frequencies = np.arange(0, stft[0].shape[0]) \
        * sample_rate / FT_frame_length / 2

    frequency_distribution = np.abs(np.mean(stft, axis=0))

    approximate_frequency = frequencies[np.argmax(frequency_distribution)]

    histogram.append(approximate_frequency)

    ''' Replace this with plot/show/display_output or something
    # Display stats
    __, axes = plt.subplots(2,1)

    axes[0].imshow(spectrogram)

    axes[0].set_title(f'{frequency:.2f} Hz')

    axes[1].plot(frequency_distribution)

    print('real frequency:', frequency)

    print('max of stft:', approximate_frequency)

    plt.show()
    '''

plt.plot(histogram)

plt.show()
