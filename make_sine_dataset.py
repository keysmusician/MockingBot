# Create a dataset of sine waves
import numpy as np
import tensorflow as tf


def make_sine_dataset(
        time_step_count, training_example_count, sample_rate,
        minimum_frequency, maximum_frequency):
    """
    Creates a dataset of sine waves uniformly distributed at random
    frequencies.

    time_step_count: The number of samples composing each wave.
    training_example_count: The number of sine waves to generate.
    sample_rate: The sample rate.
    minimum_frequency: Minimum frequency (inclusive).
    maximum_frequency: Maximum frequency (exclusive).

    Returns: `TF.data.Dataset`.
    """

    tau = np.pi * 2

    time_steps = np.arange(0, time_step_count)[None] \
        .repeat(training_example_count, 0)

    samples = np.random.uniform(
        minimum_frequency, maximum_frequency, [training_example_count])

    training_examples = np.sin(
        samples[:, None] * time_steps * tau / sample_rate)

    training_data = tf.data.Dataset.from_tensor_slices(training_examples)

    return training_data
