# Create a dataset of sine waves
import numpy as np
import tensorflow as tf


def make_sine_dataset(
        time_step_count, training_example_count, sample_rate,
        minimum_frequency, maximum_frequency, randomize_phase=True):
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

    frequencies = np.random.uniform(
        minimum_frequency, maximum_frequency, [training_example_count]
    )[:, None]

    if randomize_phase:
        phases = np.random.uniform(0, tau, [training_example_count])[:, None]
    else:
        phases = 0

    training_examples = np.sin(
        frequencies * time_steps * tau / sample_rate + phases)

    training_data = tf.data.Dataset.from_tensor_slices(training_examples)

    return training_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_dataset = make_sine_dataset(
        time_step_count=10_000,
        training_example_count=1_100,
        sample_rate=44_100,
        minimum_frequency=21,
        maximum_frequency=300,
        randomize_phase=True
    )

    display_example_count = 5
    _, axes = plt.subplots(display_example_count, 1)

    for example_number, example in enumerate(
            test_dataset.take(display_example_count)):
        axes[example_number].plot(example)

    plt.show()
