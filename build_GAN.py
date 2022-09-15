'''
Defines the GAN architecture.
'''
import os
import pathlib
import tensorflow.keras as keras
import tensorflow as tf


class GAN(keras.Model):
    '''
    A generative adversarial network.
    '''

    def __init__(self, discriminator, generator, latent_dimensions, **kwargs):
        super(GAN, self).__init__(**kwargs)

        self.discriminator = discriminator

        self.generator = generator

        self.latent_dimensions = latent_dimensions


    @property
    def metrics(self):
        return [self.generator_loss_metric, self.discriminator_loss_metric]


    def compile(
            self, discriminator_optimizer, generator_optimizer,
            discriminator_loss_function, generator_loss_function):
        super(GAN, self).compile()

        self.discriminator_optimizer = discriminator_optimizer

        self.generator_optimizer = generator_optimizer

        self.generator_loss = generator_loss_function

        self.discriminator_loss = discriminator_loss_function

        self.generator_loss_metric = keras.metrics.Mean(name="generator_loss")

        self.discriminator_loss_metric = keras.metrics.Mean(
            name="discriminator_loss")


    def save_networks(self, file_path, quiet=False):
        '''
        Saves the generator and discriminator models.

        file_path: The path at which to save the models.
        quiet: Whether to print a save confirmation or not.
        '''
        model_path = pathlib.Path(file_path) / self.name

        try:
            model_version = len(os.listdir(model_path))
        except (FileNotFoundError):
            model_version = 0

        model_path = model_path / f'v{model_version}'

        self.generator.save(model_path / 'generator')

        self.discriminator.save(model_path / 'discriminator')

        if not quiet:
            print(f'Generator and discriminator saved to: {model_path}')


    def train_step(self, targets):
        batch_size = tf.shape(targets)[0]

        latent_sample = tf.random.normal([batch_size, self.latent_dimensions])

        with tf.GradientTape() as generator_tape, \
                tf.GradientTape() as discriminator_tape:
            simulation = self.generator(latent_sample, training=True)

            target_evaluation = self.discriminator(targets, training=True)

            simulation_evaluation = self.discriminator(
                simulation, training=True)

            generator_loss = self.generator_loss(simulation_evaluation)

            discriminator_loss = self.discriminator_loss(
                target_evaluation, simulation_evaluation)

        gradients_of_generator = generator_tape.gradient(
            generator_loss, self.generator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )

        gradients_of_discriminator = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables)

        self.discriminator_optimizer.apply_gradients(
            zip(
                gradients_of_discriminator,
                self.discriminator.trainable_variables
            )
        )

        # Update metrics
        self.discriminator_loss_metric.update_state(discriminator_loss)

        self.generator_loss_metric.update_state(generator_loss)

        return {
            "discriminator_loss": self.discriminator_loss_metric.result(),
            "generator_loss": self.generator_loss_metric.result(),
        }


def build_GAN(latent_dimensions, time_steps, frequency_steps):
    '''
    Creates a generative adversarial network (GAN).

    latent_dims: An integer containing the dimensions of the latent space
        representation.
    time_steps: The resolution of the time domain of the generator's output and
        the training dataset spectrograms.
    frequency_steps: The resolution of the frequency domain of the generator's
        output and the training dataset spectrograms.

    Returns: (Generator, Discriminator, GAN):
        Generator: The generator Keras model.
        Discriminator: The discriminator Keras model.
        GAN: The complete generative adversarial network.
    '''
    # NOTE: Don't forget to rename the model if you change the architecture
    # for logging purposes. Alternatively, make it automatically detect when a
    # change has been made and come with a name so you don't log something
    # under the wrong name.

    # Generator - Dense architecture:

    relu = keras.activations.relu

    generator = keras.Sequential([
        keras.layers.Dense(
            25, activation=relu, input_shape=(latent_dimensions,)),

        keras.layers.Dense(100, activation=relu),

        keras.layers.Dense(200, activation=relu),

        keras.layers.Dense(300, activation=relu),

        keras.layers.Dense(400, activation=relu),

        keras.layers.Dense(500, activation=relu),

        keras.layers.Dense(time_steps + frequency_steps, activation=relu),

        keras.layers.Dense(time_steps * frequency_steps),

        keras.layers.Reshape((time_steps, frequency_steps))
    ])


    # Discriminator:
    discriminator = tf.keras.Sequential()

    discriminator.add(
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding='same',
            input_shape=(time_steps, frequency_steps, 1)
        )
    )

    discriminator.add(keras.layers.LeakyReLU())

    discriminator.add(keras.layers.Dropout(0.3))

    discriminator.add(
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    discriminator.add(keras.layers.LeakyReLU())

    discriminator.add(keras.layers.Dropout(0.3))

    discriminator.add(keras.layers.Flatten())

    discriminator.add(keras.layers.Dense(1))


    # GAN:
    gan = GAN(
        discriminator,
        generator,
        latent_dimensions,
        name='Dense5CentNet-ReLU' # Rename if you change the architecture
    )

    return generator, discriminator, gan


if __name__ == '__main__':
    from test_GAN import test_generator


    # Test that the model builds without errors:
    generator, discriminator, gan = build_GAN(
        latent_dimensions=128,
        time_steps=128,
        frequency_steps=512
    )

    # Ensure the forward pass behaves:
    # test_gan(gan)
    test_generator(generator)

    generator.summary()
