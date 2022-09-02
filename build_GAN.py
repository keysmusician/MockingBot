'''
Defines the GAN architecture.
'''
import tensorflow.keras as keras
import tensorflow as tf


class GAN(keras.Model):
    ''' A generative adversarial network. '''

    def __init__(self, discriminator, generator, latent_dimensions):
        super(GAN, self).__init__()

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


def build_GAN(latent_dimensions):
    '''
    Creates a generative adversarial network (GAN).

    latent_dims: An integer containing the dimensions of the latent space
        representation.

    Returns: (Generator, Discriminator, GAN):
        Generator: The generator Keras model.
        Discriminator: The discriminator Keras model.
        GAN: The complete generative adversarial network.
    '''
    # Generator - Recurrent architecture:
    frequency_dimensions, time_steps = 512, 75

    latent_features = keras.Input(shape=(latent_dimensions, 1))

    lstm = keras.layers.LSTM(frequency_dimensions)(latent_features)

    reshape = keras.layers.Reshape([frequency_dimensions, 1])(lstm)

    lstm_t = keras.layers.LSTM(frequency_dimensions)(reshape)

    reshape_2 = keras.layers.Reshape([1, frequency_dimensions, 1])(lstm_t)

    for _ in range(time_steps - 1):
        lstm_t2 = keras.layers.LSTM(frequency_dimensions)(reshape)

        reshape_cat = keras.layers.Reshape(
            [1, frequency_dimensions, 1])(lstm_t2)

        reshape_2 = keras.layers.Concatenate(axis=1)([
            reshape_2,
            reshape_cat
        ])

    generator = keras.Model(latent_features, reshape_2)

    '''
    # Output dimensions:
    a, b = 75, 512

    generator.add(
        keras.layers.Dense(
            25, activation='relu', input_shape=(latent_dimensions,))
    )

    tanh = keras.activations.tanh

    generator.add(keras.layers.Dense(25, activation=tanh))

    generator.add(keras.layers.Dense(25, activation=tanh))

    generator.add(keras.layers.Dense(25, activation=tanh))

    generator.add(keras.layers.Dense(a * b))

    generator.add(keras.layers.Reshape((a, b)))
    '''


    # Discriminator:
    discriminator = tf.keras.Sequential()

    discriminator.add(
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding='same',
            input_shape=(time_steps, frequency_dimensions, 1)
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
    gan = GAN(discriminator, generator, latent_dimensions)

    return generator, discriminator, gan

if __name__ == '__main__':
    build_GAN(10)
