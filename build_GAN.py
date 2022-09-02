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
    # Generator:
    height, width, depth = 1, 512, 12

    latent_features = keras.Input(shape=(latent_dimensions,))

    # Convolutional architecture:
    dense_1 = keras.layers.Dense(
        height * width * depth,
        use_bias=False
    )(latent_features)

    batch_normalization_1 = keras.layers.BatchNormalization()(dense_1)

    leaky_relu_1 = keras.layers.LeakyReLU()(batch_normalization_1)

    reshape_1 = keras.layers.Reshape((height, width, depth))(leaky_relu_1)

    conv2d_1 = keras.layers.Conv2DTranspose(
            12, (3, 3), strides=(5, 1), padding='same', use_bias=False
    )(reshape_1)

    assert conv2d_1.shape == (None, 5, 512, 12)

    batch_normalization_2 = keras.layers.BatchNormalization()(conv2d_1)

    leaky_relu_2 = keras.layers.LeakyReLU()(batch_normalization_2)

    conv2d_2 = keras.layers.Conv2DTranspose(
            12, (4, 4), strides=(5, 1), padding='same', use_bias=False
    )(leaky_relu_2)

    assert conv2d_2.shape == (None, 25, 512, 12)

    batch_normalization_3 = keras.layers.BatchNormalization()(conv2d_2)

    leaky_relu_3 = keras.layers.LeakyReLU()(batch_normalization_3)

    conv2d_3 = keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(5, 5),
            strides=(3, 1),
            padding='same',
            use_bias=False,
            activation='tanh'
    )(leaky_relu_3)

    assert conv2d_3.shape == (None, 75, 512, 1)

    reshape_2 = keras.layers.Reshape((75, 512))(conv2d_3)

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
            input_shape=(*generator.output_shape[1:], 1)
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
