'''
Defines the autoencoder architecture.
'''
import tensorflow as tf
import tensorflow.keras as keras


def build_autoencoder(input_dims, hidden_layer_sizes, latent_dims, batch_size):
    '''
    Creates a variational autoencoder.

    input_dims: An integer containing the dimensions of the model input.
    hidden_layer_sizes: A list containing the number of nodes for each hidden
        layer in the encoder, respectively.
    latent_dims: An integer containing the dimensions of the latent space
        representation.

    Returns: (Encoder, Decoder, Autoencoder)
        Encoder: The encoder model.
        Decoder: The decoder model.
        Autoencoder: The full autoencoder model.
    '''
    # Encoder architecture:
    input_layer = keras.Input(
        (input_dims,),
        name='encoder_input'
    )

    previous_layer = input_layer

    for node_count in hidden_layer_sizes:
        previous_layer = keras.layers.Dense(
            node_count,
            activation='relu'
        )(previous_layer)

    mean_layer = keras.layers.Dense(
        latent_dims,
        name='mean'
    )(previous_layer)

    log_variance_layer = keras.layers.Dense(
        latent_dims,
        name='log_variance'
    )(previous_layer)

    def normal_sample(inputs):
        ''' Draws samples from a normal distribution. '''
        mean, log_stddev = inputs

        # Generate a batch of random samples
        std_norm = tf.random.normal(
            shape=(batch_size, latent_dims),
            mean=0,
            stddev=1
        )  # KerasTensor

        sample = mean + tf.exp(log_stddev / 2) * std_norm

        return sample

    sample_layer = keras.layers.Lambda(normal_sample)(
        [mean_layer, log_variance_layer])

    encoder_outputs = [sample_layer, mean_layer, log_variance_layer]

    Encoder = keras.Model(input_layer, encoder_outputs, name='Encoder')

    # Decoder architecture:
    latent_space = keras.Input(
        (latent_dims,),
        name='decoder_input'
    )

    previous_layer = latent_space

    for node_count in reversed(hidden_layer_sizes):
        previous_layer = keras.layers.Dense(node_count, 'relu')(previous_layer)

    decoder_layers = keras.layers.Dense(input_dims, 'sigmoid')(previous_layer)

    Decoder = keras.Model(latent_space, decoder_layers, name='Decoder')


    # Complete autoencoder
    sample, mean, log_variance = Encoder(input_layer)

    # sample = keras.backend.print_tensor(sample, '\nsample:')

    reconstruction = Decoder(sample)

    Autoencoder = keras.Model(input_layer, reconstruction, name='autoencoder')

    def VAE_loss(inputs, reconstructions, log_variance_layer, mean_layer):
        ''' Custom loss function including a KL divergence term. '''
        '''
        reconstruction_loss = keras.losses.binary_crossentropy(
            inputs, reconstructions) * input_dims

        KL_loss = 1 + log_variance_layer - keras.backend.square(mean_layer) \
            - keras.backend.exp(log_variance_layer)

        KL_loss = keras.backend.sum(KL_loss, axis=-1) * -0.5

        total_loss = keras.backend.mean(reconstruction_loss + KL_loss)

        return total_loss
        '''
        # log_variance_layer = keras.backend.print_tensor(
        #    log_variance_layer, "\nLog variance:")

        # mean_layer = keras.backend.print_tensor(mean_layer, "Mean:")

        reconstruction_loss = keras.losses.binary_crossentropy(
            inputs, reconstructions) * input_dims

        # reconstruction_loss = tf.reduce_mean(
        #     keras.backend.square(inputs - reconstructions))

        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_variance_layer - tf.square(mean_layer) -
            tf.exp(log_variance_layer)
        )

        return keras.backend.mean(reconstruction_loss + kl_loss)

    Autoencoder.add_loss(
        VAE_loss(input_layer, reconstruction, log_variance, mean))

    return (Encoder, Decoder, Autoencoder)
