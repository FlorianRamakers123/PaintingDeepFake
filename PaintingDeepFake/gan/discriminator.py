from tensorflow.keras import layers
import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Discriminator(object):
    """
    The discriminator is a image classifier that classifies the generated images as real or fake.
    """

    def __init__(self, input_size):
        """
        Create a new Discriminator model with the specified input and output size.

        Args:
            input_shape ((int, int)): The size of the input images.
        """
        self._model = self._make_model(input_size)
        self._optimizer = tf.keras.optimizers.Adam(1e-4)

    def _make_model(self, input_shape):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[input_shape[0], input_shape[1], 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def optimizer(self):
        return self._optimizer

    def trainable_variables(self):
        return self._model.trainable_variables

    def model(self):
        return self._model

    def __call__(self, image, training=False):
        return self._model(image, training=training)
