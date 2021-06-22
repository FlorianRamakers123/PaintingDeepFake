import tensorflow as tf


@tf.function
def train_step(images, discriminator, generator):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator.loss(fake_output)
        disc_loss = discriminator.loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables())
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables())

    generator.optimizer().apply_gradients(zip(gradients_of_generator, generator.trainable_variables()))
    discriminator.optimizer().apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables()))