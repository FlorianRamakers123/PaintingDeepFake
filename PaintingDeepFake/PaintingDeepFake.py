from data.data_generator import DataGenerator
from gan.discriminator import Discriminator
from gan.generator import Generator
from gan.train import train_step
from tqdm import tqdm
import tensorflow as tf
import time

BATCH_SIZE = 10
DATA_IMAGE_SIZE = (256, 256)
EPOCHS = 35
BATCHES_PER_EPOCH = 40
SEED_SIZE = 100
START = 15 * BATCHES_PER_EPOCH * BATCH_SIZE


def train():
    data_generator = DataGenerator(BATCH_SIZE, DATA_IMAGE_SIZE, START)
    discriminator = Discriminator(DATA_IMAGE_SIZE)
    generator = Generator(SEED_SIZE, DATA_IMAGE_SIZE[0])

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer(),
                                     discriminator_optimizer=discriminator.optimizer(),
                                     generator=generator.model(),
                                     discriminator=discriminator.model())

    manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    for epoch in range(EPOCHS):
        for i in tqdm(range(BATCHES_PER_EPOCH)):
            batch = data_generator.get_next_batch()
            train_step(batch, discriminator, generator)

        if (epoch + 1) % 10 == 0:
            manager.save()

    for _ in range(10):
        arr = generator()[0]
        tf.keras.preprocessing.image.array_to_img(arr).show()


if __name__ == "__main__":
    train()