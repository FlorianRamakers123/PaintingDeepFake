from data.data_generator import DataGenerator
from gan.discriminator import Discriminator
from gan.generator import Generator
from gan.train import train_step
from tqdm import tqdm
import tensorflow as tf

BATCH_SIZE = 20
DATA_IMAGE_SIZE = (256, 256)
EPOCHS = 40
BATCHES_PER_EPOCH = 40
SEED_SIZE = 100


def train():
    data_generator = DataGenerator(BATCH_SIZE, DATA_IMAGE_SIZE)
    discriminator = Discriminator(DATA_IMAGE_SIZE)
    generator = Generator(SEED_SIZE, DATA_IMAGE_SIZE[0])

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer(),
                                     discriminator_optimizer=discriminator.optimizer(),
                                     generator=generator.model(),
                                     discriminator=discriminator.model())

    manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    should_stop = False
    for epoch in range(EPOCHS):
        for _ in tqdm(range(BATCHES_PER_EPOCH)):
            try:
                batch = data_generator.get_next_batch()
                train_step(batch, discriminator, generator)
            except:
                should_stop = True
                break
        manager.save()
        if should_stop:
            break

    for _ in range(10):
        arr = generator()[0]
        tf.keras.preprocessing.image.array_to_img(arr).show()


if __name__ == "__main__":
    train()