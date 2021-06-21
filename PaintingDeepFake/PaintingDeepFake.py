from data.data_generator import DataGenerator
from gan.discriminator import Discriminator
from gan.generator import Generator
from gan.train import train_step
import tensorflow as tf
import time

BATCH_SIZE = 256
DATA_IMAGE_SIZE = (256, 256)
EPOCHS = 50
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

    for epoch in range(EPOCHS):
        start = time.time()
        print("----------------------------------------------------------------------------")
        for i in range(BATCHES_PER_EPOCH):
            batch = data_generator.get_next_batch()
            train_step(batch, discriminator, generator)
            print (f'Batch {i + 1} completed')

        print (f'Time for epoch {epoch + 1} is {time.time()-start} sec')
        
        if (epoch + 1) % 15 == 0:
            manager.save()

if __name__ == "__main__":
    train()