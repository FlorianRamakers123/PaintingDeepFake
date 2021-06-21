import unittest
import gan.generator
import gan.discriminator
import tensorflow as tf

class test_gan(unittest.TestCase):
    def test_generator(self):
        generator = gan.generator.Generator(100, 256)
        output = generator()
        
        self.assertTrue(output.shape == (1, 256, 256, 3), f"{output.shape} != (1, 256, 256, 3).")
        tf.keras.preprocessing.image.array_to_img(output[0]).show()

    def test_discriminator(self):
        generator = gan.generator.Generator(100, 256)
        output = generator()
        self.assertTrue(output.shape == (1, 256, 256, 3), f"{output.shape} != (1, 256, 256, 3).")
        
        discriminator = gan.discriminator.Discriminator((256, 256))
        result = discriminator(output)
        self.assertTrue(result.shape == (1, 1))

if __name__ == '__main__':
    unittest.main()
