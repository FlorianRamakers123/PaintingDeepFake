import unittest
import PaintingDeepFake.gan.generator as gm
import PaintingDeepFake.gan.discriminator as dm


class test_gan(unittest.TestCase):
    def test_generator(self):
        generator = gm.Generator(100, 256)
        output = generator()
        self.assertTrue(output.shape == (1, 256, 256, 3), f"{output.shape} != (1, 256, 256, 3).")

    def test_discriminator(self):
        generator = gm.Generator(100, 256)
        output = generator()
        self.assertTrue(output.shape == (1, 256, 256, 3), f"{output.shape} != (1, 256, 256, 3).")
        discriminator = dm.Discriminator((256, 256))
        result = discriminator(output)
        self.assertTrue(result.shape == (1, 1))


if __name__ == '__main__':
    unittest.main()
