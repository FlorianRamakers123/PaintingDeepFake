import unittest

class test_gpu(unittest.TestCase):
    def test_tensorflow_gpu(self):
        import tensorflow as tf
        self.assertTrue(tf.test.is_built_with_cuda(), "Cuda not available to TensorFlow.")
        self.assertTrue(len(tf.config.list_physical_devices('GPU')) > 0, "No GPU devices found.")

    def test_pytorch_gpu(self):
        import torch
        self.assertTrue(torch.cuda.is_available(), "Cuda not available to PyTorch.")
        self.assertTrue(torch.cuda.device_count() > 0, "No GPU devices found.")


if __name__ == '__main__':
    unittest.main()
