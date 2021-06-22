import unittest
import PaintingDeepFake.data.data_generator as dgm


class test_data_generator(unittest.TestCase):
    def test_get_next_batch(self):
        dgm.CATALOG_PATH = "../" + dgm.CATALOG_PATH
        dg = dgm.DataGenerator(10, (300, 400))
        batch = dg.get_next_batch()
        self.assertTrue(batch.shape == (10, 300, 400, 3), f"{batch.shape} != (10, 300, 400, 3).")
        self.assertTrue(batch.max() <= 1.0 and batch.min() >= -1.0, f"{batch.min()} < -1.0 or {batch.max()} > 1.0")
        dg.destroy()


if __name__ == '__main__':
    unittest.main()
