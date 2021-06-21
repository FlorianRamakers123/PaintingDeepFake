import unittest
import data.data_generator


class test_data_generator(unittest.TestCase):
    def test_get_next_batch(self):
        dg = data.data_generator.DataGenerator(10, (300, 400))
        batch = dg.get_next_batch()
        self.assertTrue(batch.shape == (10, 300, 400, 3), f"{batch.shape} != (10, 300, 400, 3).")
        dg.destroy()
       
if __name__ == '__main__':
    unittest.main()
