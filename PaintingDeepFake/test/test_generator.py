import unittest
import data.generator
 
class test_generator(unittest.TestCase):
    def test__scrape_img(self):
        dg = data.generator.DataGenerator(1, None)
        url = "https://www.wga.hu/art/a/aachen/adonis.jpg"
        img_data = dg._scrape_image(url)
        self.assertTrue(len(img_data) > 0, "Retrieved image was empty.")
        dg.destroy()

    def test_get_next_batch(self):
        dg = data.generator.DataGenerator(1, (300, 400))
        batch = dg.get_next_batch()
        self.assertTrue(len(batch) == 1)
        self.assertTrue(all(item.shape[0] == 300 for item in batch), "First dimension does not match the specification.")
        self.assertTrue(all(item.shape[1] == 400 for item in batch), "Second dimension does not match the specification.")
        self.assertTrue(all(item.shape[2] == 3 for item in batch), "Third dimension does not match the specification.")
        dg.destroy()
       
if __name__ == '__main__':
    unittest.main()
