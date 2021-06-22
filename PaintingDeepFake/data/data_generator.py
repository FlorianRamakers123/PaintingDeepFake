import requests
import PIL.Image as Image
import io
import tensorflow as tf
import numpy as np

CATALOG_PATH = "data/catalog.csv"


class DataGenerator(object):
    """ 
    Class for retrieving and preprocessing a batch of images. 
    These images are resized and cropped to the appropiate size and converted to NumPy arrays.
    """
    
    def __init__(self, batch_size, input_size, start=0):
        """
        Create a new DataGenerator.
        
        Args:
            batch_size (int): The size of the batch to generate.
            input_size ((int, int)): A (height, width) tuple specifying the dimensions of the image
            start (int): the amount of images to skip
        """
        self._batch_size = batch_size
        self._input_size = input_size
        self._catalog = open(CATALOG_PATH, 'r')
        self._start = start
        self._step = 0

    def get_next_batch(self):
        """
        Get the next preprocessed batch of images.

        Returns:
            A list of preprocessed images represented as 3D NumPy arrays. 
            All the arrays in the list have the same shape. 
        """
        images = []
        while len(images) < self._batch_size:
            line = self._catalog.readline()
            url, time = line.split(",")
            time1 = int(time.split('-')[0])
            if time1 < 1650:        # only consider paintings after 1650
                continue
            url = "https://www.wga.hu/art" + url.split("html")[1] + "jpg"
            self._step += 1
            if self._step < self._start:
                continue
            try:
                img_arr = self._scrape_image(url)
            except:
                continue
            if img_arr.shape[2] != 3:       # only consider RGB paintings
                continue
            img_arr = (img_arr - 127.5) / 127.5
            images.append(img_arr)
        
        result = np.stack(images, axis=0)
        assert result.shape == (self._batch_size, self._input_size[0], self._input_size[1], 3)
        return result       
    
    def _scrape_image(self, url):
        """
        Scrape the image of the specified url
        """
        img = Image.open(io.BytesIO(requests.get(url).content))
        out_h, out_w = self._input_size
        img_w, img_h = img.size
        img_r = img_w / img_h
        new_size = [out_w, out_h]
        if out_w / out_h <= img_r: 
            new_size[0] = int(round(out_h * img_r))
        else:
            new_size[1] = int(round(out_w / img_r))
        img = img.resize(new_size, Image.ANTIALIAS)
        assert (img.size[1] >= out_h and img.size[0] == out_w) or (img.size[0] >= out_w and img.size[1] == out_h), f"Size assertion failed, ({img.size}), {(out_w, out_h)}, {new_size}"
        arr = tf.keras.preprocessing.image.img_to_array(img)
        result = arr[:out_h, :out_w, :]
        assert result.shape[:2] == (out_h, out_w), f"{result.shape[:2]} != {(out_h, out_w)}"
        return result

    def destroy(self):
        self._catalog.close()
