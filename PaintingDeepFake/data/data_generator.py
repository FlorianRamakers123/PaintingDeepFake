import requests
import PIL.Image as Image
import io
import tensorflow as tf

CATALOG_PATH = "data/catalog.csv"

class DataGenerator(object):
    """ 
    Class for retrieving and preprocessing a batch of images. 
    These images are resized and cropped to the appropiate size and converted to NumPy arrays.
    """
    
    def __init__(self, batch_size, input_size):
        """
        Create a new DataGenerator.
        
        Args:
            batch_size (int): The size of the batch to generate.
            input_size ((int, int)): A tuple specifying the dimensions of the image 
        """
        self._batch_size = batch_size
        self._input_size = input_size
        self._catalog = open(CATALOG_PATH, 'r')
        self._cached_result = None
        

    def get_next_batch(self):
        """
        Get the next preprocessed batch of images.

        Returns:
            A list of preprocessed images represented as 3D NumPy arrays. 
            All the arrays in the list have the same shape. 
        """

        lines = self._cached_result if self._cached_result else self._read_next_lines()
        urls = []
        for line in lines:
            url, time = line.split(",")
            #time1 = int(time.split('-')[0])
            #if time1 >= 1650: #https://www.wga.hu/art/a/aachen/adonis.jpg
            urls.append("https://www.wga.hu/art" + url.split("html")[1] + "jpg")
        return [self._scrape_image(url) for url in urls]


    def _read_next_lines(self):
        """
        Read the next batch of lines from the catalog.

        Returns:
            A list containing the lines of the catalog file.
            The size of the list will be equal to the batch size or smaller if the end of the file is reached.
        """
        line = self._catalog.readline()
        for _ in range(self._batch_size):
            if line:
                yield line
            else:
                break
            line = self._catalog.readline()
        
    
    def _scrape_image(self, url):
        """
        Scrape the image of the specified url
        """
        img = Image.open(io.BytesIO(requests.get(url).content))
        arr = tf.keras.preprocessing.image.img_to_array(img)
        arr = arr[:self._input_size[0], :self._input_size[1], :]
        return arr

    def is_batch_available(self):
        """
        Check if there is another full batch of images available.

        Returns:
            True if a call to get_next_batch would result in a batch with the specified size, False otherwise.
        """
        self._cached_result = self._cached_result if self._cached_result else self._read_next_lines()
        if len(self._cached_result) == self._batch_size:
            return True
        return False

    def destroy(self):
        self._catalog.close()
