import glob
from PIL import Image
import numpy as np
class Dataset():
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = self.load_images()

    def load_images(self):
        images = []
        paths =  glob.glob(self.folder_path + '/*.jpg') +  glob.glob(self.folder_path + '/*.png') +  glob.glob(self.folder_path + '/*.gif')
        for filename in paths:
            img = Image.open(filename)
            if img is not None:
                images.append(img)
        return images
    
    def ToNumpy(self):
        return [np.array(img) for img in self.images]