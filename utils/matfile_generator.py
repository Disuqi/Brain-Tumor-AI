import h5py
import numpy as np
import os
import tensorflow as tf
from scipy.ndimage import zoom
from tensorflow.keras.utils import to_categorical

class MatFileGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, image_shape=(256, 256, 3), batch_size=32, shuffle=True, rescale=1./255):
        self.directory = directory
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rescale = rescale
        self.file_list = os.listdir(directory)
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()

    def __len__(self):
        return len(self.file_list) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_files = [self.file_list[i] for i in indexes]
        return self.__data_generation(batch_files)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_files):
        images = []
        labels = []
        for file in batch_files:
            with h5py.File(os.path.join(self.directory, file), 'r') as mat:
                image = np.array(mat['cjdata']['image'])
                # Resize the image
                factors = (self.image_shape[0]/image.shape[0], self.image_shape[1]/image.shape[1])
                image = zoom(image, factors)
                # If the image is grayscale, convert it to RGB
                if image.ndim == 2:
                    image = np.stack((image,) * 3, axis=-1)
                # Rescale the image
                image = image * self.rescale
                images.append(image)
                # Adjust the labels
                label = int(mat['cjdata']['label'][0][0])
                if label == 1:
                    label = 0  # meningioma
                elif label == 2:
                    label = 1  # glioma
                elif label == 3:
                    label = 2  # pituitary
                else:
                    label = 3  # no tumor
                labels.append(label)
        images = np.array(images)
        labels = to_categorical(labels, num_classes=4)  # One-hot encoding
        return images, labels