from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import struct
import gzip

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, 'rb') as images_file:

      images_content = images_file.read()
      num_images = int.from_bytes(images_content[4:8], 'big')
      num_rows = int.from_bytes(images_content[8:12], 'big')
      num_cols = int.from_bytes(images_content[12:16], 'big')

      images = struct.unpack('>{0}B'.format(num_images * num_rows * num_cols), images_content[16:])
      X = np.array(images, dtype=np.float32).reshape(num_images, num_rows * num_cols)      
      min_x = np.min(X)
      max_x = np.max(X)
      normalized_X = (X - min_x) / (max_x - min_x)
            
    with gzip.open(label_filename, 'rb') as label_file:
      labels_content = label_file.read()
      num_labels = int.from_bytes(labels_content[4:8], 'big')
      labels = struct.unpack('>{0}B'.format(num_labels), labels_content[8:])
      y = np.array(labels, dtype=np.uint8)

    return (normalized_X, y)
    ### END YOUR SOLUTION

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        (self.X, self.y) = parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.X[index]
        if len(imgs.shape) > 1:
          imgs = np.array([ self.apply_transforms(img.reshape(28, 28, 1)).flatten() for img in imgs  ])
        else:
          imgs = self.apply_transforms(imgs.reshape(28, 28, 1)).flatten()
        y = self.y[index]
        return (imgs, y)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION