"""
Module to split image into patches, choose the patches with significant variation in labels, 
then split patches into train, validation and test sets
"""
import cv2
import os
from PIL import Image
import numpy as np
from patchify import patchify

class DataSplit:

    def __init__(self, patch_size, image_dir, dataset_dir, output_dir):
        self.patch_size = patch_size
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir

    def split_image(self, mask = False):
        # Get image names
        for root, sub_dirs, files in os.walk(self.image_dir):
            for file in files: 
                img_path = os.path.join(root, file)
                if mask:
                    image = cv2.imread(img_path, 0) # read greyscale, for mask
                else:
                    image = cv2.imread(img_path, 1) # read color, for actual image
            
                # Crop the image to the closest size divisible by 256
                image = self._crop_image(image)

                # Next, get the smaller patches
                if mask:
                    patches = patchify(image, (self.patch_size, self.patch_size), step=self.patch_size)  # we just need 1 dimension for masks
                else:
                    patches = patchify(image, (self.patch_size, self.patch_size, 3), step=self.patch_size)  # RGB (3 dimensions) for images
                
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        if mask:
                            single_patch = patches[i,j] # Single mask
                        else:
                            single_patch = patches[i,j][0] # Single image

                        if mask:
                            cv2.imwrite(self.dataset_dir+"256_patches/all/masks/" + file+"_patch_"+str(i)+str(j)+".tif", single_patch)
                        else: 
                            cv2.imwrite(self.dataset_dir+"256_patches/all/images/" + file+"_patch_"+str(i)+str(j)+".tif", single_patch)
    
    def _crop_image(self, image):
        size_x = (image.shape[1] // self.patch_size) * self.patch_size # Get the closest size
        size_y = (image.shape[0] // self.patch_size) * self.patch_size # Get the closest size
        image = Image.fromarray(image) # Using pillow, convert the image np array into a PIL image in order to use the crop function
        image = image.crop((0 ,0, size_x, size_y)) # PIL.Image.crop(box) --> box defines the "coordinate" of the crop area we want
        image = np.array(image) # Then, convert the image back to np array

        return image
        