"""
Module to split image into patches, choose the patches with significant variation in labels, 
then split patches into train, validation and test sets
"""
import cv2
import os
from PIL import Image
import numpy as np
from patchify import patchify
from tqdm import tqdm

class DataSplit:

    def __init__(self, patch_size, image_dir, mask_dir, output_dir, selection_threshold, labels_to_remove, new_label_map):
        self.patch_size = patch_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.threshold = selection_threshold
        self.labels_to_remove = labels_to_remove
        self.new_label_map = new_label_map
        # change k:v from uint8: uint8 to int: int
        self.new_label_map = {int(k): int(v) for k, v in self.new_label_map.items()}

    
    def split_and_select_patches(self):
        """Split and select images and corresponding mask, then save to folder"""
        # Get image names
        # have tqdm to show progress bar for loop
        progress_bar = tqdm(total=len(os.listdir(self.image_dir)), desc="Splitting and selecting patches")
        for root, sub_dirs, files in os.walk(self.image_dir):
            for file in files: 
                image_path = os.path.join(self.image_dir, file)
                mask_path = os.path.join(self.mask_dir, file)
                # check if mask_path exists
                if not os.path.exists(mask_path):
                    print("Mask path does not exist")
                    continue
                # Split image and masks
                image_patches = self._split_image(image_path=image_path, mask=False)
                mask_patches = self._split_image(image_path=mask_path, mask=True)
                # Select patches
                for i in range(image_patches.shape[0]):
                    for j in range(image_patches.shape[1]):
                        # If true, write
                        if self._select_patches(mask_patches[i,j]):
                            if not os.path.exists(self.output_dir):
                                os.makedirs(self.output_dir)
                            if not os.path.exists(os.path.join(self.output_dir, "images")):
                                os.makedirs(os.path.join(self.output_dir, "images"))
                            if not os.path.exists(os.path.join(self.output_dir, "masks")):
                                os.makedirs(os.path.join(self.output_dir, "masks"))
                            cv2.imwrite(os.path.join(self.output_dir, "images", file.strip('.tif')+"_patch_"+str(i)+"_"+str(j)+".tif"), image_patches[i,j][0])
                            cv2.imwrite(os.path.join(self.output_dir, "masks", file.strip('.tif')+"_patch_"+str(i)+"_"+str(j)+".tif"), mask_patches[i,j])

                progress_bar.update(1)
        progress_bar.close()


    def _split_image(self, image_path, mask = False):
        """Split and select images and corresponding mask, then save to folder"""

        if mask:
            image = cv2.imread(image_path, 0) # read greyscale, for mask
        else:
            image = cv2.imread(image_path, 1) # read color, for actual image
    
        # Crop the image to the closest size divisible by 256
        image = self._crop_image(image)

        # Next, get the smaller patches
        patches = self._divide_to_patch(image, mask)

        return patches

    def _crop_image(self, image):
        size_x = (image.shape[1] // self.patch_size) * self.patch_size # Get the closest size
        size_y = (image.shape[0] // self.patch_size) * self.patch_size # Get the closest size
        image = Image.fromarray(image) # Using pillow, convert the image np array into a PIL image in order to use the crop function
        image = image.crop((0 ,0, size_x, size_y)) # PIL.Image.crop(box) --> box defines the "coordinate" of the crop area we want
        image = np.array(image) # Then, convert the image back to np array

        return image
    
    def _divide_to_patch(self, image, mask = False):
        if mask:
            patches = patchify(image, (self.patch_size, self.patch_size), step=self.patch_size)  # we just need 1 dimension for masks
        else:
            patches = patchify(image, (self.patch_size, self.patch_size, 3), step=self.patch_size)  # RGB (3 dimensions) for images

        return patches        

    def _select_patches(self, mask_patch):
        """return true if the patch has significant variation in labels or contains labels to remove"""
        pixel_counts = np.unique(mask_patch, return_counts=True)[1]
        pixel_percentage = pixel_counts / pixel_counts.sum()
        if self.labels_to_remove is not None:
            # Remap mask labels according to a mapping dict
            mask_patch = np.vectorize(self.new_label_map.get)(mask_patch)
            # Find unique labels
            unique_labels = np.unique(mask_patch)
            # Check if unique labels are in labels_to_remove
            if np.isin(unique_labels, self.labels_to_remove).any():
                return False
        if max(pixel_percentage) < self.threshold:
            return True
        else:
            return False