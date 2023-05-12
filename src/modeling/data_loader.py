import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import glob
from PIL import Image
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):

    def __init__(self, image_dir,transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = glob.glob(self.image_dir + "/images/*.tif")
        self.mask_list = []
        for image_path in self.image_list:
            self.mask_list.append(image_path.replace("images", "masks"))

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        mask = Image.open(self.mask_list[index]).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # convert to torch tensor
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        return image, mask
    
    def __len__(self):
        return len(self.image_list)


class Transform:
    """Transform image and masks using torchvision package with a probability,
    image should go through random zoom, contrast and brightness adjustment,
    mask should go through the same random zoom, but not contrast and brightness adjustment
    """
    def __init__(self, zoom_range=(1, 2), brightness_range=0.2, contrast_range=0.2, prob=0.5):
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.prob = prob

    def __call__(self, image, mask):
        # transform image and mask with a probability
        if torch.rand(1) < self.prob:
            # random zoom using same zoom factor for both image and mask
            zoom_factor = torch.rand(1) * (self.zoom_range[1] - self.zoom_range[0]) + self.zoom_range[0]
            image = T.functional.affine(image, scale=zoom_factor, angle=0, translate=(0,0), shear=0)
            mask = T.functional.affine(mask, scale=zoom_factor, angle=0, translate=(0,0), shear=0)
            # random contrast and brightness adjustment
            image = T.ColorJitter(brightness=self.brightness_range, contrast=self.contrast_range)(image)
        
        return image, mask

    

    
if __name__ == "__main__":
    image_dir = "data/selected_data_split/train"
    # dataset with transform
    dataset = SegmentationDataset(image_dir, transform=Transform())
    # print dataset length
    print(len(dataset))
    # Dataloader and display all images and masks of the first batch in one matplotlib subplot
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # display all images and masks of the first batch in one matplotlib subplot using imshow
    count = 0
    for images, masks in dataloader:
        while count < 4:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(T.ToPILImage()(images[count]))
            ax[1].imshow(T.ToPILImage()(masks[count]))
            plt.show()
            count += 1
        break

    
