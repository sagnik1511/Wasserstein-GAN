import os
import config
from glob import glob
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split


class CustomDataset(Dataset):

    def __init__(self,
                 image_dir,
                 h=100,
                 w=100,
                 transform=None):
        super(CustomDataset, self).__init__()

        assert os.path.isdir(image_dir), "image_dir path not found!"
        self.image_dir = image_dir
        self.H, self.W = h, w
        self.transform = transform
        self.files = glob(f"{self.image_dir}/*")

    def load_image(self, path):
        image = Image.open(path)
        image = image.resize((self.H, self.W))
        return image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = self.load_image(path)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image


def create_data_patches(data_dir, h=100, w=100,
                        transform=None, batch_size=64,
                        shuffle=False):
    if not transform:
        transform = config.default_transformations
    dataset = CustomDataset(data_dir, h, w, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader


def main():
    batch_size = 2
    image_height = image_width = 224
    batch_size = 64
    loader = create_data_patches(config.IMG_FOLDER_PATH,
                                 h=image_height, w=image_width,
                                 batch_size=batch_size, shuffle=True)
    for patch in loader:
        assert patch.shape[0] == batch_size
        assert patch.shape[2] == image_height
        assert patch.shape[3] == image_width
        print("Success!!!")
        break


if __name__ == "__main__":
    main()

