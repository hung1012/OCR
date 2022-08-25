import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from utils import data_preprocessing
from utils.tokenizer import Tokenizer 

img_transforms = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, image_dir, transform = img_transforms, cate =""):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.cate = cate
        self.chars = list(' 1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.tokenizer = Tokenizer(self.chars)
    def __len__(self):
        return len(self.image_dir)
    def __getitem__(self,index):
        img_path = '../'+ self.cate + '/' + self.image_dir[index]
        s = data_preprocessing.get_label(self.image_dir[index])
        label = torch.full((15 + 2, ), self.tokenizer.EOS_token, dtype=torch.long)
        ts = self.tokenizer.tokenize(s)
        label[:ts.shape[0]] = torch.tensor(ts)
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(e)
        if self.transform:
            transformed = self.transform(image)
            image = transformed
        return (image.float(), label)