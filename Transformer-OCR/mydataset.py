import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import os
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import data_preprocessing
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

label_len = 32
vocab =  "< 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>"
# start symbol <
# end symbol >
char2token = {"PAD":0}
token2char = {0:"PAD"}
for i, c in enumerate(vocab):
    char2token[c] = i+1
    token2char[i+1] = c


def illegal(label):
    if len(label) > label_len-1:
        return True
    for l in label:
        if l not in vocab[1:-1]:
            return True
    return False

img_transforms = transforms.Compose([
    transforms.ToTensor(),
])

class MyDataset(Dataset):
    def __init__(self, image_dir, transform = img_transforms, cate =""):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.cate = cate
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index):
        img_path = '../'+ self.cate + '/' + self.image_dir[index]
        label_y_str = data_preprocessing.get_label(self.image_dir[index])
        '''
        line: image path\tlabel
        '''
        image = Image.open(img_path)
    
        if self.transform:
            transformed = self.transform(image)
            image = transformed

        label = np.zeros(label_len, dtype=int)
        for i, c in enumerate('<'+label_y_str):
            label[i] = char2token[c]
        label = torch.from_numpy(label)

        label_y = np.zeros(label_len, dtype=int)
        for i, c in enumerate(label_y_str+'>'):
            label_y[i] = char2token[c]
        label_y = torch.from_numpy(label_y) 
        
        return image.float(), label_y, label, label_y_str

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, trg_y, trg, pad=0):
        self.imgs = Variable(imgs.to(device), requires_grad=False)
        self.src_mask = Variable(torch.from_numpy(np.ones([imgs.size(0), 1, 44], dtype=np.bool)).to(device))
        if trg is not None:
            self.trg = Variable(trg.to(device), requires_grad=False)
            self.trg_y = Variable(trg_y.to(device), requires_grad=False)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask.to(device), requires_grad=False)

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, name):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.name = name
    def forward(self, x):
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name is self.name:
                b = x.size(0)
                c = x.size(1)
                return x.view(b, c, -1).permute(0, 2, 1)
        return None

# if __name__=='__main__':
#     listdataset = MyDataset('../crnn/data', img_transforms)
#     dataloader = torch.utils.data.DataLoader(listdataset, batch_size=2, shuffle=False, num_workers=0)
#     for epoch in range(1):
#         for batch_i, (imgs, labels_y, labels) in enumerate(dataloader):
#             continue


















