from cProfile import label
from sklearn import neural_network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CTCLoss
from torch.nn.functional import softmax, log_softmax
import data_preprocessing
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
import os
from torch.autograd import Variable
from collections import Iterable



class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]      # kernel size
        ps = [1, 1, 1, 1, 1, 1, 0]      # padding
        ss = [1, 1, 1, 1, 1, 1, 1]      # stride
        nm = [64, 128, 256, 256, 512, 512, 512]   #ouput channel

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        output = log_softmax(output, dim=2)

        return output


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '@'  # for `-1` index

        self.dict = {}
        self.length = 0
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
            self.length += 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [ self.dict[char.lower() if self._ignore_case else char] for char in text ]
            length = [len(text)]
        elif isinstance(text, Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

alphabet = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-.'"
converter = strLabelConverter(alphabet, False)
batch_size = 16



class MyDataset(Dataset):
    def __init__(self, image_dir, transforms = None, cate =""):
        super().__init__()
        self.image_dir = image_dir
        self.transforms = transforms
        self.cate = cate


    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image_path = '../data/'+ self.cate + '/' + self.image_dir[index]
        label = data_preprocessing.get_label(self.image_dir[index])
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(e)
        if self.transforms:
            transformed = self.transforms(image)
            image = transformed
        # for c in label:
        #     if (c=='1'):
        #         print(label)
        return (image.float(), label)

list_train = os.listdir('../data/train')
list_traindir = []
for index,image in enumerate(list_train):
    if data_preprocessing.valid_image(data_preprocessing.get_label(image)):
        list_traindir.append(image)

list_testdir = os.listdir('../data/test')
transforms = Compose([
    transforms.Resize((32,512)), 
    transforms.ToTensor()
])
train_dataset = MyDataset(list_traindir, transforms= transforms, cate = 'train')
train_loader = DataLoader(train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
            )

test_dataset = MyDataset(list_testdir, transforms=transforms, cate = "test")
test_loader = DataLoader(test_dataset,
            batch_size=8,
            shuffle=True)

print("Dataset length: {0}".format(len(train_dataset)))

model = CRNN(32,3,converter.length,128).to("cpu")
optimizer = optim.Adam(model.parameters(), lr = 0.003)
loss_fn = CTCLoss(reduction = 'mean',zero_infinity=False)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X).to("cpu")
        # pred = pred.permute(2,1,0)
        pred_size = Variable(torch.IntTensor([pred.size(0)] * batch_size)).to("cpu")
        t, l = converter.encode(y)
        t = t.to("cpu")
        l = l.to("cpu")
        loss = loss_fn(pred,t,pred_size,l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch % 20 == 0) or (batch*len(X) == size):
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    


epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
models = neural_network()
models.load_state_dict(torch.load("model.pth"))
x_test, y_test = next(iter(test_loader)) 
pred = model(x_test)
print(pred)


# x_batch,y_batch = next(iter(train_loader))
# print(x_batch.shape)
# print(y_batch)

# x_batch,y_batch = next(iter(test_loader))
# print(len(train_loader))

# target = torch.randint(low=1, high=33, size = (8,30), dtype=torch.long)
# input = torch.rand(8,3,32,128)
# input_lengths = torch.full(size=(8,), fill_value=50, dtype=torch.long)
# target_lengths = torch.randint(low=10, high=30, size=(8,), dtype=torch.long)

# print(converter.encode(y_batch)[1].shape)
# print(converter.encode(y_batch)[1])
# print(converter.encode(y_batch)[0])

# pred = model(x_batch)
# pred_size = torch.IntTensor([pred.size(1)*4]*16)
# t, l = converter.encode(y_batch)

# # print(pred.shape)
# print(l)
# print(t)
# print(y_batch)



# loss = loss_fn(output,target, input_lengths, target_lengths)
# loss.backward()
# optimizer.step()
