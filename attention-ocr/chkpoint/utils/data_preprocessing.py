import os

from numpy import append
list_train = []

def get_label(text):
    name_img = text
    label = ""
    for c in name_img:
        if (c == '_'):
            break
        else:
            label+=c
    return label

def valid_image(label):
        if len(label)<= 15:
            return True
        return False

# max = 0
# for index, label in enumerate(list_traindir):
#     if valid_image(label):
#         list_train.append(get_label(label))
# for label in list_train:
#     if max < len(label):
#         max = len(label)
# print(max)




# lexicon = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','-']
# print(len(lexicon))
# print(list_traindir[:5])
# creat_dataset.createDataset('C:/Users/HP/Desktop/crnn/lmdbdata/train', list_traindir, list_trainlabels)
# creat_dataset.createDataset('C:/Users/HP/Desktop/crnn/lmdbdata/test', list_testdir, list_testlabels)
# print(get_label(0))