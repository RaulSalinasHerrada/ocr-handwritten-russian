

import random
import Augmentor
import torch

from collections import Counter
from torchvision import transforms
from htr_model.htr_params import *

def text_to_labels(s, char2idx):
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]

# store list of images' names (in directory) and does some operations with images
class TextLoader(torch.utils.data.Dataset):
    def __init__(self, images_name, labels, transforms, char2idx, idx2char):
        """
        params
        ---
        images_name : list
            list of names of images (paths to images)
        labels : list
            list of labels to correspondent images from images_name list
        char2idx : dict
        idx2char : dict
        """
        self.images_name = images_name
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.transform = transforms

    def _transform(self, X):
        # j = np.random.randint(0, 3, 1)[0]
        j = 0
        if j == 0:
            return self.transform(X)
        # if j == 1:
        #     return tt(ld(vignet(X)))
        # if j == 2:
        #     return tt(ld(un(X)))
            

    # shows some stats about dataset
    def get_info(self):
        N = len(self.labels)
        max_len = -1
        for label in self.labels:
            if len(label) > max_len:
                max_len = len(label)
        counter = Counter(''.join(self.labels))
        counter = dict(sorted(counter.items(), key=lambda item: item[1]))
        print(
            'Size of dataset: {}\nMax length of expression: {}\nThe most common char: {}\nThe least common char: {}'.format( \
                N, max_len, list(counter.items())[-1], list(counter.items())[0]))

    def __getitem__(self, index):
        img = self.images_name[index]
        img = self.transform(img)
        img = img / img.max()
        img = img ** (random.random() * 0.7 + 0.6)

        label = text_to_labels(self.labels[index], self.char2idx)
        return (torch.FloatTensor(img), torch.LongTensor(label))

    def __len__(self):
        return len(self.labels)


# MAKE TEXT TO BE THE SAME LENGTH
class TextCollate():
    def __call__(self, batch):
        x_padded = []
        y_padded = torch.LongTensor(LENGTH, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        return x_padded, y_padded
    

p = Augmentor.Pipeline()
p.shear(max_shear_left=2, max_shear_right=2, probability=0.7)
p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)

TRAIN_TRANSFORMS = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(CHANNELS),
            p.torch_transform(),  # random distortion and shear
            transforms.ColorJitter(contrast=(0.5,1),saturation=(0.5,1)),
            transforms.RandomRotation(degrees=(-9, 9)),
            transforms.RandomAffine(10, None, [0.6 ,1] ,3 ,fillcolor=255),
            #transforms.transforms.GaussianBlur(3, sigma=(0.1, 1.9)),
            transforms.ToTensor()
        ])

TEST_TRANSFORMS = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(CHANNELS),
            transforms.ToTensor()
        ])