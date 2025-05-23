import os
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import custom_transforms as tr
import tifffile as tiff
import math


class HRSC(data.Dataset):
    def __init__(self, train=True, dataset='HRSC', crop_szie=None, val_full_img=True):  # false
        super(HRSC, self).__init__()
        self.dataset_dir = '/home/tbd/tdwc/dataset/shipdata/HRSC2016'
        self.train = train
        self.dataset = dataset
        self.val_full_img = val_full_img
        self.images = []
        self.labels = []
        self.names = []
        if crop_szie is None:
            crop_szie = [512, 512]
        self.crop_size = crop_szie
        if train:
            self.image_dir = self.dataset_dir + '/img/train'
            self.label_dir = self.dataset_dir + '/label/train'
        else:
            self.image_dir = self.dataset_dir + '/img/val'
            self.label_dir = self.dataset_dir + '/label/val'

        image_files = os.listdir(self.image_dir)
        label_files = os.listdir(self.label_dir)
        for image_file, label_file in zip(image_files, label_files):
            img_name = image_file.split('.')[0]
            image = os.path.join(self.image_dir, image_file)
            label = os.path.join(self.label_dir, img_name + '.png')
            # image = tiff.imread(image)
            image = Image.open(image)
            image = np.array(image)
            label = Image.open(label)
            label = np.array(label)
            # print(img_name)
            # print(image.shape)
            # print(label.shape)
            # print("------")
            if self.val_full_img:
                self.images.append(image)
                self.labels.append(label)
                self.names.append(img_name)
            else:
                slide_crop(image, self.crop_size, self.images)
                slide_crop(label, self.crop_size, self.labels)
        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}
        sample = self.transform(sample)
        if self.val_full_img:
            sample['name'] = self.names[index]
        return sample

    def transform(self, sample):
        if self.train:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomScaleCrop(base_size=self.crop_size, crop_size=self.crop_size),
                tr.ToTensor(add_edge=False),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.ToTensor(add_edge=False),
            ])
        return composed_transforms(sample)

    def __str__(self):
        return 'dataset:{} train:{}'.format(self.dataset, self.train)

class GE(data.Dataset):
    def __init__(self, train=True, dataset='GE', crop_szie=None, val_full_img=True):  # false
        super(GE, self).__init__()
        self.dataset_dir = '/home/tbd/tdwc/dataset/shipdata/GoogleEarth/'
        self.train = train
        self.dataset = dataset
        self.val_full_img = val_full_img
        self.images = []
        self.labels = []
        self.names = []
        if crop_szie is None:
            crop_szie = [512, 512]
        self.crop_size = crop_szie
        if train:
            self.image_dir = self.dataset_dir + '/img/train'
            self.label_dir = self.dataset_dir + '/label/train'
        else:
            self.image_dir = self.dataset_dir + '/img/val'
            self.label_dir = self.dataset_dir + '/label/val'

        image_files = os.listdir(self.image_dir)
        label_files = os.listdir(self.label_dir)
        for image_file, label_file in zip(image_files, label_files):
            img_name = image_file.split('.')[0]
            image = os.path.join(self.image_dir, image_file)
            label = os.path.join(self.label_dir, img_name + '.png')
            # image = tiff.imread(image)
            image = Image.open(image)
            image = np.array(image)
            label = Image.open(label)
            label = np.array(label)
            if self.val_full_img:
                self.images.append(image)
                self.labels.append(label)
                self.names.append(img_name)
            else:
                slide_crop(image, self.crop_size, self.images)
                slide_crop(label, self.crop_size, self.labels)
        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}
        sample = self.transform(sample)
        if self.val_full_img:
            sample['name'] = self.names[index]
        return sample

    def transform(self, sample):
        if self.train:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomScaleCrop(base_size=self.crop_size, crop_size=self.crop_size),
                tr.ToTensor(add_edge=False),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.ToTensor(add_edge=False),
            ])
        return composed_transforms(sample)

    def __str__(self):
        return 'dataset:{} train:{}'.format(self.dataset, self.train)


def slide_crop(image, label, crop_size, image_patches, label_patches, dataset,
               stride_rate=1.0/2.0, alpha=None, alpha_patches=None):
    """images shape [h, w, c]"""
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    if len(label.shape) == 2:
        label = np.expand_dims(label, axis=2)
    if alpha is not None:
        alpha = np.expand_dims(alpha, axis=2)
    stride_rate = stride_rate
    h, w, c = image.shape
    H, W = crop_size
    stride_h = int(H * stride_rate)
    stride_w = int(W * stride_rate)
    assert h >= crop_size[0] and w >= crop_size[1]
    h_grids = int(math.ceil(1.0 * (h - H) / stride_h)) + 1
    w_grids = int(math.ceil(1.0 * (w - W) / stride_w)) + 1
    for idh in range(h_grids):
        for idw in range(w_grids):
            h0 = idh * stride_h
            w0 = idw * stride_w
            h1 = min(h0 + H, h)
            w1 = min(w0 + W, w)
            if h1 == h and w1 != w:
                crop_img = image[h - H:h, w0:w0 + W, :]
                crop_label = label[h - H:h, w0:w0 + W, :]
                if alpha is not None:
                    crop_alpha = alpha[h - H:h, w0:w0 + W, :]
            if w1 == w and h1 != h:
                crop_img = image[h0:h0 + H, w - W:w, :]
                crop_label = label[h0:h0 + H, w - W:w, :]
                if alpha is not None:
                    crop_alpha = alpha[h0:h0 + H, w - W:w, :]
            if h1 == h and w1 == w:
                crop_img = image[h - H:h, w - W:w, :]
                crop_label = label[h - H:h, w - W:w, :]
                if alpha is not None:
                    crop_alpha = alpha[h - H:h, w - W:w, :]
            if w1 != w and h1 != h:
                crop_img = image[h0:h0 + H, w0:w0 + W, :]
                crop_label = label[h0:h0 + H, w0:w0 + W, :]
                if alpha is not None:
                    crop_alpha = alpha[h0:h0 + H, w0:w0 + W, :]
            crop_img = crop_img.squeeze()
            crop_label = crop_label.squeeze()
            if alpha is not None:
                crop_alpha = crop_alpha.squeeze()
            if (dataset in ['barley'] and np.any(crop_alpha > 0)) or dataset not in ['barley']:
                image_patches.append(crop_img)
                label_patches.append(crop_label)
                if alpha is not None:
                    alpha_patches.append(crop_alpha)


def label_to_RGB(image, classes=6):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    if classes == 6:  # potsdam and vaihingen
        palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    if classes == 4:  # barley
        palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    for i in range(classes):
        index = image == i
        RGB[index] = np.array(palette[i])
    return RGB


def RGB_to_label(image=None, classes=6):
    if classes == 6:  # potsdam and vaihingen
        palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    if classes == 4:  # barley
        palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    label = np.zeros(shape=[image.shape[0], image.shape[1]], dtype=np.uint8)
    for i in range(len(palette)):
        index = image == np.array(palette[i])
        index[..., 0][index[..., 1] == False] = False
        index[..., 0][index[..., 2] == False] = False
        label[index[..., 0]] = i
    return label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    remotedata_train = RemoteData(train=True, dataset='vaihingen')
    dataloader = DataLoader(remotedata_train, batch_size=1, shuffle=False, num_workers=1)
    # print(dataloader)

    for ii, sample in enumerate(dataloader):
        im = sample['label'].numpy().astype(np.uint8)
        pic = sample['image'].numpy().astype(np.uint8)
        print(im.shape)
        im = np.squeeze(im, axis=0)
        pic = np.squeeze(pic, axis=0)
        print(im.shape)
        im = np.transpose(im, axes=[1, 2, 0])[:, :, 0:3]
        pic = np.transpose(pic, axes=[1, 2, 0])[:, :, 0:3]
        print(im.shape)
        im = np.squeeze(im, axis=2)
        # print(im)
        im = label_to_RGB(im)
        plt.imshow(pic)
        plt.show()
        plt.imshow(im)
        plt.show()
        if ii == 10:
            break
