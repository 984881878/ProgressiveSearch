import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *


class Cifar10DataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=8, resize_scale=0.08, distort_color=None, resize_to_224=False, cutout=False,
                 cutout_length=16):
        self._save_path = save_path
        self.resize_to_224 = resize_to_224

        train_transform, valid_transform = self.build_transform(distort_color, resize_scale, resize_to_224, cutout,
                                                                cutout_length)
        tmp_dataset = datasets.CIFAR10(root=self.save_path, train=True, download=True)
        train_dataset = datasets.CIFAR10(root=self.save_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=self.save_path, train=False, download=True, transform=valid_transform)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(tmp_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            train_indexes, valid_indexes = self.random_sample_valid_set(
                [cls for _, cls in tmp_dataset], valid_size, self.n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            valid_dataset = datasets.CIFAR10(root=self.save_path, train=True, download=True, transform=valid_transform)

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '~/Datasets/cifar10'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download cifar10')

    # @property
    # def train_path(self):
    #     return os.path.join(self.save_path, 'train')

    # @property
    # def valid_path(self):
    #     return os.path.join(self.save_path, 'val')

    @property
    def normalize(self):
        if self.resize_to_224:
            return transforms.Normalize(mean=[0.49139944, 0.48215830, 0.44653127],
                                        std=[0.24125858, 0.23781145, 0.25642931])
        else:
            return transforms.Normalize(mean=[0.49140003, 0.48215839, 0.44653109],
                                        std=[0.24703231, 0.24348512, 0.26158798])

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        if self.resize_to_224:
            return 224
        return 32

    def build_transform(self, distort_color, resize_scale, resize_to_224, cutout, cutout_length):
        if resize_to_224:
            print('Color jitter: %s' % distort_color)
            if distort_color == 'strong':
                color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            elif distort_color == 'normal':
                color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
            else:
                color_transform = None

            if color_transform is None:
                train_transforms = transforms.Compose([
                    transforms.Resize(224),     # regard cifar10 img as 224 x 224
                    transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ])
            else:
                train_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    color_transform,
                    transforms.ToTensor(),
                    self.normalize,
                ])

            valid_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.Resize(self.resize_value),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    self.normalize,
                ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            if cutout:
                train_transforms.transforms.append(Cutout(cutout_length))

            valid_transforms = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

        return train_transforms, valid_transforms


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
