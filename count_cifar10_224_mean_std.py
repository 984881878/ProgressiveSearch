from torchvision import datasets, transforms
import torch.utils.data

# trans = transforms.Compose([transforms.ToTensor()])
# image_size = 32
trans = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
image_size = 224

train_data = datasets.CIFAR10('~/Datasets/cifar10', train=True, download=True, transform=trans)

data_loder = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=4, pin_memory=True)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    tmp = torch.tensor([0., 0., 0.])
    tmp = tmp.to(device=device)
    for batch_data, _ in data_loder:
        batch_data = batch_data.to(device=device)
        tmp_mean = batch_data.sum(dim=(0, 2, 3))
        # print(tmp_mean.shape)
        tmp += tmp_mean
        # print(tmp.shape)
    train_mean = tmp / (len(train_data) * image_size * image_size)
    print('train_mean: [' + ', '.join(['{0:.8f}'.format(e.item()) for e in train_mean]) + ']')

    tmp = torch.tensor([0., 0., 0.])
    tmp = tmp.to(device=device)
    for batch_data, _ in data_loder:
        batch_data = batch_data.to(device=device)
        tmp[0] += ((batch_data[:, 0, :, :] - train_mean[0]) ** 2).sum()
        tmp[1] += ((batch_data[:, 1, :, :] - train_mean[1]) ** 2).sum()
        tmp[2] += ((batch_data[:, 2, :, :] - train_mean[2]) ** 2).sum()
    train_std = (tmp / (len(train_data) * image_size * image_size)).sqrt()
    print('train_std: [' + ', '.join(['{0:.8f}'.format(e.item()) for e in train_std]) + ']')
else:
    'you\'d better compute it use gpu!\n'
