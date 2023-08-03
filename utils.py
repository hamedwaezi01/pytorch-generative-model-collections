import os, gzip, torch, csv
import torch.nn as nn
import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):
    
    def __init__(self, data=None, targets=None, original_set=None, csv_file=None, transform=None):
        super().__init__()
        self.transform = transform
        if csv_file is None or len(csv_file) == 0:
            self.data = torch.cat((original_set.data, data))
            self.targets = torch.cat((original_set.targets ,targets))
        elif data is not None or targets is not None or original_set is not None:
            raise ValueError()
        else:
            targets = []
            data = []
            with open(csv_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    targets.append(row[0])
                    data.append(row[1:])
            self.data = torch.tensor(data)
            self.targets = torch.tensor(targets)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return self.data.size(0)

    def save(self, path):
        data = self.data.reshape(-1, self.data.shape[1]*self.data.shape[2])
        with open(path, "w") as f:
            writer = csv.writer(f)
            for index, image in enumerate(data):
                writer.writerow((int(self.targets[index]), image))


def imshow_normalized(tensor):
    plt.imshow(((tensor + 1) * (255.0/2)).cpu().squeeze(), interpolation='nearest')

def generate_augmentation(model, label, count, batch_size=64):
    labels = torch.tensor([label] * batch_size).cuda()
    data = torch.tensor([]).cuda()
    model.G.eval()
    for i in range((count//batch_size) + (count%batch_size>0)):
        current_size = batch_size
        if i == count // batch_size:
            current_size = count%batch_size
        y_ = torch.zeros((current_size, model.class_num)).scatter_(1, labels[:current_size].type(torch.LongTensor).unsqueeze(1), 1).cuda()
        with torch.no_grad():
            z_ = torch.rand((current_size, model.z_dim)).cuda()
            new = model.G(z_, y_)
            data = torch.cat([data, new])

    return data

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    image = Image.fromarray(image * 255)
    return image.convert("RGB").save(path) if images.shape[-1] == 1 else image.save(path)
    # return imageio.imwrite(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, duration=200)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()