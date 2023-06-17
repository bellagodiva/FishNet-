import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import torchvision.transforms as T
import train

parser = argparse.ArgumentParser(description='fishnet cifar10')
parser.add_argument('--root_dir', type=str, default="/mnt/hard2/bella/cifar10",
                    help='root dir of dataset')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs (default: 40)')

args = parser.parse_args()


if torch.cuda.is_available():
    print('cuda')
    use_cuda = True

####################################################################
#
# Load the dataset 
#
####################################################################
# base framework from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Cifar10_Dataset(Dataset):
    def __init__(self, root_dir, split=None, transform=None):
        """
        Arguments:
            root_dir (string): Directory which contains train and test dir
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self._load_dict()
    
    def _load_dict(self):
        # extracts (image, label) dictionary following https://www.cs.toronto.edu/~kriz/cifar.html
        total_data = np.empty((0,3072))
        total_labels = np.empty((0))
        for batch in os.listdir(self.root_dir + '/' + self.split):
            with open(self.root_dir + '/' + self.split + '/' + batch, 'rb') as fo:
                batch_dict = pickle.load(fo, encoding='bytes')
                data = batch_dict[b'data']
                labels = batch_dict[b'labels']
                total_data = np.concatenate((total_data, data), axis=0)
                total_labels = np.concatenate((total_labels, labels), axis=0)
        
        # reshape images' np array from num_samples, 3072 -> num_samples, 3(RGB), 32(height), 32(width)
        self.num_samples = total_data.shape[0]
        images = np.reshape(total_data, (self.num_samples,3,32,32))
        # convert pixel values from 0-255 to 0-1 
        images = images/255
        # and then subtracting the mean and dividing the variance for each channel of the RGB respectively.
        images = images.transpose(1,0,2,3) #reshape into channel first (3, num_samples, 32, 32)
        c0_mean = np.mean(images[0])
        c1_mean = np.mean(images[1])
        c2_mean = np.mean(images[2])
        c0_std = np.std(images[0])
        c1_std = np.std(images[1])
        c2_std = np.std(images[2])
        images[0] = (images[0]-c0_mean)/c0_std
        images[1] = (images[1]-c1_mean)/c1_std
        images[2] = (images[2]-c2_mean)/c2_std
        images = images.transpose(1,0,2,3) #shape(n,c,h,w)
        
        total_images = torch.FloatTensor(images.astype(float))
        total_labels = torch.Tensor(total_labels)
        ## Page 7 random crop, horizontal flip and standard color augmentation [17]) used in [9] for fair comparison. 
        if self.split == 'train':
            transforms = T.Compose([
                    T.RandomGrayscale(0.5),
                    T.RandomHorizontalFlip(0.5)])
            augment_images = transforms(total_images)
            total_images = torch.concat((total_images, augment_images), dim=0)
            total_labels = torch.concat((total_labels, total_labels), dim=0)
        self.images = total_images
        self.labels = total_labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        sample = {'image': self.images[idx], 'label': self.labels[idx]}
        return sample

print("Start loading the data....")
train_path = os.path.join(args.root_dir, 'train_augmentplus.dt') 
val_path = os.path.join(args.root_dir, 'val_augmentplus.dt') 
test_path = os.path.join(args.root_dir, 'test_augmentplus.dt') 
if not os.path.exists(train_path):
    print("  - Creating new train data")
    train_data = Cifar10_Dataset(
        root_dir = args.root_dir,
        split = "train",
        )
    val_data = Cifar10_Dataset(
        root_dir = args.root_dir,
        split = "val",
        )
    test_data = Cifar10_Dataset(
       root_dir = args.root_dir,
        split = "test",
        )
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(test_data, test_path)
else:
    print("  - Found cached data")
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    test_data = torch.load(test_path)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
print('Finish loading the data....')


####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.use_cuda = use_cuda
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(val_data), len(test_data)
hyp_params.num_cls = 10
hyp_params.n_stage = 4
hyp_params.clip = 0.8
# parameters reference: https://github.com/kevin-ssy/FishNet/blob/master/models/net_factory.py
# fishnet150
hyp_params.n_channel = [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600]
hyp_params.n_res_block = [2, 4, 8, 4, 2, 2, 2, 2, 2, 4]
hyp_params.n_trans_block = [2, 2, 2, 2, 2, 4]

if __name__ == '__main__':
    test_loss = train.train_model(hyp_params, train_loader, valid_loader, test_loader)