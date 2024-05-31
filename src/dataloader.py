import torch.utils.data
from src.dataset import BDD100K
from torchvision import transforms

from torch.utils.data import DataLoader
import torch


generator = torch.Generator()
generator.manual_seed(123)

data_mean = torch.tensor([0.485, 0.456, 0.406]) # need to find real one later
data_std = torch.tensor([0.229, 0.224, 0.225]) # need to calculate 

normalize = transforms.Normalize(
            mean=list(data_mean),
            std=list(data_std),
        )
train_transform = transforms.Compose(
                [
                 transforms.RandomHorizontalFlip(p=0.5), 
                 transforms.RandomCrop((32,32), padding=4),
                 transforms.PILToTensor(),
                 normalize
                ])

transform = transforms.Compose([transforms.PILToTensor(), 
                                 normalize])


def collection(data):
    images = []
    targets = []
    paths = []
    indices = []
    for sample in data:
        images += [sample[0]]
        targets += [sample[1]]
        paths += [sample[2]]
        indices += [sample[3]]
    return images, targets, paths, indices


def get_pool_loader(data_dir, batch_size, shuffle):
    ''' I should implement the custom dataset mean and std later.
    trainset = CustomDataset(train='train', path=data_dir, transform=None)
    data_mean, data_std = calculate_mean_std(trainset)    
    CustomDataset.data_mean = data_mean
    CustomDataset.data_std = data_std'''
    
    # set the mini-batch training
    trainset = BDD100K(root=data_dir, annFile="/home/siyu/Documents/al/scapstone/bdd100k/labels/det_20/det_train_coco.json", transform=transforms.PILToTensor())
    
    label_pool, unlabel_pool = torch.utils.data.random_split(dataset=trainset, lengths=[0.05, 0.95], generator=generator)

    # # For test
    # label_pool, unlabel_pool, _ = torch.utils.data.random_split(dataset=trainset, lengths=[20, 10, trainset.__len__() - 30], generator=generator)
    train_loader = DataLoader(label_pool, batch_size, shuffle=shuffle, collate_fn=collection, pin_memory=True)
    valid_loader = DataLoader(unlabel_pool, batch_size, shuffle=shuffle, collate_fn=collection, pin_memory=True)

    return train_loader, valid_loader


def get_train_loader(root, batch_size, shuffle):
    
    ''' I should implement the custom dataset mean and std later.
    trainset = CustomDataset(train='train', path=data_dir, transform=None)
    data_mean, data_std = calculate_mean_std(trainset)    
    CustomDataset.data_mean = data_mean
    CustomDataset.data_std = data_std'''
    
    # set the mini-batch training
    trainset = BDD100K(root=root, annFile="/home/siyu/Documents/al/scapstone/bdd100k/labels/det_20/det_train_coco.json", transform=transforms.PILToTensor())
    train_loader = DataLoader(trainset, batch_size, shuffle=shuffle, collate_fn=collection, pin_memory=True)
    return train_loader

def get_valid_loader(data_dir, batch_size, shuffle):
    # data_dir = os.path.join(data_dir, 'validsub')
    validset = BDD100K(root=data_dir, annFile="/home/siyu/Documents/al/scapstone/bdd100k/labels/det_20/det_val_coco.json", transform=transform)
    
    # set the mini-batch training
    valid_loader = DataLoader(validset, batch_size, shuffle=shuffle, collate_fn=collection)
    return valid_loader
    

# Only for inference
def get_test_loader(data_dir, batch_size, shuffle):
    
    # data_dir=os.path.join(data_dir,'valid')
    testset = BDD100K(root=data_dir, transform=transform)
    test_loader = DataLoader(testset, batch_size, shuffle=shuffle, collate_fn=collection)
    return test_loader