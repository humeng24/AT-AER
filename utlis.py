import logging
import os

import numpy as np
import torch
import torchattacks
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F
from tqdm import tqdm


def mkdir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)

def rmdir(path):
    if(os.path.exists(path)):
        os.removedirs(path)


def getmean_std(dataset='cifar10'):
    if(dataset == 'cifar10'):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif(dataset == 'cifar100'):
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    elif(dataset == 'svhn'):
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.25, 0.25, 0.25)
    return mean, std

class Net(nn.Module):
    def __init__(self, backbone, dataset='cifar10'):
        super().__init__()
        mean, std = getmean_std(dataset=dataset)
        self.normalize = transforms.Normalize(mean, std)
        self.backbone = backbone
    def forward(self, data):
        out = self.normalize(data)
        out = self.backbone(out)
        return out

def dataloader(dataset = 'cifar10', train_bz=128, test_bz=128):
    transform_ = transforms.Compose([transforms.ToTensor()])
    train_transform_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    if(dataset == 'cifar10'):
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/cifar10', train=True, download=True, transform=train_transform_),
                batch_size=train_bz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data/cifar10', train=False, download=True, transform=transform_),
            batch_size=test_bz, shuffle=False)
        return train_loader, test_loader
    elif(dataset == 'cifar100'):
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=train_transform_),
                batch_size=train_bz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=transform_),
            batch_size=test_bz, shuffle=False)
        return train_loader, test_loader
    elif(dataset=='svhn'):
        train_loader = torch.utils.data.DataLoader(datasets.svhn.SVHN('../data/svhn', split= "train", download=True, transform=train_transform_),
                batch_size=train_bz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.svhn.SVHN('../data/svhn', split= "test", download=True, transform=transform_),
            batch_size=test_bz, shuffle=False)
        return train_loader, test_loader
    else:
        raise Exception()

def evaluate(out, target):
    pred = out.data.max(dim= 1)[1]
    return pred.eq(target.data).cpu().sum()

def test_cln(model, test_data_loader, device):
    clncorrect = 0
    allsample = 0
    model.eval()
    for data, target in tqdm(test_data_loader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            clncorrect += evaluate(model(data), target)
        allsample += len(target)
    return clncorrect, allsample

def test_PGD10(model, test_data_loader, device, steps = 10):
    adversary = torchattacks.PGD(model=model, steps = steps)
    clncorrect = 0
    advcorrect = 0
    allsample = 0
    model.eval()
    for data, target in tqdm(test_data_loader):
        data = data.to(device)
        target = target.to(device)
        advdata = adversary(data, target)
        with torch.no_grad():
            advout = model(advdata)
        advcorrect += evaluate(advout, target)
        clncorrect += evaluate(model(data), target)
        allsample += len(target)
    return advcorrect, clncorrect, allsample

def getstrength(epochs, epoch, maxstrength, minstrength):
    strength = (np.cos(epoch/epochs*np.pi-np.pi) + 1)/2 * (maxstrength-minstrength) + minstrength
    return strength

def getstep(epochs, epoch, maxstep, minstep):
    step = int((maxstep - minstep)/epochs*(epoch +1)+ minstep + 0.9999)
    return step

def getlr(epochs, epoch, maxlr, minlr, alpha=0.8):
    lr = (np.cos(epoch/epochs*np.pi) + 1)/2 * (maxlr-minlr) + minlr
    scale = (epochs-epoch)/epochs
    if(epoch % 2==0):
        lr *= (1-alpha)*scale
    else:
        lr *= (1+alpha)*scale
    return lr

def managerlr_wd(optimizer, epoch, epochs, max_lr, min_lr, max_wd, min_wd, alpha=0.8):
    lr = getlr(epochs, epoch, maxlr=max_lr, minlr=min_lr, alpha=alpha)
    wd = getstrength(epochs, epoch, max_wd, min_wd)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd
    return lr, wd

def pag_attacks(model, x_natural, y, step_size=2/255, epsilon=8/255, perturb_steps=10):
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits = model(x_adv)
            cost = F.cross_entropy(logits, y, reduction='mean')
        grad = torch.autograd.grad(cost, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv.detach_()
    return x_adv

def get_unique_filename(filename):
    base_name, extension = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_name}_{counter}{extension}"
        counter += 1
    return filename

def create_logger(logpath='./logs',logfile= 'log.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    mkdir(logpath)
    name = get_unique_filename(os.path.join(logpath, logfile))
    file = os.path.abspath(name)
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("logs are saved in "+file)
    return logger
