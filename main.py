import copy
import time

import torch
import numpy as np
import argparse

from tqdm import tqdm
import torch.nn.functional as F

import os

from models import resnet18
from utlis import mkdir, Net, dataloader, test_cln, evaluate, getstrength, managerlr_wd, getstep, test_PGD10, \
    pag_attacks, create_logger, rmdir


def train_adv(model, optimizer, train_loader, epoch, device, strength, step):
    allloss = []
    correct = 0
    nums = 0
    for batch_index, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        advdatalist = []
        targetlist = []
        if(epoch > 0):
            data_catche = torch.load(f"{catche_filename}/_batch_{batch_index}.pt")
            cdata = data_catche["data"].to(device)
            ctarget = data_catche["target"].to(device)
            model.eval()
            cout = model(cdata)
            pred = cout.data.max(dim= 1)[1]
            effective_sign = pred == ctarget
            edata = cdata[effective_sign]
            etarget = ctarget[effective_sign]
            ##
            if(len(edata) > 0):
                advdatalist.append(edata)
                targetlist.append(etarget)
        advdatalist.append(data)
        targetlist.append(target)
        train_data = torch.cat(advdatalist, dim=0)
        train_target = torch.cat(targetlist, dim=0)
        steps = step
        advdata = pag_attacks(model, train_data, train_target, step_size=strength/steps*3, epsilon=strength, perturb_steps=steps)
        bz = len(data)
        save_data = {"data":advdata[-bz:], "target":train_target[-bz:]}
        torch.save(save_data, f"{catche_filename}/_batch_{batch_index}.pt")
        model.train()
        optimizer.zero_grad()
        advout = model(advdata)
        loss = F.cross_entropy(advout, train_target, reduction='mean') #+ clw_weight * F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()
        model.eval()
        allloss.append(loss.item())
        nums += len(train_target)
        correct += evaluate(advout, train_target)
    logger.info(f"loss= {np.mean(allloss)}, train adv acc[{correct}/{nums}]{np.round(correct/nums*100, 2)}%")

def train_cln(model, optimizer, train_loader, device):
    allloss = []
    correct = 0
    nums = 0
    model.train()
    for batch_index, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target, reduction='mean')
        loss.backward()
        optimizer.step()
        allloss.append(loss.item())
        nums += len(target)
        correct += evaluate(out, target)
    logger.info(f"loss= {np.mean(allloss)}, train cln acc[{correct}/{nums}]{np.round(correct/nums*100, 2)}%")
def main():
    modelname = f"{args.dataset}-{args.epochs}-{args.max_lr}-{args.max_wd}-{args.swa}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numclass = 100 if args.dataset =='cifar100' else 10
    model = Net(resnet18(input_channels=3, num_classes=numclass), dataset=args.dataset).to(device)
    train_loader, test_loader = dataloader(dataset = args.dataset, train_bz=args.batch_size, test_bz=args.batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, momentum=0.9, weight_decay=0, nesterov=False)
    logger.info(optimizer)
    gcorrect = 0
    gcln = 0
    gepoch = -1
    swamodel = torch.optim.swa_utils.AveragedModel(model)
    swamodel.eval()
    optimizercln = torch.optim.SGD(model.parameters(), lr=0.1*args.max_lr, momentum=0.9, weight_decay=0,nesterov=True)
    logger.info(optimizercln)
    start_time = time.time()
    for epoch in range(10):
        logger.info(f"Cln training epoch:{epoch}")
        train_cln(model, optimizercln, train_loader, device)
        clncorrect, allsample = test_cln(model, test_loader, device)
        logger.info(f">> test acc [{clncorrect}]/[{allsample}][{clncorrect/allsample*100}]%\n")
    for epoch in range(args.epochs):
        logger.info(f"Adv training epoch:{epoch}")
        lr, wd = managerlr_wd(optimizer, epoch, args.epochs, max_lr=args.max_lr, min_lr=args.min_lr, max_wd=args.max_wd, min_wd=args.min_wd, alpha=args.scale)
        strength = getstrength(args.epochs, epoch, args.max_strength, args.min_strength)
        step = getstep(args.epochs, epoch, maxstep=10, minstep=0)
        logger.info(f"LR={lr}, WD={wd}, Strength={strength}, Step={step}")
        train_adv(model, optimizer, train_loader, epoch, device, strength, step)
        testmodel = model
        if(epoch >= args.swa):
            swamodel.update_parameters(model)
            torch.optim.swa_utils.update_bn(train_loader, swamodel, device)
            testmodel = swamodel
        advcorrect, clncorrect, allsample = test_PGD10(testmodel, test_loader, device)
        logger.info(f">> test acc [{clncorrect}]/[{allsample}][{clncorrect/allsample*100}]% PGD10 acc [{advcorrect}]/[{allsample}][{advcorrect/allsample*100}]%\n")
        if(advcorrect > gcorrect and epoch >= args.swa):
            gepoch = epoch
            gcorrect = advcorrect
            gcln = clncorrect
            bestmodel = copy.deepcopy(testmodel.state_dict())
    end_time = time.time()
    torch.save(bestmodel, os.path.join(args.savemodels, f'{modelname}-bestmodel.pt'))
    torch.save(testmodel.state_dict(), os.path.join(args.savemodels, f'{modelname}-last.pt'))
    logger.info(f">> the best model in {gepoch} epoch and test the cln acc [{gcln}]/[{allsample}][{gcln/allsample*100}]%, best acc PGD10 acc [{gcorrect}]/[{allsample}][{gcorrect/allsample*100}]%\n")
    logger.info(f"The running time is :{end_time- start_time}s each epoch:{(end_time- start_time)/args.epochs}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Learning Rate and Weight Decay Scheduler")
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train for')
    parser.add_argument('--max-lr', type=float, default=0.33, help='Maximum learning rate')
    parser.add_argument('--min-lr', type=float, default=0, help='Minimum learning rate')
    parser.add_argument('--max-wd', type=float, default=1.2e-3, help='Maximum weight decay')
    parser.add_argument('--min-wd', type=float, default=0, help='Minimum weight decay')
    parser.add_argument('--max-strength', type=float, default=8, help='Maximum attack strength')
    parser.add_argument('--min-strength', type=float, default=4, help='Minimum attack strength')
    parser.add_argument('--scale', type=float, default=0.75, help='Scale factor for learning rate and weight decay')
    parser.add_argument('--cuda', type=int, default=1, help='CUDA device ID to use (negative value to disable CUDA)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--swa', type=int, default=100, help='Batch size for training')
    parser.add_argument('--savemodels', type=str, default='./savemodels-cifar10', help='folder of models')
    parser.add_argument('--logs', type=str, default='./logs-cifar10', help='folder of logs')
    args = parser.parse_args()
    args.max_strength = args.max_strength/255
    args.min_strength = args.min_strength/255
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.cuda}'
    catche_filename = f'catche-{args.cuda}'
    mkdir(catche_filename)
    mkdir(args.savemodels)
    logger = create_logger(logpath=args.logs,logfile= f'{args.dataset}-{args.swa}.log')
    logger.info(args)
    main()
    rmdir(catche_filename)