import argparse
import os
import random as rd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

import augment
import loss
import net
import logit_calibration
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="fer2013", type=str, help="fer2013 or ferplus or cifar100")
parser.add_argument('--dataset_path', default="data", type=str, help="path to dataset")
parser.add_argument('--num_classes', default=7, type=int, help="number of classes",choices=[7, 100])
parser.add_argument('--model', default="resnet18", type=str, help="resnet18 or resnet50" , choices=['resnet18', 'resnet50','resnet34','resnet101'])
parser.add_argument('--num_workers', default=4, type=int, help="number of workers")
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=250, type=int, help="number of epochs to train for (should be 120 for FER2013 and 250 for CIFAR100)")
parser.add_argument('--lr', default=0.1, type=float, help="initial learning rate")
parser.add_argument('--gpu', default=0, type=int, help="gpu id")
parser.add_argument('--seed', default=2024, type=int, help="random seed")
parser.add_argument('--method', default="CrossEntropy", type=str, help="loss function", choices=['CrossEntropy', 'DistilKL', 'Loca', 'LogitCalibration'])
parser.add_argument('--temp', default=3, type=float, help="temperature for softmax distillation")
parser.add_argument('--optimizer', default="SGD", type=str, help="optimizer", choices=['SGD', 'AdamW'])
parser.add_argument('--loss_coeff', default=0.3, type=float, help="loss coefficient")
parser.add_argument('--feature_loss_coeff', default=0.03, type=float, help="feature loss coefficient")
parser.add_argument('--save_path', default="checkpoints", type=str, help="path to save checkpoints")
parser.add_argument('--use_wandb', default=False, type=bool, help="use wandb")
args = parser.parse_args()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience (int): how many epochs of no improvement until termination
            min_delta (float): minimum difference in loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def setseed(seed):
    # torch.backends.cudnn.deterministic = True
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_model(model_name):
    if model_name == "resnet18":
        model = net.resnet18(num_classes=args.num_classes)
    elif model_name == "resnet34":
        model = net.resnet34(num_classes=args.num_classes)
    elif model_name == "resnet50":
        model = net.resnet50(num_classes=args.num_classes)
    elif model_name == "resnet101":
        model = net.resnet101(num_classes=args.num_classes)
    else:
        raise NotImplementedError
    return model

def setup_loss(loss_name):
    if loss_name == "CrossEntropy":
        loss_fn = loss.CEDistill(args.temp)
    elif loss_name == "DistilKL":
        loss_fn = loss.DistilKL(args.temp)
    elif loss_name == "Loca":
        loss_fn = logit_calibration.Loca(args.temp)
    elif loss_name == "LogitCalibration":
        loss_fn = logit_calibration.LogitCalibration2(args.temp)
    else:
        raise NotImplementedError
    return loss_fn

def setup_optimizer(optimizer_name, model):
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        raise NotImplementedError
    return optimizer
 
def setup_dataloader():
    if args.dataset == "fer2013":
        trainset = torchvision.datasets.ImageFolder(args.dataset_path+"/train", transform=augment.data_transforms_FER['train'])
        valset = torchvision.datasets.ImageFolder(args.dataset_path+"/val", transform=augment.data_transforms_FER['val'])
        testset = torchvision.datasets.ImageFolder(args.dataset_path+"/test", transform=augment.data_transforms_FER['test'])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == "ferplus":
        trainset = torchvision.datasets.ImageFolder(args.dataset_path+"/train", transform=augment.data_transforms_FER['train'])
        valset = torchvision.datasets.ImageFolder(args.dataset_path+"/val", transform=augment.data_transforms_FER['val'])
        testset = torchvision.datasets.ImageFolder(args.dataset_path+"/test", transform=augment.data_transforms_FER['test'])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(args.dataset_path, download=True,transform=augment.data_transforms_CIFAR['train'])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valset = torchvision.datasets.CIFAR100(args.dataset_path, download=True,transform=augment.data_transforms_CIFAR['val'])
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers)
        testloader = valloader
    else:
        raise NotImplementedError
    return trainloader, valloader, testloader
if args.use_wandb:
    import wandb
    wandb.init(project="BYOT-FER", config=args)
# ------------------------------------------- Set Up --------------------------------------------------
setseed(args.seed)
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
model = setup_model(args.model)
model = model.to(device)
loss_fn = setup_loss(args.method)
criterion = nn.CrossEntropyLoss()
setup_optimizer = setup_optimizer(args.optimizer, model)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(setup_optimizer, milestones=[args.epochs*1/3, args.epochs*2/3, args.epochs - 10], gamma=0.1)
trainloader, valloader, testloader = setup_dataloader()

global init 
init = False
# ------------------------------------------- Training --------------------------------------------------

def train(net, trainloader, optimizer, criterion, args, device, use_wandb, epoch, init):
    correct = [0 for _ in range(5)]
    predicted = [0 for _ in range(5)]

    net.train()
    sum_loss, total = 0.0, 0.0

    for i, data in tqdm(enumerate(trainloader, 0)):
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, outputs_feature = net(inputs)
        ensemble = sum(outputs)/len(outputs)
        ensemble.detach_()

        if init is False:
            layer_list = []
            teacher_feature_size = outputs_feature[0].size(1)
            for index in range(1, len(outputs_feature)):
                student_feature_size = outputs_feature[index].size(1)
                layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
            net.adaptation_layers = nn.ModuleList(layer_list)
            net.adaptation_layers.cuda()
            optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
            init = True

        loss = torch.FloatTensor([0.]).to(device)
        loss += criterion(outputs[0], labels)

        teacher_output = outputs[0].detach()
        teacher_feature = outputs_feature[0].detach()

        if args.method == "LogitCalibration":
            calibrated_logit, teachertemp = loss_fn(outputs[0], teacher_output)


        for index in range(1, len(outputs)):
            if args.method == "LogitCalibration":
                loss += F.kl_div(F.log_softmax(outputs[index]/teachertemp.unsqueeze(1), dim=1), F.softmax(calibrated_logit/teachertemp.unsqueeze(1), dim=1), reduction='batchmean')*args.loss_coefficient
                loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)
            else :
                loss += loss_fn(outputs[index], teacher_output) * args.loss_coefficient
                loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)

            if index != 1:
                loss += torch.dist(net.adaptation_layers[index-1](outputs_feature[index]), teacher_feature) * \
                        args.feature_loss_coefficient

        sum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(labels.size(0))
        outputs.append(ensemble)

        for classifier_index in range(len(outputs)):
            _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
            correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())

    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
          ' Ensemble: %.2f%%' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                  100 * correct[0] / total, 100 * correct[1] / total,
                                  100 * correct[2] / total, 100 * correct[3] / total,
                                  100 * correct[4] / total))

    if use_wandb:
        wandb.log({"epoch":epoch+1,
                    "loss":sum_loss/(i+1),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "Acc_4/4":100 * correct[0] / total,
                    "Acc_3/4":100 * correct[1] / total,
                    "Acc_2/4":100 * correct[2] / total,
                    "Acc_1/4":100 * correct[3] / total,
                    "Acc_Ensemble":100 * correct[4] / total
                    })


def validate(net, val_loader, criterion, args, device, use_wandb, epoch):
    # Save model with current best accuracy on validation 4/4 or lowest val loss
    best_accuracy = 0.0
    best_loss = float('inf')
    total_loss = 0.0
    correct = [0] * 5
    total = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, outputs_feature = net(inputs)
            ensemble = sum(outputs) / len(outputs)
            outputs.append(ensemble)
            total += float(labels.size(0))

            teacher_output = outputs[0].detach()
            teacher_feature = outputs_feature[0].detach()

            loss = torch.FloatTensor([0.]).to(device)
            loss += criterion(outputs[0], labels)

            if args.method == "LogitCalibration":
                calibrated_logit, teachertemp = loss_fn(outputs[0], teacher_output)

            for index in range(1, len(outputs)):
                if args.method == "LogitCalibration":
                    loss += F.kl_div(F.log_softmax(outputs[index]/teachertemp.unsqueeze(1), dim=1), F.softmax(calibrated_logit/teachertemp.unsqueeze(1), dim=1), reduction='batchmean')*args.loss_coefficient
                    loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)
                else :
                    loss += loss_fn(outputs[index], teacher_output) * args.loss_coefficient
                    loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)
                

                if index - 1 < len(net.adaptation_layers):
                    loss += torch.dist(net.adaptation_layers[index-1](outputs_feature[index]), teacher_feature) * args.feature_loss_coefficient
            total_loss += loss.item()

            for classifier_index in range(len(outputs)):
                _, predicted = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted.eq(labels.data).cpu().sum())

    accuracy = [100 * correct[index] / total for index in range(len(correct))]

    print('Validation Set Loss: %.03f | Accuracy: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
          ' Ensemble: %.4f%%' % (total_loss / (len(val_loader)), *accuracy))

    if use_wandb:
        wandb.log({"Validation Loss": total_loss / (len(val_loader)),
                   "Validation Accuracy 4/4": accuracy[0],
                   "Validation Accuracy 3/4": accuracy[1],
                   "Validation Accuracy 2/4": accuracy[2],
                   "Validation Accuracy 1/4": accuracy[3],
                   "Validation Accuracy Ensemble": accuracy[4],
                   })

    if accuracy[0] > best_accuracy or total_loss / len(val_loader) < best_loss:
        best_accuracy = accuracy[0]
        best_loss = total_loss / len(val_loader)
        # Check if save_path exists, if not create it
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        # Add more info to  model_best_epoch _ accuracy_ loss.pth: epoch count,accuracy and loss

        torch.save(
            net.state_dict(),
            os.path.join(args.save_path, 'model_best_epoch_%d_accuracy_%.3f_loss_%.3f.pth' % (epoch, accuracy[0], total_loss / len(val_loader))))

    return total_loss/ len(val_loader)

def test(net, test_loader, criterion, args, device, use_wandb, epoch):
    """
    Evaluate the model on the test set.
    """
    net.eval()
    correct = [0] * 5
    total = 0.0
    predictions = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = net(images)
            ensemble = sum(outputs) / len(outputs)
            outputs.append(ensemble)
            total += float(labels.size(0))
            _, predicted = torch.max(outputs[-1].data, 1)
            correct[-1] += float(predicted.eq(labels.data).cpu().sum())
            predictions.extend(predicted.cpu().numpy())

    accuracy = [100 * correct[index] / total for index in range(len(correct))]

    print('Test Set Accuracy: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
          ' Ensemble: %.4f%%' % tuple(accuracy))

    if use_wandb:
        wandb.log({
            "TestAcc_4/4": accuracy[0],
            "TestAcc_3/4": accuracy[1],
            "TestAcc_2/4": accuracy[2],
            "TestAcc_1/4": accuracy[3],
            "TestAccEnsemble": accuracy[-1],
        })


for epoch in range(1, args.epochs + 1):
    train(model, trainloader, setup_optimizer, criterion, args, device, args.use_wandb, epoch, init)
    val_loss = validate(model, valloader, criterion, args, device, args.use_wandb, epoch)
    lr_scheduler.step()
    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    if early_stopping(val_loss):
        print("Early stopping")
        # Test
        test(model, testloader, criterion, args, device, args.use_wandb, epoch)
        break
    # Test per 5 epoch
    if epoch % 5 == 0:
        test(model, testloader, setup_loss, args, device, args.use_wandb, epoch)
    