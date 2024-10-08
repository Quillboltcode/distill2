import os
import random as rd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger, CSVLogger

import augment
import loss
import net
import logit_calibration

class ResNetAdaptation(nn.Module):
    def __init__(self, model_name, num_classes ):
        super().__init__()
        self.num_classes = num_classes
        self.model = self.setup_model(model_name)
        self.adaptation_layers = None
        self.init_adaptation_layers()

    def setup_model(self, model_name):
        if model_name == "resnet18":
            model = net.resnet18(num_classes=self.num_classes)
        elif model_name == "resnet34":
            model = net.resnet34(num_classes=self.num_classes)
        elif model_name == "resnet50":
            model = net.resnet50(num_classes=self.num_classes)
        elif model_name == "resnet101":
            model = net.resnet101(num_classes=self.num_classes)
        else:
            raise NotImplementedError
        return model

    def init_adaptation_layers(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        _, outputs_feature = self.model(dummy_input)
        layer_list = []
        teacher_feature_size = outputs_feature[0].size(1)
        for index in range(1, len(outputs_feature)):
            student_feature_size = outputs_feature[index].size(1)
            layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
        self.adaptation_layers = nn.ModuleList(layer_list)
    def forward(self, x):
        return self.model(x) 


class LitModel(LightningModule):
    def __init__(self, model_name="resnet18", 
                 method="CrossEntropy",
                 dataset="fer2013",
                 dataset_path="data", 
                 lr=0.1, 
                 optimizer="SGD", batch_size=256,
                 num_workers=4,
                 temp=3, loss_coefficient=0.3,
                 feature_loss_coefficient=0.03,
                 save_path="checkpoints",
                 num_classes=7):

        super(LitModel, self).__init__()
        # self.args = args
        self.best_accuracy = 0.0
        self.best_loss = 9999
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hparams.model_name = model_name
        self.hparams.method = method
        self.hparams.dataset = dataset
        self.hparams.dataset_path = dataset_path
        self.hparams.lr = lr
        self.hparams.optimizer = optimizer
        self.hparams.batch_size = batch_size
        self.hparams.num_workers = num_workers
        self.hparams.temp = temp
        self.hparams.loss_coefficient = loss_coefficient
        self.hparams.feature_loss_coefficient = feature_loss_coefficient
        self.hparams.save_path = save_path
        self.hparams.num_classes = num_classes

        # Setup model, loss, optimizer, and scheduler
        self.model = self.setup_model(model_name)
        self.loss_fn = self.setup_loss(method)
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()
        
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/17182
        self.validation_step_outputs = []
        # self.test_step_outputs = []

    def setup_model(self, model_name):
        return ResNetAdaptation(model_name, self.hparams.num_classes)



    def setup_loss(self, loss_name):
        if loss_name == "CrossEntropy":
            return loss.CEDistill(self.hparams.temp)
        elif loss_name == "DistilKL":
            return loss.DistilKL(self.hparams.temp)
        elif loss_name == "Loca":
            return logit_calibration.Loca(self.hparams.temp)
        elif loss_name == "LogitCalibration":
            return logit_calibration.LogitCalibration2(self.hparams.temp)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        else:
            raise NotImplementedError

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[80,160,240], 
            gamma=0.1,
            verbose=False
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        trainset = torchvision.datasets.ImageFolder(self.hparams.dataset_path+"/train", transform=augment.data_transforms_FER['train'])
        return DataLoader(trainset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        valset = torchvision.datasets.ImageFolder(self.hparams.dataset_path+"/val", transform=augment.data_transforms_FER['val'])
        return DataLoader(valset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        testset = torchvision.datasets.ImageFolder(self.hparams.dataset_path+"/test", transform=augment.data_transforms_FER['test'])
        return DataLoader(testset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, outputs_feature = self.model(inputs)

        # Initialize adaptation layers if not done


        # Loss calculation
        loss = torch.FloatTensor([0.]).to(self.device)
        loss += self.criterion(outputs[0], labels)
        teacher_output = outputs[0].detach()
        teacher_feature = outputs_feature[0].detach()

        if self.hparams.method == "LogitCalibration":
            calibrated_logit, teachertemp = self.loss_fn(teacher_output, labels)
        elif self.hparams.method == "Loca":
            calibrated_logit = self.loss_fn(teacher_output, labels)

        for index in range(1, len(outputs)):
            if self.hparams.method == "LogitCalibration":
                loss += F.kl_div(
                    F.log_softmax(outputs[index] / teachertemp.unsqueeze(1), dim=1),
                    F.softmax(calibrated_logit / teachertemp.unsqueeze(1), dim=1),
                    reduction='batchmean'
                ) * self.hparams.loss_coefficient
                loss += self.criterion(outputs[index], labels) * (1 - self.hparams.loss_coefficient)
            elif self.hparams.method == "Loca":
                loss += F.kl_div(
                    F.log_softmax(outputs[index] , dim=1),
                    F.softmax(calibrated_logit , dim=1),
                    reduction='batchmean'
                ) * self.hparams.loss_coefficient
                loss += self.criterion(outputs[index], labels) * (1 - self.hparams.loss_coefficient)
            else:
                loss += self.loss_fn(outputs[index], teacher_output) * self.hparams.loss_coefficient
                loss += self.criterion(outputs[index], labels) * (1 - self.hparams.loss_coefficient)

            if index != 1:
                loss += torch.dist(self.model.adaptation_layers[index - 1](outputs_feature[index]), teacher_feature) * \
                        self.hparams.feature_loss_coefficient

        # Accuracy calculation
        # self.log_accuracy(outputs, labels)

        # Log the training loss and accuracy
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    # def log_accuracy(self, outputs, labels):
    #     for classifier_index in range(len(outputs)):
    #         _, self.predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
    #         self.correct[classifier_index] += float(self.predicted[classifier_index].eq(labels.data).cpu().sum())
    #     self.log('train_acc', 100 * self.correct[-1] / len(labels), on_step=True, on_epoch=True, prog_bar=True, logger=True , sync_dist=True)

    def on_training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, prog_bar=True , sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, outputs_feature = self.model(inputs)


        total_loss = 0.0
        correct = [0] * 4
        total = float(labels.size(0))

        teacher_output = outputs[0].detach()
        teacher_feature = outputs_feature[0].detach()

        loss = torch.FloatTensor([0.]).to(self.device)
        loss += self.criterion(outputs[0], labels)

        if self.hparams.method == "LogitCalibration":
            calibrated_logit, teachertemp = self.loss_fn(teacher_output, labels)
        elif self.hparams.method == "Loca":
            calibrated_logit = self.loss_fn(teacher_output, labels)

        for index in range(1, len(outputs)):
            if self.hparams.method == "LogitCalibration":
                loss += F.kl_div(
                    F.log_softmax(outputs[index] / teachertemp.unsqueeze(1), dim=1),
                    F.softmax(calibrated_logit / teachertemp.unsqueeze(1), dim=1),
                    reduction='batchmean'
                ) * self.hparams.loss_coefficient
                loss += self.criterion(outputs[index], labels) * (1 - self.hparams.loss_coefficient)
            elif self.hparams.method == "Loca":
                loss += F.kl_div(
                    F.log_softmax(outputs[index] , dim=1),
                    F.softmax(calibrated_logit , dim=1),
                    reduction='batchmean'
                ) * self.hparams.loss_coefficient
                loss += self.criterion(outputs[index], labels) * (1 - self.hparams.loss_coefficient)
            else:
                loss += self.loss_fn(outputs[index], teacher_output) * self.hparams.loss_coefficient
                loss += self.criterion(outputs[index], labels) * (1 - self.hparams.loss_coefficient)

            if index - 1 < len(self.model.adaptation_layers):
                loss += torch.dist(self.model.adaptation_layers[index - 1](outputs_feature[index]), teacher_feature) * \
                        self.hparams.feature_loss_coefficient

        total_loss += loss.item()

        # Accuracy calculation for individual outputs
        for classifier_index in range(len(outputs)):
            _, predicted = torch.max(outputs[classifier_index].data, 1)
            correct[classifier_index] += float(predicted.eq(labels.data).cpu().sum())

        accuracy = [100 * correct[index] / total for index in range(len(correct))]

        # Logging validation loss and accuracy 
        self.log('val_loss', total_loss / (len(outputs)), prog_bar=True, sync_dist=True)
        
        for index, acc in enumerate(accuracy):
            self.log(f'val_accuracy_{len(accuracy)-index}/{len(accuracy)}', acc, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append(torch.tensor(total_loss / (len(outputs)), device=self.device))
        return {'val_loss': total_loss}

    def on_validation_epoch_end(self):
        # Calculate average validation loss across all batches
            # Calculate average validation loss across all batches
        if len(self.validation_step_outputs) > 0:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            self.log('avg_val_loss', avg_loss, prog_bar=True, sync_dist=True)

        # Log the average validation loss
        self.log('avg_val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs, _ = self.model(images)
        total_samples = labels.size(0)
        correct = [0] * 4

        # Compute the accuracy for each output
        for i, output in enumerate(outputs):
            _, predicted = torch.max(output.data, 1)
            correct[i] += predicted.eq(labels.data).cpu().sum().item()

        accuracy = [100 * correct[i] / total_samples for i in range(len(correct))]

        # Logging the accuracies for each classifier
        self.log('test_acc_4/4', accuracy[0], prog_bar=True, sync_dist=True)
        self.log('test_acc_3/4', accuracy[1], prog_bar=True, sync_dist=True)
        self.log('test_acc_2/4', accuracy[2], prog_bar=True, sync_dist=True)
        self.log('test_acc_1/4', accuracy[3], prog_bar=True, sync_dist=True)
        





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="fer2013", type=str, choices=['fer2013', 'ferplus', 'cifar100'])
    parser.add_argument('--dataset_path', default="data", type=str)
    parser.add_argument('--num_classes', default=7, type=int, choices=[7, 100])
    parser.add_argument('--model', default="resnet18", type=str, choices=['resnet18', 'resnet50','resnet34','resnet101'])
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--method', default="CrossEntropy", type=str, choices=['CrossEntropy', 'DistilKL', 'Loca', 'LogitCalibration'])
    parser.add_argument('--temp', default=3, type=float)
    parser.add_argument('--optimizer', default="SGD", type=str, choices=['SGD', 'AdamW'])
    parser.add_argument('--loss_coefficient', default=0.3, type=float, help="loss coefficient")
    parser.add_argument('--feature_loss_coefficient', default=0.03, type=float, help="feature loss coefficient")
    parser.add_argument('--save_path', default="checkpoints", type=str, help="path to save checkpoints")
    parser.add_argument('--use_wandb', default=False, type=bool)
    args = parser.parse_args()

    # Seed for reproducibility
    pl.seed_everything(args.seed)

    # Logger (optional)
    wandb_logger = WandbLogger(project="BYOT-FER") if args.use_wandb else CSVLogger("logs")

    # Model setup
    model = LitModel(
        model_name=args.model,
        method=args.method,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        lr=args.lr, batch_size=args.batch_size,
        num_workers=args.num_workers,
        temp=args.temp,
        loss_coefficient=args.loss_coefficient,
        feature_loss_coefficient=args.feature_loss_coefficient,
        save_path=args.save_path,
        num_classes=args.num_classes
    )

    # Trainer with early stopping
    # early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0000, verbose=True, mode='min')
    # check if dir exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(monitor='val_loss', 
                                    dirpath=args.save_path,
                                    filename='model_best_epoch_{epoch}_{val_loss:.3f}', mode='min', save_top_k=1)
    trainer = Trainer(
        max_epochs=args.epochs, 
        logger=wandb_logger, 
        callbacks=[
            # early_stopping_callback,
            ckpt_callback,
            lr_monitor], 
        accelerator='gpu',            # Use GPU accelerator
    strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto', 
    devices=args.gpu if torch.cuda.device_count() == 1 else torch.cuda.device_count(),  # Automatically detect the number of available GPUs)
    )
    # Training
    trainer.fit(model)
    # Load best model
    # print(ckpt_callback.best_model_path)
    # Load from model with the best val loss
    model = LitModel.load_from_checkpoint(ckpt_callback.best_model_path)
    

    model.eval()
    # print(model)
    trainer.test(model)