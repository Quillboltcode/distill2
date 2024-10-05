# distill2
Experiment on kaggle enviroment with 2xT4GPU

usage: train_lightning.py [-h] [--dataset {fer2013,ferplus,cifar100}] [--dataset_path DATASET_PATH]
                          [--num_classes {7,100}] [--model {resnet18,resnet50,resnet34,resnet101}]
                          [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
                          [--gpu GPU] [--seed SEED] [--method {CrossEntropy,DistilKL,Loca,LogitCalibration}]
                          [--temp TEMP] [--optimizer {SGD,AdamW}] [--loss_coefficient LOSS_COEFFICIENT]
                          [--feature_loss_coefficient FEATURE_LOSS_COEFFICIENT] [--save_path SAVE_PATH]
                          [--use_wandb USE_WANDB]

options:
  -h, --help            show this help message and exit
  --dataset {fer2013,ferplus,cifar100}
  --dataset_path DATASET_PATH
  --num_classes {7,100}
  --model {resnet18,resnet50,resnet34,resnet101}
  --num_workers NUM_WORKERS
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --lr LR
  --gpu GPU
  --seed SEED
  --method {CrossEntropy,DistilKL,Loca,LogitCalibration} Acompa paper using DistilKL while BYOT(2021) using CrossEntropy 
  --temp TEMP
  --optimizer {SGD,AdamW}
  --loss_coefficient LOSS_COEFFICIENT
                        loss coefficient
  --feature_loss_coefficient FEATURE_LOSS_COEFFICIENT
                        feature loss coefficient
  --save_path SAVE_PATH
                        path to save checkpoints
  --use_wandb USE_WANDB