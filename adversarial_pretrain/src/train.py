import argparse
import os
import pathlib
import json
import shutil

import numpy as np
import torch
import wandb
import torchvision

from adversarial_pretrain.src.models.timm_model_loader import MixBNModelBuilder

torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda')


class AdvPreTrain:
    def __init__(self, config):
        seed = np.random.randint(99999)
        assert pathlib.Path(config).is_file(), f"provided config({config}) doesn't exist"
        with open(config, 'r') as fh:
            self.config = json.load(fh)

        self.project_name = self.config['project_name']
        self.model_type = self.config['model_type']
        self.train_type = self.config['train_type']
        self.pretrain_dataset = self.config['pretrain_dataset']
        self.pretrain_classes = self.config['pretrain_classes']
        self.finetune_dataset = self.config['finetune_dataset']
        self.finetune_classes = self.config['finetune_classes']
        self.save_path = self.config['save_path']
        self.exp_name = f"{self.train_type}_{self.model_type}_seed_{seed}_pretrain_{self.pretrain_dataset}"\
                        f"_finetune_{self.finetune_dataset}"

        # define model and pretraining dataset
        self.model = MixBNModelBuilder(model_type=self.model_type, num_classes=self.pretrain_classes,
                                       pretrained=False, mix_bn=False).to(DEVICE)

        base_dir = pathlib.Path(f"{self.save_path}/{self.project_name}/{self.exp_name}")
        base_dir.mkdir(parents=True, exist_ok=True)

        self.pretrain_cur_model = base_dir / "pretrain_cur.pth"
        self.pretrain_best_model = base_dir / "pretrain_best.pth"
        self.finetune_cur_model = base_dir / "finetune_cur.pth"
        self.finetune_best_model = base_dir / "finetune_best.pth"

        # initialize wandb
        self.wandb_run = wandb.init(project=self.project_name, dir=os.getenv('TMPDIR'),
                                    name=self.exp_name, id=self.exp_name, reinit=True)

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    def snapshot_gpu(model, optimizer, epoch, path):
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'cur_epoch': epoch}, path)

    @staticmethod
    def load_snapshot(model, path, device):
        snapshot = torch.load(path, map_location=device)
        model.load_state_dict(snapshot['model'])
        model.to(device)

    def pretrain_imagenet(self):
        pretrain_epochs = 100
        batch_size = 198
        pretrain_lr = 1e-2
        pretrain_wd = 1e-3

        # copy imagenet data locally
        imagenet_root = pathlib.Path(f"{os.getenv('TMPDIR')}/imagenet_data")
        imagenet_root.mkdir(exist_ok=True, parents=True)
        if not (imagenet_root / "ILSVRC2012_img_train.tar").is_file():
            shutil.copy("/nas/vista-ssd01/batl/public_datasets/ImageNet/ILSVRC2012_img_train.tar", str(imagenet_root))
        if not (imagenet_root / "ILSVRC2012_img_val.tar").is_file():
            shutil.copy("/nas/vista-ssd01/batl/public_datasets/ImageNet/ILSVRC2012_img_val.tar", str(imagenet_root))
        if not (imagenet_root / "ILSVRC2012_devkit_t12.tar.gz").is_file():
            shutil.copy("/nas/vista-ssd01/batl/public_datasets/ImageNet/ILSVRC2012_devkit_t12.tar.gz",
                        str(imagenet_root))
        if not (imagenet_root / "meta.bin").is_file():
            shutil.copy("/nas/vista-ssd01/batl/public_datasets/ImageNet/meta.bin",
                        str(imagenet_root))

        pretrain_train_dataset = torchvision.datasets.ImageNet(root=imagenet_root,
                                                               split="train",
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.RandomResizedCrop(224),
                                                                   torchvision.transforms.RandomHorizontalFlip(),
                                                                   torchvision.transforms.ToTensor()
                                                               ]))
        train_dataloader = torch.utils.data.DataLoader(pretrain_train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                                                       worker_init_fn=lambda wid: np.random.seed(
                                                           np.random.get_state()[1][0] + wid)
                                                       )
        pretrain_valid_dataset = torchvision.datasets.ImageNet(root=imagenet_root,
                                                               split="val",
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.Resize(256),
                                                                   torchvision.transforms.CenterCrop(224),
                                                                   torchvision.transforms.ToTensor()
                                                               ]))
        valid_dataloader = torch.utils.data.DataLoader(pretrain_valid_dataset, batch_size=batch_size,
                                                       shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=pretrain_lr, weight_decay=pretrain_wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        scaler = torch.cuda.amp.GradScaler()

        best_val_acc1 = -np.inf

        for epoch in range(pretrain_epochs):
            losses = list()
            acc1s = list()
            acc5s = list()
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                data, labels = data.float().to(DEVICE, non_blocking=True), labels.long().to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = self.model(data)
                    loss = criterion(preds, labels)

                acc1, acc5 = AdvPreTrain.accuracy(preds, labels, topk=(1, 5))
                print(f"epoch:{epoch + 1}, batch:{batch_idx + 1}/{len(train_dataloader)}, "
                      f"loss:{loss.item():.6f}, acc1:{acc1.item()}, acc5:{acc5.item()}")
                wandb.log({'pretrain/loss': loss.item()})
                losses.append(loss.item())
                acc1s.append(acc1.item())
                acc5s.append(acc5.item())

                # scale the loss so that no underflow happens and then call backward
                scaler.scale(loss).backward()
                # unscale the gradients and then decide to do the step if no NaN
                scaler.step(optimizer)
                # reset the scaler for next iteration
                scaler.update()

            wandb.log({'pretrain/epoch_avg_loss': np.mean(losses),
                       'pretrain/epoch_avg_acc1': np.mean(acc1s),
                       'pretrain/epoch_avg_acc5': np.mean(acc5s)})
            scheduler.step()

            # validate
            with torch.no_grad():
                # note this averaging is not exactly correct because of change in batch size amongst validation
                # but this should be good enough for validation purposes
                val_losses = list()
                val_acc1s = list()
                val_acc5s = list()
                for idx, (data, labels) in enumerate(valid_dataloader):
                    data, labels = data.float().to(DEVICE), labels.long().to(DEVICE)
                    preds = self.model(data)
                    acc1, acc5 = AdvPreTrain.accuracy(preds, labels, topk=(1, 5))
                    loss = criterion(preds, labels)
                    val_losses.append(loss.item())
                    val_acc1s.append(acc1.item())
                    val_acc5s.append(acc5.item())
                wandb.log({'pretrain/validation_loss:': np.mean(val_losses),
                           'pretrain/validation_acc1:': np.mean(val_acc1s),
                           'pretrain/validation_acc5': np.mean(val_acc5s)})

                if np.mean(val_acc1s) > best_val_acc1:
                    best_val_acc1 = np.mean(val_acc1s)
                    AdvPreTrain.snapshot_gpu(self.model, optimizer, epoch, self.pretrain_best_model)

    def finetune(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for BATL adv. training')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('cuda is not available exiting....')
        exit(1)

    trainer = AdvPreTrain(args.config)
    trainer.pretrain_imagenet()
