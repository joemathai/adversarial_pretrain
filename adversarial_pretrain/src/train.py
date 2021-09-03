import argparse
import os
import pathlib
import json
import shutil
import time

import numpy as np
import torch
import wandb
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist
from torchinfo import summary

from functional_attacks import pgd_attack_linf, validation
from adversarial_pretrain.src.models.timm_model_loader import MixBNModelBuilder
from adversarial_pretrain.src.models.adv_perturbation_net import RandomAdvPerturbationNetwork
from adversarial_pretrain.src.utils.loss import LabelSmoothingCrossEntropy

torch.backends.cudnn.benchmark = True


class AdvPreTrain:
    def __init__(self, config):
        seed = np.random.randint(99999)
        assert pathlib.Path(config).is_file(), f"provided config({config}) doesn't exist"
        with open(config, 'r') as fh:
            self.config = json.load(fh)

        self.project_name = self.config['project_name']
        self.exp_prefix = self.config['exp_prefix']
        self.model_type = self.config['model_type']
        self.train_type = self.config['train_type']
        self.pretrain_dataset_name = self.config['pretrain_dataset_name']
        self.pretrain_classes = self.config['pretrain_classes']
        self.pretrain_adversary = self.config['pretrain_adversary']
        self.adversary_config = self.config['adversary_config']
        self.finetune_dataset = self.config['finetune_dataset']
        self.finetune_classes = self.config['finetune_classes']
        self.save_path = self.config['save_path']

        self.exp_name = f"{self.exp_prefix}_{self.train_type}_{self.model_type}_pretrain_{self.pretrain_dataset_name}"\
                        f"_finetune_{self.finetune_dataset}"
        base_dir = pathlib.Path(f"{self.save_path}/{self.project_name}/{self.exp_name}")
        base_dir.mkdir(parents=True, exist_ok=True)

        self.pretrain_cur_model = base_dir / "pretrain_cur.pth"
        self.pretrain_best_model = base_dir / "pretrain_best.pth"
        self.finetune_cur_model = base_dir / "finetune_cur.pth"
        self.finetune_best_model = base_dir / "finetune_best.pth"

    @staticmethod
    def worker_init(wid):
        return np.random.seed(np.random.get_state()[1][0] + wid)

    @staticmethod
    @torch.no_grad()
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
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
    def snapshot_gpu(model, optimizer, scheduler, epoch, best_val, path):
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'cur_epoch': epoch,
                    'best_val': best_val}, path)

    @staticmethod
    def load_snapshot(model, optimizer, scheduler, path, device):
        snapshot = torch.load(path, map_location=device)
        model.load_state_dict(snapshot['model'])
        optimizer.load_state_dict(snapshot['optimizer'])
        scheduler.load_state_dict(snapshot['scheduler'])
        return snapshot['cur_epoch'], snapshot['best_val']

    def get_adversarial_batch(self, model, batch, labels, pgd_iterations, device):
        model.eval()
        adversary = RandomAdvPerturbationNetwork(batch.shape, random_init=True, config=self.adversary_config).to(device)
        adv_batch = pgd_attack_linf(adversary, model, batch, labels, num_iterations=pgd_iterations,
                                    validator=validation, loss_fn_type='xent')
        success_rate = validation(model, batch, adv_batch, labels)
        model.train()
        return adv_batch, success_rate

    def pretrain_imagenet(self, gpu, ngpus):
        print(f'pretraining worker gpu/rank:{gpu} ngpus/world_size:{ngpus}')
        dist.init_process_group(backend='nccl', rank=gpu, world_size=ngpus)

        pretrain_epochs = 100
        batch_size = 224
        pretrain_lr = 3e-4
        pretrain_wd = 1e-4
        device = torch.device(f"cuda:{gpu}")

        # copy imagenet data locally
        imagenet_root = pathlib.Path(f"{os.getenv('TMPDIR')}/imagenet_data")
        if gpu == 0:
            wandb.init(project=self.project_name, dir=os.getenv('TMPDIR'), name=self.exp_name, id=self.exp_name, reinit=True, resume=True)
            imagenet_root.mkdir(exist_ok=True, parents=True)
            if not (imagenet_root / "ILSVRC2012_img_train.tar").is_file():
                print('copying ImageNet train.tar to LFS')
                shutil.copy(f"{self.config['pretrain_dataset_root']}/ILSVRC2012_img_train.tar", str(imagenet_root))
            if not (imagenet_root / "ILSVRC2012_img_val.tar").is_file():
                print('copying ImageNet val.tar to LFS')
                shutil.copy(f"{self.config['pretrain_dataset_root']}/ILSVRC2012_img_val.tar", str(imagenet_root))
            if not (imagenet_root / "ILSVRC2012_devkit_t12.tar.gz").is_file():
                print('copying ImageNet devkit.tar to LFS')
                shutil.copy(f"{self.config['pretrain_dataset_root']}/ILSVRC2012_devkit_t12.tar.gz",
                            str(imagenet_root))
            if not (imagenet_root / "meta.bin").is_file():
                shutil.copy(f"{self.config['pretrain_dataset_root']}/meta.bin",
                            str(imagenet_root))
            # unpack the data using torchvision dataset utils
            torchvision.datasets.ImageNet(root=str(imagenet_root), split="train")
            torchvision.datasets.ImageNet(root=str(imagenet_root), split="val")

        dist.barrier()  # wait for gpu:0 to copy the files to unpack them
        pretrain_train_dataset = torchvision.datasets.ImageFolder(str(imagenet_root / 'train'),
                                                                  transform=torchvision.transforms.Compose([
                                                                      torchvision.transforms.RandomResizedCrop(224),
                                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                                      torchvision.transforms.ToTensor()
                                                                  ]))
        pretrain_valid_dataset = torchvision.datasets.ImageFolder(str(imagenet_root / 'val'),
                                                                  transform=torchvision.transforms.Compose([
                                                                      torchvision.transforms.Resize(256),
                                                                      torchvision.transforms.CenterCrop(224),
                                                                      torchvision.transforms.ToTensor()
                                                                  ]))
        # setup the model
        model = MixBNModelBuilder(model_type=self.model_type, num_classes=self.pretrain_classes,
                                  pretrained=False, mix_bn=False).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

        # print model architecture
        if gpu == 0:
            summary(model)

        # setup the dataloader
        train_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_train_dataset, rank=gpu, shuffle=True, drop_last=True)
        train_dataloader = torch.utils.data.DataLoader(pretrain_train_dataset, batch_size=batch_size,
                                                       num_workers=4, pin_memory=True,
                                                       worker_init_fn=self.worker_init,
                                                       prefetch_factor=batch_size // 4,
                                                       persistent_workers=True,
                                                       sampler=train_sampler
                                                       )
        valid_dataloader = torch.utils.data.DataLoader(pretrain_valid_dataset, batch_size=batch_size,
                                                       shuffle=False, num_workers=4, pin_memory=False, drop_last=False)

        cur_iter = 0
        best_val_acc1 = -np.inf
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=pretrain_wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 95], gamma=0.1)

        if self.pretrain_cur_model.is_file():
            cur_iter, best_val_acc1 = self.load_snapshot(model, optimizer, scheduler,
                                                         self.pretrain_cur_model, device=device)
            model.train()
            print(f'gpu:{gpu} loaded the previous state trained to iter:{cur_iter} with best_val:{best_val_acc1}')

        start_batch = time.time()
        adv_success_rate = 0.0
        train_dataloader_iterator = iter(train_dataloader)
        for iter_idx in range(cur_iter, len(train_dataloader) * pretrain_epochs):

            # for every epoch
            if (iter_idx + 1) % len(train_dataloader) == 0:
                scheduler.step()
                train_sampler.set_epoch(iter_idx / len(train_dataloader))

            try:
                data, labels = next(train_dataloader_iterator)
            except StopIteration as err:
                train_dataloader_iterator = iter(train_dataloader)
                data, labels = next(train_dataloader_iterator)

            data, labels = data.float().to(device, non_blocking=True), labels.long().to(device, non_blocking=True)

            if self.pretrain_adversary:
                half_batch_size = data.shape[0] // 2
                half_adv_batch, adv_success_rate = self.get_adversarial_batch(model, data[half_batch_size:],
                                                                              labels[half_batch_size:],
                                                                              pgd_iterations=5, device=device)
                with torch.no_grad():
                    data[half_batch_size:].copy_(half_adv_batch)

            preds = model(data)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2.0)
            optimizer.step()

            print(f"gpu: {gpu}/{ngpus} epoch: {iter_idx + 1 // len(train_dataloader)},"
                  f" batch: {iter_idx % len(train_dataloader)}/{len(train_dataloader)},"
                  f" loss: {loss.item():.7f},"
                  f" batch_time: {time.time() - start_batch}"
                  f" adv_success_rate: {adv_success_rate}")

            if gpu == 0:
                wandb.log({'pretrain/ce_smooth_loss': loss.item(),
                           'pretrain/adv_success_rate': adv_success_rate})

            # periodically plot the accuracies on train batch
            if gpu == 0 and (iter_idx + 1) % 200 == 0:
                acc1, acc5 = self.accuracy(preds, labels, topk=(1, 5))
                wandb.log({'pretrain/acc1': acc1.item(), 'pretrain/acc5': acc5.item(),
                           'pretrain/batch_time:': time.time() - start_batch}, commit=False)

            # validate and checkpoint
            if (iter_idx + 1) % 1000 == 0:
                # if rank 0 then do validation
                if gpu == 0:
                    print(f'gpu:{gpu} running validation')
                    with torch.no_grad():
                        # note this averaging is not exactly correct because of change in batch size amongst validation
                        # but this should be good enough for validation purposes
                        val_losses = list()
                        val_acc1s = list()
                        val_acc5s = list()
                        model.eval()
                        for idx, (data, labels) in enumerate(valid_dataloader):
                            data, labels = data.float().to(device), labels.long().to(device)
                            # note: validate using model.module because for some reason the forward pass on a DDP model here causes
                            # the barrier not to work at the end of validation
                            preds = model.module(data)
                            acc1, acc5 = self.accuracy(preds, labels, topk=(1, 5))
                            loss = criterion(preds, labels)
                            print(f"validation idx:{idx + 1}/{len(valid_dataloader)} loss:{loss.item()}, "
                                  f"acc1:{acc1.item()}, acc5:{acc5.item()}")
                            val_losses.append(loss.item())
                            val_acc1s.append(acc1.item())
                            val_acc5s.append(acc5.item())
                        wandb.log({'pretrain/validation_loss:': np.mean(val_losses),
                                   'pretrain/validation_acc1:': np.mean(val_acc1s),
                                   'pretrain/validation_acc5': np.mean(val_acc5s)}, commit=False)
                        model.train()
                        if np.mean(val_acc1s) > best_val_acc1:
                            best_val_acc1 = np.mean(val_acc1s)
                            self.snapshot_gpu(model, optimizer, scheduler, iter_idx + 1, best_val_acc1, self.pretrain_best_model)
                    # save cur state
                    self.snapshot_gpu(model, optimizer, scheduler, iter_idx + 1, best_val_acc1, self.pretrain_cur_model)

                entry_time = time.time()
                dist.barrier() # wait for all other ranks to get to this point

            start_batch = time.time()

    def finetune(self):
        pass

    def __call__(self, gpu, ngpus):
        self.pretrain_imagenet(gpu, ngpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for BATL adv. training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('cuda is not available exiting....')
        exit(1)

    if torch.cuda.device_count() != args.ngpus:
        print("cuda device count and input gpu count mismatch")
        exit(1)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(AdvPreTrain(args.config), nprocs=args.ngpus, args=(args.ngpus,))
