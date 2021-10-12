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
from sklearn.metrics import f1_score, roc_auc_score

from functional_attacks import pgd_attack_linf, validation
from adversarial_pretrain.src.models.timm_model_loader import MixBNModelBuilder
from adversarial_pretrain.src.models.adv_perturbation_net import RandomAdvPerturbationNetwork
from adversarial_pretrain.src.utils.loss import LabelSmoothingCrossEntropy
from adversarial_pretrain.src.dataloaders.batl_data import get_dataloader


torch.backends.cudnn.benchmark = True


class AdvPreTrain:
    def __init__(self, config):
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
        self.adversary = None
        self.finetune_dataset = self.config['finetune_dataset']
        self.finetune_classes = self.config['finetune_classes']
        self.save_path = self.config['save_path']

        self.exp_name = f"{self.exp_prefix}_{self.train_type}_{self.model_type}_pretrain_{self.pretrain_dataset_name}" \
                        f"_finetune_{self.finetune_dataset}"
        base_dir = pathlib.Path(f"{self.save_path}/{self.project_name}/{self.exp_name}")
        base_dir.mkdir(parents=True, exist_ok=True)

        self.pretrain_cur_model = base_dir / "pretrain_cur.pth"
        self.pretrain_best_model = base_dir / "pretrain_best.pth"

        self.finetune_cur_model = dict()
        self.finetune_best_model = dict()
        for dataset in ["BATL", "HQ-WMCA", "WMCA"]:
            finetune_base_dir = base_dir / dataset
            finetune_base_dir.mkdir(exist_ok=True, parents=True)
            self.finetune_cur_model[dataset] = finetune_base_dir / "finetune_cur.pth"
            self.finetune_best_model[dataset] = finetune_base_dir / "finetune_best.pth"

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
                    'optimizer': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'cur_epoch': epoch,
                    'best_val': best_val}, path)

    @staticmethod
    def load_snapshot(model, optimizer, scheduler, path, device, strict=True):
        snapshot = torch.load(path, map_location=device)
        model.load_state_dict(snapshot['model'], strict=strict)
        if optimizer is not None:
            optimizer.load_state_dict(snapshot['optimizer'])
        if scheduler is not None:
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

    def pretrain_imagenet(self, batch_size=224, epochs=100, lr=1e-3, wd=1e-4):
        gpu = 0
        ngpus = 1
        device = torch.device(f"cuda")
        pretrain_epochs = epochs
        batch_size = batch_size
        pretrain_lr = lr
        pretrain_wd = wd

        # copy imagenet data locally
        imagenet_root = pathlib.Path(f"{os.getenv('TMPDIR')}/imagenet_data")
        imagenet_root.mkdir(exist_ok=True, parents=True)
        if not (imagenet_root / "ILSVRC2012_img_train.tar").is_file():
            print('[pretrain] copying ImageNet train.tar to LFS')
            shutil.copy(f"{self.config['pretrain_dataset_root']}/ILSVRC2012_img_train.tar", str(imagenet_root))
        if not (imagenet_root / "ILSVRC2012_img_val.tar").is_file():
            print('[pretrain] copying ImageNet val.tar to LFS')
            shutil.copy(f"{self.config['pretrain_dataset_root']}/ILSVRC2012_img_val.tar", str(imagenet_root))
        if not (imagenet_root / "ILSVRC2012_devkit_t12.tar.gz").is_file():
            print('[pretrain] copying ImageNet devkit.tar to LFS')
            shutil.copy(f"{self.config['pretrain_dataset_root']}/ILSVRC2012_devkit_t12.tar.gz", str(imagenet_root))
        if not (imagenet_root / "meta.bin").is_file():
            shutil.copy(f"{self.config['pretrain_dataset_root']}/meta.bin", str(imagenet_root))
            
        # unpack the data using torchvision dataset utils
        torchvision.datasets.ImageNet(root=str(imagenet_root), split="train")
        torchvision.datasets.ImageNet(root=str(imagenet_root), split="val")

        pretrain_train_dataset = torchvision.datasets.ImageFolder(str(imagenet_root / 'train'),
                                                                  transform=torchvision.transforms.Compose([
                                                                      torchvision.transforms.RandomResizedCrop(224),
                                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                                      torchvision.transforms.ColorJitter(brightness=0.3,
                                                                                                         contrast=0.3,
                                                                                                         saturation=0.3),
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
                                  pretrained=True, mix_bn=False).to(device)
        summary(model)

        # setup the dataloader
        train_dataloader = torch.utils.data.DataLoader(pretrain_train_dataset, batch_size=batch_size,
                                                       num_workers=4, pin_memory=True,
                                                       worker_init_fn=self.worker_init,
                                                       prefetch_factor=batch_size // 8,
                                                       persistent_workers=True,
                                                       shuffle=True
                                                       )
        valid_dataloader = torch.utils.data.DataLoader(pretrain_valid_dataset, batch_size=batch_size,
                                                       shuffle=False, num_workers=4, pin_memory=False, drop_last=False)

        cur_iter = 0
        best_val_acc1 = -np.inf
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=pretrain_wd)
        optimizer = torch.optim.SGD(model.parameters(), lr=pretrain_lr, weight_decay=pretrain_wd, momentum=0.9)
        scheduler = None

        if self.pretrain_cur_model.is_file():
            cur_iter, best_val_acc1 = self.load_snapshot(model, optimizer, scheduler,
                                                         self.pretrain_cur_model, device=device)
            model.train()
            print(
                f'[pretrain] gpu:{gpu} loaded the previous state trained to iter:{cur_iter}'
                f' epoch:{cur_iter // len(train_dataloader)} with best_val:{best_val_acc1}'
            )

        start_batch = time.time()
        adv_success_rate = 0.0
        train_dataloader_iterator = iter(train_dataloader)
        iterations_trained = 0
        for iter_idx in range(cur_iter, len(train_dataloader) * pretrain_epochs):
            batch_load_time = time.time()
            iterations_trained += 1

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
                                                                              pgd_iterations=10, device=device)
                with torch.no_grad():
                    data[half_batch_size:].copy_(half_adv_batch)

            preds = model(data)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2.0)
            optimizer.step()

            if iter_idx % 10 == 0:
                print(f"[pretrain] gpu: {gpu}/{ngpus} epoch: {(iter_idx + 1) // len(train_dataloader)},"
                      f" batch: {iter_idx % len(train_dataloader)}/{len(train_dataloader)},"
                      f" loss: {loss.item():.7f},"
                      f" batch_time: {time.time() - start_batch:.5f}, batch_load_time:{batch_load_time - start_batch:.5f}"
                      f" model_time: {time.time() - batch_load_time:.5f}"
                      f" adv_success_rate: {adv_success_rate}, lr:{pretrain_lr}, wd:{pretrain_wd}, batch_size:{batch_size}")
                wandb.log({'pretrain/train_ce_smooth_loss': loss.item(),
                           'pretrain/train_adv_success_rate': adv_success_rate})

            # periodically plot the accuracies on train batch
            if (iter_idx + 1) % 500 == 0:
                acc1, acc5 = self.accuracy(preds, labels, topk=(1, 5))
                wandb.log({'pretrain/train_acc1': acc1.item(), 'pretrain/train_acc5': acc5.item(),
                           'pretrain/batch_time:': time.time() - start_batch,
                           'pretrain/epoch': (iter_idx + 1) // len(train_dataloader)}, commit=False)
                # save cur state
                self.snapshot_gpu(model, optimizer, scheduler, iter_idx + 1, acc1.item(), self.pretrain_cur_model)

            # validate and checkpoint
            if (iter_idx + 1) % (10 * len(train_dataloader)) == 0:
                # if rank 0 then do validation
                if gpu == 0:
                    print(f'[pretrain] gpu:{gpu} running validation')
                    with torch.no_grad():
                        # note this averaging is not exactly correct because of change in batch size amongst validation
                        # but this should be good enough for validation purposes
                        val_losses = list()
                        val_acc1s = list()
                        val_acc5s = list()
                        model.eval()
                        for idx, (data, labels) in enumerate(valid_dataloader):
                            data, labels = data.float().to(device), labels.long().to(device)
                            # note: validate using model.module because for some reason the forward pass on a DDP model
                            # causes the barrier not to work at the end of validation
                            preds = model.module(data)
                            acc1, acc5 = self.accuracy(preds, labels, topk=(1, 5))
                            loss = criterion(preds, labels)
                            print(f"[pretrain] validation idx:{idx + 1}/{len(valid_dataloader)} loss:{loss.item()}, "
                                  f"acc1:{acc1.item()}, acc5:{acc5.item()}, lr:{pretrain_lr}, wd:{pretrain_wd}")
                            val_losses.append(loss.item())
                            val_acc1s.append(acc1.item())
                            val_acc5s.append(acc5.item())
                        wandb.log({'pretrain/validation_loss:': np.mean(val_losses),
                                   'pretrain/validation_acc1:': np.mean(val_acc1s),
                                   'pretrain/validation_acc5': np.mean(val_acc5s)}, commit=False)
                        model.train()
                        if np.mean(val_acc1s) > best_val_acc1:
                            best_val_acc1 = np.mean(val_acc1s)
                            # note: saving the model without the DDP wrapper
                            self.snapshot_gpu(model.module, optimizer, scheduler, iter_idx + 1, best_val_acc1,
                                              self.pretrain_best_model)
                            # save the model along with acc1 tag for fine-tuning experiments
                            self.snapshot_gpu(model.module, optimizer, scheduler, iter_idx + 1, best_val_acc1,
                                              self.pretrain_best_model.parent / "pretrain_acc1_{:.2f}.pth")

            start_batch = time.time()
        return iterations_trained

    def finetune(self, train_dataset, gpu=0):
        print(f'[finetune] gpu:{gpu} starting finetunining on {train_dataset}...')
        device = torch.device(f'cuda:{gpu}')
        finetune_epochs = 150
        lr = 3e-4
        weight_decay = 1e-3
        img_size = (224, 224)
        batch_size = 96

        (train_dataloader, valid_dataloader, test_dataloader), mean, std, (
            patch_dataset, train_partition, valid_partition, test_partition) = get_dataloader(train_dataset,
                                                                                              batch_size=batch_size,
                                                                                              img_size=img_size)
        model = MixBNModelBuilder(model_type=self.model_type, num_classes=self.pretrain_classes,
                                  pretrained=False, mix_bn=False).to(device)
        if self.pretrain_best_model.is_file():
            cur_iter, best_val_acc1 = self.load_snapshot(model, None, None, self.pretrain_best_model, device=device)
            print(f'[finetune] {train_dataset} gpu:{gpu} loaded the best trained to iter:{cur_iter} with best_val:{best_val_acc1}')
        # change the classifier layer of the model
        model.model.classifier = torch.nn.Linear(1280, self.finetune_classes, device=device)
        summary(model)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay,
                                      amsgrad=False)
        optimizer.zero_grad()

        class_weights = None
        criterion = None
        best_val_f1 = None
        for epoch in range(finetune_epochs):
            
            # validation
            if epoch % 10 == 0 or epoch + 1 == finetune_epochs:
                step_name = "validation"
                if epoch + 1 == finetune_epochs:
                    step_name = "test"
                    print("[finetune] loading the best model on validation for final test")
                    self.load_snapshot(model, None, None, self.finetune_best_model[train_dataset], device=device)
                if epoch == 0:
                    print('[finetune] evaluate the model without finetuning...')
                    step_name = "no_finetune"
                    criterion = torch.nn.CrossEntropyLoss().to(device)
                    
                with torch.no_grad():
                    model.eval()
                    preds = list()
                    gt = list()
                    for batch_idx, (data, labels, weights, multi_labels) in enumerate(valid_dataloader):
                        data, labels, weights, multi_labels = data.float(), labels.long(), weights.float(), multi_labels.float()
                        data, labels = data.to(device), labels.to(device)
                        rgb_test_data = data.flip(dims=(1,))  # bgr to rgb
                        if rgb_test_data.shape[2] != 224:  # this is needed for WMCA dataset which is native 128x128
                            rgb_test_data = torch.nn.functional.interpolate(rgb_test_data, size=224)
                        pred = model(rgb_test_data)
                        loss = criterion(pred, labels)
                        # if step_name == "validation":
                        #     wandb.log({f"fine-tune/{step_name}_{train_dataset}_loss": loss.item()})
                        pred_probability = np.squeeze(torch.nn.functional.softmax(pred, dim=1).cpu().data.numpy()[:, 1])
                        preds.append(pred_probability)
                        gt.append(labels.cpu().data.numpy())
                    gt = np.concatenate(gt, axis=0).flatten()
                    preds = np.concatenate(preds, axis=0).flatten()
                    f1 = f1_score(gt, preds > 0.5)
                    auc_score = roc_auc_score(gt, preds)
                    wandb.log({f"fine-tune/{step_name}_{train_dataset}_f1": f1})
                    wandb.log({f"fine-tune/{step_name}_{train_dataset}_auc": auc_score})
                    wandb.run.summary[f"{step_name}_{train_dataset}_f1"] = f1
                    wandb.run.summary[f"{step_name}_{train_dataset}_auc"] = auc_score
                    model.train()
                    if best_val_f1 is None or best_val_f1 < f1:
                        best_val_f1 = f1
                        self.snapshot_gpu(model, optimizer, None, epoch, best_val_f1,
                                          self.finetune_best_model[train_dataset])
                # snapshot the current model
                self.snapshot_gpu(model, optimizer, None, epoch, f1, self.finetune_cur_model[train_dataset])

            # fine-tuning 
            for batch_idx, (data, labels, weights, multi_labels) in enumerate(train_dataloader):
                if class_weights is None:
                    try:
                        class_weights = torch.tensor([weights[labels == 0][0], weights[labels == 1][0]],
                                                     device=device, dtype=torch.float32)
                        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
                    except Exception as err:
                        print("batch has only one type of samples, moving to next...")
                        continue

                data, labels, weights, multi_labels = data.float(), labels.long(), weights.float(), multi_labels.float()
                data, labels = data.to(device), labels.to(device)
                with torch.no_grad():
                    rgb_train_data = data.flip(dims=(1,))  # bgr to rgb
                    if np.random.random() > 0.5:
                        rgb_train_data = torch.flip(rgb_train_data, [-1])
                    if rgb_train_data.shape[2] != 224:  # this is needed for WMCA dataset which is native 128x128
                        rgb_train_data = torch.nn.functional.interpolate(rgb_train_data, size=224)

                preds = model(rgb_train_data)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"[finetune] {train_dataset} epoch:{epoch}, idx:{batch_idx}/{len(train_dataloader)} loss:{loss.item()}")
                wandb.log({f"fine-tune/train_{train_dataset}_loss": loss.item()})

    def __call__(self):
        wandb_save_dir = pathlib.Path(os.getenv('TMPDIR')) / 'wandb'
        wandb_save_dir.mkdir(exist_ok=True, parents=True)
        wandb.init(project=self.project_name, dir=str(wandb_save_dir), name=self.exp_name,
                   id=self.exp_name, reinit=True, resume=True)
        
        for batch_size, epochs, lr, wd in [(256, 250, 1e-1, 1e-4)]:
            iterations_trained = self.pretrain_imagenet(batch_size=batch_size, epochs=epochs, lr=lr, wd=wd)
        wandb.finish()
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for BATL adv. training')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('cuda is not available exiting....')
        exit(1)

    AdvPreTrain(args.config)()
    print("done")
