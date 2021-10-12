import os
import itertools
import wandb
import pathlib
import argparse
import torchvision
import torch
import numpy as np
import time
import copy
from torchsummary import summary
from adversarial_pretrain.src.dataloaders.batl_data import get_dataloader
from adversarial_pretrain.src.utils.metrics import ROCMetrics
from sklearn.metrics import f1_score, roc_auc_score
from batl_utils_modules.pad_algorithms.pad_algorithm import PADAlgorithm, test_pad_algorithm_on_partition
from tqdm import tqdm


torch.backends.cudnn.benchmark = True


def snapshot_gpu(model, optimizer, scheduler, epoch, best_val, path):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict() if optimizer is not None else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'cur_epoch': epoch,
                'best_val': best_val}, path)


@torch.no_grad()
def evaluate(model, dataloaders, prefix='validation', gpu=0):
    device = torch.device(f'cuda:{gpu}')
    model.eval()
    preds = list()
    gt = list()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]
    for dataloader_name, dataloader in dataloaders:
        for batch_idx, (data, labels, weights, multi_labels) in enumerate(dataloader):
            data, labels, weights, multi_labels = data.float(), labels.long(), weights.float(), multi_labels.float()
            data, labels = data.to(device), labels.to(device)
            rgb_test_data = data.flip(dims=(1,))  # bgr to rgb
            if rgb_test_data.shape[2] != 224:  # this is needed for WMCA dataset which is native 128x128
                rgb_test_data = torch.nn.functional.interpolate(rgb_test_data, size=224, mode='bilinear')
            # normalize the data
            rgb_test_data = torchvision.transforms.functional.normalize(rgb_test_data,
                                                                        mean=(0.485, 0.456, 0.406),
                                                                        std=(0.229, 0.224, 0.225),
                                                                        inplace=True)
            pred = model(rgb_test_data)
            pred_probability = np.squeeze(torch.nn.functional.softmax(pred, dim=1).cpu().data.numpy()[:, 1])
            preds.append(pred_probability)
            gt.append(labels.cpu().data.numpy())
    # run batl metrics
    gt = np.concatenate(gt, axis=0).flatten()
    preds = np.concatenate(preds, axis=0).flatten()
    roc_metrics = ROCMetrics(fprs=(0.0, 0.002, 0.02))
    batl_metrics = roc_metrics.calculate_metrics(preds, gt, pos_label=1)
    batl_metrics["f1"] = f1_score(gt, preds > 0.5)
    model.train()
    return {f'{prefix}_{k}': v for k, v in batl_metrics.items()}


def finetune(model, train_dataloader, val_dataloader, test_dataloaders, gpu=0, mode='fixed_feature', write_path='', prefix=""):
    device = torch.device(f'cuda:{gpu}')
    lr = 3e-4
    wd = 1e-2
    model_path = pathlib.Path(write_path) / train_dataloader[0]
    model_path.mkdir(exist_ok=True, parents=True)
    best_model_path = model_path / 'best.pth'
    cur_model_path = model_path / 'cur.pth'
    cur_epoch = 0

    if mode == 'fixed_feature':
        finetune_epochs = 60
        # optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=wd)
    else:
        finetune_epochs = 120
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    optimizer.zero_grad()

    class_weights = None
    best_val_f1 = None
    if best_model_path.is_file():
        snapshot = torch.load(str(cur_model_path), map_location=device)
        cur_epoch = snapshot['cur_epoch']
        best_val_f1 = snapshot['best_val']
        optimizer.load_state_dict(snapshot['optimizer'])
        model.load_state_dict(snapshot['model'], strict=True)

    # setup the model
    model.to(device)
    model.train()

    for epoch in tqdm(range(cur_epoch, finetune_epochs)):
        model.train()
        avg_epoch_loss = list()
        start_epoch = time.time()
        train_dataset_name, dataloader = train_dataloader
        for bidx, (data, labels, weights, multi_labels) in enumerate(dataloader):

            # assign class weights to solve imbalance
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
                    rgb_train_data = torch.nn.functional.interpolate(rgb_train_data, size=224, mode='bilinear')
                # normalize the data
                rgb_train_data = torchvision.transforms.functional.normalize(rgb_train_data,
                                                                             mean=(0.485, 0.456, 0.406),
                                                                             std=(0.229, 0.224, 0.225),
                                                                             inplace=True)
            preds = model(rgb_train_data)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_epoch_loss.append(loss.item())
            wandb.log({f"{prefix}_loss": loss.item()})

        # if (epoch + 1) % 50 == 0:
        #     print(f"epoch:{epoch} dataset:{train_dataset_name} avg_loss:{np.mean(avg_epoch_loss)} avg_time:{time.time() - start_epoch}")

        val_metrics = evaluate(model, val_dataloader)
        if best_val_f1 is None or best_val_f1 < val_metrics['validation_f1']:
            best_val_f1 = val_metrics['validation_f1']
            snapshot_gpu(model, optimizer, None, epoch, best_val_f1, str(best_model_path))

    snapshot_gpu(model, optimizer, None, epoch, best_val_f1, str(cur_model_path))
    # load the best model from validation
    snapshot = torch.load(str(best_model_path), map_location=device)
    model.load_state_dict(snapshot['model'], strict=True)

    results = []
    for test_prefix, dataloaders  in test_dataloaders:
        test_metrics = evaluate(model, dataloaders, prefix=test_prefix, gpu=gpu)
        results.append(test_metrics)

    return results



pytorch_models = {
    'resnet18': torchvision.models.resnet18,
    'resnet50': torchvision.models.resnet50,
    'wide_resnet50_2': torchvision.models.wide_resnet50_2
}

parser = argparse.ArgumentParser()
parser.add_argument('--train-dataset', type=str, default='batl')
parser.add_argument('--arch', type=str, default='model architecture')
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--mode', default='fixed_feature')
args = parser.parse_args()


project="adv_prior_microsoft_v4"
exp_name = f"{args.arch}_{args.mode}_OG_128x128_bilinear"
wandb.init(reinit=True, project=project, name=exp_name)
write_path_root = f"/nas/vista-ssd01/users/jmathai/batl/{project}/{exp_name}"


(batl_train_dataloader, batl_valid_dataloader, batl_test_dataloader), mean, std, (
    batl_patch_dataset, batl_train_partition, batl_valid_partition, batl_test_partition) = get_dataloader('BATL',
                                                                                                          batch_size=64,
                                                                                                          img_size=(224, 224),
                                                                                                          num_frames=5)
(hqwmca_train_dataloader, hqwmca_valid_dataloader, hqwmca_test_dataloader), mean, std, (
    hqwmca_patch_dataset, hqwmca_train_partition, hqwmca_valid_partition, hqwmca_test_partition) = get_dataloader('HQ-WMCA',
                                                                                                                  batch_size=64,
                                                                                                                  img_size=(224, 224),
                                                                                                                  num_frames=5)
(wmca_train_dataloader, wmca_valid_dataloader, wmca_test_dataloader), mean, std, (
    wmca_patch_dataset, wmca_train_partition, wmca_valid_partition, wmca_test_partition) = get_dataloader('WMCA',
                                                                                                          batch_size=64,
                                                                                                          img_size=(224, 224),
                                                                                                          num_frames=5)

train_datasets = {
    'batl': ('batl_train', batl_train_dataloader),
    'hqwmca': ('hqwmca_train', hqwmca_train_dataloader),
    'wmca': ('wmca_train', wmca_train_dataloader)
}
val_datasets = {
    'batl': ('batl_val', batl_valid_dataloader),
    'hqwmca': ('hqwmca_val', hqwmca_valid_dataloader),
    'wmca': ('wmca_val', wmca_valid_dataloader)
}
test_datasets = {
    'batl': ('batl_val', batl_test_dataloader),
    'hqwmca': ('hqwmca_val', hqwmca_test_dataloader),
    'wmca': ('wmca_val', wmca_test_dataloader)
}


class Tester:
    
    @staticmethod
    def _test_algorithm_with_batl_framework(exp_name, model, reader_patch_dataset, combined_partition, result_file_path):
        class TmpTestAlg(PADAlgorithm):
            def __init__(self, model, exp_name, reader_patch_dataset):
                super().__init__()
                self.model = model
                self.name = exp_name
                self.test_data_loader = None
                self.reader_patch_dataset = reader_patch_dataset

            def get_name(self):
                return self.name

            def load_model(self):
                pass

            def train(self, train_file_readers: list, save_path: str, valid_file_readers: (None, list) = None):
                pass

            def retrain(self, pre_trained_model_path: str, train_file_readers: list, save_path: str,
                        valid_file_readers: (None, list) = None):
                raise NotImplementedError

            def __validate_data__(self, file):
                pass

            @torch.no_grad()
            def predict_pa(self, data) -> tuple:
                # model.apply(to_clean_status)
                model.eval()
                if self.test_data_loader is None:
                    self.test_data_loader = copy.deepcopy(self.reader_patch_dataset)
                    self.test_data_loader.prepare_data_storage([data])
                sample_patches = self.test_data_loader.get_sample_patches(data)
                rgb_data = torch.from_numpy(np.stack(sample_patches, axis=0).astype(np.float32)).to(device).flip(dims=(1,))
                if rgb_data.shape[2] != 224:  # this is needed for WMCA dataset which is native 128x128
                    rgb_data = torch.nn.functional.interpolate(rgb_data, size=224, mode='bilinear')
                # normalize the data
                rgb_data = torchvision.transforms.functional.normalize(rgb_data,
                                                                       mean=(0.485, 0.456, 0.406),
                                                                       std=(0.229, 0.224, 0.225),
                                                                       inplace=True)
                frame_preds = np.squeeze(torch.nn.functional.softmax(model(rgb_data), dim=1).cpu().data.numpy()[:, 1])
                results = list()
                for frame_no, frame_pred in enumerate(frame_preds):
                    results.append({'pad_score': frame_pred, 'pad_threshold': 0.5,
                                    'identifier': data.get_reader_identifier(),
                                    'pad_algorithm': exp_name + f'_frame_{frame_no}',
                                    'frame_no': frame_no,
                                    'seed': exp_name.split('_')[-1]})
                model.train()
                return results

            def requires_test_file_reader(self):
                return True

        pad_alg = TmpTestAlg(model, exp_name, reader_patch_dataset)
        test_pad_algorithm_on_partition(combined_partition, pad_alg, str(result_file_path))



# random_trials = 5
# final_results = dict()
# device = torch.device('cuda')
# for source in ['hqwmca', 'wmca', 'batl']:
#     for arch in ['resnet18', 'resnet50', 'wide_resnet50_2']:
#         for mode in ['fixed_feature']:
#             for eps in ['eps0.0', 'eps0.5', 'eps1.0', 'eps2.0', 'eps4.0', 'eps8.0']:
#                 for seed in range(random_trials):
#                     prefix = f"{eps}_{source}_{seed}/{source}_train/best.pth"
#                     model_path = pathlib.Path(os.getenv('TMPDIR')) / 'results' / f'{source}' / f'{arch}_{mode}' / prefix
#                     print(model_path)
                    
#                     if not pathlib.Path(model_path).is_file():
#                         print(f"{model_path} doesn't exist moving along...")
                    
#                     model = pytorch_models[args.arch](pretrained=True)
#                     model.fc = torch.nn.Linear(512 * 1 if args.arch == 'resnet18' else 512 * 4, 2)
#                     snapshot = torch.load(model_path, map_location=torch.device('cpu'))
#                     model.load_state_dict(snapshot['model'], strict=True)
#                     model.to(device)

#                     if source == 'hqwmca':
#                         test_partitions = list(itertools.chain(*[hqwmca_test_partition]))
#                         test_patch_dataset = hqwmca_patch_dataset
#                         cross_partitions = list(itertools.chain(*[wmca_train_partition, wmca_valid_partition, wmca_test_partition]))
#                         cross_patch_dataset = wmca_patch_dataset
#                         cross_partitions2 = list(itertools.chain(*[batl_train_partition, batl_valid_partition, batl_test_partition]))
#                         cross_patch_dataset2 = batl_patch_dataset
#                         test_results_save_path = f"{source}_{eps}_{seed}_hqwmca_test.csv"
#                         cross_results_save_path = f"{source}_{eps}_{seed}_wmca_all.csv"
#                         cross_results_save_path2 = f"{source}_{eps}_{seed}_batl_all.csv"
#                     elif source == "wmca":
#                         test_partitions = list(itertools.chain(*[wmca_test_partition]))
#                         test_patch_dataset = wmca_patch_dataset
#                         cross_partitions = list(itertools.chain(*[hqwmca_train_partition, hqwmca_valid_partition, hqwmca_test_partition]))
#                         cross_patch_dataset = hqwmca_patch_dataset
#                         cross_partitions2 = list(itertools.chain(*[batl_train_partition, batl_valid_partition, batl_test_partition]))
#                         cross_patch_dataset2 = batl_patch_dataset
#                         test_results_save_path = f"results/{source}_{eps}_{mode}_{seed}_wmca_test.csv"
#                         cross_results_save_path = f"results/{source}_{eps}_{mode}_{seed}_hqwmca_all.csv"
#                         cross_results_save_path2 = f"{source}_{eps}_{seed}_batl_all.csv"
#                     else:
#                         test_partitions = list(itertools.chain(*[batl_test_partition]))
#                         test_patch_dataset = batl_patch_dataset
#                         cross_partitions = list(itertools.chain(*[hqwmca_train_partition, hqwmca_valid_partition, hqwmca_test_partition]))
#                         cross_patch_dataset = hqwmca_patch_dataset
#                         cross_partitions2 = list(itertools.chain(*[wmca_train_partition, wmca_valid_partition, wmca_test_partition]))
#                         cross_patch_dataset2 = wmca_patch_dataset
#                         test_results_save_path = f"results/{source}_{eps}_{mode}_{seed}_batl_test.csv"
#                         cross_results_save_path = f"results/{source}_{eps}_{mode}_{seed}_hqwmca_all.csv"
#                         cross_results_save_path2 = f"{source}_{eps}_{seed}_wmca_all.csv"

#                     exp_name = "prior_test"
#                     Tester._test_algorithm_with_batl_framework(exp_name, model, test_patch_dataset, test_partitions,
#                                                                test_results_save_path)
#                     Tester._test_algorithm_with_batl_framework(exp_name, model, cross_patch_dataset, cross_partitions,
#                                                                cross_results_save_path)
#                     Tester._test_algorithm_with_batl_framework(exp_name, model, cross_patch_dataset2, cross_partitions2,
#                                                                cross_results_save_path2)


random_trials = 5
final_results = dict()
device = torch.device('cuda')
for root, dirs, files in os.walk(args.model_path):
    for file in sorted(files):
        eps_prefix = file.split('.ckpt')[0].split('_')[-1]
        for source in ['batl', 'hqwmca', 'wmca']:
            cross = sorted(list(set(['batl', 'hqwmca', 'wmca']) - set([source])))
            aggregated_results = dict()
            print(f"\n\n-------------------------------{source}-----------------------------------\n\n")
            print(f"model path:", file)
            np.random.random()
            for trial in range(random_trials):
                # reinitialize the model for each run
                print("loading model file :", file)
                model = pytorch_models[args.arch](pretrained=True)
                if 'eps0.0' not in file:
                    snapshot = torch.load(os.path.join(args.model_path, file), map_location=torch.device('cpu'))
                    # extract the weights of the model
                    inner_model_weights = {k.replace('module.model.', ''): v
                                           for k, v in snapshot['model'].items() if 'module.model' in k}
                    model.load_state_dict(inner_model_weights, strict=True)
                model.fc = torch.nn.Linear(512 * 1 if args.arch == 'resnet18' else 512 * 4, 2)
                model.to(device)
                
                prefix = file.split('.ckpt')[0].split('_')[-1] + f"_{source}_{trial}"
                write_path = pathlib.Path(write_path_root) / 'results' / f'{source}' / f'{args.arch}_{args.mode}' / prefix
                write_path.mkdir(exist_ok=True, parents=True)
                results = finetune(model,
                                   train_datasets[source],
                                   val_datasets[source],
                                   [
                                       (f'{source}_test', [test_datasets[source]]),
                                       (f'{cross[0]}_all', [train_datasets[cross[0]], val_datasets[cross[0]], test_datasets[cross[0]]]),
                                       (f'{cross[1]}_all', [train_datasets[cross[1]], val_datasets[cross[1]], test_datasets[cross[1]]])
                                   ],
                                   gpu=0, write_path=write_path, prefix=prefix,
                                   mode=args.mode)
                for result in results:
                    for k, v in result.items():
                        aggregated_results.setdefault(k, list()).append(v)

            for k, v in aggregated_results.items():
                if 'auc' in k:
                    print(f"{k}: {np.mean(v):.3f} +- {np.std(v):.3f}")
                    final_results.setdefault(source, dict()).setdefault(eps_prefix, dict())[k] = (np.mean(v), np.std(v))


print("\n------------------------------------------------------------------------------------------")                    
print("\n\n\n")
for eps in ['eps0.0', 'eps0.5', 'eps1.0', 'eps2.0', 'eps4.0', 'eps8.0']:
    print(f"\t{eps}")
    for source in ['batl', 'hqwmca', 'wmca']:
        print(f"\t\t{source}")
        test = sorted(list(set(['batl', 'hqwmca', 'wmca']) - set([source])))
        metric = f"{source}_test_auc"
        mean, std = final_results[source][eps][metric]
        print(f"\t\t\t{source} {eps} {metric} = {mean:.2f} +- {std:.3f}")
        for t in test:
            metric = f"{t}_all_auc"
            mean, std = final_results[source][eps][metric]
            print(f"\t\t\t{source} {eps} {metric} = {mean:.2f} +- {std:.3f}")

