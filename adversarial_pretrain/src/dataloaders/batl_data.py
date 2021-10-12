import copy
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from batl_utils_modules.dataIO.odin_h5_dataset import Odin5MultiDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from batl_isi_pad_modules.utils.sampler import ListBatchSampler
from batl_isi_pad_modules.dataset_utils.face_datasets.odin_h5_face_datasets_helpers import create_odin_face_dataset, \
    FaceOdinMultiLabelData
from batl_isi_pad_modules.utils.ReaderPatchDatasets import ReaderPatchDataset, get_default_general_storage_func
from batl_isi_pad_modules.utils.ReaderPatchGenerators import IdentityPatchGenerator
from batl_isi_pad_modules.dataset_utils.FaceDetector import FaceAlignmentSingleFaceDetector


def save_mem_collate_fc(batch):
    return default_collate(batch.pop())


def worker_init(wid):
    return np.random.seed(np.random.randint(1, 1000) + wid)


def square_face_detector(device: (None, str) = None, flexible: bool = True):
    face_detector = FaceAlignmentSingleFaceDetector(
        params={'device': device,  # -> Use gpu if available.
                'faceDetector': 'sfd',
                'resizeRatio': 0.25,  # -> Resize image by 0.25 before applying face detection.
                'interpolationMethod': 'cubic',  # -> Use bicubic interpolation for resizing.
                'borderExpansion': (0.0, 0.0, 0.0, 0.0),  # -> Expand the detected bounding box by 0.25 on the top.
                'returnLandmarks': True,  # -> Use landmarks for calculating transformations.
                'acceptMaxBoundingBox': flexible},
        check_bounds=False,
        input_color_format='bgr')
    return face_detector


class CustomRandomSampler:
    def __init__(self, num_samples: int = None, generator=None, frequency=1) -> None:
        self._num_samples = num_samples
        self.sample_frequency = frequency

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self):
        n = self._num_samples
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        random_sample_indices = torch.randperm(n, generator=generator) * self.sample_frequency + \
                                torch.randint(0, self.sample_frequency, (n,), generator=generator)
        yield from random_sample_indices.tolist()

    def __len__(self) -> int:
        return self.num_samples


def _get_data(db_directories, gt_files, db_partitions, data_type="COLOR", batch_size=32, num_workers=2,
              check_file_existence=False, num_frames=1, frames_spacing='sequential', stats=False, img_size=(320, 256),
              copy_data_locally=False, load_data_in_memory=True):
    datasets = []
    multi_label_data = FaceOdinMultiLabelData()
    extractor_params = {'num_frames': num_frames, 'frames_spacing': frames_spacing}
    face_detector = square_face_detector()
    for i, dbd in enumerate(db_directories):
        print("extracting db_directory:", dbd)
        datasets.append(create_odin_face_dataset(data_type, db_path=dbd, gt_path=gt_files[i],
                                                 part_path=db_partitions[i],
                                                 pai_code_label_mapping=multi_label_data.pai_code_label_mapping,
                                                 multi_label=True,
                                                 check_file_existence=check_file_existence,
                                                 face_detector=face_detector,
                                                 output_shape=img_size,
                                                 extractor_params=extractor_params))  # ['sequential', 'equally_spaced']
    dataset = Odin5MultiDataset(datasets=datasets)
    """
    GETTING A SPECIFIC PARTITION - Returns of DataReader objects
    """
    train_partition = dataset.get_partition('train')
    valid_partition = dataset.get_partition('valid')
    test_partition = dataset.get_partition('test')
    print(len(train_partition), len(test_partition), len(valid_partition))
    """
    CREATING A DATA LOADER
    """
    # note: this is used because the batl-isi-pad was modified to extract 3 equally spaced frames per transaction trial
    # file: batl_isi_pad_modules/dataset_utils/face_datasets/odin_h5_face_datasets_helpers.py
    # patch_generator = GeneralPatchGenerator(patch_size=GuidedAdvAugMix.IMG_SIZE, stride=GuidedAdvAugMix.IMG_SIZE)
    patch_generator = IdentityPatchGenerator()
    storage_func = get_default_general_storage_func(max_groups_per_file=2000, compression_level=3,
                                                    max_lock_attempts=1000000, verbose=False)
    patch_dataset = ReaderPatchDataset(data_storage_func=storage_func,
                                       patch_generator=patch_generator,
                                       make_transform=None,
                                       vista_subpath='BATL_FACE',
                                       vista_output_dir="/nas/vista-hdd01/batl/batl_isi_pad_datasets",
                                       copy_data_locally=copy_data_locally,
                                       load_data_in_memory=load_data_in_memory,
                                       dataset_storage_name="WMCA_RGB_Preprocessed" if 'WMCA' in db_directories[
                                           0] else None,
                                       num_classes=multi_label_data.get_num_classes(),
                                       getitem_vars=('data', 'labels_binary', 'weights_binary', 'labels_multi'),
                                       verbose=True)

    data_loaders = list()
    transform = None
    for idx, partition in enumerate((train_partition, valid_partition, test_partition)):
        if len(partition) == 0:
            data_loaders.append(None)
            continue

        tmp_patch_dataset = copy.deepcopy(patch_dataset)
        # for train compute the transform and use the transform for valid, test
        if idx == 0:
            tmp_patch_dataset.prepare_dataset(file_readers=partition)
            transform = tmp_patch_dataset.transform
        else:
            tmp_patch_dataset.prepare_dataset(file_readers=partition, transform_to_use=transform)

        if not tmp_patch_dataset.save_mem or (tmp_patch_dataset.save_mem and tmp_patch_dataset.load_data_in_memory):
            data_loaders.append(
                DataLoader(
                    dataset=tmp_patch_dataset, batch_size=batch_size,
                    shuffle=(idx == 0), pin_memory=True,
                    num_workers=num_workers, drop_last=(idx == 0),
                    worker_init_fn=worker_init,
                    persistent_workers=True
                )
            )

        else:
            data_loaders.append(
                DataLoader(
                    dataset=tmp_patch_dataset,
                    pin_memory=True,
                    collate_fn=save_mem_collate_fc,
                    batch_sampler=ListBatchSampler(
                        CustomRandomSampler(len(tmp_patch_dataset) // num_frames, frequency=num_frames) if (
                                idx == 0) else SequentialSampler(tmp_patch_dataset),
                        batch_size=batch_size,
                        drop_last=(idx == 0)),
                    num_workers=num_workers,
                    worker_init_fn=worker_init,
                    persistent_workers=True
                )
            )

    if stats:
        return data_loaders, transform.mean, transform.std, \
               (patch_dataset, train_partition, valid_partition, test_partition)
    return data_loaders, None, None, (patch_dataset, train_partition, valid_partition, test_partition)


def get_dataloader(dataset_name, batch_size, img_size=(224, 224), num_frames=5, data_type='COLOR'):
    db_directories = {
        "BATL": [
            # "/nas/vista-hdd01/batl/odin-phase-2-self-test-3/FACE",
            "/nas/vista-hdd01/batl/GCT-2/FACE",
            "/nas/vista-hdd01/batl/odin-phase-2-self-test-4/FACE",
            "/nas/vista-hdd01/batl/GCT-3/FACE",
            "/nas/vista-hdd01/batl/GCT-4/FACE",
        ],
        "HQ-WMCA": ["/nas/vista-hdd01/batl/IDIAP-DATA/FACE"],
        "WMCA": ["/nas/vista-hdd01/public_datasets/WMCA/WMCA/preprocessed-face-station_RGB"]
    }
    gt_files = {
        "BATL": [
            # "/nas/vista-ssd01/batl/odin-phase-2-self-test-3/batl_st3_partitions/st3_0225_0308_FACE_ground_truth.csv",
            "/nas/vista-ssd01/batl/GCT-2/batl_gct2_partitions/gct2_0508_0523_FACE_ground_truth.csv",
            "/nas/vista-ssd01/batl/odin-phase-2-self-test-4/batl_st4_partitions/st4_0903_0910_FACE_ground_truth.csv",
            "/nas/vista-ssd01/batl/GCT-3/batl_gct3_partitions/gct3_1025_1115_FACE_ground_truth.csv",
            "/nas/vista-ssd01/batl/GCT-4/batl_gct4_partitions/gct4_1106_0308_FACE_ground_truth.csv",
        ],
        "HQ-WMCA": ["/nas/vista-ssd01/batl/IDIAP-DATA/batl_idiap_partitions/idiap_0226_1011_FACE_ground_truth.csv"],
        "WMCA": ["/nas/vista-hdd01/public_datasets/WMCA/WMCA/batl_wmca_partitions/wmca_FACE_ground_truth.csv"]
    }
    # no consolidated partitions for gct4 and idiap
    db_partitions = {
        "BATL": [
            # "/nas/vista-ssd01/batl/odin-phase-2-self-test-3/batl_st3_partitions/consolidated_partitions/st3_dataset_partition_FACE_3folds_part0.csv",
            "/nas/vista-ssd01/batl/GCT-2/batl_gct2_partitions/consolidated_partitions/gct2_dataset_partition_FACE_3folds_part0.csv",
            "/nas/vista-ssd01/batl/odin-phase-2-self-test-4/batl_st4_partitions/consolidated_partitions/st4_dataset_partition_FACE_3folds_part0.csv",
            "/nas/vista-ssd01/batl/GCT-3/batl_gct3_partitions/consolidated_partitions/gct3_dataset_partition_FACE_3folds_part0.csv",
            "/nas/vista-ssd01/batl/GCT-4/batl_gct4_partitions/gct4_dataset_partition_FACE_3folds_part0.csv",
        ],
        "HQ-WMCA": [
            "/nas/vista-ssd01/batl/IDIAP-DATA/batl_idiap_partitions/idiap_dataset_partition_FACE_grandtest.csv"],
        "WMCA": [
            "/nas/vista-hdd01/public_datasets/WMCA/WMCA/batl_wmca_partitions/wmca_dataset_partition_FACE_grandtest.csv"]
    }

    return _get_data(db_directories[dataset_name], gt_files[dataset_name], db_partitions[dataset_name],
                     data_type=data_type, batch_size=batch_size, num_frames=num_frames, frames_spacing='equally_spaced',
                     img_size=img_size, copy_data_locally=True, load_data_in_memory=False)

