import os
import sys
import re
import six
import time
import math
import random

from natsort import natsorted
import PIL
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


class Batch_Balanced_Dataset(object):
    def __init__(
        self, opt, dataset_root, select_data, batch_ratio, log, learn_type=None
    ):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        self.opt = opt
        dashed_line = "-" * 80
        print(dashed_line)
        log.write(dashed_line + "\n")
        print(
            f"dataset_root: {dataset_root}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}"
        )
        log.write(
            f"dataset_root: {dataset_root}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}\n"
        )
        assert len(select_data) == len(batch_ratio)

        _AlignCollate = AlignCollate(self.opt)

        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(select_data, batch_ratio):
            _batch_size = max(round(self.opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + "\n")
            _dataset, _dataset_log = hierarchical_dataset(
                root=dataset_root, opt=self.opt, select_data=[selected_d],
            )
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            """
            number_dataset = int(
                total_number_dataset * float(self.opt.total_data_usage_ratio)
            )
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [
                Subset(_dataset, indices[offset - length : offset])
                for offset, length in zip(_accumulate(dataset_split), dataset_split)
            ]
            selected_d_log = f"num total samples of {selected_d}: {total_number_dataset} x {self.opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n"

            selected_d_log += f"num samples of {selected_d} per batch: {self.opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}"
            print(selected_d_log)
            log.write(selected_d_log + "\n")
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            # for faster training, we multiply small datasets itself.
            if len(_dataset) < 50000:
                multiple_times = int(50000 / len(_dataset))
                dataset_self_multiple = [_dataset] * multiple_times
                _dataset = ConcatDataset(dataset_self_multiple)

            _data_loader = torch.utils.data.DataLoader(
                _dataset,
                batch_size=_batch_size,
                shuffle=True,
                num_workers=int(self.opt.workers),
                collate_fn=_AlignCollate,
                pin_memory=False,
                drop_last=False,
            )

            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f"{dashed_line}\n"
        batch_size_sum = "+".join(batch_size_list)
        Total_batch_size_log += (
            f"Total_batch_size: {batch_size_sum} = {Total_batch_size}\n"
        )
        Total_batch_size_log += f"{dashed_line}"
        self.opt.Total_batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + "\n")

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_labels = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, label = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_labels += label
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, label = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_labels += label
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_labels


def hierarchical_dataset(root, opt, select_data="/", mode="train"):
    """select_data='/' contains all sub-directory of root directory"""
    dataset_list = []
    dataset_log = f"dataset_root:    {root}\t dataset: {select_data[0]}"
    print(dataset_log)
    dataset_log += "\n"
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt, mode=mode)
                sub_dataset_log = f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
                print(sub_dataset_log)
                dataset_log += f"{sub_dataset_log}\n"
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):
    def __init__(self, root, opt, mode="train"):

        self.root = root
        self.opt = opt
        self.mode = mode
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                label = txn.get(label_key).decode("utf-8")

                # length filtering
                length_of_label = len(label)
                if length_of_label > opt.batch_max_length:
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = txn.get(label_key).decode("utf-8")
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))
                label = "[dummy_label]"

            # For length of label >= 3 and height > width image, rotate the image to make it horizontal.
            # To  avoid rotate texts such as 'it', '1l', 'yi', and so on.
            if not self.opt.no_rotate:
                width, height = img.size
                if len(label) >= 3 and height > width:
                    img = img.transpose(PIL.Image.ROTATE_90)

        return (img, label)


class RawDataset(Dataset):
    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == ".jpg" or ext == ".jpeg" or ext == ".png":
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            img = PIL.Image.open(self.image_path_list[index]).convert("RGB")

        except IOError:
            print(f"Corrupted image for {index}")
            # make dummy image and dummy label for corrupted image.
            img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))

            # For length of label >= 3 and height > width image, rotate the image to make it horizontal.
            # To  avoid rotate texts such as 'it', '1l', 'yi', and so on.
            if not self.opt.no_rotate:
                width, height = img.size
                if len(label) >= 3 and height > width:
                    img = img.transpose(PIL.Image.ROTATE_90)

        return (img, self.image_path_list[index])


class AlignCollate(object):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.mode = mode
        self.transform = ResizeNormalize((opt.imgW, opt.imgH))

    def __call__(self, batch):
        images, labels = zip(*batch)

        if self.mode == "test" and self.opt.SARdecode:

            image_tensors = []
            image_tensors_90 = []  # rotated 90 degree
            image_tensors_270 = []  # rotated 270 degree (-90 degree)

            for image in images:
                image_tensors.append(self.transform(image))

                """
                Decoding technique from SAR paper: if height > width, we rotate image 90, 270 degrees.
                """
                width, height = image.size
                if height > width:
                    image_tensors_90.append(
                        self.transform(image.transpose(PIL.Image.ROTATE_90))
                    )
                    image_tensors_270.append(
                        self.transform(image.transpose(PIL.Image.ROTATE_270))
                    )
                else:
                    image_tensors_90.append(self.transform(image))
                    image_tensors_270.append(self.transform(image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
            image_tensors_90 = torch.cat([t.unsqueeze(0) for t in image_tensors_90], 0)
            image_tensors_270 = torch.cat(
                [t.unsqueeze(0) for t in image_tensors_270], 0
            )

            image_tensors = torch.cat(
                [image_tensors, image_tensors_90, image_tensors_270], dim=0
            )
            labels = labels + labels + labels

            return image_tensors, labels

        else:

            image_tensors = [self.transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            # # image check
            # from utils import tensor2im, save_image

            # os.makedirs(f"./image_test/{self.opt.imgH}", exist_ok=True)
            # for i, (img_tensor, label) in enumerate(zip(image_tensors, labels)):
            #     img = tensor2im(img_tensor)
            #     save_image(img, (f"./image_test/{self.opt.imgH}/{i}_{label}.jpg"))
            # print(asdf)

            return image_tensors, labels


class ResizeNormalize(object):
    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        # CAUTION: it should be (width, height). different from size of transforms.Resize (height, width)
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, image):
        image = image.resize(self.size, self.interpolation)
        image = self.toTensor(image)
        image.sub_(0.5).div_(0.5)
        return image
