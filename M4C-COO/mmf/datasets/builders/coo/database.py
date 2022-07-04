# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import os

# from mmf.common.registry import registry
from mmf.datasets.databases.annotation_database import AnnotationDatabase
from mmf.common.registry import registry


class COOAnnotationDatabase(AnnotationDatabase):
    def __init__(self, config, path, dataset_type, *args, **kwargs):
        path = path.split(",")
        self.dataset_type = dataset_type
        super().__init__(config, path, *args, **kwargs)

    def load_annotation_db(self, path):
        cfg = registry.get("config")
        if self.dataset_type == "train":
            gt_rel_path = cfg.dataset_config.coo.annotations.train[0]
        elif self.dataset_type == "val":
            gt_rel_path = cfg.dataset_config.coo.annotations.val[0]
        elif self.dataset_type == "test":
            gt_rel_path = cfg.dataset_config.coo.annotations.test[0]

        gt_path = os.path.join(cfg.dataset_config.coo.data_dir, gt_rel_path)
        print("gt_path", self.dataset_type, gt_path)

        with open(gt_path, "r", encoding="utf-8-sig") as gt_file:
            gt_list = gt_file.readlines()

        data = []
        annotation = {}

        for gt_line in gt_list:
            split_line = gt_line.strip().split("\t")
            if len(split_line) == 1:
                manga_name_index = split_line[0]
                sentence_text = ["<pad>"]  # ["</s>"]  # ["s"]  # []
                link_text = ["</s>"]  # ["s"]  # []
                link_order = ["</s>"]  # ["s"]  # []
            elif len(split_line) == 2:
                manga_name_index = split_line[0]
                sentence_text = split_line[1].split(" ")
                link_text = ["</s>"]  # ["s"]  # []
                link_order = ["</s>"]  # ["s"]  # []
            elif len(split_line) == 4:
                manga_name_index = split_line[0]
                sentence_text = split_line[1].split(" ")
                link_text = [split_line[2]]
                link_order = [split_line[3]]

            annotation["image_id"] = manga_name_index
            annotation["ocr_tokens"] = sentence_text
            annotation["link_tokens"] = link_text
            annotation["link_order"] = link_order
            data.append(copy.deepcopy(annotation))

        self.data = data
