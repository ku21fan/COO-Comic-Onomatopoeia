# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.coo.dataset import COODataset
from mmf.datasets.builders.textvqa.builder import TextVQABuilder


@Registry.register_builder("coo")
class COOBuilder(TextVQABuilder):
    def __init__(self, dataset_name="coo", dataset_class=COODataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/coo/defaults.yaml"

    def load(self, config, *args, **kwargs):
        dataset = super().load(config, *args, **kwargs)
        dataset.dataset_name = self.dataset_name
        return dataset
