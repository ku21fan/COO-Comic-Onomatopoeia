# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import object_to_byte_tensor
from mmf.datasets.builders.coo.database import COOAnnotationDatabase


class COODataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            "coo", config, dataset_type, annotation_database=COOAnnotationDatabase
        )
        self.dataset_name = "textcaps"
        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info

    # from textcaps
    def preprocess_sample_info(self, sample_info):
        # add dummy questions to train with M4C (for TextVQA)
        sample_info["question_str"] = ""  # empty question
        sample_info["question_id"] = sample_info["image_id"]
        return sample_info

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        current_sample.link_order = sample_info["link_order"]
        current_sample.question_id = sample_info["question_id"]

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]
        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)
            # print(current_sample.image_id, current_sample.image_info_0.image_id)
            assert current_sample.image_id == current_sample.image_info_0.image_id

        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead
        if hasattr(current_sample, "image_info_0"):
            for k in list(current_sample.image_info_0):
                if k != "max_features":
                    current_sample.image_info_0.pop(k)
        if hasattr(current_sample, "image_info_1"):
            for k in list(current_sample.image_info_1):
                if k != "max_features":
                    current_sample.image_info_1.pop(k)

        return current_sample

    def add_sample_details(self, sample_info, sample):
        sample.image_id = object_to_byte_tensor(sample.image_id)

        # 1. Load text (question words)
        question_str = (
            sample_info["question"]
            if "question" in sample_info
            else sample_info["question_str"]
        )
        text_processor_args = {"text": question_str}

        if "question_tokens" in sample_info:
            text_processor_args["tokens"] = sample_info["question_tokens"]

        processed_question = self.text_processor(text_processor_args)

        if "input_ids" in processed_question:
            sample.text = processed_question["input_ids"]
            sample.text_len = torch.tensor(
                len(processed_question["tokens"]), dtype=torch.long
            )
        else:
            # For GLoVe based processors
            sample.text = processed_question["text"]
            sample.text_len = processed_question["length"]

        # 2. Load object
        # object bounding box information
        if "obj_normalized_boxes" in sample.image_info_0 and hasattr(
            self, "copy_processor"
        ):
            sample.obj_bbox_coordinates = self.copy_processor(
                {"blob": sample.image_info_0.obj_normalized_boxes}
            )["blob"]

        # 3. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info["ocr_tokens"] = []
            sample_info["ocr_info"] = []
            # if "ocr_normalized_boxes" in sample_info:
            if "ocr_normalized_boxes" in sample.image_info_0:
                sample.image_info_0["ocr_normalized_boxes"] = np.zeros(
                    (0, 4), np.float32
                )
            # clear OCR visual features
            if "image_feature_1" in sample:
                sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
            return sample

        # Preprocess OCR tokens
        if hasattr(self, "ocr_token_processor"):
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample_info["ocr_tokens"]
            ]
        else:
            ocr_tokens = sample_info["ocr_tokens"]

        # Get FastText embeddings for OCR tokens
        context = self.context_processor({"tokens": ocr_tokens})
        sample.context = context["text"]
        sample.ocr_tokens = context["tokens"]

        sample.context_tokens = object_to_byte_tensor(context["tokens"])
        sample.context_feature_0 = context["text"]
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]

        # Get PHOC embeddings for OCR tokens
        if hasattr(self, "phoc_processor"):
            context_phoc = self.phoc_processor({"tokens": ocr_tokens})
            sample.context_feature_1 = context_phoc["text"]
            sample.context_info_1 = Sample()
            sample.context_info_1.max_features = context_phoc["length"]

        # OCR bounding box information
        if "ocr_normalized_boxes" in sample.image_info_0 and hasattr(
            self, "copy_processor"
        ):
            # New imdb format: OCR bounding boxes are already pre-computed
            max_len = self.config.processors.answer_processor.params.max_length
            sample.ocr_bbox_coordinates = self.copy_processor(
                {"blob": sample.image_info_0["ocr_normalized_boxes"]}
            )["blob"][:max_len]

        return sample

    def add_answer_info(self, sample_info, sample):
        # Load real answers from sample_info
        answer_processor = self.config.processors.answer_processor.type
        if answer_processor == "m4c_coo":
            answers = sample_info.get("link_order", [])
        else:
            answers = sample_info.get("link_tokens", [])

        answer_processor_arg = {"answers": answers}

        answer_processor_arg["tokens"] = sample.pop("ocr_tokens", [])

        processed_answers = self.answer_processor(answer_processor_arg)

        assert not self.config.fast_read, (
            "In TextVQADataset, online OCR sampling is incompatible "
            "with fast_read, so fast_read is currently not supported."
        )

        sample.update(processed_answers)
        sample.answers = object_to_byte_tensor(answers)

        if "answers_scores" in sample:
            sample.targets = sample.pop("answers_scores")

        return sample
