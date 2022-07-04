# modified from https://github.com/facebookresearch/mmf/blob/main/tools/scripts/features/extract_features_vmb.py
# Copyright (c) Facebook, Inc. and its affiliates.

# Requires vqa-maskrcnn-benchmark (https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark)
# to be built and installed. Category mapping for visual genome can be downloaded from
# https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json

# When the --background flag is set, the index saved with key "objects" in
# info_list will be +1 of the Visual Genome category mapping above and 0
# is the background class. When the --background flag is not set, the
# index saved with key "objects" in info list will match the Visual Genome
# category mapping.
import argparse
import os

import cv2
import numpy as np
import torch
from maskrcnn_benchmark.config import cfg

# from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image

from tools.extraction_utils import chunks, get_image_files

from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from natsort import natsorted

pooler = Pooler(
    # setting from configs/mixtrain/seg_rec_poly_fuse_feature/yaml
    output_size=(32, 32),  # (resolution_h, resolution_w),
    scales=(0.25,),  # scales,
    sampling_ratio=2,  # sampling_ratio,
)

from shapely.geometry import Polygon, MultiPoint


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(-1, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(-1, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            iou = float(inter_area) / (union_area + 1e-6)
        except shapely.geos.TopologicalError:
            print("shapely.geos.TopologicalError occured, iou set to 0")
            iou = 0
    return iou


def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    polygon_points = np.array(line).reshape(-1, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def nms(polygon, score_list, overlap=0.2):
    rec_scores = score_list
    indices = sorted(range(len(rec_scores)), key=lambda k: -rec_scores[k])
    polygon_num = len(polygon)
    nms_flag = [True] * polygon_num
    for i in range(polygon_num):
        ii = indices[i]
        if not nms_flag[ii]:
            continue
        for j in range(polygon_num):
            jj = indices[j]
            if j == i:
                continue
            if not nms_flag[jj]:
                continue
            poly1 = polygon_from_list(polygon[ii])
            poly2 = polygon_from_list(polygon[jj])
            box1_score = rec_scores[ii]
            box2_score = rec_scores[jj]
            iou = polygon_iou(polygon[ii], polygon[jj])
            thresh = overlap

            if iou > thresh:
                if box1_score > box2_score:
                    nms_flag[jj] = False
                if box1_score == box2_score and poly1.area > poly2.area:
                    nms_flag[jj] = False
                if box1_score == box2_score and poly1.area <= poly2.area:
                    nms_flag[ii] = False
                    break

    return nms_flag


class FeatureExtractor:

    MAX_SIZE = 4000
    MIN_SIZE = 1440

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.detection_model = self._build_detection_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_file",
            # default=None,
            # default="model_best_95000_0.68513.pth",
            default="MTSv3.pth",
            type=str,
            # help="Detectron model file. This overrides the model_name param.",
            # MTS modelfile
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        # parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--batch_size", type=int, default=1, help="Batch size"
        )  # for convenience
        parser.add_argument(
            "--num_features",
            type=int,
            # default=100,
            default=75,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder",
            type=str,
            default="./M4C_vis_emb/tmp",
            help="Output folder",
        )
        parser.add_argument("--use_gt", action="store_true", help="Use gt file or not")
        parser.add_argument(
            "--hard_roi", action="store_true", help="Use hard roi masking or not"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
            + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)

        if im.shape[-1] > 3:
            im = np.array(img.convert("RGB")).astype(np.float32)

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        # self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
        self,
        proposals,
        feats,
        score_list,
        im_scales,
        im_infos,
        conf_thresh=0,
        use_gt=False,
    ):

        filtered_feat_list = []
        filtered_bbox = []
        info_list = []

        if feats == []:
            # print("empty gt")
            filtered_feat_list.append([])
            info_list.append(
                {
                    "bbox": [],
                    "num_boxes": 0,  # num_boxes.item(),
                    "objects": [],
                    "cls_prob": [],  # scores[keep_boxes][:, start_index:].cpu().numpy(),
                    "image_width": im_infos[0]["width"],
                    "image_height": im_infos[0]["height"],
                }
            )
            return filtered_feat_list, info_list

        polygon_list = proposals[0].get_field("masks").to_np_polygon()
        bbox_list = proposals[0].bbox.tolist()
        if len(feats) != len(polygon_list):
            print("ignore weird feats", len(feats), len(polygon_list), bbox_list)
            assert (
                len(polygon_list) == 0
            )  # len(feats) != len(polygon_list) and len(polygon_list) != 0
            assert sum(bbox_list[0]) == 0

        if polygon_list == []:
            # print("empty gt")
            filtered_feat_list.append([])
            info_list.append(
                {
                    "bbox": [],
                    "num_boxes": 0,  # num_boxes.item(),
                    "objects": [],
                    "cls_prob": [],  # scores[keep_boxes][:, start_index:].cpu().numpy(),
                    "image_width": im_infos[0]["width"],
                    "image_height": im_infos[0]["height"],
                }
            )
            return filtered_feat_list, info_list

        """thresholding with score 0.4 (same with STD task) """
        # print("score threshold before", len(polygon_list))
        polygon_tmp = []
        feats_tmp = []
        score_tmp = []
        for polygon, feat, score in zip(polygon_list, feats, score_list):
            if score <= 0.4:
                continue
            else:
                polygon_tmp.append(polygon)
                feats_tmp.append(feat)
                score_tmp.append(score)

        polygon_list = polygon_tmp
        feats = feats_tmp
        score_list = score_tmp
        # print("score threshold after", len(polygon_list))

        """nms same with STD task"""
        polygon_tmp = []
        feats_tmp = []
        score_tmp = []

        nms_flag = nms(polygon_list, score_list)
        for k in range(len(polygon_list)):
            dt = polygon_list[k].tolist()
            if nms_flag[k]:
                if dt not in polygon_tmp:
                    polygon_tmp.append(dt)
                    feats_tmp.append(feats[k])
                    score_tmp.append(score_list[k])
        # print("nms after", len(polygon_tmp))

        polygon_list = polygon_tmp
        feats = feats_tmp
        score_list = score_tmp

        # after score / NMS filtering, some cases become polygon_list = []
        if polygon_list == []:
            # print("empty gt")
            filtered_feat_list.append([])
            info_list.append(
                {
                    "bbox": [],
                    "num_boxes": 0,  # num_boxes.item(),
                    "objects": [],
                    "cls_prob": [],  # scores[keep_boxes][:, start_index:].cpu().numpy(),
                    "image_width": im_infos[0]["width"],
                    "image_height": im_infos[0]["height"],
                }
            )
            return filtered_feat_list, info_list

        """ take top 75 scores """
        left_page = []
        right_page = []
        page_half_x = int(im_infos[0]["width"]) / 2

        tmp_feats_list = []
        tmp_bbox_list = []

        for polygon, feat in zip(polygon_list, feats):
            x_list = polygon[0::2]
            y_list = polygon[1::2]
            bbox = [min(x_list), min(y_list), max(x_list), max(y_list)]

            tmp_feats_list.append(feat)
            tmp_bbox_list.append(bbox)

        if use_gt:
            filtered_feat_list = tmp_feats_list[: self.args.num_features]
            filtered_bbox = tmp_bbox_list[: self.args.num_features]

        else:
            if len(score_list) > 75:
                print("over 75")
                sorted_scores = sorted(score_list, reverse=True)
                threshold = sorted_scores[74]

                for feat, box, score in zip(tmp_feats_list, tmp_bbox_list, score_list):
                    if score < threshold:
                        print(threshold, score)
                        continue
                    else:
                        filtered_feat_list.append(feat)
                        filtered_bbox.append(box)

            else:
                filtered_feat_list = tmp_feats_list
                filtered_bbox = tmp_bbox_list

            assert (
                len(filtered_bbox) <= self.args.num_features
                and len(filtered_feat_list) <= self.args.num_features
            )

        objects = torch.zeros(len(score_list), 1)

        filtered_feat_list = [torch.stack(filtered_feat_list)]

        info_list.append(
            {
                "bbox": filtered_bbox / im_scales[0],
                "num_boxes": len(filtered_bbox),  # num_boxes,  # num_boxes.item(),
                "objects": objects.cpu().numpy(),
                "cls_prob": objects.cpu().numpy(),  # scores[keep_boxes][:, start_index:].cpu().numpy(),
                "image_width": im_infos[0]["width"],
                "image_height": im_infos[0]["height"],
            }
        )

        return filtered_feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []

        for image_path in image_paths:
            print(image_path)
            im, im_scale, im_info = self._image_transform(image_path)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")
        # print("0", current_img_list.image_sizes)

        if self.args.use_gt:

            image_size = (im_info["width"], im_info["height"])

            # batch 1
            gt_path = image_paths[0].replace("_images", "_gts") + ".txt"
            # print(gt_path)
            with open(gt_path, "r", encoding="utf-8") as gt_file:
                gt_list = gt_file.readlines()

            gt_bbox_list = []
            gt_polygon_list = []
            for line in gt_list:
                point_list = line.strip().split(",")[:-1]
                point_list = list(map(int, point_list))  # change to 'int' type
                x_list = point_list[0::2]
                y_list = point_list[1::2]
                gt_polygon_list.append([point_list])
                gt_bbox_list.append(
                    [min(x_list), min(y_list), max(x_list), max(y_list)]
                )
                # print(x_list, y_list, gt_bbox_list)

            if not gt_bbox_list:
                proposals = []
                emb_features = []
                score_list = []

            else:
                box_num = len(gt_bbox_list)
                height, width = im.size(1), im.size(2)

                tmp_box = BoxList(gt_bbox_list, image_size, mode="xyxy").to("cuda")
                masks = SegmentationMask(gt_polygon_list, image_size)
                tmp_box.add_field("masks", masks)
                resized_box = tmp_box.resize((width, height))

                proposals = [resized_box]
                # print(proposals)

                with torch.no_grad():
                    features = self.detection_model.backbone(
                        current_img_list.tensors
                    )  # results of FPN
                    ch1_features, fuse_feature = self.detection_model.proposal.head(
                        features
                    )
                    pooled_features = pooler([ch1_features], proposals)
                    # print(len(pooled_features), pooled_features.size())

                    emb_features = pooled_features.view(
                        box_num, -1
                    )  # use this. 32x32 = 1024 dim!

                score_list = [1] * box_num

        else:

            with torch.no_grad():

                features = self.detection_model.backbone(
                    current_img_list.tensors
                )  # results of FPN
                (proposals, seg_results), _ = self.detection_model.proposal(
                    current_img_list, features, None
                )
                # SEGHead module
                ch1_features, fuse_feature = self.detection_model.proposal.head(
                    features
                )
                pooled_features = pooler([ch1_features], proposals)
                # print(len(pooled_features), pooled_features.size())

                emb_features = pooled_features.view(
                    pooled_features.size(0), -1
                )  # use this. 32x32 = 1024 dim!
                # print(emb_features.size())

                score_list = seg_results["scores"][0].tolist()
                # print(len(seg_results["scores"][0].tolist()))
                # print(len(proposals), len(emb_features))

        feat_list = self._process_feature_extraction(
            # output,
            proposals,
            emb_features,
            score_list,
            im_scales,
            im_infos,
            # self.args.feature_name,
            self.args.confidence_threshold,
            use_gt=self.args.use_gt,
        )

        return feat_list

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        if feature == []:
            feature_save = []
        else:
            feature_save = feature.cpu().numpy()

        np.save(os.path.join(self.args.output_folder, file_base_name), feature_save)
        np.save(os.path.join(self.args.output_folder, info_file_base_name), info)

    def extract_features(self):
        image_dir = self.args.image_dir
        if os.path.isfile(image_dir):
            features, infos = self.get_detectron_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
        else:

            files = get_image_files(
                self.args.image_dir,
                exclude_list=self.args.exclude_list,
                start_index=self.args.start_index,
                end_index=self.args.end_index,
                output_folder=self.args.output_folder,
            )

            files = natsorted(files)

            finished = 0
            total = len(files)

            for chunk, begin_idx in chunks(files, self.args.batch_size):
                features, infos = self.get_detectron_features(chunk)
                # print(features.size(), infos.size())
                for idx, file_name in enumerate(chunk):
                    # print(
                    #     idx,
                    #     len(features),
                    #     len(features[idx]),
                    #     len(infos),
                    #     infos[0]["num_boxes"],
                    # )
                    self._save_feature(file_name, features[idx], infos[idx])
                finished += len(chunk)

                if finished % 200 == 0:
                    print(f"Processed {finished}/{total}")


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
