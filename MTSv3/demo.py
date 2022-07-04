import os
import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.chars import getstr_grid, get_tight_rect

from PIL import Image
import numpy as np
import argparse

import glob
import tqdm
import time

import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.pyplot as plt

from yacs.config import CfgNode


def add_custom_configs(cfg: CfgNode):
    _C = cfg
    _C.SOLVER.VALID_DETECT_THRESH = 0.4
    _C.SEED = 456


class TextDemo(object):
    def __init__(
        self, cfg, confidence_threshold=0.7, min_image_size=224, output_polygon=True
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        checkpointer = DetectronCheckpointer(cfg, self.model)
        print("Load from:", cfg.MODEL.WEIGHT)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.output_polygon = output_polygon

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
        Returns:
            result_polygons (list): detection results
            result_words (list): recognition results
        """
        # result_polygons, result_words = self.compute_prediction(image)
        # return result_polygons, result_words
        return self.compute_prediction(image)

    def compute_prediction(self, original_image):
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions, _, etcs = self.model(image_list)
        # print(len(predictions), predictions, b, etcs)
        # print(len(predictions), predictions, b, etcs)
        global_predictions = predictions
        # global_predictions = predictions[0]
        # char_predictions = predictions[1]
        # char_mask = char_predictions['char_mask']
        # char_boxes = char_predictions['boxes']
        # words, rec_scores = self.process_char_mask(char_mask, char_boxes)
        # seq_words = char_predictions['seq_outputs']
        # seq_scores = char_predictions['seq_scores']

        global_predictions = [o.to(self.cpu_device) for o in global_predictions]

        # always single image is passed at a time
        global_prediction = global_predictions[0]
        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        global_prediction = global_prediction.resize((width, height))
        # print("resized", global_prediction)
        # boxes = global_prediction.bbox.tolist()
        polygons = global_prediction.get_field("masks").to_np_polygon()
        # scores = global_prediction.get_field("scores").tolist()
        scores = etcs["scores"][0].tolist()

        result_polygons = []
        for k, polygon in enumerate(polygons):
            # score = scores[k]
            if scores[k] < 0.4:  # same as the score when evaluation
                continue
            result_polygons.append(polygon)
        #     word = words[k]
        #     rec_score = rec_scores[k]
        #     seq_word = seq_words[k]
        #     seq_char_scores = seq_scores[k]
        #     seq_score = sum(seq_char_scores) / float(len(seq_char_scores))
        #     if seq_score > rec_score:
        #         result_words.append(seq_word)
        #     else:
        #         result_words.append(word)
        # return result_polygons, result_words
        return result_polygons

    def process_char_mask(self, char_masks, boxes, threshold=192):
        texts, rec_scores = [], []
        for index in range(char_masks.shape[0]):
            box = list(boxes[index])
            box = list(map(int, box))
            text, rec_score, _, _ = getstr_grid(
                char_masks[index, :, :, :].copy(), box, threshold=threshold
            )
            texts.append(text)
            rec_scores.append(rec_score)
        return texts, rec_scores

    def mask2polygon(self, mask, box, im_size, threshold=0.5, output_polygon=True):
        # mask 32*128
        image_width, image_height = im_size[1], im_size[0]
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cls_polys = (mask * 255).astype(np.uint8)
        poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
        poly_map = poly_map.astype(np.float32) / 255
        poly_map = cv2.GaussianBlur(poly_map, (3, 3), sigmaX=3)
        ret, poly_map = cv2.threshold(poly_map, 0.5, 1, cv2.THRESH_BINARY)
        if output_polygon:
            SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            poly_map = cv2.erode(poly_map, SE1)
            poly_map = cv2.dilate(poly_map, SE1)
            poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
            try:
                _, contours, _ = cv2.findContours(
                    (poly_map * 255).astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE,
                )
            except:
                contours, _ = cv2.findContours(
                    (poly_map * 255).astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE,
                )
            if len(contours) == 0:
                print(contours)
                print(len(contours))
                return None
            max_area = 0
            max_cnt = contours[0]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_cnt = cnt
            perimeter = cv2.arcLength(max_cnt, True)
            epsilon = 0.01 * cv2.arcLength(max_cnt, True)
            approx = cv2.approxPolyDP(max_cnt, epsilon, True)
            pts = approx.reshape((-1, 2))
            pts[:, 0] = pts[:, 0] + box[0]
            pts[:, 1] = pts[:, 1] + box[1]
            polygon = list(pts.reshape((-1,)))
            polygon = list(map(int, polygon))
            if len(polygon) < 6:
                return None
        else:
            SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            poly_map = cv2.erode(poly_map, SE1)
            poly_map = cv2.dilate(poly_map, SE1)
            poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
            idy, idx = np.where(poly_map == 1)
            xy = np.vstack((idx, idy))
            xy = np.transpose(xy)
            hull = cv2.convexHull(xy, clockwise=True)
            # reverse order of points.
            if hull is None:
                return None
            hull = hull[::-1]
            # find minimum area bounding box.
            rect = cv2.minAreaRect(hull)
            corners = cv2.boxPoints(rect)
            corners = np.array(corners, dtype="int")
            pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
            polygon = [x * 1.0 for x in pts]
            polygon = list(map(int, polygon))
        return polygon

    # ABCNet like visualization
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/visualizer.py
    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                # edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
                pass
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            # linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        # self.output.ax.add_patch(polygon)
        return polygon

    # ABCNet like visualization
    def visualization(self, path, polygons):
        image = plt.imread(path)
        # fig, ax = plt.subplots()

        # set size https://stackoverflow.com/a/8218887
        fig = plt.figure(frameon=False)
        fig.set_size_inches(image.shape[1] / 100, image.shape[0] / 100)

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        # plt.axis("off")
        # ax.set_axis_off()
        im = ax.imshow(image)
        for polygon in polygons:
            polygon = polygon.reshape((-1, 2))
            # print(polygon)
            # pts = np.array(polygon, np.int32)
            vis_output = self.draw_polygon(polygon, "green", edge_color=None, alpha=0.5)
            ax.add_patch(vis_output)

        return fig


def main(args):
    add_custom_configs(cfg)
    # update the config options with the config file
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # manual override some options
    # cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    text_demo = TextDemo(
        cfg, min_image_size=800, confidence_threshold=0.7, output_polygon=True
    )
    # load image and then run prediction

    # ABCNet like visualization
    if os.path.isdir(args.input[0]):
        args.input = [
            os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])
        ]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"

    # result_polygons, result_words = text_demo.run_on_opencv_image(image)
    # text_demo.visualization(image, result_polygons, result_words)

    for path in tqdm.tqdm(args.input, disable=not args.output):
        image = cv2.imread(path)
        # print(path, image)
        result_polygons = text_demo.run_on_opencv_image(image)

        # make output folder here?
        os.makedirs(args.output, exist_ok=True)

        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            out_filename = os.path.join(args.output, os.path.basename(path))
        else:
            assert len(args.input) == 1, "Please specify a directory with args.output"
            out_filename = args.output

        # visualized_output = text_demo.visualization(image, result_polygons)
        # cv2.imwrite(out_filename, visualized_output)
        # ABCNet like visualization
        visualized_output = text_demo.visualization(path, result_polygons)
        visualized_output.savefig(out_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameters for demo")
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/mixtrain/seg_rec_poly_fuse_feature.yaml",
    )
    # parser.add_argument("--image_path", type=str, default="./demo_images/demo.jpg")
    # parser.add_argument("--visu_path", type=str, default="./demo_images/demo_results.jpg")
    parser.add_argument(
        "--input", nargs="+", help="A list of space separated input images"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    main(args)
