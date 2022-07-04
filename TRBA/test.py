import os
import sys
import time
import string
import argparse
import re
from datetime import date

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):

    if opt.eval_type == "benchmark":
        """evaluation with 6 benchmark evaluation datasets"""
        # eval_data_list = ["lmdb/test", "hardROI/test"]
        eval_data_list = ["lmdb/test"]
        opt.eval_data = "../COO-data/TRBA_data/"

    if calculate_infer_time:
        eval_batch_size = (
            1  # batch_size should be 1 to calculate the GPU inference time per image.
        )
    else:
        eval_batch_size = opt.batch_size

    if opt.SARdecode:
        validation_function = validation_SARdecode
    else:
        validation_function = validation

    accuracy_list = []
    total_forward_time = 0
    total_eval_data_number = 0
    total_correct_number = 0
    log = open(f"./result/{opt.exp_name}/log_all_evaluation.txt", "a")
    dashed_line = "-" * 80
    print(dashed_line)
    log.write(dashed_line + "\n")
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_eval = AlignCollate(opt, mode="test")
        eval_data, eval_data_log = hierarchical_dataset(
            root=eval_data_path, opt=opt, mode="test"
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_eval,
            pin_memory=True,
        )

        (
            _,
            accuracy_by_best_model,
            _,
            _,
            _,
            infer_time,
            length_of_data,
        ) = validation_function(
            model, criterion, eval_loader, converter, opt, tqdm_position=0
        )
        accuracy_list.append(f"{accuracy_by_best_model:0.2f}")
        total_forward_time += infer_time
        total_eval_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        log.write(eval_data_log)
        print(f"Acc {accuracy_by_best_model:0.2f}")
        log.write(f"Acc {accuracy_by_best_model:0.2f}\n")
        print(dashed_line)
        log.write(dashed_line + "\n")

    averaged_forward_time = total_forward_time / total_eval_data_number * 1000
    total_accuracy = total_correct_number / total_eval_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    eval_log = "accuracy: "
    for name, accuracy in zip(eval_data_list, accuracy_list):
        eval_log += f"{name}: {accuracy}\t"
    eval_log += f"total_accuracy: {total_accuracy:0.2f}\t"
    eval_log += f"averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.2f}"
    print(eval_log)
    log.write(eval_log + "\n")

    # for convenience
    print("\t".join(accuracy_list))
    print(f"Total_accuracy:{total_accuracy:0.2f}")
    log.write("\t".join(accuracy_list) + "\n")
    log.write(f"Total_accuracy:{total_accuracy:0.2f}" + "\n")
    log.close()

    # for convenience
    today = date.today()
    if opt.log_multiple_test:
        log_all_model = open(f"./evaluation_log/log_multiple_test_{today}.txt", "a")
        log_all_model.write("\t".join(accuracy_list) + "\n")
    else:
        log_all_model = open(
            f"./evaluation_log/log_all_model_evaluation_{today}.txt", "a"
        )
        log_all_model.write(
            f"./result/{opt.exp_name}\tTotal_accuracy:{total_accuracy:0.2f}\n"
        )
        log_all_model.write("\t".join(accuracy_list) + "\n")
    log_all_model.close()

    return total_accuracy, eval_data_list, accuracy_list


def validation(model, criterion, eval_loader, converter, opt, tqdm_position=1):
    """validation or evaluation"""
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in tqdm(
        enumerate(eval_loader),
        total=len(eval_loader),
        position=tqdm_position,
        leave=False,
    ):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        labels_index, labels_length = converter.encode(
            labels, batch_max_length=opt.batch_max_length
        )

        if "CTC" in opt.Prediction:
            start_time = time.time()
            preds = model(image)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(
                preds.log_softmax(2).permute(1, 0, 2),
                labels_index,
                preds_size,
                labels_length,
            )

        else:
            text_for_pred = (
                torch.LongTensor(batch_size).fill_(converter.dict["[SOS]"]).to(device)
            )

            start_time = time.time()
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            target = labels_index[:, 1:]  # without [SOS] Symbol
            cost = criterion(
                preds.contiguous().view(-1, preds.shape[-1]),
                target.contiguous().view(-1),
            )

        # make unknown chracter to [UNK] token.
        if "Attn" in opt.Prediction:
            labels_string = converter.decode(
                labels_index[:, 1:], labels_length
            )  # without [SOS] Symbol
        elif "CTC" in opt.Prediction:
            labels_string = converter.decode_gt(labels_index, labels_length)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_size = torch.IntTensor([preds.size(1)] * preds_index.size(0)).to(device)
        preds_str = converter.decode(preds_index, preds_size)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, prd, prd_max_prob in zip(labels_string, preds_str, preds_max_prob):
            if "Attn" in opt.Prediction:
                gt = gt[: gt.find("[EOS]")]
                prd_EOS = prd.find("[EOS]")
                prd = prd[:prd_EOS]  # prune after "end of sentence" token ([EOS])
                prd_max_prob = prd_max_prob[:prd_EOS]

            """we only evaluate the characters included in char_set (=opt.character)."""
            gt = gt.replace("[UNK]", "")
            prd = prd.replace("[UNK]", "")

            if opt.NED:
                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(prd) == 0:
                    norm_ED += 0
                elif len(gt) > len(prd):
                    norm_ED += 1 - edit_distance(prd, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(prd, gt) / len(prd)

            else:
                if prd == gt:
                    n_correct += 1

            # calculate confidence score (= multiply of prd_max_prob)
            try:
                confidence_score = prd_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([EOS])
            confidence_score_list.append(confidence_score)

    if opt.NED:
        # ICDAR2019 Normalized Edit Distance. In web page, they report % of norm_ED (= norm_ED * 100).
        score = norm_ED / float(length_of_data) * 100
    else:
        score = n_correct / float(length_of_data) * 100  # accuracy

    return (
        valid_loss_avg.val(),
        score,
        preds_str,
        confidence_score_list,
        labels_string,
        infer_time,
        length_of_data,
    )


def validation_SARdecode(
    model, criterion, eval_loader, converter, opt, tqdm_position=1
):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in tqdm(
        enumerate(eval_loader),
        total=len(eval_loader),
        position=tqdm_position,
        leave=False,
    ):
        batch_size = image_tensors.size(0)

        """
        To use decoding technique from SAR paper,
        A mini-batch consists of original image, 90 rotated image, and 270 rotated image.
        (= torch.cat([image, 90 degree rotated image, 270 degree rotated image], dim=0))
        """
        batch_size_third = int(
            batch_size / 3
        )  # 3 for original image, 90 rotated image, 270 rotated image.

        image = image_tensors.to(device)
        # For max length prediction
        labels_original_img = labels[
            :batch_size_third
        ]  # use only original image, excluding 90 and 270 rotated images
        labels_index, labels_length = converter.encode(
            labels_original_img, batch_max_length=opt.batch_max_length
        )

        if "CTC" in opt.Prediction:
            start_time = time.time()
            preds = model(image)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size_third)
            # permute 'preds' to use CTCloss format
            preds_original_img = (
                preds[:batch_size_third].log_softmax(2).permute(1, 0, 2)
            )
            cost = criterion(
                preds_original_img, labels_index, preds_size, labels_length
            )

        else:
            text_for_pred = (
                torch.LongTensor(batch_size).fill_(converter.dict["[SOS]"]).to(device)
            )

            start_time = time.time()
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            target = labels_index[:, 1:].contiguous().view(-1)  # without [SOS] Symbol
            preds_original_img = (
                preds[:batch_size_third].contiguous().view(-1, preds.shape[-1])
            )
            cost = criterion(preds_original_img, target)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        # decoding technique from SAR paper.
        three_preds_index = torch.cat(
            [
                preds_index[:batch_size_third].unsqueeze(1),  # origin images
                preds_index[batch_size_third : batch_size_third * 2].unsqueeze(
                    1
                ),  # 90 degree rotated images
                preds_index[batch_size_third * 2 :].unsqueeze(1),
            ],
            dim=1,
        )  # 270 degree rotated images
        three_preds_max_prob = torch.cat(
            [
                preds_max_prob[:batch_size_third].unsqueeze(1),  # origin images
                preds_max_prob[batch_size_third : batch_size_third * 2].unsqueeze(
                    1
                ),  # 90 degree rotated images
                preds_max_prob[batch_size_third * 2 :].unsqueeze(1),
            ],
            dim=1,
        )  # 270 degree rotated images

        preds_index = []
        confidence_score_list = []
        for three_prd_idx, three_prd_prob in zip(
            three_preds_index, three_preds_max_prob
        ):

            three_confidence_score = (
                []
            )  # for select top confidence score among 0, 90, 270 rotated images
            for prd_idx, prd_prob in zip(three_prd_idx, three_prd_prob):
                if "Attn" in opt.Prediction:
                    try:
                        prd_EOS_idx = torch.nonzero(
                            prd_idx == opt.eos_token_index, as_tuple=False
                        )[0]
                        prd_prob = prd_prob[:prd_EOS_idx]
                    except:
                        prd_prob = prd_prob

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = prd_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([EOS])
                three_confidence_score.append(confidence_score)

            confidence_score_list.append(max(three_confidence_score))
            top_confidence_index = three_confidence_score.index(
                max(three_confidence_score)
            )

            pred = three_prd_idx[top_confidence_index]
            preds_index.append(pred)

        preds_index = torch.stack(preds_index)
        length_of_data = length_of_data + preds_index.size(0)

        # make unknown chracter to [UNK] token.
        if "Attn" in opt.Prediction:
            labels_string = converter.decode(
                labels_index[:, 1:], labels_length
            )  # without [SOS] Symbol
        elif "CTC" in opt.Prediction:
            labels_string = converter.decode_gt(labels_index, labels_length)

        # calculate accuracy
        preds_size = torch.IntTensor([preds.size(1)] * preds_index.size(0)).to(device)
        preds_str = converter.decode(preds_index, preds_size)
        for gt, prd in zip(labels_string, preds_str):
            if "Attn" in opt.Prediction:
                gt = gt[: gt.find("[EOS]")]
                prd = prd[: prd.find("[EOS]")]

            """we only evaluate the characters included in char_set (=opt.character)."""
            gt = gt.replace("[UNK]", "")
            prd = prd.replace("[UNK]", "")

            if opt.NED:
                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(prd) == 0:
                    norm_ED += 0
                elif len(gt) > len(prd):
                    norm_ED += 1 - edit_distance(prd, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(prd, gt) / len(prd)

            else:
                if prd == gt:
                    n_correct += 1

    if opt.NED:
        # ICDAR2019 Normalized Edit Distance. In web page, they report % of norm_ED (= norm_ED * 100).
        score = norm_ED / float(length_of_data) * 100
    else:
        score = n_correct / float(length_of_data) * 100  # accuracy

    return (
        valid_loss_avg.val(),
        score,
        preds_str,
        confidence_score_list,
        labels_string,
        infer_time,
        length_of_data,
    )


def test(opt):
    """model configuration"""
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
        opt.sos_token_index = converter.dict["[SOS]"]
        opt.eos_token_index = converter.dict["[EOS]"]
    opt.num_class = len(converter.character)

    model = Model(opt)
    print(
        "model input parameters",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print("loading pretrained model from %s" % opt.saved_model)
    try:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    except:
        print(
            "*** pretrained model not match strictly *** and thus load_state_dict with strict=False mode"
        )
        # pretrained_state_dict = torch.load(opt.saved_model)
        # for name in pretrained_state_dict:
        #     print(name)
        model.load_state_dict(
            torch.load(opt.saved_model, map_location=device), strict=False
        )

    opt.exp_name = "_".join(opt.saved_model.split("/")[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f"./result/{opt.exp_name}", exist_ok=True)
    # os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if "CTC" in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        # ignore [PAD] token
        criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.dict["[PAD]"]).to(
            device
        )

    """ evaluation """
    model.eval()
    with torch.no_grad():
        if (
            opt.eval_type
        ):  # evaluate 6 benchmark evaluation datasets or 7 additionally collected evaluation datasets
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            log = open(f"./result/{opt.exp_name}/log_evaluation.txt", "a")
            AlignCollate_eval = AlignCollate(opt, mode="test")
            eval_data, eval_data_log = hierarchical_dataset(
                root=opt.eval_data, opt=opt, mode="test"
            )
            eval_loader = torch.utils.data.DataLoader(
                eval_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_eval,
                pin_memory=True,
            )

            if opt.SARdecode:
                validation_function = validation_SARdecode
            else:
                validation_function = validation

            _, score_by_best_model, _, _, _, _, _ = validation_function(
                model, criterion, eval_loader, converter, opt
            )
            log.write(eval_data_log)
            print(f"{score_by_best_model:0.2f}")
            log.write(f"{score_by_best_model:0.2f}\n")
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", help="path to evaluation dataset")
    parser.add_argument(
        "--eval_type",
        type=str,
        help="evaluate 6 benchmark evaluation datasets or 7 additionally collected evaluation datasets |benchmark|addition|",
    )
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=512, help="input batch size")
    parser.add_argument(
        "--saved_model", required=True, help="path to saved_model to evaluation"
    )
    parser.add_argument(
        "--log_multiple_test", action="store_true", help="log_multiple_test"
    )
    """ Data processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        help="character label",
    )
    parser.add_argument(
        "--char_set",
        type=str,
        default="../COO-data/Onomatopoeia_train_char_set.txt",
        help="character set for target language",
    )
    parser.add_argument(
        "--NED", action="store_true", help="For Normalized edit_distance"
    )
    """ Model Architecture """
    parser.add_argument("--model_name", type=str, required=True, help="CRNN|TRBA")
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
    parser.add_argument("--SARdecode", action="store_true", help="use SAR decode")
    parser.add_argument(
        "--no_rotate", action="store_true", help="no use rotate for vertical"
    )
    parser.add_argument("--twoD", action="store_true", help="use twoDimension")

    opt = parser.parse_args()

    if opt.char_set:
        with open(opt.char_set, "r", encoding="utf-8-sig") as char_set:
            opt.character = char_set.readlines()[0].strip()

    if opt.model_name == "CRNN":
        opt.Transformation = "None"
        opt.FeatureExtraction = "VGG"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "CTC"

    elif opt.model_name == "TRBA":
        opt.Transformation = "TPS"
        opt.FeatureExtraction = "ResNet"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "Attn"

    elif opt.model_name == "RBA":  # RBA
        opt.Transformation = "None"
        opt.FeatureExtraction = "ResNet"
        opt.SequenceModeling = "BiLSTM"
        opt.Prediction = "Attn"

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        print(
            "We recommend to use 1 GPU, check your GPU number, you would miss CUDA_VISIBLE_DEVICES=0 or typo"
        )
        print("To use multi-gpu setting, remove or comment out these lines")
        sys.exit()

    if sys.platform == "win32":
        opt.workers = 0

    os.makedirs(f"./evaluation_log", exist_ok=True)

    test(opt)
