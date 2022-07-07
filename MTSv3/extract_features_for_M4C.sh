#!/bin/bash

# ### extract emb from GT
CUDA_VISIBLE_DEVICES=0 python extract_features_vmb.py --config_file configs/best_M4C.yaml --image_dir ../COO-data/MTSv3_data/train_images/ --output_folder ../COO-data/M4C_feature/emb/train_gt_emb/ --use_gt
CUDA_VISIBLE_DEVICES=0 python extract_features_vmb.py --config_file configs/best_M4C.yaml --image_dir ../COO-data/MTSv3_data/val_images/ --output_folder ../COO-data/M4C_feature/emb/val_gt_emb/ --use_gt
CUDA_VISIBLE_DEVICES=0 python extract_features_vmb.py --config_file configs/best_M4C.yaml --image_dir ../COO-data/MTSv3_data/test_images/ --output_folder ../COO-data/M4C_feature/emb/test_gt_emb/ --use_gt

