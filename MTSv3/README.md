## Acknowledgements
This implementation has been based on the repository [Mask TextSpotter v3](https://github.com/MhLiao/MaskTextSpotterV3).

## Visualization
The figures show the ground truth, prediction by ABCNet v2, and prediction by MTS v3, respectively.
The green regions are the predicted regions and the red circles are failures.

<p align="center">
   <img src="../ABCNetv2/vis-det1.jpg" width=100%>
</p>

<p align="center">
   <img src="../ABCNetv2/vis-det2.jpg" width=100%>
</p>

## Installation
Recommended OS: Linux <br>
Install conda >= 4.11.0  (python >= 3.8) and then run following commands. <br>
(if your conda version is lower than 4.11.0, update conda first, with the command `conda update -n base -c defaults conda`)

### Requirements (from original MTSv3 repository):
- GCC >= 4.9 (This is very important!)
- CUDA >= 9.0

```bash
  conda create -n MTSv3 python=3.8 -y
  conda activate MTSv3

  pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

  # python dependencies
  pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 editdistance natsort

  export INSTALL_DIR=$PWD

  # install pycocotools
  cd $INSTALL_DIR
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py build_ext install

  # install apex
  cd $INSTALL_DIR
  git clone https://github.com/NVIDIA/apex.git
  cd apex
  python setup.py install --cuda_ext --cpp_ext

  cd $INSTALL_DIR
  rm -r cocoapi/  # Please be careful when use `rm -r` command
  rm -r apex/     # Please be careful when use `rm -r` command

  # build MTSv3
  # from https://github.com/facebookresearch/maskrcnn-benchmark/issues/1236#issuecomment-903606515
  # cuda_dir="maskrcnn_benchmark/csrc/cuda"
  # perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
  python setup.py build develop

  unset INSTALL_DIR
```

## Dataset
Run [data_for_MTSv3.ipynb](https://github.com/ku21fan/COO-Comic-Onomatopoeia/blob/main/COO-data/data_for_MTSv3.ipynb) in the COO-data folder to make train/val/test data.

## Pretrained models
Download the pretrained models in [Dropbox](https://www.dropbox.com/sh/lx61z7gq5yzkp02/AAAEyzVuVqVy_-EvtqTOJTaXa?dl=0)

## Demo with pretrained model [MTSv3.pth](https://www.dropbox.com/s/u0rnep52nshfukx/MTSv3.pth)
```
CUDA_VISIBLE_DEVICES=0 python demo.py --config-file configs/best_test.yaml \
--input demo_images/test/ --output demo_results/test/ MODEL.WEIGHT MTSv3.pth
```

## Training (= Finetune)
Check the initial weights in the config file.
We finetune the model from the trained model on CTW1500 ([MTSv3_CTW1500_finetuned_model.pth](https://www.dropbox.com/s/hhwwgjuvbv6nl8j/MTSv3_CTW1500_finetuned_model.pth))

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train_net.py \
--config-file configs/COO_default.yaml SEED 456 OUTPUT_DIR output/seed456
```

## Evaluation
### Check the config file ([configs/best_test.yaml](https://github.com/ku21fan/COO-Comic-Onomatopoeia/blob/main/MTSv3/configs/best_test.yaml)) for some parameters.
test dataset: ```TEST.DATASETS```;
input size: ```INPUT.MIN_SIZE_TEST```;
output directory: ```OUTPUT_DIR```

1. run test code with pretrained model [MTSv3.pth](https://www.dropbox.com/s/u0rnep52nshfukx/MTSv3.pth)
   ```
   CUDA_VISIBLE_DEVICES=0 python test_net.py --config-file configs/best_test.yaml MODEL.WEIGHT MTSv3.pth
   ```

2. evaluation in COO_eval folder (check the path of `results_dir` in [script.py](https://github.com/ku21fan/COO-Comic-Onomatopoeia/blob/main/MTSv3/COO_eval/detect_only/script.py))
   ```
   cd COO_eval/detect_only/
   python script.py
   ```

## Citing the related works

Please cite the related works in your publications if it helps your research:

    @inproceedings{liao2020mask,
      title={Mask TextSpotter v3: Segmentation Proposal Network for Robust Scene Text Spotting},
      author={Liao, Minghui and Pang, Guan and Huang, Jing and Hassner, Tal and Bai, Xiang},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      year={2020}
    }

## License
License is described in [here (CC BY-NC 4.0)](https://github.com/MhLiao/MaskTextSpotterV3/blob/master/LICENSE.md)
