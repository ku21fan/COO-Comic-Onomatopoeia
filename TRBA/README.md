
## Acknowledgements
This implementation has been based on the repository [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) and its advanced version [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels).

## Getting Started (will be updated)

### Installation
Recommended OS: Linux <br>
Install conda >= 4.11.0  (python >= 3.8) and then run following commands. <br>
(if your conda version is lower than 4.11.0, update conda first, with the command `conda update -n base -c defaults conda`)
```
conda create -n TRBA python=3.8 -y
conda activate TRBA

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


### Datasest
Run `data_for_TRBA.ipynb` in the COO-data folder to make train/val/test data.

### Training
1. Train TRBA model with SAR decoding
   ```
   CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name TRBA --exp_name TRBA_SARdecode --SARdecode
   ```

2. Train TRBA model with SAR decoding + half of batch filled with HardROI + Image height100 + 2D attn
   ```
   CUDA_VISIBLE_DEVICES= python3 train.py --model_name TRBA --exp_name TRBA_SARdecode_HardROIhalf_H100_2D \
   --SARdecode --imgH 100 --twoD --train_data ../COO-data/TRBA_data/ --select_data lmdb/train-hardROI/train
   ```

### Evaluation
1. Test TRBA model with SAR decoding
   ```
   CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name TRBA --eval_type benchmark --SARdecode \
   --saved_model saved_models/TRBA_SARdecode/best_score.pth
   ```
2. Test TRBA model with SAR decoding + half of batch filled with HardROI + Image height100 + 2D attn
   ```
   CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name TRBA --eval_type benchmark --SARdecode --imgH 100 --twoD \
   --saved_model saved_models/TRBA_SARdecode_HardROIhalf_H100_2D/best_score.pth
   ```


<h3 id="pretrained_models"> Run demo with pretrained model (will be updated) <a href="https://colab.research.google.com/github/ku21fan/STR-Fewer-Labels/blob/master/demo_in_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> </h3>

1. [Download pretrained model](https://www.dropbox.com/sh/lx61z7gq5yzkp02/AAAEyzVuVqVy_-EvtqTOJTaXa?dl=0) <br>
There are 2 different models of TRBA

    Model | Description
    -- | --
    TRBA_SARdecode.pth | TRBA + SARdecode
    TRBA_SARdecode_HardROIhalf_H100_2D.pth | TRBA + SARdecode + half of batch filled with HardROI + Image height100 + 2D attn

2. Add image files to test into `demo_image/`
3. Run demo.py with [TRBA_SARdecode.pth](https://www.dropbox.com/s/07dx4846sbnd8vv/TRBA_SARdecode.pth?dl=0)
   ```
   CUDA_VISIBLE_DEVICES=0 python3 demo.py --model_name TRBA --SARdecode \
   --image_folder demo_image/ --saved_model TRBA_SARdecode.pth
   ```
   or run demo.py with [TRBA_SARdecode_HardROIhalf_H100_2D.pth](https://www.dropbox.com/s/fwizki6halwsqty/TRBA_SARdecode_HardROIhalf_H100_2D.pth?dl=0)
   ```
   CUDA_VISIBLE_DEVICES=0 python3 demo.py --model_name TRBA --SARdecode --twoD --imgH 100 \
   --image_folder demo_image/ --saved_model TRBA_SARdecode_HardROIhalf_H100_2D.pth
   ```


#### prediction results
| demo images | [TRBA_SARdecode](https://www.dropbox.com/s/07dx4846sbnd8vv/TRBA_SARdecode.pth?dl=0) | [TRBA_SARdecode_HardROIhalf_H100_2D](https://www.dropbox.com/s/fwizki6halwsqty/TRBA_SARdecode_HardROIhalf_H100_2D.pth?dl=0) |
| ---         |     ---      |          --- |
| <img src="./demo_image/LoveHina_vol14/1-0.jpg">    |   ババッ | ババッ   |
| <img src="./demo_image/LoveHina_vol14/2-0.jpg">    |   カアッ・・・       | カアッ・・・        |
| <img src="./demo_image/LoveHina_vol14/2-2.jpg">    |   ぐぎぎぎ       | ぐぎぎぎっ       |
| <img src="./demo_image/LoveHina_vol14/2-4.jpg">    |   ドドド        | ドドド        |
| <img src="./demo_image/LoveHina_vol14/3-3.jpg">    |   ガラガラ     | ガガ   |
| <img src="./demo_image/LoveHina_vol14/5-6.jpg">    |   ドキえッ!     | ドキイッ!      |
| <img src="./demo_image/LoveHina_vol14/5-10.jpg">    |   ぎゅっ   | ぎゅっ・・・    |
| <img src="./demo_image/LoveHina_vol14/6-0.jpg">    |   こくっ・・・      | こくっ・・・       |
| <img src="./demo_image/LoveHina_vol14/6-3.jpg">    |   ドーン      | ドーン      |
| <img src="./demo_image/LoveHina_vol14/6-4.jpg">   |   カチン | カチン |


## Citation
Please consider citing this work in your publications if it helps your research.
```
@inproceedings{baek2021STRfewerlabels,
  title={What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels},
  author={Baek, Jeonghun and Matsui, Yusuke and Aizawa, Kiyoharu},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## License
For code: MIT.

