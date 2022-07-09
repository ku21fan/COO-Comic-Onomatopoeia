
## Acknowledgements
This implementation has been based on the repository [mmf (1.0.0rc12)](https://github.com/facebookresearch/mmf), especially [M4C](https://mmf.sh/docs/projects/m4c/).

## Installation
Recommended OS: Linux <br>
Install conda >= 4.11.0  (python >= 3.8) and then run following commands. <br>
(if your conda version is lower than 4.11.0, update conda first, with the command `conda update -n base -c defaults conda`)
```
conda create -n M4C-COO python=3.8 -y
conda activate M4C-COO

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install --editable .
```

## Dataset
To train M4C-COO data, we need train/val/test gt annotation, fastText model, and FRCN features.
1. GT annotations of train/val/test are provided in [COO-data/M4C_feature folder](https://github.com/ku21fan/COO-Comic-Onomatopoeia/tree/main/COO-data/M4C_feature).
2. Download fastText model [model_300.bin](https://www.dropbox.com/s/c0nzo2lgcm2epo1/model_300.bin), then place it in [COO-data/M4C_feature folder](https://github.com/ku21fan/COO-Comic-Onomatopoeia/tree/main/COO-data/M4C_feature).
3. To obtain FRCN features for M4C-COO, go to [MTSv3 folder](https://github.com/ku21fan/COO-Comic-Onomatopoeia/tree/main/MTSv3) and then install MTSv3.
4. Download pretrained model [MTSv3.pth](https://www.dropbox.com/s/u0rnep52nshfukx/MTSv3.pth)
5. Run `sh extract_features_for_M4C.sh` to obtain FRCN features for train/val/test data.


## Training
**Caution:** our metric function [COO_PRHmean](https://github.com/ku21fan/COO-Comic-Onomatopoeia/blob/77be6e809db5ff40e161e2f012870670ddaf5473/M4C-COO/mmf/modules/metrics.py#L1393) does not support multi-GPU training. If you plan to multi-GPU training, modify this metric part. 
With one NVIDIA Tesla V100 GPU, training of M4C-COO takes about 2 hours.

Train M4C-COO model (default setting)
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=default.yaml run_type=train_val datasets=coo model=m4c_coo \
env.save_dir=./save/default model_config.m4c_coo.obj.remove_obj_bbox=True model_config.m4c_coo.obj.remove_obj_frcn=True
```

Train M4C-COO model with 11,640 vocabularies.
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=vocab11640.yaml run_type=train_val datasets=coo model=m4c_coo \
env.save_dir=./save/vocab11640 model_config.m4c_coo.obj.remove_obj_bbox=True model_config.m4c_coo.obj.remove_obj_frcn=True
```

## Evaluation
Evaluate M4C-COO model (default setting)
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=default.yaml run_type=test datasets=coo model=m4c_coo \
env.save_dir=./save_eval checkpoint.resume_file=./save/default/best.ckpt \
model_config.m4c_coo.obj.remove_obj_bbox=True model_config.m4c_coo.obj.remove_obj_frcn=True
```

Evaluate M4C-COO model with 11,640 vocabularies.
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=vocab11640.yaml run_type=test datasets=coo model=m4c_coo \
env.save_dir=./save_eval checkpoint.resume_file=./save/vocab11640/best.ckpt \
model_config.m4c_coo.obj.remove_obj_bbox=True model_config.m4c_coo.obj.remove_obj_frcn=True
```

Evaluate with pretrained model [M4C-COO_default.ckpt](https://www.dropbox.com/s/9gzglorqt0muu5h/M4C-COO_default.ckpt)
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=default.yaml run_type=test datasets=coo model=m4c_coo \
env.save_dir=./save_eval checkpoint.resume_file=M4C-COO_default.ckpt \
model_config.m4c_coo.obj.remove_obj_bbox=True model_config.m4c_coo.obj.remove_obj_frcn=True
```

Evaluate with pretrained model [M4C-COO_vocab11640.ckpt](https://www.dropbox.com/s/4vn0jgegu4p6qso/M4C-COO_vocab11640.ckpt)
```
CUDA_VISIBLE_DEVICES=0 mmf_run config=vocab11640.yaml run_type=test datasets=coo model=m4c_coo \
env.save_dir=./save_eval checkpoint.resume_file=M4C-COO_vocab11640.ckpt \
model_config.m4c_coo.obj.remove_obj_bbox=True model_config.m4c_coo.obj.remove_obj_frcn=True
```

## Documentation

Learn more about MMF [here](https://mmf.sh/docs).

## Citation

If you use MMF in your work or use any models published in MMF, please cite:

```bibtex
@misc{singh2020mmf,
  author =       {Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and
                 Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  title =        {MMF: A multimodal framework for vision and language research},
  howpublished = {\url{https://github.com/facebookresearch/mmf}},
  year =         {2020}
}
```

## License

MMF is licensed under BSD license available in [LICENSE](LICENSE) file
