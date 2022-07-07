# COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts
We provide COmic Onomatopoeia dataset (COO) and the source codes we used in our paper. <br>
We hope that this work will facilitate future work on recognizing various types of texts.

<img src="teaser.png">


<br>

## Dataset: Comic Onomatopoeia (COO)
We provide the [annotations of the COO](https://github.com/ku21fan/COO-Comic-Onomatopoeia/tree/main/COO-data/annotations). <br>

#### Prerequisites: Download Manga109 images
Following [the license of Manga109](http://www.manga109.org/en/download.html),  the redistribution of the image files of Manga109 data is not permitted. <br> 
Thus, you should download the image files of Manga109 data via [Manga109 webpage](http://www.manga109.org/en/download.html). <br> 

After downloading, unzip `Manga109.zip` and then move `images` folder of Manga109 data into `COO-data` folder. <br>
= We need `images` folder in `COO-data` folder (i.e. `COO-data/images`) for further data preparation with image files. 


#### Preprocessing for each model
See the section `dataset` in each model folder.


<br>

## Codes
For text detection, we used ABCNetv2 and MTSv3. <br>
For text recognition, we used TRBA. <br> 
For link prediction, we used M4C-COO (a variant of M4C) <br>

<br>

## Leaderboard
We will list the results of SOTA methods that provide the official code.

### Text detection

<br>

### Text recognition

<br>

### Link prediction


<br>


## Citation
When using annotations of comic onomatopoeia dataset (COO) or if you find this work useful for your research, please cite our paper.
```
@inproceedings{baek2022COO,
  title={COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts},
  author={Baek, Jeonghun and Matsui, Yusuke and Aizawa, Kiyoharu},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Contact
Feel free to contact us if there is any question: Jeonghun Baek ku21fang@gmail.com

## License
For the dataset, annotation data of COO is licensed under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). <br>
The license of image data of Manga109 is described [here](http://www.manga109.org/en/download.html). <br>

For the codes made by us: MIT. <br>
After examining the licenses of original source codes of each method we used in our work, we found that the redistribution of source codes is permitted. <br>
Thus, to facilitate future work, we provide the source codes in this repository. <br>
Please let us know if there is a license issue with code redistribution. If so, we will remove the code and provide the instructions to reproduce our work.
