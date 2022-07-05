# COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts
We provide COmic Onomatopoeia dataset (COO) and the source codes we used in our paper. <br>
We hope that this work will facilitate future work on recognizing various types of texts.


<br>

## Dataset: COmic Onomatopoeia (COO)
To facilitate future work, we provide the preprocessed annotation of the COO. <br>

### Prerequisites: Download Manga109 images
Following [the license of Manga109](http://www.manga109.org/en/download.html),  the redistribution of the image files of Manga109 data is not permitted. <br> 
Thus, you should download the image files of Manga109 data via [Manga109 webpage](http://www.manga109.org/en/download.html). <br> 

After downloading, unzip `Manga109.zip` and then move `images` folder of Manga109 data into `COO-data` folder. <br>
= We need `images` folder in `COO-data` folder (i.e. `COO-data/images`) for further data preparation with image files. 

<br>

## Codes
For text detection, we used ABCNetv2 and MTSv3. <br>
For text recognition, we used TRBA. <br> 
For link prediction, we used M4C-COO (a variant of M4C) <br>


<br>

## Citation
If you find this work useful for your research, please cite:
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
For the dataset, annotation data of COO is licensed under the Creative Commons Attribution 4.0 International (CC-BY-4.0) license, and image data from Manga109 is described [here](http://www.manga109.org/en/download.html). <br>

For the codes made by us: MIT. <br>
After examining the licenses of original source codes of each method we used in our work, we found that the redistribution of source codes is permitted. <br>
Thus, to facilitate future work, we provide the source codes in this repository. <br>
Please let us know if there is a license issue with code redistribution. If so, we will remove the code and provide the instructions to reproduce our work.
