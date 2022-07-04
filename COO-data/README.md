데이터셋 준비할때 ipynb 들을 다 그냥 .py 버전으로도 만들어두자! 혹은 .py 만 공개하자! 

`pip install Flask==2.0.2 Shapely==1.8.0 manga109api==0.3.1 pillow` <br>

`preprocessed_data` folder contains preprocessed annotation data of COO. <br>
`books*.txt` contains the book titles of Manga109 data. <br>
`Onomatopoeia_train_char_set.txt` contains the character set of train data. <br>

To prepare the data for ABCNetv2, MTSv3, and TRBA, use `data_for_ABCNetv2.ipynb`, `data_for_MTSv3.ipynb`, and `data_for_TRBA.ipynb`, respectivly. <br>

For data analysis, use `data_analysis.ipynb` <br>
For data visualization, run `python vis.py`, then the web page made by Flask is running on 0.0.0.0:6006 (or localhost: 6006)  <br>
You can visualize multiple pages by using the format "{start_number}-{end_number}" in the `Page index` field. <br>
For exmaple, 
- Page index: 2 → Visualize page 2. 
- Page index: 2-5 → Visualize page 2, 3, 4, 5.
