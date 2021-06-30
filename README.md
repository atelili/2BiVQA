# 2BVQA

2BVQA is a no-reference deep learning based algorithm to predict visual video quality scores.

## 1-Installation
```python
pip install -r requirements.txt
```

## 2-Features extraction

Please note that the meta-data should be a csv file with two columns: video name and MOS.
### a- Extract frames from videos:

```python
python video2frames.py --video_dir "path_to_videos_directory"  --nbr_frame "number_of_frames_per_videos_to_be_extracted"
```

### b- Features extraction:

```python
python extract_features.py --frame_dir 'path to frames directory' --csv_file 'path to meta-data csv file'  --num_patch 'number of patches (224*224) to be extracted from frames --overlapping 'overlapping between patches'
```

ResNet50 is used for features extractions.





## 3-Train (optional):

This step can be skipped, and directly test the model in the next section with pre-trained models. 

To train your own model:

You can train the first bi-lstm to do spatial pooling between patches, or uses our spatial bi-lstm which was trained on KONIQ 10-k dataset. 

```python
python train_spatial_bilstm.py --x_train 'path to train npy file'
```
Then you need to train the second Bi-lstm to do temporal pooling between frames:

```python
python train.py --x_train 'path to train npy file' --n 'number of frames per video' --spatial_weights 'path to spatial bi-lstm model'
```

## 4-Test: 


To test model:


```python
python evaluate_model.py  --input_final_model 'final model' --sp_model_weights 'path sp model'  --x_test 'path to npy file' --n 'number of frames per video'
```




For testing our model on KonViD-1k:
You can download KonViD-1K test features [here](https://drive.google.com/drive/folders/1hDXz0TIpmayBWb1afuclTg1Ca8PR_o4R?usp=sharing).

Or :

```python
import wget
x_test_konvid.npy = wget.download('http://openvvc.insa-rennes.fr/models/KonViD-1k/y_test_konvid.npy')
y_test_konvid.npy = wget.download('http://openvvc.insa-rennes.fr/models/KonViD-1k/y_test_konvid.npy')
```

Then:


```python
python evaluate_model.py  --input_final_model konvid_model1.h5 --sp_model_weights res-bi-sp_koniq.h5  --x_test ./features/x_test_konvid.npy --n 30
```


|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| ResNet50-Bi-LSTM     | 0.8463         | 0.8404          | 0.6529   | 0.3620  |  


<p align="center">
  <img width="640" height="480" src="https://github.com/Tlili-ahmed/BVQA/blob/master/figures/mos_sroc%20%3D0.8463255562480931.png">
</p>


## 5-Performance Benchmark:


###### KonViD-1K:

  
|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.6567          | 0.6576          | 0.4761   | 0.4813   |  
| V-BLIINDS    | 0.7101          | 0.7037           | 0.5188  | 0.4595 |
| TLVQM      | 0.7729   | 0.7688      | 0.5770 | 0.4102 |
| VIDEVAL       | 0.7832   | 0.7803      | 0.5845 | 0.4026 |
| ResNet-50      | 0.8018  | 0.8104      | - | - |
| VGG-19      | 0.7741   | 0.7845      | - | - |
| KonCept512       | 0.7349   | 0.7489      | - | - |
| VIDEVAL+KonCept512         | 0.8149   | 0.8169      | - | - |
| RAPIQUE       | 0.8031   | 0.8175  | - | 0.3623  |
| NR-QM UGC    | 0.8134  | 0.8143 | 0.6201 | 0.3695 |
| ResNet-50 + Bi-LSTM   | 0.8463  | 0.8404     | 0.6529 | 0.3620 |

###### LIVE VQC:

|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.5925           | 0.6380          | -  | -   |  
| V-BLIINDS    | 0.6939           | 0.7178           | -  | - |
| TLVQM      | 0.7988    | 0.8025       | - | - |
| VIDEVAL       | 0.7522    | 0.7514       | - | - |
| ResNet-50      | 0.6636   | 0.7205       | - | - |
| VGG-19      | 0.6568    | 0.7160       | - | - |
| KonCept512       | 0.6645    | 0.7278       | - | - |
| VIDEVAL+KonCept512         | 0.7849      | 0.8010       | - | - |
| RAPIQUE       | 0.7548   | 0.7863  | - | 10.518  |
| ResNet-50 + Bi-LSTM   | 0.7614  | 0.8325     | 0.6212 | 9,9799 |



###### UGC-VQA:

|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.8346          | 0.8886          | 0.4173   | 0.5587   |  
| NIQE    | 0.5929         | 0.5977           |0.4173 | 0.9764 |
| VSFA      | 0.8925   | 0.9587      | 0.7280 | 0.3463 |
| RAPIQUE       | 0.8647   | 0.9247  | 0.6898 | 0.4742  |
| NR-QM UGC    | 0.9352 | 0.9826 | 0.7937 | 0.2260 |
| ResNet-50 + Bi-LSTM   | 0.9886  | 0.9987     | 0.9276 | 0.0630 |





