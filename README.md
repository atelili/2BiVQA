# 2BiVQA

2BiVQA is a no-reference deep learning based video quality assessment metric.


<p align="center">
  <img src="https://github.com/Tlili-ahmed/2BiVQA/blob/master/figures/2BiVQA_f.drawio.png">
</p>


This repository contains the code for our paper on [2BiVQA: Double Bi-LSTM based Video Quality Assessment of UGC Videos]. 
If you use any of our code, please cite:
```
@article{Telili2021,
  title = {2BiVQA: Double Bi-LSTM based Video Quality Assessment of UGC Videos},
  author = {Ahmed Telili and Sid Ahmed Fezza and Wassim Hamidouche },
  booktitle={IEEE TRANSACTIONS ON IMAGE PROCESSING},
  year = {2022}
}
```


* [2BiVQA: Double Bi-LSTM based Video Quality Assessment of UGC Videos]
  * [Requirements](#requirements)
  * [Features extraction](#features-extraction)
  * [Model Training](#model-training)
  * [Test](#test)
      * [On KonViD-1K](#on-KonViD-1K)
      * [LIVE_VQC](#live-vqc)
  * [Demo](#demo)
  * [Evaluate](#evaluate)
  * [Performance Benchmark](#performance-benchmark)
  * [Reference](#reference)
    
<!-- /code_chunk_output -->



## 1-Requirements
```python
pip install -r requirements.txt
```

## 2-Features extraction

Please note that the meta-data should be a csv file with two columns: video name and MOS.


```
python3 extract_features.py [-h] [-v 'path to videos directory']
                                   [-f 'path to meta-data csv file']
                                   [-np 'number of patches (224*224) to be extracted from frames']
                                   [-o 'overlapping between patches']
```

ResNet50 is used for features extractions.





## 3-Model Training (optional):

This step can be skipped, and directly test the model in the next section with pre-trained models. 

To train your own model:



```python
python train.py --x_train 'path to train npy file' --n 'number of frames per video' --spatial_weights 'path to spatial bi-lstm model'
```

## 4-Test: 


To test model:

#### a-On KonViD-1K:

```python
python test_model.py --dataset konvid 
```

|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| 2BiVQA     | 0.8463         | 0.8404          | 0.6529   | 0.3620  |  


<p align="center">
  <img src="https://github.com/Tlili-ahmed/BVQA/raw/master/figures/mos_KonViD-1K.png">
</p>

#### b-On LIVE_VQC: 

```python
python test_model.py --dataset live  
```

|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| 2BiVQA   | 0.7614  | 0.8325     | 0.6212 | 9.9799 |


<p align="center">
  <img src="https://github.com/Tlili-ahmed/BVQA/raw/master/figures/mos_LIVE.png">
</p>



## 5-Demo:

To predict quality for your own dataset using pre-trained model:

```python
python demo.py  --video_dir 'path to your dataset folder'
```

## 6-Evaluate:

To evaluate model:

Please note that your csv file should have two columns: 'Mos' and 'Predicted'.

```python
python evaluate.py  --mos_pred konvid.csv
```



## 7-Performance Benchmark:


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
| 2BiVQA   | 0.8463  | 0.8404     | 0.6529 | 0.3620 |

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
| 2BiVQA   | 0.7614  | 0.8325     | 0.6212 | 9,9799 |








