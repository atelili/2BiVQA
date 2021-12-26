# 2BiVQA

2BiVQA: Double Bi-LSTM based Video Quality Assessment of UGC Videos

<p align="center">
  <img src="https://github.com/Tlili-ahmed/2BiVQA/blob/master/figures/2BiVQA_f.drawio.png">
</p>


This repository contains the code for our paper on [2BiVQA: Double Bi-LSTM based Video Quality Assessment of UGC Videos](#2bivqa:_double_bi-lstm_based_video_quality_assessment_of_ugc_videos). 
If you use any of our code, please cite:
```
@article{Telili2021,
  title = {2BiVQA: Double Bi-LSTM based Video Quality Assessment of UGC Videos},
  author = {Ahmed Telili and Sid Ahmed Fezza and Wassim Hamidouche },
  booktitle={IEEE TRANSACTIONS ON IMAGE PROCESSING},
  year = {2022}
}
```



  * [Requirements](#requirements)
  * [Features extraction](#features-extraction)
  * [Model Training](#model-training)
  * [Test](#test)
      * [On KonViD-1K](#a-on-Konvid-1k)
      * [LIVE_VQC](#b-on-live_vqc)
  * [Demo](#demo)
  * [Evaluate](#evaluate)
  * [Performance Benchmark](#performance-benchmark)
  * [Reference](#reference)
    
<!-- /code_chunk_output -->



## Requirements
```python
pip install -r requirements.txt
```

## Features extraction

Please note that the meta-data should be a csv file with two columns: video name and MOS.


```
python3 extract_features.py [-h] [-v 'path to videos directory']
                                   [-f 'path to meta-data csv file']
                                   [-o 'overlapping between patches']
```

ResNet50 is used for features extractions.





## Model Training (optional):

This step can be skipped, and directly test the model in the next section with pre-trained models. 

To train your own model:



```python
python End2End_train.py [-h] [-nf number of frames to be extracted] [-b batch_size]
                     
```

## Test: 


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



## Demo:

To predict quality for your own dataset using pre-trained model:

```python
python demo.py  [-h] [-nf number of frames to be extracted] [-m path to pretrained model] [-f path to videos dir]
```

## Evaluate:

To evaluate model:

Please note that your csv file should have two columns: 'Mos' and 'Predicted'.

```python
python evaluate.py  --mos_pred konvid.csv
```



## Performance Benchmark:


###### KonViD-1K [1]:

  
|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.6567       | 0.6576          | 0.4761   | 0.4813   |  
| NIQE      | 0.5417   | 0.5530       | 0.3790     | 0.5336   |
| ILNIQE      | 0.5264  | 0.5400 | 0.3692 | 0.5406 |
| VIIDEO      |0.2988 | 0.3002 | 0.2036 | 0.6101 |
| GM-LOG      | 0.6578 | 0.6636 | 0.4770 | 0.4818 |
| HIGRADE      |0.7206 | 0.7269 | 0.5319 | 0.4391|
| FRIQUEE     | 0.7472 | 0.7482 | 0.5509 | 0.4252 |
| CORNIA     |0.7169 | 0.7135 | 0.5231 | 0.4486 |
| HOSA    | 0.7654 | 0.7664 | 0.5690 | 0.4142 |
| V-BLIINDS    | 0.7101          | 0.7037           | 0.5188  | 0.4595 |
| TLVQM      | 0.7729   | 0.7688      | 0.5770 | 0.4102 |
| ResNet-50      | 0.8018  | 0.8104      | 0.6100 | 0.3749 |
| VGG-19      | 0.7741   | 0.7845      | 0.5841 | 0.3958 |
| KonCept512       | 0.7349   | 0.7489      | 0.5425 | 0.4260 |
| VIDEVAL       | 0.7832   | 0.7803      | 0.5845 | 0.4026 |
| RAPIQUE       | 0.8072  | 0.8175  | 0.6189 | 0.3623  |
| 2BiVQA   | 0.8463  | 0.8404     | 0.6529 | 0.3620 |

###### LIVE VQC [2]:

|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.5925           | 0.6380          | 0.4162 | 13.100     |  
| NIQE      |0.5957 | 0.6286 | 0.4252 | 13.110   |
| ILNIQE      | 0.5037 | 0.5437 | 0.3555 | 14.148 |
| VIIDEO      |0.0332 | 0.0231 | 0.2146 | 16.654 |
| GM-LOG      | 0.5881 | 0.6212 | 0.4180 | 13.223 |
| HIGRADE      |0.6103 | 0.6332 | 0.4391 | 13.027|
| FRIQUEE     | 0.6579 | 0.7000 | 0.4770 | 12.198 |
| CORNIA     |0.6719 | 0.7183 | 0.4849 | 11.832|
| HOSA    | 0.6873 | 0.7414 | 0.5033 | 11.353 |
| V-BLIINDS    | 0.6939           | 0.7178           | 0.5078 | 11.765 |
| TLVQM      | 0.7988    | 0.8025       | 0.6080 | 10.145 |
| ResNet-50      | 0.6636   | 0.7205       | 0.4786 | 11.591 |
| VGG-19      | 0.6568    | 0.7160       | 0.4722 | 11.783 |
| KonCept512       | 0.6645    | 0.7278       | 0.4793 | 11.626 |
| VIDEVAL       | 0.7522    | 0.7514       | 0.5639 | 11.100|
| RAPIQUE       | 0.7415 | 0.7659 | 0.5576 | 10.6653 |
| 2BiVQA   | 0.7614  | 0.8325     | 0.6212 | 9.9799 |



###### YouTube-UGC [3]:

|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.3820 | 0.3952 | 0.2635 | 0.5919     |  
| NIQE      |0.2379 | 0.2776 | 0.1600 | 0.6174|
| ILNIQE      |0.2918 | 0.3302  | 0.1980 | 0.6052|
| VIIDEO      |0.0580 | 0.1534 | 0.0389 | 0.6359 |
| GM-LOG      | 0.3678 | 0.3920  | 0.2517 |  0.5896 |
| HIGRADE      |0.7376 | 0.7216 | 0.5478 | 0.4471|
| FRIQUEE     | 0.7652 | 0.7571 | 0.5688 | 0.4169 |
| CORNIA     |0.5972  | 0.6057  | 0.4211 | 0.5136 |
| HOSA    | 0.6025 |  0.6047 |  0.4257  | 0.5132|
| V-BLIINDS    | 0.5590 |  0.5551 |  0.3899  | 0.5356 |
| TLVQM      | 0.6693  | 0.6590 |  0.4816 |  0.4849 |
| ResNet-50      | 0.7183 |  0.7097 |  0.5229 |  0.4538 |
| VGG-19      | 0.7025 |  0.6997 |  0.5091 |  0.4562 |
| KonCept512       | 0.5872 |  0.5940 |  0.4101 |  0.5135 |
| VIDEVAL       | 0.7787 |  0.7733 |  0.5830 |  0.4049|
| RAPIQUE       | 0.7610 |  0.7620 |  0.5610 |  0.4060  |
| 2BiVQA   | 0.7716 |  0.7904 |  0.5812 |  0.4047  |



###### All-Combined:

|    Methods   |SROCC            | PLCC            | KROCC        | RMSE |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.5695  |  0.5861 |  0.4030 |  0.5617     |  
| NIQE      |0.4622 |  0.4773 |  0.322 |  0.6112   |
| ILNIQE      | 0.4592 |  0.4741 |  0.3213  | 0.6119 |
| VIIDEO      |0.1039 |  0.1621 |  0.0688 |  0.6804 |
| GM-LOG      | 0.5650 |  0.5942 |  0.3995 |  0.5588 |
| HIGRADE      |0.7398 |  0.7368 |  0.5471 |  0.4674|
| FRIQUEE     |0.7568 |  0.7550  | 0.5651 |  0.4549 |
| CORNIA     |0.6764 |  0.6974 |  0.4846 |  0.4946|
| HOSA    | 0.6957  | 0.7082 |  0.5038 |  0.4893|
| V-BLIINDS    | 0.6545 |  0.6599  | 0.4739 |  0.5200 |
| TLVQM      | 0.7271 |  0.7342  | 0.5347  | 0.4705 |
| ResNet-50      | 0.7557 |  0.7747 |  0.5613 |  0.4385|
| VGG-19      |0.7321  | 0.7482 |  0.5399  | 0.4610|
| KonCept512       | 0.6608 |  0.6763 |  0.4759 |  0.5091 |
| VIDEVAL       | 0.7960 |  0.7939 |  0.6032 |  0.4268|
| RAPIQUE       | 0.8086 |  0.8186 |  0.6148 |  0.4076 |
| 2BiVQA   | 0.8003 |  0.7941 |  0.6088 |  0.4218 |




## Reference


```
[1] V. Hosu, F. Hahn, M. Jenadeleh, H. Lin, H. Men, T. Szirányi, S. Li,and D. Saupe, “The konstanz natural video database (konvid-1k),” in2017 Ninth international conference on quality of multimedia experience(QoMEX).  IEEE, 2017, pp. 1–6.

[2] Z. Sinno and A. C. Bovik, “Large-scale study of perceptual videoquality,”IEEE Transactions on Image Processing, vol. 28, no. 2, pp.612–627, 2018.

[3] Y. Wang, S. Inguva, and B. Adsumilli, “Youtube ugc dataset for videocompression research,” in2019 IEEE 21st International Workshop onMultimedia Signal Processing (MMSP).  IEEE, 2019, pp. 1–5.
```




