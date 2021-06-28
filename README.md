# BVQA
BVQA is a deep learning algorithm to predict video quality assessment.

## Installation
```python
pip install -r requirements.txt
```

## Usage
To extract frames from videos:
```python
python video2frames.py --video_dir "path_to_videos_directory"  --nbr_frame "number_of_frames_per_videos_to_be_extracted"
```


To compute features:

```python
python extract_features.py --frame_dir 'path to frames directory' --csv_file 'path to meta-data csv file'  --num_patch 'number of patches (224*224) to be extracted from frames --overlapping 'overlapping between patches'
```
Please note that the meta-data should be a csv file with two columns: video name and MOS also we use ResNet50 for features extractions.


To evaluate model:


You can download konvid test features [here](https://drive.google.com/drive/folders/1hDXz0TIpmayBWb1afuclTg1Ca8PR_o4R?usp=sharing).

```python
python evaluate_model.py  ----input_final_model 'path to final model' --sp_model_weights 'path sp model'  --x_test 'path to npy file' --n 'number of frames per video'
```

<p align="center">
  <img width="640" height="480" src="https://github.com/Tlili-ahmed/BVQA/blob/master/figures/mos_sroc%20%3D0.8463255562480931.png">
</p>

## Train

To train model:

First we need to train a bi-lstm to do spatial pooling between patches
```python
python train_spatial_bilstm.py --x_train 'path to train npy file'
```
Then we need to use this wieghts to train the final model.

```python
python train.py --x_train 'path to train npy file' --n 'number of frames per video' --spatial_weights 'path to spatial bi-lstm model'
```

