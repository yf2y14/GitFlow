=====================

Notes on DLC
@author: joseph-2023-10-01

=======================

Installation Deeplabcut

> conda create -n dlc-live python=3.7 tensorflow-gpu==1.13.1 # if using GPU
> conda create -n dlc-live python=3.7 tensorflow==1.13.1 # if not using GPU
> conda activate dlc-live
> pip install deeplabcut

E:\DeepLabCut
├─csv_dataset
├─myvideos
├─topic-joseph-2023-10-01
    ├─dlc-models
    ├─labeled-data
    ├─training-datasets
    ├─videos
	└─config.yaml
	
	
step 1: create_dlc_project.py  #you don't need to do this if the project has been already existed
step 2: python video2csv.py --gen_video --gen_train_data


		 python video2csv.py --video_file "E:\DeepLabCut\upload_video\mydog456.mp4"

		csv_dataset\xxx_normalized.csv
					xxx_normalized.csv

step 3: inference.py  # do prediction from csv file  
