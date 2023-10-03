# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:03:06 2023

@author: joseph@艾鍗學院

I try to make it to more organized and clear.

https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/standardDeepLabCut_UserGuide.md


pip install deeplabcut[gui]


"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from video2csv import *
from normalized import csv_std


graph1=tf.Graph()

Model=r"C:\JoePython\AidySchool\DeepLabCut\tok.h5"
#The project_path need to modify manually to fit your situation
project_path=r'C:\JoePython\AidySchool\DeepLabCut\topic-joseph-2023-10-01'  
config_path=os.path.join(project_path,'config.yaml')


print('project_path:',project_path)
print('config_path:',config_path)


def predict(mymodel,x):
 
    t_df=x
    print(t_df.head())
    
    
    with graph1.as_default():
        lstm_model = tf.keras.models.load_model(mymodel)
        t_df = t_df.iloc[:,:40].values.astype(float)
        t_df=t_df[np.newaxis,:,:]
        m_ans=lstm_model.predict(t_df)
        idx=np.argmax(m_ans)
        

    return idx,m_ans[0][idx]



def jf_inf(video_file):
    analyze_video([video_file],True)
    feature_csvfile=gen_single_normalized_csv(video_file)
    std_csv = pd.read_csv(feature_csvfile,header=[1],index_col=0)  
    std_data=std_csv
    cls,conf=predict(Model,std_data)
    return int(cls),float(conf)



if __name__ == "__main__":

    video_file=r"C:\Users\Joe\Desktop\deepzoo\newwalk(5)d.mp4"
   
    
    analyze_video([video_file],True)
    print('video_file:',[video_file])
    feature_csvfile=gen_single_normalized_csv(video_file)
    print(feature_csvfile)

    std_csv = pd.read_csv(feature_csvfile,header=[1],index_col=0)
  
    std_data=std_csv
    print(std_data)
    cls,conf=predict(Model,std_data)
    print('Predict {} (conf:{:.2f})'.format(cls,conf))
    
