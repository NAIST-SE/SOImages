import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from keras.preprocessing import image
import pandas as pd
import cv2 
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
from skimage import io
from utils import *

df = pd.read_excel('../Dataset/RQ2/RQ2_data.xlsx')
extract_df = pd.read_excel('../Dataset/RQ2/Image_text_extract_paddleocr_RQ2.xlsx')
p = Path("../Dataset/RQ2/Image_RQ2_purpose")
dirs = p.glob("*")
labels_dict = {'context':0, 'desired output':1, 'undesired output':1}
image_data, labels, body_data, tag_data, title_data, text_in_image_data = preprocess_img(dirs,labels_dict,df,extract_df)
image_data_array = np.array(image_data, dtype='float32')/255.0
labels = np.array(labels)
M = image_data_array.shape[0]
image_data_array = image_data_array.reshape(M,-1)

body_data_after_preprocess = preprocess_text(body_data)
tag_data_after_preprocess = preprocess_text(tag_data)
title_data_after_preprocess = preprocess_text(title_data)
text_in_image_data_after_preprocess = preprocess_text(text_in_image_data)

body_data_after_wordvector = word_vector(body_data_after_preprocess)
tag_data_after_wordvector = word_vector(tag_data_after_preprocess)
title_data_after_wordvector = word_vector(title_data_after_preprocess)
text_in_image_data_after_wordvector = word_vector(text_in_image_data_after_preprocess)

from sklearn.preprocessing import StandardScaler
std_slc = StandardScaler()

image_data_array = std_slc.fit_transform(image_data_array)
body_data_after_wordvector = std_slc.fit_transform(body_data_after_wordvector)
tag_data_after_wordvector = std_slc.fit_transform(tag_data_after_wordvector)
title_data_after_wordvector = std_slc.fit_transform(title_data_after_wordvector)
text_in_image_data_after_wordvector = std_slc.fit_transform(text_in_image_data_after_wordvector)

total_data = np.concatenate((image_data_array, body_data_after_wordvector),axis=1)
total_data = np.concatenate((total_data, tag_data_after_wordvector),axis=1)
total_data = np.concatenate((total_data, title_data_after_wordvector),axis=1)
total_data = np.concatenate((total_data, text_in_image_data_after_wordvector),axis=1)

number_of_classes = len(np.unique(labels))
data = classWiseData(total_data, labels, number_of_classes)

X, y = total_data, labels
auto_sklearn_model(X,y)
