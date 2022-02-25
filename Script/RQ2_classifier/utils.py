import numpy as np
import os
from pathlib import Path
from keras.preprocessing import image
import pandas as pd
import cv2 
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
from skimage import io
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from gensim.models.keyedvectors import KeyedVectors
word_vect = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn import svm
import csv
import autosklearn.classification
from imblearn.over_sampling import SMOTE


def preprocess_img(dirs,labels_dict,df,extract_df):
    image_data = []
    labels = []    
    body_data = []
    tag_data = []
    title_data = []
    text_in_image_data = []
    for folder_dir in dirs:
        label = str(folder_dir).split("/")[-1]
        for img_path in folder_dir.glob("*.jpg"):
            Id = str(img_path).split('/')[-1].replace(".jpg","")
            body_data.append(str(df.loc[df['Q_id'] == int(Id)]['Body'].values))
            tag_data.append(str(df.loc[df['Q_id'] == int(Id)]['Tags'].values))
            title_data.append(str(df.loc[df['Q_id'] == int(Id)]['Title'].values))
            img = image.load_img(img_path, target_size=(32,32))
            img_array = image.img_to_array(img)
            image_data.append(img_array)
            labels.append(labels_dict[label])
            extract_text = extract_df.loc[extract_df['Q_id'] == int(Id)]['Q_image_text'].values
            if str(extract_text) == '[nan]':
                extract_text = ''
            else:
                extract_text = str(extract_text)
            text_in_image_data.append(extract_text)
    return image_data, labels, body_data, tag_data, title_data, text_in_image_data



def preprocess_text(X):
    documents = []
    stemmer = WordNetLemmatizer()
    lancaster = LancasterStemmer()

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)


        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    return documents



def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return np.zeros(200)
    
def word_vector(docs):
    doc_vec = []
    for doc in docs:
        vec = get_mean_vector(word_vect, doc.split(' '))

        doc_vec.append(vec)
    return doc_vec

    
def classWiseData(x, y, number_of_classes):
    data = {}
    
    for i in range(number_of_classes):
        data[i] = []
        
    for i in range(x.shape[0]):
        data[y[i]].append(x[i])
        
    for k in data.keys():
        data[k] = np.array(data[k])
        
    return data

def auto_sklearn_model(X, y):

    sss = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    print(sss)
    runner = 0

    for train_index, test_index in sss.split(X,y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_test_class = (np.unique(y_test))
        automl = autosklearn.classification.AutoSklearnClassifier(
            memory_limit=1024*32, time_left_for_this_task = 30*60, metric=autosklearn.metrics.f1_weighted
                )

        automl.fit(X_train.copy(), y_train.copy())
        automl.refit(X_train.copy(), y_train.copy())
        y_hat = automl.predict(X_test)
        predict_proba = automl.predict_proba(X_test)

        if len(np.unique(y)) == 2: 
            roc_auc = roc_auc_score(y_test,predict_proba[:,1])
        else: 
            roc_auc = roc_auc_score(y_test, predict_proba ,average='weighted',multi_class='ovr',labels=y_test_class)
        print("round:",runner,"Classification report", classification_report(y_test, y_hat))
        print("round:",runner,"ROC AUC", roc_auc)
        print("round:",runner,"Confusion matrix", confusion_matrix(y_test, y_hat))
        print("show_models",automl.show_models())
        print("sprint_statistics",automl.sprint_statistics())
        runner += 1
