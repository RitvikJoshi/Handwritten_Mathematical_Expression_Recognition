"""
Training Random Forest classifier for different Classification task
Run
Training.py <Feature.csv> <sym/seg/par indicator>

Input:
Feature.csv - Feature files to train random forest 
sym - To train for symbol classifier[Works only with symbol feature file (SymbolFeatures.csv)] 
seg - To train for segmentor classifier[Works only with segmentor feature file (SegmentFeatures.csv)]
par - To train for Parser classifier[Works only with Parser feature file (ParserFeatures.csv)]	
Output:
RandomForest.pickle - Returns trained random forest pickle file

Author
Ritvik Joshi
Rahul Dashora
"""

from csv import reader
import numpy as np
import pickle

import sys
from sklearn.ensemble import RandomForestClassifier
import pickle

from sklearn.externals import joblib


def read_data(filename):
    input_file = open(filename,'r')
    data = list(reader(input_file))
    UID=[]
    
    mat_data = np.zeros((len(data),len(data[0])-2))
    classes = np.zeros((len(data)))
    print(mat_data.shape)
    for i in range(len(data)):
        UID.append(data[i][0])
        mat_data[i] = np.asarray(data[i][1:len(data[i])-1],dtype='float')
        classes[i]=np.asarray(data[i][len(data[i])-1],dtype='float')
    print(mat_data)
    print(classes)
    return mat_data,classes


def Parserrandomforest(filepath):
    #filename='C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\SegmentorFeature.csv'
    print(filepath)
    filename =filepath
    data_array,result_array=read_data(filename)
    rdtree = RandomForestClassifier(n_estimators=100)
    rdtree =rdtree.fit(data_array,result_array)
    pickle.dump(rdtree, open('ParserClassifier_v64.p', 'wb'))



def Segmentorrandomforest(filepath):
    #filename='C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\SegmentorFeature.csv'
    print(filepath)
    filename =filepath
    data_array,result_array=read_data(filename)
    rdtree = RandomForestClassifier(n_estimators=100)
    rdtree =rdtree.fit(data_array,result_array)
    pickle.dump(rdtree, open('SegmentorClassifier_v64.p', 'wb'))



def symbolTrainRandomforest(filepath):
    #filename='SymbolFeature.csv'
    filename = filepath
    print(filepath)
    data_array,result_array=readSymbol(filename)

    rf = RandomForestClassifier(n_estimators=50,n_jobs=4)
    rf=rf.fit(data_array,result_array)
    joblib.dump(rf, open('SymbolClassifier_v64.p', 'wb'))


def readSymbol(filename):
    input_file = open(filename, 'r')
    data = list(reader(input_file))
    UID = []

    mat_data = np.zeros((len(data), len(data[0]) - 2))
    classes = []#np.zeros((len(data)))
    print(mat_data.shape)
    for i in range(len(data)):
        #UID.append(data[i][0])
        #print(data[i])
        mat_data[i] = np.asarray(data[i][1:len(data[i]) - 1], dtype='float32')
        classes.append(data[i][-1])
        #np.asarray(data[i][len(data[i]) -1], dtype='float')

    print(mat_data)
    print(classes)

    return mat_data, classes

#symbolTrainRandomforest()


def main():
    if(len(sys.argv)<3):
        print("Please use the following way to run the program")
        print("Training.py <feature.csv> <sym/seg/par indicator>")
        print("feature.csv - Feature file for training random forest")
        print("sym - indicator for training symbol classifier")
        print("seg - indicator for training segmentor classifier")
        print("par - indicator for training parser classifier")

    else:
        filename = sys.argv[1]
        ind = sys.argv[2]

        if ind == 'sym':
            symbolTrainRandomforest(filename)
        elif ind == 'seg':
            Segmentorrandomforest(filename)
        elif ind == 'par':
            Parserrandomforest(filename)
        else:
            print("INVALID indicator for classifier")
            print("Please use the following way to run the program")
            print("Training.py <feature.csv> <sym/seg/par indicator>")
            print("feature.csv - Feature file for training random forest")
            print("sym - indicator for training symbol classifier")
            print("seg - indicator for training segmentor classifier")
            print("par - indicator for training parser classifier")


main()