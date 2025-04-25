import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logger_config import logger

def manupulate_missing_data(X):
    col_no = len(X[0])
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 0:col_no])
    X[:, 0:col_no] = imputer.transform(X[:, 0:col_no])
    return(X)

def endcoding_catagorical_data(X):
    for index in range(len(X[0])):
        if type(X[0][index]) is str:
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [index])], remainder='passthrough')
            X = np.array(ct.fit_transform(X))
    return(X)

def data_preprocessing(inout_data):           
    logger.debug(f"Inside data_preprocessing function")
    dataset = pd.read_csv(inout_data)        
    logger.debug(f"Created data frame")
    

    #----------- to do --------#
    #----- Exception for the missing input data file -----------# 

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values      
    logger.debug(f"Created X and y data frames")
    
    #endcoding catagorical data usnig OneHotEncoder
    X = endcoding_catagorical_data(X)      
    logger.debug(f"Endcoding catagorical data usnig OneHotEncoder -> Done")
    #manupulate missing data using mean strategy
    X = manupulate_missing_data(X)  
    logger.debug(f"Manupulate missing data using mean strategy -> Done")
    
    #Splitting data set into training and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
    logger.debug(f"Splitting data set into training and test  -> Done")       
    return(X_train, X_test, y_train, y_test)

class DataPreProcessing:
    def __init__(self, ml_method, dataset):
        self.ml_method = ml_method
        self.dataset = pd.read_csv(dataset)

    def data_processing_clustering(self, col_no):
        if col_no == -1: return self.dataset.iloc[:, :].values
        else: return self.dataset.iloc[:, col_no].values
        # --- to do need to add exception if col no does not exist 