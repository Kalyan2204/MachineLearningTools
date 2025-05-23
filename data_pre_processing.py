import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logger_config import logger
from utility import clock

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

def scaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)     
    return(X_train, X_test)

class DataPreProcessing:
    def __init__(self, ml_method, dataset):
        self.ml_method = ml_method
        self.dataset = pd.read_csv(dataset)

    def data_processing_clustering(self, col_no):
        if col_no == -1: return self.dataset.iloc[:, :].values
        else: return self.dataset.iloc[:, col_no].values
        # --- to do need to add exception if col no does not exist 

    def data_processing_regression(self):
        logger.debug(f"Inside data_processing_regression function of DataPreProcessing object")
        X = self.dataset.iloc[:, :-1].values
        y = self.dataset.iloc[:, -1].values      
        logger.debug(f"Created X and y data frames")    
           
        X = endcoding_catagorical_data(X)      
        logger.debug(f"Endcoding catagorical data usnig OneHotEncoder -> Done")
        X = manupulate_missing_data(X)  
        logger.debug(f"Manupulate missing data using mean strategy -> Done")
    
        #Splitting data set into training and test 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
        logger.debug(f"Splitting data set into training and test  -> Done")
        return(X_train, X_test, y_train, y_test)   
    
    def data_processing_classification(self):
        logger.debug(f"Inside data_processing_classification function of DataPreProcessing object")
        X_train, X_test, y_train, y_test = self.data_processing_regression()
        X_train, X_test =  scaling(X_train, X_test) 
        logger.debug(f"Scaling X_train and X_test  -> Done") 
        return(X_train, X_test, y_train, y_test)     
    
    
    def task(self, transactions, col_no, i):
        transactions.append([str(self.dataset.values[i,j]) for j in range(0, col_no)])
        return transactions
    
    def data_processing_association(self):
        logger.debug(f"Inside data_processing_association function of DataPreProcessing object")
        row_no = len(self.dataset)
        col_no = len(self.dataset.columns)
        logger.info(f"In input the .csv file has row_no = {row_no} and col_no = {col_no}")
        transactions = []   
        #row_no = 100
        #col_no = 10

        for i in range(0, row_no):
            transactions.append([str(self.dataset.values[i,j]) for j in range(0, col_no)])        
        return(transactions)       