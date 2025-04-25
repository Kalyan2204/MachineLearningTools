import numpy as np
from data_pre_processing import data_preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from utility import clock
from logger_config import logger
from abc import ABC, abstractmethod
from data_pre_processing import DataPreProcessing

clasification_algo_disc = {}

def clasification(clasification_algo):
    logger.debug(f"Inside @clasification decorators -> input clasification_algo = {clasification_algo.__name__}")
    clasification_algo_disc[clasification_algo] = None
    def inner(*args, **kwargs):         
        logger.debug(f"Inside inner method of @clasification decorators -> input args  =  {args}, kwargs = {kwargs}") 
        return clasification_algo(*args, **kwargs)
    return inner

class BuildClasificationAlgo():  
    def __init__(self, args):
        logger.debug(f"Constructor BuildClasificationAlgo with args = {args}")
        self.args = args
        self.call_clasification_method()

    def generate_result(self, clasification_algo = None):
        logger.debug(f"Inside generate_result")
        if clasification_algo is None:
            logger.debug(f"Inside if block of generate_result method")
            for clasification_algo in clasification_algo_disc.keys(): 
                score = clasification_algo_disc[clasification_algo] [0]
                logger.debug(f"[Clasification Algo Name] : {clasification_algo.__name__} \t ->  Accuracy Score = {score:0.4f}")                                  
            best_clasification_algo = max(clasification_algo_disc, key=lambda k: clasification_algo_disc[k][0])            
            logger.info(f"Best Clasification Algorithm = {best_clasification_algo.__name__}")
            logger.info(f"Accuracy Score = {clasification_algo_disc[best_clasification_algo][0]}")
            #logger.info(f"Confusion Matrix = {clasification_algo_disc[best_clasification_algo][1]}") 
            #logger.info(f"Model Parameter = {clasification_algo_disc[best_clasification_algo][2]}") 
        else:                      
            logger.info(f"Clasification Algorithm = {clasification_algo.__name__}")
            logger.info(f"Accuracy Score = {clasification_algo_disc[clasification_algo][0]}")
            #logger.info(f"Confusion Matrix = {clasification_algo_disc[clasification_algo][1]}") 
            #logger.info(f"Model Parameter = {clasification_algo_disc[clasification_algo][2]}") 

    @clock('[{elapsed:0.8f}s] : {name}')    
    def call_clasification_method(self):
        algo = self.args.algo
        logger.debug(f"Inside call_clasification_method -> algo = {algo}")
        flag_got_algo = False        
        if algo == "all": 
            flag_got_algo = True 
            logger.debug(f"Calling clasification_method() for all clasification algorithms")  
            logger.debug(f"clasification_algo_disc =  {clasification_algo_disc}")
            for clasification_algo in clasification_algo_disc.keys():
                logger.debug(f"clasification_algo = {clasification_algo.__name__}")
                clasification_algo_disc[clasification_algo] = clasification_algo(self.args).clasification_method()  
            self.generate_result()       
        else:         
            for clasification_algo in clasification_algo_disc.keys():
                if algo == clasification_algo.__name__:
                    flag_got_algo = True
                    logger.debug(f"Trying to call the {clasification_algo.__name__}(self.args).clasification_method()")
                    clasification_algo_disc[clasification_algo] = clasification_algo(self.args).clasification_method()
                    self.generate_result(clasification_algo) 
        if flag_got_algo is False:
            logger.error(f"You have entered the clasification algorithm  {algo}. \n"+
                         f"Possible algorithms are: {[x.__name__ for x in clasification_algo_disc.keys()]}") 

class ClasificationBase(ABC):
    def __init__(self, args):
        logger.debug(f"Constructor ClasificationBase: This is an Interface")
        obj =  DataPreProcessing("Clasification", args.dataset) 
        self.X_train, self.X_test, self.y_train, self.y_test = data_preprocessing(args.dataset) 
        self.args = args    

    @abstractmethod    
    def clasification_method(self):         
        logger.debug(f"Inside clasification_method: This is an Abstruct Method")
        pass

@clasification
class Logistic(ClasificationBase):
    def clasification_method(self):
        logger.debug(f"Inside clasification_method in Logistic Object")
        return(self.logistic_regression())
    
    def logistic_regression(self):
        logger.debug(f"Inside logistic_regression method")
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(self.X_train, self.y_train)    
        y_pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        score = accuracy_score(self.y_test, y_pred)
        return(score, cm, classifier)

@clasification
class KNN(ClasificationBase):
    def clasification_method(self):
        logger.debug(f"Inside clasification_method in KNN Object")
        return(self.knn_clasification())
    def knn_clasification(self):
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(self.X_train, self.y_train)   
        y_pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        score = accuracy_score(self.y_test, y_pred)
        return(score, cm, classifier)

@clasification
class SVM_Linear(ClasificationBase):
    def clasification_method(self):
        logger.debug(f"Inside clasification_method in SVM_Linear Object")
        return(self.svm_linear_clasification())

    def svm_linear_clasification(self):
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        y_pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        score = accuracy_score(self.y_test, y_pred)
        return(score, cm, classifier)

@clasification
class SVM_Kenel(ClasificationBase):
    def clasification_method(self):
        logger.debug(f"Inside clasification_method in SVM_Kenel Object")
        return(self.kernel_svm_Clasification())

    def kernel_svm_Clasification(self):
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        y_pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        score = accuracy_score(self.y_test, y_pred)
        return(score, cm, classifier)

@clasification
class NaiveNayes(ClasificationBase):
    def clasification_method(self):
        logger.debug(f"Inside clasification_method in NaiveNayes Object")
        return(self.naive_bayes_clasification())

    def naive_bayes_clasification(self):
        classifier = GaussianNB()
        classifier.fit(self.X_train, self.y_train)
        y_pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        score = accuracy_score(self.y_test, y_pred)
        return(score, cm, classifier)

@clasification
class DecisionTree(ClasificationBase):
    def clasification_method(self):
        logger.debug(f"Inside clasification_method in DecisionTree Object")
        return(self.decision_tree_clasification())
    
    def decision_tree_clasification(self):
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        y_pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        score = accuracy_score(self.y_test, y_pred)
        return(score, cm, classifier)

@clasification
class RandomForest(ClasificationBase):
    def clasification_method(self):
        logger.debug(f"Inside clasification_method in RandomForest Object")
        return(self.random_forest_clasification())
    
    def random_forest_clasification(self):
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        y_pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        score = accuracy_score(self.y_test, y_pred)
        return(score, cm, classifier)
