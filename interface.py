from abc import ABC, abstractmethod
from logger_config import logger
import clustering
import regression
import classification

class InterfaceML(ABC):
    @abstractmethod
    def machine_learning_method(self, args): 
        pass
    
class Clustering(InterfaceML):
    def machine_learning_method(self, args):
        clustering.BuildClusteringAlgo(args)
           
class Regression(InterfaceML):
    def machine_learning_method(self, args):
        regression.BuildRegressionAlgo(args) 
           
class Classification(InterfaceML):
    def machine_learning_method(self, args):
        classification.BuildClasificationAlgo(args) 

