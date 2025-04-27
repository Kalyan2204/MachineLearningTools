import numpy as np
import pandas as pd
from utility import clock
from logger_config import logger
from abc import ABC, abstractmethod
from apyori import apriori
from data_pre_processing import DataPreProcessing

MIN_SUPPORT = 0.003     # Form your domain knowledge 
MIN_CONFIDENCE = 0.2    # Form your domain knowledge
MIN_LIFT = 3            # Generally good trade-off value
MIN_LENGTH = 2          # Since we want the rule of 2
MAX_LENGTH = 2          # Since we want the rule of 2

association_algo_disc = {}

def association(association_algo):
    logger.debug(f"Inside @association decorators -> input association_algo = {association_algo.__name__}")
    association_algo_disc[association_algo] = None
    def inner(*args, **kwargs):         
        logger.debug(f"Inside inner method of @association decorators -> input args  =  {args}, kwargs = {kwargs}") 
        return association_algo(*args, **kwargs)
    return inner

def inspect(results, algo):
    logger.debug(f"Inside inspect method to represent the result")
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    if algo == "apriori": 
        return list(zip(lhs, rhs, supports, confidences, lifts))
    elif algo == "eclat":
        return list(zip(lhs, rhs, supports))
    else: return -1
    

class BuildAssociationAlgo():  
    def __init__(self, args):
        logger.debug(f"Constructor BuildAssociationAlgo with args = {args}")
        self.args = args
        self.call_association_method()      

    @clock('[{elapsed:0.8f}s] : {name}')    
    def call_association_method(self):
        algo = self.args.algo
        logger.debug(f"Inside call_association_method -> algo = {algo}")
        flag_got_algo = False        
        if algo == "all": 
            flag_got_algo = True 
            logger.debug(f"Calling association_method() for all association algorithms")  
            logger.debug(f"association_algo_disc =  {association_algo_disc}")
            for association_algo in association_algo_disc.keys():
                logger.debug(f"association_algo = {association_algo.__name__}")
                association_algo_disc[association_algo] = association_algo(self.args).association_method()  
            #self.generate_result()       
        else:         
            for association_algo in association_algo_disc.keys():
                if algo == association_algo.__name__:
                    flag_got_algo = True
                    logger.debug(f"Trying to call the {association_algo.__name__}(self.args).association_method()")
                    association_algo_disc[association_algo] = association_algo(self.args).association_method()
                    #self.generate_result(association_algo) 
        if flag_got_algo is False:
            logger.error(f"You have entered the association algorithm  {algo}. \n"+
                         f"Possible algorithms are: {[x.__name__ for x in association_algo_disc.keys()]}") 
            
class AssociationBase(ABC):
    def __init__(self, args):
        logger.debug(f"Constructor AssociationBase: This is an Interface")
        obj =  DataPreProcessing("Association", args.dataset) 
        self.transactions = obj.data_processing_association()
        self.args = args   

    @abstractmethod    
    def association_method(self):         
        logger.debug(f"Inside association_method: This is an Abstruct Method")
        pass

@clock
@association
class Apriori(AssociationBase):
    def association_method(self):
        logger.debug(f"Inside association_method in Apriori Object")
        return(self.apriori())

    def apriori(self): 
        logger.debug(f"Inside apriori method")
        rules = apriori(transactions = self.transactions, min_support = MIN_SUPPORT, min_confidence = MIN_CONFIDENCE, 
                        min_lift = MIN_LIFT, min_length = MIN_LENGTH, max_length = MAX_LENGTH)
        results = list(rules)
        logger.debug(f"Featched Association Rules for the data set")
        resultsinDataFrame = pd.DataFrame(inspect(results, "apriori"), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
        resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
        logger.info(f"Final result of the Apriori-based Association Rule: \n{resultsinDataFrame}")
        return   
      
@clock
@association
class Eclat(AssociationBase):
    def association_method(self):
        logger.debug(f"Inside association_method in Eclat Object")
        return(self.eclat())

    def eclat(self): 
        logger.debug(f"Inside eclat method")
        rules = apriori(transactions = self.transactions, min_support = MIN_SUPPORT, min_confidence = MIN_CONFIDENCE, 
                        min_lift = MIN_LIFT, min_length = MIN_LENGTH, max_length = MAX_LENGTH)
        results = list(rules)
        logger.debug(f"Featched Association Rules for the data set")
        resultsinDataFrame = pd.DataFrame(inspect(results, "eclat"), columns = ['Product 1', 'Product 2', 'Support'])
        resultsinDataFrame.nlargest(n = 10, columns = 'Support')
        logger.info(f"Final result of the Eclat-based Association Rule: \n{resultsinDataFrame}")
        return 