import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from utility import clock
from logger_config import logger
from abc import ABC, abstractmethod
from data_pre_processing import DataPreProcessing

regression_algo_disc = {}

def regression(regression_algo):
    logger.debug(f"Inside @regression decorators -> input regression_algo = {regression_algo.__name__}")
    regression_algo_disc[regression_algo] = None
    def inner(*args, **kwargs):         
        logger.debug(f"Inside inner method of @regression decorators -> input args  =  {args}, kwargs = {kwargs}") 
        return regression_algo(*args, **kwargs)
    return inner

class BuildRegressionAlgo():  
    def __init__(self, args):
        logger.debug(f"Constructor BuildRegressionAlgo with args = {args}")
        self.args = args
        self.call_regression_method()

    def generate_result(self, regression_algo = None):
        logger.debug(f"Inside generate_result")
        if regression_algo is None:
            logger.debug(f"Inside if ---regression_algo_disc = {regression_algo_disc} ")
            for regression_algo in regression_algo_disc.keys(): 
                score = regression_algo_disc[regression_algo] [0]
                logger.debug(f"[Regression Algo Name] : {regression_algo.__name__} \t ->  R2-score = {score:0.4f}")                                  
            best_regression_algo = max(regression_algo_disc, key=lambda k: regression_algo_disc[k][0])            
            logger.info(f"Best Regression Algorithm = {best_regression_algo.__name__}")
            logger.info(f"R2 Score = {regression_algo_disc[best_regression_algo][0]}")
            logger.info(f"Model Parameter = {regression_algo_disc[best_regression_algo][1]}") 
        else:                      
            logger.info(f"Regression Algorithm = {regression_algo.__name__}")
            logger.info(f"R2 Score = {regression_algo_disc[regression_algo][0]}")
            logger.info(f"Model Parameter = {regression_algo_disc[regression_algo][1]}") 


    @clock('[{elapsed:0.8f}s] : {name}')    
    def call_regression_method(self):
        algo = self.args.algo
        logger.debug(f"Inside call_regression_method -> algo = {algo}")
        flag_got_algo = False        
        if algo == "all": 
            flag_got_algo = True 
            logger.debug(f"Calling regression_method() for all regression algorithms")  
            logger.debug(f"regression_algo_disc =  {regression_algo_disc}")
            for regression_algo in regression_algo_disc.keys():
                logger.debug(f"regression_algo = {regression_algo.__name__}")
                regression_algo_disc[regression_algo] = regression_algo(self.args).regression_method()  
            self.generate_result()       
        else:         
            for regression_algo in regression_algo_disc.keys():
                if algo == regression_algo.__name__:
                    flag_got_algo = True
                    logger.debug(f"Trying to call the {regression_algo.__name__}(self.args).regression_method()")
                    regression_algo_disc[regression_algo] = regression_algo(self.args).regression_method()
                    self.generate_result(regression_algo) 
        if flag_got_algo is False:
            logger.error(f"You have entered the regression algorithm  {algo}. \n"+
                         f"Possible algorithms are: {[x.__name__ for x in regression_algo_disc.keys()]}") 

class RegressionBase(ABC):
    def __init__(self, args):
        logger.debug(f"Constructor RegressionBase: This is an Interface")
        obj =  DataPreProcessing("Regression", args.dataset) 
        self.X_train, self.X_test, self.y_train, self.y_test = obj.data_processing_regression()
        self.args = args    

    @abstractmethod    
    def regression_method(self):         
        logger.debug(f"Inside regression_method: This is an Abstruct Method")
        pass

@regression
class Linear(RegressionBase):
    def regression_method(self):
        logger.debug(f"Inside regression_method in LinerRegression Object")
        return(self.liner_regression())

    def liner_regression(self): 
        logger.debug(f"Inside liner_regression method")
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        score = r2_score(self.y_test, y_pred)
        return(score, regressor)

@regression
class Polynomial(RegressionBase):
    def regression_method(self):
        logger.debug(f"Inside regression_method in PolynomialRegression Object")
        return(self.polynomial_regression())

    def polynomial_regression(self): 
        logger.debug(f"Inside polynomial_regression method")
        poly_reg = PolynomialFeatures(degree = 4)
        X_poly = poly_reg.fit_transform(self.X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train)
        y_pred = regressor.predict(poly_reg.transform(self.X_test))
        np.set_printoptions(precision=2)
        score = r2_score(self.y_test, y_pred)
        return(score, regressor)  
    
@regression
class SupportVector(RegressionBase):
    def regression_method(self):
        logger.debug(f"Inside regression_method in SupportVectorRegression Object")
        return(self.support_vector_regression())

    def support_vector_regression(self): 
        logger.debug(f"Inside support_vector_regression method")
        #Feature Scaling
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.y_train = sc_y.fit_transform(self.y_train.reshape(-1, 1))

        regressor = SVR(kernel = 'rbf')
        regressor.fit(self.X_train, self.y_train.reshape(len(self.y_train)))
        y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(self.X_test)).reshape(-1,1))
        np.set_printoptions(precision=2)
        score = r2_score(self.y_test, y_pred)
        return(score, regressor)
    
@regression
class DecisionTree(RegressionBase):
    def regression_method(self):
        logger.debug(f"Inside regression_method in DecisionTreeRegression Object")
        return(self.decision_tree_regression())

    def decision_tree_regression(self):
        logger.debug(f"Inside decision_tree_regression method")
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        score = r2_score(self.y_test, y_pred)
        return(score, regressor)   
    
@regression
class RandomForest(RegressionBase):
    def regression_method(self):
        logger.debug(f"Inside regression_method in RandomForestRegression Object")
        return(self.random_forest_regression())
    
    def random_forest_regression(self):
        logger.debug(f"Inside random_forest_regression method")
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        score = r2_score(self.y_test, y_pred)
        return(score, regressor)




