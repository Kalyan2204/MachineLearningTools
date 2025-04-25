from utility import argument_parser
import interface
import sys
from logger_config import logger


class UseFactory():
    def build_ml(ml_type):
        if ml_type == "Regression":
            return interface.Regression()    
        if ml_type == "Classification":
            return interface.Classification()   
        if ml_type == "Clustering":
            return interface.Clustering()   
        
if __name__ == "__main__":
    logger.debug(f"Inside main")
    args = argument_parser()
    logger.info(f"args = {args}")
    ml_obj = UseFactory.build_ml(args.ml_method)
    ml_obj.machine_learning_method(args)




    #dataset = sys.argv[1]
    #ml_method = sys.argv[2]    
    #ml_obj = UseFactory.build_ML(ml_method)
    #ml_obj.machine_learning_method(dataset)

    
    



