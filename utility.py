import time
import argparse
from logger_config import logger

DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args})  -> {result}'
def clock(fmt=DEFAULT_FMT):
    def decorate(func):
        def clocked(*_args):
            t0 = time.time()
            _result = func(*_args)
            elapsed = time.time() - t0
            name = func.__name__
            args = ', '.join(repr(arg) for arg in _args)
            result = repr(_result)
            logger.debug(fmt.format(**locals()))
            return _result
        return clocked
    return decorate

def argument_parser():    
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--dataset", dest='dataset', metavar='string',
                        help='Data Set for the Machine Learning Tool in .csv format. User must provide full path of the .csv file otherwise the program will return nothing')
    parser.add_argument('-m', '--ml_method', dest='ml_method', default = "Regression", metavar='string', choices=['Regression', 'Classification', 'Clustering', 'Association'],
                        help='Specify which kind of ML Tool is Required. Possoble Values: Regression, Classification, Clustering. Default = "Regression"')
    parser.add_argument('-n', '--nos', dest='col_no', default = "-1", type=int, nargs='+',
                        help='Specify which columns you want in your training from the dataset in int, default = all' )    
    parser.add_argument('-a', '--algo', dest='algo', default = "all", metavar='string',
                        help='Specify which algorithm you want for your ML Tool. Like Liner Regression, SVM, etc, default = all' )
    args = parser.parse_args()

    # Check arguments and return
    assert args.dataset is not None, "User must provide full path of the .csv file as an argument. Example: -d Data.csv"
    return(args)