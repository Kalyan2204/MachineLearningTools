import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from data_pre_processing import DataPreProcessing
from sklearn.cluster import KMeans, AgglomerativeClustering
from autoelbow_rupakbob import autoelbow
from logger_config import logger
from abc import ABC, abstractmethod

MAX_CLUSTER_K_MEANS = 10
RANDOM_STATE = 42
COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'magenta']
clustering_algo_disc = {}

def clustering(clustering_algo):
    logger.debug(f"Inside @clustering decorators -> input clustering_algo = {clustering_algo.__name__}")
    clustering_algo_disc[clustering_algo] = None
    def inner(*args, **kwargs):         
        logger.debug(f"Inside inner method of @clustering decorators -> input args  =  {args}, kwargs = {kwargs}") 
        return clustering_algo(*args, **kwargs)
    return inner

def dispaly_cluster(X, y_kmeans, n_clusters, col_no, title =""):
    logger.debug(f"Inside dispaly_cluster -> input n_clusters = {n_clusters}, col_no = {col_no}, title = {title}")
    for i in range(n_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 50, c = COLORS[i], label = f'Cluster {i+1}')   
    logger.debug(f"Inside dispaly_cluster -> Plotting the Image")
    plt.xlabel(f'Column no {col_no[0]}')
    plt.ylabel(f'Column no {col_no[1]}')
    plt.title(title)
    plt.legend()
    plt.show()
    return

class BuildClusteringAlgo():  
    def __init__(self, args):
        logger.debug(f"Constructor BuildClusteringAlgo with args = {args}")
        self.args = args
        self.call_clustering_method()

    def call_clustering_method(self):
        algo = self.args.algo
        logger.debug(f"Inside call_clustering_method -> algo = {algo}")
        flag_got_algo = False        
        if algo == "all": 
            flag_got_algo = True 
            logger.debug(f"Calling clustering_method() for all clustering algorithms")  
            logger.debug(f"clustering_algo_disc =  {clustering_algo_disc}")
            for clustering_algo in clustering_algo_disc.keys():
                logger.debug(f"clustering_algo = {clustering_algo.__name__}")
                clustering_algo(self.args).clustering_method()        
        else:         
            for clustering_algo in clustering_algo_disc.keys():
                if algo == clustering_algo.__name__:
                    flag_got_algo = True
                    logger.debug(f"Trying to call the {clustering_algo.__name__}(self.args).clustering_method()")
                    clustering_algo(self.args).clustering_method()
        if flag_got_algo is False:            
            logger.error(f"You have entered the clustering algorithm {algo}. \n"+
                         f"Possible algorithms are: {[x.__name__ for x in clustering_algo_disc.keys()]}")
        
class ClusteringBase(ABC):
    def __init__(self, args):
        logger.debug(f"Constructor ClusteringBase: This is an Interface")
        obj =  DataPreProcessing("Clustering", args.dataset)
        self.X_train = obj.data_processing_clustering(args.col_no)
        self.args = args

    @abstractmethod    
    def clustering_method(self):         
        logger.debug(f"Inside clustering_method: This is an Abstruct Method")
        pass

@clustering
class KMeansManual(ClusteringBase):
    def clustering_method(self):
        logger.debug(f"Inside clustering_method in KMeansClusteringManual Object")
        self.k_means_clustering_manual()

    def elbow_method_manual(self):        
        logger.debug(f"Inside elbow_method_manual in KMeansClusteringManual Object")
        wcss = []
        for i in range(1, MAX_CLUSTER_K_MEANS+1):
            kmeans = self.k_means_func(i, False)
            kmeans.fit(self.X_train)
            wcss.append(kmeans.inertia_)      
        logger.debug(f"Plotting the graph for Elbow Method")
        plt.plot(range(1, MAX_CLUSTER_K_MEANS+1), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        return

    def k_means_func(self, n_clusters, display=True):
        logger.debug(f"Inside k_means_func in KMeansClusteringManual Object -> input n_clusters = {n_clusters}, display = {display}")
        kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = RANDOM_STATE)
        y_kmeans = kmeans.fit_predict(self.X_train) 
        if display is True and len(self.X_train[0]) == 2:
            dispaly_cluster(self.X_train, y_kmeans, n_clusters, self.args.col_no, 
                            f"K-Means Clustering Manual Method with {n_clusters} clusters")
        return kmeans     
    
    def k_means_clustering_manual(self, display=True):
        logger.debug(f"Inside k_means_clustering_manual in KMeansClusteringManual Object -> input display = {display}")         
        self.elbow_method_manual()        
        n_clusters = int(input("Enter a number of clusters: "))
        logger.info(f"You have entered number of clusters = {n_clusters}")
        kmeans = self.k_means_func(n_clusters, display)
        return kmeans
    
@clustering
class KMeansAuto(ClusteringBase):
    def clustering_method(self):
        logger.debug("Inside clustering_method in KMeansClusteringAuto")
        self.k_means_clustering_auto_elbow()

    def k_means_func(self, n_clusters, display=True):
        logger.debug(f"Inside k_means_func in KMeansClusteringAuto Object -> input n_clusters = {n_clusters}, display = {display}")
        kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = RANDOM_STATE)
        y_kmeans = kmeans.fit_predict(self.X_train)  
        if display is True and len(self.X_train[0]) == 2:
            dispaly_cluster(self.X_train, y_kmeans, n_clusters, self.args.col_no, 
                            f"K-Means Clustering Manual Method with {n_clusters} clusters")
        return kmeans 

    ## Based on the Elbo Method it will identify the number of clusters and return the result accordingly
    def k_means_clustering_auto_elbow(self, display=True): 
        logger.debug(f"Inside k_means_clustering_auto_elbow in KMeansClusteringAuto Object -> input display = {display}")     
        n_clusters =autoelbow.auto_elbow_search(self.X_train)
        logger.info(f"Optimized number of clusters based on the autoelbow.auto_elbow_search: {n_clusters}")
        kmeans = self.k_means_func(n_clusters, display)
        return kmeans
    
@clustering
class Hierarchical(ClusteringBase):  
    def clustering_method(self):
        logger.debug("Inside clustering_method in HierarchicalClustering")
        self.hierarchical_clustering_manual()

    def find_dendrogram(self):
        logger.debug("Inside find_dendrogram in HierarchicalClustering")
        dendrogram = sch.dendrogram(sch.linkage(self.X_train, method = 'ward')) 
        logger.debug("Plotting Dendrogram")       
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()
    
    def hierarchical_clustering_manual(self, display=True):
        logger.debug("Inside hierarchical_clustering_manual in HierarchicalClustering")
        self.find_dendrogram()      
        n_clusters = int(input("Enter a number of clusters: "))
        logger.info(f"You have entered number of clusters = {n_clusters}")
        hc = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'ward')
        y_hc = hc.fit_predict(self.X_train)
        if display is True and len(self.X_train[0]) == 2:
            dispaly_cluster(self.X_train, y_hc, n_clusters, self.args.col_no,
                            f"Hierarchical Clustering Manual Method with {n_clusters} clusters")