import pandas as pd

from constants.dataset import TARGETVAR
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod


class ClusterPrediction(ABC):

    def __init__(self, model_f, cluster=None, n_clusters=None, params_kmeans=None):
        """
        :param model_f: () -> {sklearn interface model}  a function that return a new instance of the model
        :param cluster: An instance of cluster algorithm. Must support 'fit_predict' that given
                        in input the numpy matrix return for each row the cluster which it is in.
                        Otherwise use KMeans clustering algorithm and the two following params
        :param n_clusters: the number of clusters. It is ignored if cluster is specified
        :param params_kmeans: additional params for kmeans algorithm. It is ignored
                              if cluster is specified
        """
        self.n_clusters = n_clusters or cluster.n_cluster
        self.model_f = model_f
        if cluster == None:
            params_kmeans = params_kmeans or dict()
            self.cluster_obj = KMeans(n_clusters, **params_kmeans)
        else:
            self.cluster_obj = cluster
        self.clusters_test = None
        self.models = None

    def fit(self, train: pd.DataFrame, test: pd.DataFrame, target=None, fit_params=None):
        target = target or TARGETVAR
        fit_params = fit_params or dict()
        clusters = self.cluster_obj.fit_predict(X=pd.concat([train.drop(columns=TARGETVAR), test]))
        clusters_train, clusters_test = clusters[:len(train)], clusters[len(train):]
        models = []
        for i in range(self.cluster_obj.n_clusters):
            model = self.model_f()
            cluster = train.loc[clusters_train == i, :]
            X_train, y_train = cluster.drop(columns=target), cluster[target]
            model.fit(X_train, y_train.values, **fit_params)
            models.append(model)
        self.clusters_test = clusters_test
        self.models = models
        return self

    @abstractmethod
    def predict(self, test: pd.DataFrame):
        pass


class ClusterValuePrediction(ClusterPrediction):
    """
    Predict a value usign clustering
    """

    def predict(self, test: pd.DataFrame):
        indx_clusters = []
        for i in range(self.n_clusters):
            cluster = test.loc[self.clusters_test == i, :]
            prediction = self.models[i].predict(cluster.values)
            indx_clusters.append((cluster.index, prediction))
        dfs = [pd.DataFrame(p, index=indx_cluster, columns=['p']) for indx_cluster, p in indx_clusters]
        yhat = pd.concat(dfs).sort_index().values
        return yhat


class ClusterProbabilityPrediction(ClusterPrediction):
    """
    Predict the probability for each class using clustering
    """

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        indx_clusters = []
        for i in range(self.n_clusters):
            cluster = test.loc[self.clusters_test == i, :]
            prediction = self.models[i].predict_proba(cluster.values)
            indx_clusters.append((cluster.index, prediction))
        nclasses = indx_cluster[0][1].shape[1]
        dfs = [pd.DataFrame(p, index=indx_cluster,
                            columns=[f'p_{i}' for i in range(nclasses)]) for indx_cluster, p in indx_clusters]
        predictions = pd.concat(dfs).sort_index().values
        return predictions


__all__ = ['ClusterProbabilityPrediction', 'ClusterValuePrediction']
