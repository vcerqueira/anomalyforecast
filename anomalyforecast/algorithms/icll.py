from typing import List
from collections import Counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class NoGreyZoneError(ValueError):
    pass


class ICLL:

    def __init__(self, model_l1, model_l2, cluster_linkage: str = 'ward'):
        self.model_l1 = model_l1
        self.model_l2 = model_l2
        self.clusters = []
        self.grey_ind_arr = np.array([])
        self.linkage = cluster_linkage

    def fit(self, X: pd.DataFrame, y):

        if isinstance(y, pd.Series):
            y = y.values

        print('clustering')
        self.clusters = self.clustering(X=X, method=self.linkage)

        self.grey_ind_arr = self.three_way_grouping(clusters=self.clusters, y=y)
        print('m1')
        y_l1 = y.copy()
        y_l1[self.grey_ind_arr] = 1

        X_l2 = X.loc[self.grey_ind_arr, :]
        y_l2 = y[self.grey_ind_arr]

        self.model_l1.fit(X, y_l1)
        print('m2')
        self.model_l2.fit(X_l2, y_l2)

    def predict(self, X):

        yh_l1, yh_l2 = self.model_l1.predict(X), self.model_l2.predict(X)

        yh_f = np.asarray([x1 * x2 for x1, x2 in zip(yh_l1, yh_l2)])

        return yh_f

    def predict_proba(self, X):

        yh_l1_p = self.model_l1.predict_proba(X)
        try:
            yh_l1_p = np.array([x[1] for x in yh_l1_p])
        except IndexError:
            yh_l1_p = yh_l1_p.flatten()
        yh_l2_p = self.model_l2.predict_proba(X)
        yh_l2_p = np.array([x[1] for x in yh_l2_p])

        yh_fp = np.asarray([x1 * x2 for x1, x2 in zip(yh_l1_p, yh_l2_p)])

        return yh_fp

    def predict_l2(self, X):

        return self.model_l2.predict(X)

    def predict_proba_l2(self, X):

        yh_l2_p = self.model_l2.predict_proba(X)
        yh_l2_p = np.array([x[1] for x in yh_l2_p])

        return yh_l2_p

    def predict_proba_l1(self, X):

        yh_l1_p = self.model_l1.predict_proba(X)
        yh_l1_p = np.array([x[1] for x in yh_l1_p])

        return yh_l1_p

    @classmethod
    def three_way_grouping(cls, clusters: List[np.ndarray], y: np.ndarray) -> np.ndarray:

        whites, blacks, greys = [], [], []
        for clst in clusters:
            try:
                y_clt = y[np.asarray(clst)]

                if len(Counter(y_clt)) == 1:
                    if y_clt[0] == 0:
                        whites.append(clst)
                    else:
                        blacks.append(clst)
                else:
                    greys.append(clst)
            except ValueError:
                raise ValueError('Error when creating three-way groups.')

        if len(greys) < 1:
            raise NoGreyZoneError('No grey zone error.')

        grey_ind = np.array(sorted(np.concatenate(greys).ravel()))
        grey_ind = np.unique(grey_ind)

        if len(blacks) > 0:
            black_ind = np.array(sorted(np.concatenate(blacks).ravel()))
        else:
            black_ind = np.array([])

        greyb_ind = np.unique(np.concatenate([grey_ind, black_ind])).astype(int)

        return greyb_ind

    @classmethod
    def clustering(cls, X, method='ward'):
        d = pdist(X)  # PAIRWISE DISTANCE BETWEEN INSTANCES

        Z = linkage(d, method)  # calculate the instances links for the agglomerative clustering
        Z[:, 2] = np.log(1 + Z[:, 2])
        sZ = np.std(Z[:, 2])  # standard deviation of the distance between each cluster linkage
        mZ = np.mean(Z[:, 2])  # mean of the distance between each cluster linkage

        clustLabs = fcluster(Z, mZ + sZ,
                             criterion='distance')  # for the clusters using merger distance threshold as mZ+stds*sZ

        clusters = []
        for lab in np.unique(clustLabs):  # produce a clusters data structure to work with
            clusters.append(np.where(clustLabs == lab)[0])

        return clusters
