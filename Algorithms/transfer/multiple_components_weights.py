import numpy as np
from Utils.KMM import kmm
from sklearn.cluster import KMeans
from cvxopt import matrix, solvers, spdiag
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, matthews_corrcoef, recall_score


"""
   In original paper, the author uses spectral clustering algorithm. But spectral clustering 
   algorithm cannot scale to large scale (more than 10000 dimension), so we replace it with k-means.
   
   KMM计算很耗时
"""


class MCWs():
    def __init__(self, model, k=4, sigmma=1.0, lamb=1):
        self.k = k
        self.lamb = lamb
        self.gamma = sigmma
        self.model = model
        self.res = []


    def _clustering(self, x):
        # clustering = SpectralClustering(n_clusters=self.k)
        clustering = KMeans(n_clusters=self.k)
        clustering.fit(x)
        cluster = clustering.labels_

        res = []
        for i in range(self.k):
            res.append(np.where(cluster == i))
        return res


    def _compute_KMM_weight(self, x, train):
        weight = kmm(x, train, sigma=self.gamma)
        return weight


    def _getPerformance(self, x_validation, L_validation, alpha, ensemble, MultiObj):
        if len(alpha) == 0:
            if MultiObj == False:
                return [0]
            else:
                return [0, 0, 0, 0, 0, 0, 0]

        res = np.zeros(len(L_validation))

        prediction = np.zeros((len(ensemble), len(L_validation)))
        for i in range(len(ensemble)):
            prediction[i] = ensemble[i].predict(x_validation).reshape(1, -1)
        prediction[prediction<0] = 0
        for i in range(len(L_validation)):
            res[i] = np.argmax(np.bincount(prediction[:, i].astype(np.int), weights=alpha))


        if MultiObj == False:
            return [roc_auc_score(L_validation, res)]
        else:

            if -1 in L_validation and 0 in res:
                L_validation[L_validation == -1] = 0
            elif 0 in L_validation and -1 in res:
                res[res == -1] = 0


            # AUC, F1, ACC, Recall, ERR, PREC, MCC
            all_tres = [0, 0, 0, 0, 0, 0, 0]

            try:
                all_tres[0] = roc_auc_score(L_validation, res)
            except Exception:
                pass

            try:
                all_tres[1] = f1_score(L_validation, res)
            except Exception:
                pass

            try:
                all_tres[2] = accuracy_score(L_validation, res)
            except Exception:
                pass

            try:
                all_tres[3] = recall_score(L_validation, res)
            except Exception:
                pass

            if all_tres[2] != 0:
                all_tres[4] = 1 - all_tres[2]

            try:
                all_tres[5] = precision_score(L_validation, res)
            except Exception:
                pass

            try:
                all_tres[6] = matthews_corrcoef(L_validation, res)
            except Exception:
                pass

            return all_tres


    def _build_ensemble(self, xs, ys, train, l_train, test, l_test, MultiObj):
        cluster = self._clustering(xs)
        tmp_res = []
        tmp_prediction = []
        ensemble = []
        for item in cluster:
            x = np.concatenate((xs[item], train), axis=0)
            y = np.concatenate((ys[item], l_train), axis=0)
            weight = np.asarray(self._compute_KMM_weight(x, train)).reshape(-1)
            try:
                self.model.fit(x, y, sample_weight=weight)
                pd = self.model.predict(train)
            except:
                continue
            tmp_prediction.append(pd)
            tmp_res.append(accuracy_score(l_train, pd))
            ensemble.append(self.model)


        tmp_res = np.asarray(tmp_res)
        tmp_prediction = np.asarray(tmp_prediction)
        init_weight = tmp_res / np.sum(tmp_res)
        # optimize the weight of base classifiers
        A = matrix(np.ones((1, len(cluster))))
        m, n = A.size

        def F(x=None, z=None):
            if x is None: return 0, matrix(1.0, (n, 1))
            if min(x) <= 0.0: return None
            f = 0
            x1 = np.asarray(x).reshape(-1)
            for i in range(len(l_train)):
                f = f + self.lamb * np.linalg.norm(x1 - init_weight) + \
                    np.linalg.norm(np.dot(tmp_prediction[:, i].reshape(-1, 1), x1.reshape(1, -1)) - l_train[i])
            Df = -(x ** -1).T
            if z is None: return f, Df
            H = spdiag(z[0] * x ** -2)
            return f, Df, H

        optimal_weight = solvers.cp(F, A=A, b=matrix(np.ones((1, 1))), options={'show_progress': False})['x']
        optimal_weight = np.asarray(optimal_weight).reshape(-1)


        self.res.append(self._getPerformance(test, l_test, optimal_weight, ensemble, MultiObj))


    def run(self, Xs, Ys, test, l_test, train, l_train, loc, MultiObj):
        self.Xs = np.asarray(Xs)
        self.Ys = np.asarray(Ys)

        for i in range(len(loc)):
            if i < len(loc) - 1:
                xs = self.Xs[loc[i]:loc[i + 1]]
                ys = self.Ys[loc[i]:loc[i + 1]]
            else:
                xs = self.Xs[loc[i]:]
                ys = self.Ys[loc[i]:]

            self._build_ensemble(xs, ys, train, l_train, test, l_test, MultiObj)

        return np.mean(self.res, axis=0)

