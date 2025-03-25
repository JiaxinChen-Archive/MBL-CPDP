import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, matthews_corrcoef, recall_score



class HISNN(object):
    def __init__(self, model, MinHam=1.0):
        self.MinHam = MinHam
        self.model = model


    def _MahalanobisDist(self, data, base):

        covariance = np.cov(base.T)  # calculate the covarince matrix
        inv_covariance = np.linalg.pinv(covariance)
        mean = np.mean(base, axis=0)
        dist = np.zeros((np.asarray(data)).shape[0])
        for i in range(dist.shape[0]):
            dist[i] = distance.mahalanobis(data[i], mean, inv_covariance)
        return dist


    def _TrainInstanceFiltering(self):
        # source outlier remove based on source
        dist = self._MahalanobisDist(self.Xsource, self.Xsource)
        threshold = np.mean(dist) * 3 * np.std(dist)
        outliers = []
        for i in range(len(dist)):
            if dist[i] > threshold:
                outliers.append(i)  # index of the outlier
        self.Xsource = np.delete(self.Xsource, outliers, axis=0)
        self.Ysource = np.delete(self.Ysource, outliers, axis=0)

        # source outlier remove based on target
        dist = self._MahalanobisDist(self.Xsource, self.Xtarget)
        threshold = np.mean(dist) * 3 * np.std(dist)
        outliers = []
        for i in range(len(dist)):
            if dist[i] > threshold:
                outliers.append(i)  # index of the outlier
        self.Xsource = np.delete(self.Xsource, outliers, axis=0)
        self.Ysource = np.delete(self.Ysource, outliers, axis=0)


        #Did not delete data_selection
        # NN data_selection for source data based on target
        neigh = NearestNeighbors(radius=self.MinHam, metric='hamming')
        neigh.fit(self.Xsource)
        res = neigh.radius_neighbors(self.Xtarget, return_distance=False)

        tmp = np.concatenate((self.Xsource, self.Ysource.reshape(-1, 1)), axis=1)
        x = tmp[res[0]]
        for item in res[1:]:
            x = np.concatenate((x, tmp[item]), axis=0)
            x = np.unique(x, axis=0)

        self.Xsource = x[:, :-1]
        self.Ysource = x[:, -1]

    def predict(self, MultiObj):
        predict = np.zeros(self.Xtarget.shape[0])
        neigh = NearestNeighbors(radius=self.MinHam, metric='hamming')
        neigh.fit(self.Xsource)
        res = neigh.radius_neighbors(self.Xtarget, return_distance=False)
        for i in range(res.shape[0]):
            # case 1
            if len(res[i]) == 1:
                subRes = neigh.radius_neighbors(self.Xsource[res[i][0]])
                # case 1-1
                if len(subRes) == 1:
                    predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))
                else:
                    tmp = np.unique(self.Ysource[subRes])
                    # case 1-2
                    if len(tmp) == 1:
                        predict[i] = tmp[0]
                    # case 1-3
                    else:
                        predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))
            else:
                tmp = np.unique(self.Ysource[res[i]])
                # case 2
                if len(tmp) == 1:
                    predict[i] = tmp[0]
                # case 3
                else:
                    predict[i] = self.model.predict(self.Xtarget[i].reshape(1, -1))

        if MultiObj == False:
            self.AUC = [roc_auc_score(self.Ytarget, predict)]
        else:

            if -1 in self.Ytarget and 0 in predict:
                self.Ytarget[self.Ytarget == -1] = 0
            elif 0 in self.Ytarget and -1 in predict:
                predict[predict == -1] = 0


            all_tres = [0, 0, 0, 0, 0, 0, 0]

            try:
                all_tres[0] = roc_auc_score(self.Ytarget, predict)
            except Exception:
                pass

            try:
                all_tres[1] = f1_score(self.Ytarget, predict)
            except Exception:
                pass

            try:
                all_tres[2] = accuracy_score(self.Ytarget, predict)
            except Exception:
                pass

            try:
                all_tres[3] = recall_score(self.Ytarget, predict)
            except Exception:
                pass


            if all_tres[2] != 0:
                all_tres[4] = 1 - all_tres[2]

            try:
                all_tres[5] = precision_score(self.Ytarget, predict)
            except Exception:
                pass

            try:
                all_tres[6] = matthews_corrcoef(self.Ytarget, predict)
            except Exception:
                pass

            self.AUC = all_tres


    def run(self, Xs, Xt, Ys, Yt, MultiObj):
        self.Xsource = np.asarray(Xs)
        self.Xtarget = np.asarray(Xt)
        self.Ysource = np.asarray(Ys)
        self.Ytarget = np.asarray(Yt)

        #######
        self._TrainInstanceFiltering()

        try:
            self.model.fit(np.log(self.Xsource + 1), self.Ysource)
            self.predict(MultiObj)
            return self.AUC

        except:
            if MultiObj==False:
                return [0]
            else:
                return [0, 0, 0, 0, 0, 0, 0]


