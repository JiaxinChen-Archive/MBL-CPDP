import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,\
    precision_score, matthews_corrcoef, recall_score

# the efficiency of over-sampling should be optimized!

class CDE_SMOTE():
    def __init__(self, model, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.model = model
        self.flag = 0


    def _over_sampling(self, x, idx, num):
        x_over = x[idx]
        knn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        knn.fit(x_over)
        neighbors = knn.kneighbors(x_over, return_distance=False)
        if x_over.shape[0] > num:
            idx = np.random.choice(x_over.shape[0], num, replace=False)
        else:
            idx = np.random.choice(x_over.shape[0], num, replace=True)
        for i in idx:
            i = int(i)
            rnd = int(neighbors[i][int(np.random.choice(self.k, 1))])
            xnew = x_over[i] + np.random.random() * (x_over[i] - x[rnd])
            x = np.concatenate((x, xnew.reshape(1, -1)), axis=0)
        return x


    def _class_distribution_estimation(self):
        m = np.bincount(self.Ysource)
        x = self._over_sampling(self.Xsource, np.where(self.Ysource == 1)[0], m[0] - m[1])
        y = np.concatenate((self.Ysource, np.ones(m[0] - m[1])), axis=0)
        self.model.fit(x, y)
        prediction = self.model.predict(self.Xtarget).astype(np.int)
        return np.bincount(prediction)


    def _class_distribution_modification(self, n):
        m = np.bincount(self.Ysource)
        num = int(m[0] * n[0] / n[1]) - m[1]
        self.Xsource = self._over_sampling(self.Xsource, np.where(self.Ysource == 1)[0], num)
        self.Ysource = np.concatenate((self.Ysource, np.ones(num)), axis=0)
        self.model.fit(self.Xsource, self.Ysource)


    def run(self, Xs, Xt, Ys, Yt, MultiObj):
        self.Xsource = np.asarray(Xs)
        self.Xtarget = np.asarray(Xt)
        self.Ysource = np.asarray(Ys).astype(np.int)
        self.Ytarget = np.asarray(Yt)

        try:
            n = self._class_distribution_estimation()
            self._class_distribution_modification(n)
            prediction = self.model.predict(self.Xtarget)

            if MultiObj == False:
                return [roc_auc_score(self.Ytarget, prediction)]
            else:


                if -1 in self.Ytarget and 0 in prediction:
                    self.Ytarget[self.Ytarget == -1] = 0
                elif 0 in Yt and -1 in prediction:
                    prediction[prediction == -1] = 0


                # AUC, F1, ACC, Recall, ERR, PREC, MCC
                all_tres = [0, 0, 0, 0, 0, 0, 0]

                try:
                    all_tres[0] = roc_auc_score(self.Ytarget, prediction)
                except Exception:
                    pass

                try:
                    all_tres[1] = f1_score(self.Ytarget, prediction)
                except Exception:
                    pass

                try:
                    all_tres[2] = accuracy_score(self.Ytarget, prediction)
                except Exception:
                    pass

                try:
                    all_tres[3] = recall_score(self.Ytarget, prediction)
                except Exception:
                    pass

                if all_tres[2] != 0:
                    all_tres[4] = 1 - all_tres[2]

                try:
                    all_tres[5] = precision_score(self.Ytarget, prediction)
                except Exception:
                    pass

                try:
                    all_tres[6] = matthews_corrcoef(self.Ytarget, prediction)
                except Exception:
                    pass

                return all_tres

        except:
            if MultiObj == False:
                return [0]

            else:
                return [0, 0, 0, 0, 0, 0, 0]

