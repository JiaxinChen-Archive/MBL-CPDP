import numpy as np
import collections, copy
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, \
    matthews_corrcoef, recall_score


# No FEATURES sets
class GIS():
    def __init__(self, model, model_name, mProb, mCount, popsize=30, chrmsize=0.02, numgens=20,
                 numparts=5):
        self.popsize = popsize
        self.chrmsize = chrmsize
        self.numgens = numgens
        self.numparts = numparts
        self.model = model
        self.p = mProb
        self.c = mCount
        self.iteration = 1
        self.res = []
        self.allres = []
        self.modelname = model_name



    def _NNfilter(self, train, test, n_neighbors=10):
        xtrain = train[:, :-1]
        ytrain = train[:, -1]
        xtest = test[:, :-1]

        knn = NearestNeighbors(metric='euclidean')
        knn.fit(xtrain)
        data = []

        for item in xtest:
            tmp = knn.kneighbors(item.reshape(1, -1), n_neighbors, return_distance=False)[0]
            for i in tmp:
                data.append(list(train[i]))
        if len(data) == 0:
            return []
        Xs, idx = np.unique(np.asanyarray(data), axis=0, return_index=True)
        return Xs


    def _evaluate(self, train, test):
        if train.shape[0] < 2 or test.shape[0] < 2 or np.unique(train[:, -1]).shape[0] == 1:
            return 0
        self.model.fit(train[:, :-1], train[:, -1])
        try:
            pre = self.model.predict(test[:, :-1])
            f1 = f1_score(test[:, -1], pre)
            g = np.sqrt(precision_score(test[:, -1], pre) * recall_score(test[:, -1], pre))
            return f1 * g
        except:
            return 0


    def _crossover(self, DS1, DS2):
        nDS1 = []
        nDS2 = []
        point = int(np.random.randint(0, DS1.shape[0], 1))

        for i in range(point):
            nDS1.append(DS1[i])
            nDS2.append(DS2[i])

        for i in range(point, DS1.shape[0]):
            nDS2.append(DS1[i])
            nDS1.append(DS2[i])

        nDS1 = np.asarray(nDS1)
        nDS2 = np.asarray(nDS2)
        # label instances of nDS1 and nDS2 when conflict occurs
        for item in nDS1:
            label = nDS1[nDS1 == item]
            if label.ndim == 1:
                label = label[-1]
            else:
                label = label[:, -1]
            if len(np.unique(label)) == 2:
                res = collections.Counter(label)
                item[-1] = sorted(res.items(), key=lambda x: x[1], reverse=True)[0][0]

        for item in nDS2:
            label = nDS2[nDS2 == item]
            if label.ndim == 1:
                label = label[-1]
            else:
                label = label[:, -1]
            if len(np.unique(label)) == 2:
                res = collections.Counter(label)
                item[-1] = sorted(res.items(), key=lambda x: x[1], reverse=True)[0][0]

        return nDS1, nDS2


    def _mutation(self, DS):
        if DS.shape[0] < self.c:
            return DS
        r = np.random.random()
        if r < self.p:
            idx = np.random.choice(DS.shape[0], self.c, replace=False)
            for i in range(self.c):
                # reverse the labels of selected instances
                if DS[idx[i]][-1] == 1:
                    DS[DS == DS[idx[i]]][-1] = 0
                else:
                    DS[DS == DS[idx[i]]][-1] = 1
        return DS


    def _generate(self, DataSets):
        DTs = copy.deepcopy(DataSets)
        DT = []
        for i in range(len(DTs)):
            if len(DTs[i]) == 0:
                continue
            DTs[i] = self._mutation(DTs[i])
        i = 0
        while i < self.popsize:
            idx = np.random.choice(self.popsize, 2, replace=False)
            d1, d2 = self._crossover(DTs[idx[1]], DTs[idx[0]])
            DT.append(d1)
            DT.append(d2)
            i += 2

        return DT


    def run(self, Xs, Xt, Ys, Yt, MultiObj):
        self.Xs = np.asarray(Xs)
        self.Xt = np.asarray(Xt)
        self.Ys = np.asarray(Ys)
        self.Yt = np.asarray(Yt)

        TEST = np.concatenate((self.Xt, self.Yt.reshape(-1, 1)), axis=1)
        TRAIN = np.concatenate((self.Xs, self.Ys.reshape(-1, 1)), axis=1)

        for i in range(self.iteration):
            idx = sorted(np.random.choice(TEST.shape[0], self.numparts - 1, replace=False))
            TestParts = np.split(TEST, idx)
            prediction = []

            for testPart in TestParts:

                vSet = self._NNfilter(TRAIN, testPart)

                if len(vSet) == 0:
                    prediction = np.concatenate((prediction, np.random.randint(0, 2, testPart.shape[0])))
                    continue
                TrainDataSets = []
                self.fitness = np.zeros(self.popsize)
                for _ in range(self.popsize):
                    idx = sorted(np.random.choice(TEST.shape[0], int(self.chrmsize * TEST.shape[0]), replace=True))
                    TrainDataSets.append(TEST[idx])

                for td in range(len(TrainDataSets)):
                    self.fitness[td] = self._evaluate(TrainDataSets[td], vSet)

                # create a generation using operators and elite from current generation
                # combine the two generations and extract a new generation
                for g in range(self.numgens):
                    DT = self._generate(TrainDataSets)
                    fitness = np.zeros(self.popsize)
                    for j in range(self.popsize):
                        fitness[j] = self._evaluate(DT[i], vSet)
                    fitness = np.concatenate((self.fitness, fitness))
                    idx = np.argsort(-fitness)
                    DS = []
                    for k in range(self.popsize):
                        if idx[k] < self.popsize:
                            DS.append(TrainDataSets[idx[k]])
                        else:
                            DS.append(DT[idx[k]-self.popsize])
                    self.best = DS[0]

                # select top dataset from last generation
                # evaluate bestDS on testParts  and append the results to the
                # pool of results
                try:
                    self.model.fit(self.best[:, :-1], self.best[:, -1])
                    prediction = np.concatenate((prediction, self.model.predict(testPart[:, :-1])))
                except:
                    prediction = np.concatenate((prediction, np.random.randint(0, 2, testPart.shape[0])))


            if MultiObj == False:
                tres = [roc_auc_score(TEST[:, -1], prediction)]
                all_tres = tres
            else:

                if -1 in TEST[:, -1] and 0 in prediction:
                    TEST[:, -1][TEST[:, -1] == -1] = 0
                elif 0 in TEST[:, -1] and -1 in prediction:
                    prediction[prediction == -1] = 0


                all_tres = [0, 0, 0, 0, 0, 0, 0]

                try:
                    all_tres[0] = roc_auc_score(TEST[:, -1], prediction)
                except Exception:
                    pass  # 捕获异常但不执行任何操作

                try:
                    all_tres[1] = f1_score(TEST[:, -1], prediction)
                except Exception:
                    pass

                try:
                    all_tres[2] = accuracy_score(TEST[:, -1], prediction)
                except Exception:
                    pass

                try:
                    all_tres[3] = recall_score(TEST[:, -1], prediction)
                except Exception:
                    pass

                if all_tres[2] != 0:
                    all_tres[4] = 1 - all_tres[2]

                try:
                    all_tres[5] = precision_score(TEST[:, -1], prediction)
                except Exception:
                    pass

                try:
                    all_tres[6] = matthews_corrcoef(TEST[:, -1], prediction)
                except Exception:
                    pass

                tres = all_tres[0]


            if i > 0 and abs(tres - self.res[-1]) < 0.0001:
                self.res.append(tres)
                self.allres.append(all_tres)

                if MultiObj == False:
                    return [np.median(np.asarray(self.res))]
                else:
                    return np.median(np.asarray(self.allres), axis=0)
            else:
                self.res.append(tres)
                self.allres.append(all_tres)

        if MultiObj == False:
            return [np.median(np.asarray(self.res))]
        else:
            return np.median(np.asarray(self.allres), axis=0)





