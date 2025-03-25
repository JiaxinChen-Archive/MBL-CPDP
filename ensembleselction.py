import os
import time
import traceback
import numpy as np
from iteration_utilities import deepflatten
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from Algorithms.Framework_Ensemble import cpdp_Ensemble
from func_timeout import func_set_timeout, FunctionTimedOut


ESTimeBudget = 400
every_Timebuget = 100
class ES():
    def __init__(self, tabuList, EnsemblePool, f, range, rval, dir, allparam, earlystop, EnsemblePool_Size, ESSelected_Size,
                 ULPopSize, xsource, ysource, xtarget, ytarget, loc):

        self.tabuList = tabuList
        self.EnsemblePool = EnsemblePool
        self.param_Range = range
        self.param_RVal = rval
        self.allparam = allparam
        self.earlystop = earlystop
        self.EnsemblePool_Size = EnsemblePool_Size
        self.ESSelected_Size = ESSelected_Size
        self.ULPopSize = ULPopSize
        self.objFunc = f
        self.dir = dir
        self.xsource = xsource
        self.ysource = ysource
        self.xtarget = xtarget
        self.ytarget = ytarget
        self.loc = loc




    def ensemble_f(self, params):
        self.p_ES = cpdp_Ensemble(clf=self.clfType, adpt=self.adpt, fs=self.fsType, clf_loc=self.clf_loc,
                                  earlystop=self.earlystop, repeat=5)
        self.p_ES.set_params_es(**params)

        try:
            Result, clf_loc = self.p_ES.run_ensemble(
                self.xsource, self.ysource, self.xtarget, self.ytarget, self.loc,
                self.ESSelected_Size, MultiObj=True)
        except:
            Result = [0, 0, 0, 0, 0, 0, 0]
            clf_loc = None

        res = Result[0]
        self.clf_loc = clf_loc

        return {'loss': -res, 'status': STATUS_OK, 'result': res, 'MulObj': Result, 'Clf_loc': self.clf_loc}


    # the process of lower-level optimization (TPE)
    @func_set_timeout(every_Timebuget)
    def ensemble_ParameterOptimization(self):

        self.gclf = self.clfType
        paramSpace = dict()

        for i in range(len(self.paramName)):
            if self.paramType[i] == 'i':
                if len(self.paramRange[i]) == 3:
                    paramSpace[self.paramName[i]] = hp.choice(self.paramName[i],
                                                              range(self.paramRange[i][0], self.paramRange[i][1],
                                                                    self.paramRange[i][2]))
                else:
                    paramSpace[self.paramName[i]] = hp.choice(self.paramName[i],
                                                              range(self.paramRange[i][0], self.paramRange[i][1]))
            if self.paramType[i] == 'c':
                paramSpace[self.paramName[i]] = hp.choice(self.paramName[i], self.paramRVal[i])
            if self.paramType[i] == 'r':
                paramSpace[self.paramName[i]] = hp.uniform(self.paramName[i], self.paramRange[i][0],
                                                           self.paramRange[i][1])

        if 'SVM' in self.gclf:
            tmpparamSpace = {
                'SVCkernel': hp.choice('SVCkernel', [
                    {'kernel': 'linear', 'max_iter': -1},
                    {'kernel': 'poly', 'degree': hp.choice('degree', range(1, 5)),
                     'polycoef0': hp.uniform('polycoef0', 0, 10),
                     'polygamma': hp.uniform('polygamma', 1e-2, 100),
                     'max_iter': 10},
                    {'kernel': 'sigmoid', 'sigcoef0': hp.uniform('sigcoef0', 0, 10),
                     'siggamma': hp.uniform('siggamma', 1e-2, 100),
                     'max_iter': 10},
                    {'kernel': 'rbf', 'rbfgamma': hp.uniform('rbfgamma', 1e-2, 100),
                     'max_iter': 10}
                ]),
                'svmC': hp.uniform('C', 0.001, 10),
            }
            paramSpace = dict(paramSpace, **tmpparamSpace)


        if 'NB' in self.gclf:
            tmpparamSpace = {
                'NBparam': hp.choice('NBparam', [
                    {'NBType': 'gaussian'},
                    {'NBType': 'multinomial', 'malpha': hp.uniform('malpha', 0, 10)},
                    {'NBType': 'complement', 'calpha': hp.uniform('calpha', 0, 10),
                     'norm': hp.choice('norm', [True, False])}])

            }
            paramSpace = dict(paramSpace, **tmpparamSpace)


        if self.adpt == 'TD':
            adptparamSpace = {
                'TDparam': hp.choice('TDparam', [
                    {'TD_strategy': 'NN', 'TD_num': hp.choice('TD_num', range(1, len(self.loc)))},
                    {'TD_strategy': 'EM'}
                ])
            }
            paramSpace = dict(paramSpace, **adptparamSpace)

        best = fmin(self.ensemble_f, space=paramSpace, algo=tpe.suggest, max_evals=5,
                    trials=self.trails, show_progressbar=False)


    def ensemble_run(self):

        for i in range(5):
            try:
                self.ensemble_ParameterOptimization()
            except:
                 best = []

            # save the running history
            his = dict()
            try:
                his['name'] = list(self.trails.trials[0]['misc']['vals'].keys())
                i = 0
                for item in self.trails.trials:
                    if item['state'] == 2:
                        results = list(deepflatten(item['misc']['vals'].values()))
                        results.append(item['result']['result'])
                        his[i] = results
                        i += 1
                if i > 0:
                    inc_value = self.trails.best_trial['result']['result']
                    best = self.trails.best_trial['misc']['vals']
                    MulObj = self.trails.best_trial['result']['MulObj']
                    Clf_loc = self.trails.best_trial['result']['Clf_loc']
                else:
                    try:
                        inc_value = self.trails.best_trial['result']['result']
                        best = self.trails.best_trial['misc']['vals']
                        MulObj = self.trails.best_trial['result']['MulObj']
                        Clf_loc = self.trails.best_trial['result']['Clf_loc']
                    except:
                        inc_value = 0
                        best = []
                        MulObj = [0, 0, 0, 0, 0, 0, 0]
                        Clf_loc = []
            except:
                inc_value = 0
                best = []
                MulObj = [0, 0, 0, 0, 0, 0, 0]
                Clf_loc = []


            if inc_value <= self.earlystop:
                break

        return -inc_value, best, MulObj, Clf_loc



    def EnsembleSearch(self, filtered_EnsemblePool, ES_num):

        Classifier_loc = [int(key[-1]) for key in filtered_EnsemblePool.keys()]
        unique_elements = list(dict.fromkeys(Classifier_loc))
        clf_loc = unique_elements
        self.clf_loc = unique_elements
        adpt_fs_loc = [key[:2] for key in filtered_EnsemblePool.keys()][ES_num]
        self.adpt = self.param_RVal[0][adpt_fs_loc[0]]
        self.fsType = self.param_RVal[1][adpt_fs_loc[1]]
        adpt_fs_name = [self.adpt, self.fsType]


        while len(self.clf_loc) >= 1:

            self.FinalES = dict()
            self.clfType = [self.param_RVal[2][x] for _, x in enumerate(self.clf_loc)]
            adpt_fs_clf_name = adpt_fs_name + self.clfType

            params = dict()
            for j in range(len(adpt_fs_clf_name)):
                if adpt_fs_clf_name[j] in ['SVM', 'TD', 'NB']:
                    continue
                if adpt_fs_clf_name[j] != 'None':
                    params = dict(params, **(self.allparam[adpt_fs_clf_name[j]]))

            self.paramName = dict()  # the name of parameters
            self.paramType = dict()  # the type of parameters (integer, real, categery)
            self.paramRVal = dict()  # the range of parameters (origin)
            self.paramRange = dict()  # the range of parameters (transfered)
            for z, (k, v) in enumerate(params.items()):
                self.paramName[z] = k
                self.paramType[z] = v[0]
                self.paramRVal[z] = v[1]
                self.paramRange[z] = v[1] if v[0] in ['i', 'r'] else [0, len(v[1]) - 1]

            self.trails = Trials()

            inc_value, best, MulObj_AUC_F1, Clf_loc = self.ensemble_run()

            if inc_value == 0:
                mul_res = [0, 0]
            else:
                mul_res = [inc_value, -(1 - np.sqrt(np.abs(inc_value)))]

            if Clf_loc is not None:
                self.FinalES[tuple((*adpt_fs_loc, *Clf_loc))] = [mul_res, best, MulObj_AUC_F1]
            else:
                self.FinalES[tuple(adpt_fs_loc)] = [mul_res, best, MulObj_AUC_F1]


            if inc_value is not None:
                """ Final Result """
                folder_path = self.dir
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                with open(folder_path, 'a+') as f:
                    if Clf_loc is not None:
                        print('adpt_fs_clf_loc:', tuple((*adpt_fs_loc, *Clf_loc)), file=f)
                    else:
                        print('adpt_fs_clf_loc:', tuple(adpt_fs_loc), file=f)
                    print('AUC and (1-sqrt(AUC)):', mul_res, file=f)
                    print('MulObj_AUC_F1:', MulObj_AUC_F1, file=f)
                    print('Parameter:', best, file=f)


            if self.FinalES != {}:
                tabuList = dict()
                tabuList.update(self.tabuList)

                for key in self.FinalES:
                    if key in tabuList:
                        if self.FinalES[key][0][0] < tabuList[key][0][0]:
                            tabuList[key] = self.FinalES[key]
                    else:
                        tabuList.update(self.FinalES)
                self.tabuList.clear()
                self.tabuList.update(tabuList)

            clf_loc.pop()
            self.clf_loc = clf_loc

        return


    @func_set_timeout(ESTimeBudget)
    def EnsembleSelection(self):

        filtered_EnsemblePool = dict(sorted(
            ((k, v) for k, v in self.EnsemblePool.items() if v[0][0] < -0.5),
            key=lambda x: x[1][0][0]
        )[:self.EnsemblePool_Size])

        if len(filtered_EnsemblePool) < 1:
            return

        EStime = time.time()
        ES_num = 0

        while time.time() - EStime < ESTimeBudget:
            self.EnsembleSearch(filtered_EnsemblePool, ES_num)
            ES_num += 1
            if ES_num > len(filtered_EnsemblePool) - 1:
                ES_num = 0

        return
