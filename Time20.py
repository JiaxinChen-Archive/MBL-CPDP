import os
import errno
import signal
import logging
import warnings
import traceback
import multiprocessing
#import numpy as np
import platform
import subprocess
import copy
from LS import *
from numpy import *
from Utils.File import *
from ensembleselction import *
from Algorithms.Framework import cpdp
from Utils.helper import MfindCommonMetric
from iteration_utilities import deepflatten
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from func_timeout import func_set_timeout, FunctionTimedOut


""" kill the zombie child process """


def wait_child(signum, frame):
    logging.info('receive SIGCHLD')
    try:
        while True:
            # -1 表示任意子进程
            # os.WNOHANG 表示如果没有可用的需要 wait 退出状态的子进程，立即返回不阻塞
            cpid, status = os.waitpid(-1, os.WNOHANG)
            if cpid == 0:
                logging.info('no child process was immediately available')
                break
            exitcode = status >> 8
            logging.info('child process %s exit with exitcode %s', cpid, exitcode)
    except OSError as e:
        if e.errno == errno.ECHILD:
            logging.warning('current process has no existing unwaited-for child processes.')
        else:
            raise
    logging.info('handle SIGCHLD end')



#signal.signal(signal.SIGCHLD, wait_child)
# Time budge
PObudget = 20



class Llevel(object):
    """
        up    : given upper-level variables, format as {'clf':x, 'adpt':y, 'fs':z}
        params: the responding lower-level variables when given upper-level variables
        ldir  : the path where history will be saved
    """

    def __init__(self, up, params, xsource, ysource, xtarget, ytarget, loc,
                 fe, ldir, earlystop):

        self.xsource = xsource
        self.ysource = ysource
        self.xtarget = xtarget
        self.ytarget = ytarget
        self.train = None
        self.Ltrain = None
        self.adpt = up['adpt']
        self.fs = up['fs']
        self.gclf = up['clf']
        self.fe = fe
        self.loc = loc
        self.dir = ldir
        self.trails = Trials()
        self.earlystop = earlystop
        self.paramName = dict()  # the name of parameters
        self.paramType = dict()  # the type of parameters (integer, real, categery)
        self.paramRVal = dict()  # the range of parameters (origin)
        self.paramRange = dict()  # the range of parameters (transfered)
        i = 0
        for k, v in params.items():
            self.paramName[i] = k
            self.paramType[i] = v[0]
            self.paramRVal[i] = v[1]
            if v[0] == 'i' or v[0] == 'r':
                self.paramRange[i] = v[1]
            elif v[0] == 'c':
                self.paramRange[i] = [0, len(v[1]) - 1]
            i += 1

        return


    # the objective function (run the algorithm)
    def f(self, params):

        self.p = cpdp(clf=self.gclf, adpt=self.adpt, fs=self.fs, earlystop=self.earlystop, repeat=1)
        self.p.set_params(**params)
        Result = self.p.run(self.xsource, self.ysource, self.xtarget, self.ytarget, self.loc, MultiObj=True)
        res = Result[0]

        return {'loss': -res, 'status': STATUS_OK, 'result': res, 'all_res': Result}


    def ParmSpace(self):
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

        return paramSpace


    # the process of lower-level optimization (TPE)
    @func_set_timeout(PObudget)
    def ParameterOptimization(self):

        paramSpace = self.ParmSpace()

        best = fmin(self.f, space=paramSpace, algo=tpe.suggest, max_evals=self.fe,
                    trials=self.trails, show_progressbar=False)


    def run(self):

        """ Lower-level function """

        for i in range(5):
            try:
                self.ParameterOptimization()
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
                    mul_res = self.trails.best_trial['result']['all_res']
                else:
                    try:
                        inc_value = self.trails.best_trial['result']['result']
                        best = self.trails.best_trial['misc']['vals']
                        mul_res = self.trails.best_trial['result']['all_res']
                    except:
                        inc_value = 0
                        best = []
                        mul_res = [0, 0, 0, 0, 0, 0, 0]
            except:
                inc_value = 0
                best = []
                mul_res = [0, 0, 0, 0, 0, 0, 0]

            if inc_value < self.earlystop:
                break

        return -inc_value, best, mul_res



class Ulevel(object):
    """
        params:  the whole variables that contain two-level variables
        method:  the name of method that is used to perform upper-level optimization
    """

    def __init__(self, parameters, xsource, ysource, xtarget, ytarget, ULPopSize, loc, UFE=1000,
                 LFE=1000, EnsemblePool_Size=5, ESSelected_Size=3, method='vns', fname=None
                 ):

        self.xsource = xsource
        self.ysource = ysource
        self.xtarget = xtarget
        self.ytarget = ytarget
        self.ULPopSize = ULPopSize
        self.ULMultiObj = True
        self.loc = loc
        self.params = parameters
        params = parameters['up']
        self.paramName = dict()  # the name of parameters
        self.paramType = dict()  # the type of parameters (integer, real, categery, constant)
        self.paramRVal = dict()  # the range of parameters (origin)
        self.paramRange = dict()  # the range of parameters (transfered)
        i = 0
        for k, v in params.items():
            self.paramName[i] = k
            self.paramType[i] = v[0]
            self.paramRVal[i] = v[1]
            if v[0] == 'i' or v[0] == 'r':
                self.paramRange[i] = v[1]
            elif v[0] == 'c':
                self.paramRange[i] = [0, len(v[1]) - 1]
            i += 1

        self.FE = UFE
        self.LFE = LFE
        self.method = method
        self.fname = fname
        if self.fname is None:
            self.ldir = os.getcwd() + '/BL-history/lower/' + str(time.time())
        else:
            self.ldir = os.getcwd() + '/BL-history/lower/' + self.fname
        if not os.path.exists(self.ldir):
            os.makedirs(self.ldir)

        udir = os.getcwd() + '/BL-history/upper/'
        if not os.path.exists(udir):
            os.makedirs(udir)

        if not os.path.exists(udir + '/' + self.fname):
            os.makedirs(udir + '/' + self.fname)

        self.ufname = udir + '/' + self.fname + 'Time20.txt'
        self.startTime = time.time()
        self.EnsemblePool_Size = EnsemblePool_Size
        self.ESSelected_Size = ESSelected_Size
        self.earlystop = 0.5



    def f(self, x):

        para = dict()
        for i in range(len(self.paramName)):
            if self.paramType[i] == 'r':
                para[self.paramName[i]] = x[i]
            elif self.paramType[i] == 'i':
                para[self.paramName[i]] = int(round(x[i]))
            elif self.paramType[i] == 'c':
                para[self.paramName[i]] = self.paramRVal[i][int(round(x[i]))]

        if para['adpt'] in ['VCB', 'MCWs'] and para['clf'] in ['KNN', 'NCC', 'RNC', 'MLP', 'PAC', 'RF']:
            return [0, 0], {}, [0, 0, 0, 0, 0, 0, 0]

        params = dict()
        for i in range(len(self.paramName)):
            if list(para.values())[i] in ['SVM', 'TD', 'NB']:
                continue
            if list(para.values())[i] != 'None':
                params = dict(params, **(self.params['lp'][list(para.values())[i]]))

        # lower level optimization
        ex = Llevel(para, params, xsource=self.xsource, ysource=self.ysource, xtarget=self.xtarget,
                    ytarget=self.ytarget, loc=self.loc, fe=self.LFE, ldir=self.ufname, earlystop=self.earlystop)

        res, best, MulObj = ex.run()

        if res == 0:
            return [0, 0], best, MulObj
        else:
            mul_res = []
            mul_res.append(-np.abs(res))
            mul_res.append(-(1 - np.sqrt(np.abs(res))))
            return mul_res, best, MulObj


    def run(self):

        with open(self.ufname, 'a+') as f:
            print('###############################Upper Level###############################', file=f)

        if self.method == 'paratabu':
            exc = paraTabu(f=self.f, range=self.paramRange, dir=self.ufname, max=self.FE, stime=self.startTime,
                           earlystop=self.earlystop, EnsemblePool_Size=self.EnsemblePool_Size,
                           ESSelected_Size=self.ESSelected_Size,
                           ULPopSize=self.ULPopSize)
            try:
                [fres, location] = exc.run()
            except FunctionTimedOut:
                pass

            for p in exc.processList:
                try:
                    if platform.system() == "Windows":
                        # Windows System
                        subprocess.call(["taskkill", "/F", "/PID", str(p)])
                    else:
                        # Linux and the other Unix system
                        os.kill(p, signal.SIGKILL)
                except:
                    pass

            es = ES(tabuList=exc.tabuList, EnsemblePool=exc.EnsemblePool, f=self.f, range=self.paramRange,
                    rval=self.paramRVal, dir=self.ufname, allparam=self.params['lp'], earlystop=self.earlystop,
                    EnsemblePool_Size=self.EnsemblePool_Size,
                    ESSelected_Size=self.ESSelected_Size, ULPopSize=self.ULPopSize,
                    xsource=self.xsource, ysource=self.ysource, xtarget=self.xtarget, ytarget=self.ytarget,
                    loc=self.loc)
            try:
                es.EnsembleSelection()
            except FunctionTimedOut:
                pass

            es.tabuList = {k: v for k, v in es.tabuList.items() if v[0] != [0, 0]}
            allfunc = np.array(list([row[0] for row in es.tabuList.values()]))
            Nondominated = NSGAII.offspringselect([row[0] for row in allfunc], [row[1] for row in allfunc],
                                                  self.ULPopSize, FixedNum='True')
            new_dict = {list(es.tabuList.keys())[i]: list(es.tabuList.values())[i] for i in Nondominated[0]}
            fres = np.asarray(list(new_dict.values()))
            location = np.asarray(list(new_dict.keys()))


            with open(self.ufname, 'a+') as f:
                print(fres, file=f)
                print('location:', location, file=f)
                for k, v in new_dict.items():
                    print('location:', k, file=f)
                    try:
                        print('MulObj:', v[0], file=f)
                        print('Parameter:', v[1], file=f)
                        print('MulObj_AUC_F1_ACC_Recall_ERR_PREC_MCC:', v[2], file=f)
                    except:
                        print('MulObj+Parameter+MulObj_AUC_F1_ACC_Recall_ERR_PREC_MCC:', v, file=f)


            allfunc = np.array(list(fres))
            try:
                F = [row[0] for row in allfunc]
                x = [row[1] for row in allfunc]
                MulObj_AUC_F1 = [row[2] for row in allfunc]
            except:
                F = [0, 0]
                x = {}
                MulObj_AUC_F1 = [0, 0]

            return F, x, fres, location, MulObj_AUC_F1



warnings.filterwarnings('ignore')


# the function to perform a bi-level optimization
def bl(Xsource, Lsource, Xtarget, Ltarget, ULPopSize, loc, fname, method, repeat=5):
    #   parameters template: {name: (type, range)}
    up = \
        {
            'adpt': ('c',
                     ['NNfilter', 'GIS', 'UM', 'CLIFE', 'FSS_bagging',
                      'MCWs', 'TD', 'VCB', 'HISNN', 'CDE_SMOTE', 'None']),
            'fs': ('c', ['LASSO', 'PCAmining', 'RFImportance', 'FeSCH', 'FSFilter', 'None']),
            'clf': ('c', ['RF', 'KNN', 'SVM', 'LR', 'DT', 'NB', 'Ridge', 'PAC',
                          'Perceptron', 'MLP', 'RNC', 'NCC', 'EXtree', 'adaBoost', 'bagging', 'EXs'])
        }

    lp = \
        {
            'NNfilter': {'NNn_neighbors': ('i', [1, 100]),
                         'NNmetric': ('c', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])},
            'GIS': {'mProb': ('r', [0.02, 0.1]), 'chrmsize': ('r', [0.02, 0.1]), 'popsize': ('i', [2, 31, 2]),
                    'numparts': ('i', [2, 7]), 'numgens': ('i', [5, 21]), 'mCount': ('i', [3, 11])},
            'CDE_SMOTE': {'CDE_k': ('i', [1, 100]),
                          'CDE_metric': ('c', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis'])},
            'CLIFE': {'Clife_n': ('i', [1, 100]), 'Cliff_alpha': ('r', [0.05, 0.2]), 'Clife_beta': ('r', [0.2, 0.4]),
                      'percentage': ('r', [0.6, 0.9])},
            'FeSCH': {'Fesch_nt': ('i', [1, Xsource.shape[1]]), 'Fesch_strategy': ('c', ['SFD', 'LDF', 'FCR'])},
            'FSS_bagging': {'FSS_topn': ('i', [1, len(loc)]), 'FSS_ratio': ('r', [0.1, 0.9]),
                            'FSS_score_thre': ('r', [0.3, 0.7])},
            'HISNN': {'HISNNminham': ('i', [1, Xsource.shape[1]])},
            'MCWs': {'MCW_k': ('i', [2, len(loc)]), 'MCW_sigmma': ('r', [0.01, 10]), 'MCW_lamb': ('r', [1e-6, 1e2])},
            'VCB': {'VCB_M': ('i', [2, 30]), 'VCB_lamb': ('r', [0.5, 1.5])},
            'UM': {'pvalue': ('r', [0.01, 0.1])},
            'TCAplus': {'kernelType': ('c', ['primal', 'linear', 'rbf', 'sam']),
                        'dim': ('i', [5, max(Xsource.shape[1], Xtarget.shape[1])]),
                        'lamb': ('r', [1e-6, 1e2]), 'gamma': ('r', [1e-5, 1e2])},
            'PCAmining': {'pcaDim': ('i', [5, max(Xsource.shape[1], Xtarget.shape[1])])},

            'RF': {'RFn_estimators': ('i', [10, 200]), 'RFcriterion': ('c', ['gini', 'entropy']),
                   'RFmax_features': ('r', [0.2, 1.0]),
                   'RFmin_samples_split': ('i', [2, 40]), 'RFmin_samples_leaf': ('i', [1, 20])},
            'KNN': {'KNNneighbors': ('i', [1, 10]), 'KNNp': ('i', [1, 5])},
            'DT': {'DTcriterion': ('c', ['gini', 'entropy']), 'DTmax_features': ('r', [0.2, 1.0]),
                   'DTmin_samples_split': ('i', [2, 40]),
                   'DTsplitter': ('c', ['best', 'random']), 'DTmin_samples_leaf': ('i', [1, 20])},
            'LR': {'penalty': ('c', ['l1', 'l2']), 'lrC': ('r', [0.001, 10]), 'maxiter': ('i', [50, 200]),
                   'fit_intercept': ('c', [True, False])},
            'Ridge': {'Ridge_alpha': ('r', [0.001, 100]), 'Ridge_fit': ('c', [True, False]),
                      'Ridge_tol': ('r', [1e-5, 0.1])},
            'PAC': {'PAC_c': ('r', [1e-3, 100]), 'PAC_fit': ('c', [True, False]),
                    'PAC_tol': ('r', [1e-5, 0.1]), 'PAC_loss': ('c', ['hinge', 'squared_hinge'])},
            'Perceptron': {'Per_penalty': ('c', ['l1', 'l2']), 'Per_alpha': ('r', [1e-5, 0.1]),
                           'Per_fit': ('c', [True, False]), 'Per_tol': ('r', [1e-5, 0.1])},
            'MLP': {'MLP_hidden': ('i', [50, 200]),
                    'MLP_activation': ('c', ['identity', 'logistic', 'tanh', 'relu']),
                    'MLP_maxiter': ('i', [100, 250]), 'solver': ('c', ['lbfgs', 'sgd', 'adam'])},
            'RNC': {'RNC_radius': ('r', [0, 10000]), 'RNC_weights': ('c', ['uniform', 'distance'])},
            'NCC': {'NCC_metric': ('c', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis']),
                    'NCC_shrink_thre': ('r', [0, 10])},
            'EXtree': {'EX_criterion': ('c', ['gini', 'entropy']), 'EX_splitter': ('c', ['random', 'best']),
                       'EX_max_feature': ('r', [0.2, 1.0]), 'EX_min_split': ('i', [2, 40]),
                       'EX_min_leaf': ('i', [1, 20])},
            'adaBoost': {'ada_n': ('i', [10, 200]), 'ada_learning_rate': ('r', [0.01, 10])},
            'bagging': {'bag_n': ('i', [10, 200]), 'bag_max_samples': ('r', [0.7, 1.0]),
                        'bag_max_features': ('r', [0.7, 1.0])},
            'EXs': {'EXs_criterion': ('c', ['gini', 'entropy']), 'EXs_n_estimator': ('i', [10, 200]),
                    'EXs_max_feature': ('r', [0.2, 1.0]), 'EXs_min_samples_split': ('i', [2, 40]),
                    'EXs_min_samples_leaf': ('i', [1, 20])},

            'LASSO': {'LASSOC': ('r', [0.1, 1]), 'LASSOPenalty': ('c', ['l1', 'l2'])},
            'RFImportance': {'RF_n_estimators': ('i', [10, 200]), 'RFmax_depth': ('i', [10, 200]),
                             'RFImportance_threshold': ('c', ['mean', 'median'])},
            'FSFilter': {'FS_Strategy': ('c', ['Variance', 'MIF', 'CSF']), 'FS_threshold': ('r', [0.6, 0.9])}

        }

    his = []
    create_dir('resBL')
    fnameList('resBL', his)
    curr = 'resBL/' + fname + '-Time20.txt'

    print(curr)

    if curr in his:
        with open(curr, 'r') as f:
            lines = f.readlines()
            if len(lines) / 4 >= repeat:
                return
            else:
                repeat = int(repeat - len(lines) / 4)

    stime = time.time()
    for i in range(repeat):
        params = {'up': up, 'lp': lp}

        ex = Ulevel(params, xsource=Xsource, ysource=Lsource, xtarget=Xtarget, ytarget=Ltarget, ULPopSize=ULPopSize,
                    loc=loc, UFE=10000, LFE=5, EnsemblePool_Size=6, ESSelected_Size=3, method=method, fname=fname)


        res, inc, fres, location, MulObj_AUC_F1 = ex.run()

        """Final Result"""
        folder_path = ex.ufname[:-10]
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = 'fres_{}.npy'.format(i)
        file_path = folder_path + '/' + file_name
        np.save(file_path, fres)


        path = create_dir('resBL' + '/' + fname.split('\\')[0])
        with open(path + fname.split('\\')[-1] + '.txt', 'a+') as f:
            print('AUC and (1-sqrt(AUC)):', res, file=f)
            print('MulObj_AUC_F1_ACC_Recall_ERR_PREC_MCC:', MulObj_AUC_F1, file=f)
            print('location:', location.tolist(), file=f)
            print('Parameter:', inc, file=f)
            print('---------------------', '\n', file=f)



# main function
if __name__ == '__main__':
    begin_num = 1
    end_num = 20

    ULPopSize = 10    # ULPopSize = 5
    flist = []
    group = sorted(['ReLink', 'AEEEM', 'JURECZKO'])

    for g_i in range(len(group)):
        tmp = []
        fnameList('data/' + group[g_i], tmp)
        tmp = sorted(tmp)
        flist.append(tmp)

    for c in range(begin_num, end_num + 1):
        if c in range(6):
            tmp = flist[0].copy()
            target = tmp.pop(c - 1)
        if c in range(6, 18):
            tmp = flist[1].copy()
            target = tmp.pop(c - 6)
        if c in range(18, 21):
            tmp = flist[2].copy()
            target = tmp.pop(c - 18)

        Xsource, Lsource, Xtarget, Ltarget, loc = MfindCommonMetric(tmp, target, split=True)
        bl(Xsource, Lsource, Xtarget, Ltarget, ULPopSize, loc, target.split('/')[-1].split('.')[0],
           repeat=31, method='paratabu')


    print('done')
    os.wait()
    os._exit(0)



