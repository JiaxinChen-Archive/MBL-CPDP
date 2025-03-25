import numpy as np
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from sklearn.metrics import mutual_info_score
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import train_test_split

from Algorithms.classifier import *
from Algorithms.feature_selection import LASSO, PCAmining, RFImportance, FeSCH, FSFilter
from Algorithms.transfer import TCAplus, GIS, UM, CDE_SMOTE, FSS_bagging, HISNN, multiple_components_weights, \
    Value_Cognitive_Boosting, NNfilter, CLIFE_MORPH, Training_Data_Selection
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, \
    precision_score, matthews_corrcoef, recall_score




class cpdp_Ensemble(object):
    def __init__(self, kernelType='linear', dim=5, lamb=1, gamma=1,  # TCA
                 NNn_neighbors=5, NNmetric='euclidean',  # NNfilter
                 pcaDim=5,  # PCAmining
                 pvalue=0.05,  # Universal
                 mProb=0.05, mCount=5, popsize=30, chrmsize=0.02, numgens=20, numparts=5,  # GIS
                 CDE_k=3, CDE_metric='euclidean',  # CDE_SMOTE
                 Clife_n=10, Cliff_alpha=0.15, Clife_beta=0.35, percentage=0.8,  # CLIFE_SMOTE
                 Fesch_nt=1, Fesch_strategy='SFD',  # FeSCH
                 FSS_topn=10, FSS_ratio=0.1, FSS_score_thre=0.5,  # FSS_bagging
                 HISNNminham=1.0,  # HISNN
                 MCW_k=4, MCW_sigmma=1.0, MCW_lamb=1.0,  # MCWs
                 TD_strategy='NN', TD_num=3,  # Training-Data-Selection
                 VCB_M=30, VCB_lamb=1.0,  # VCB
                 RFn_estimators=10, RFcriterion='gini', RFmax_features=1.0, RFmin_samples_split=2,
                 RFmin_samples_leaf=1,  # RF
                 DTcriterion='gini', DTmax_features=1.0, DTsplitter='best',
                 DTmin_samples_split=2, DTmin_samples_leaf=1,  # Decision-Tree
                 KNNneighbors=5, KNNp=2,  # KNN
                 penalty='l2', lrC=1.0, maxiter=100, fit_intercept=True,  # Logistic regression
                 Ridge_alpha=0.001, Ridge_fit=False, Ridge_tol=1e-3,  # Ridge
                 PAC_c=1e-3, PAC_fit=False, PAC_tol=1e-3, PAC_loss='hinge',
                 Per_penalty='l2', Per_alpha=1, Per_fit=False, Per_tol=1e-3,
                 MLP_hidden=100, MLP_activation='relu', MLP_maxiter=200, solver='adam',
                 RNC_radius=1, RNC_weights='uniform',
                 NCC_metric='euclidean', NCC_shrink_thre=0,
                 EX_criterion='gini', EX_splitter='best', EX_max_feature=1.0, EX_min_split=2,
                 EX_min_leaf=1,
                 ada_n=100, ada_learning_rate=0.1,
                 bag_n=100, bag_max_samples=1.0, bag_max_features=1.0,
                 EXs_criterion='gini', EXs_max_feature=1.0, EXs_min_samples_split=2, EXs_min_samples_leaf=1,
                 EXs_n_estimator=100,
                 LASSOC = 1, LASSOPenalty ='l1',
                 RF_n_estimators = 10, RFmax_depth = 10,  RFImportance_threshold = 0.1,
                 FS_threshold = 0.9, FS_Strategy = 'Variance',
                 clf='SVM',
                 adpt='TCA',  # TL and Classifier
                 fs='LASSO',
                 clf_loc=[1, 2, 3],
                 earlystop=0.5,
                 repeat=5
                 ):
        # the number of repetition of algorithm
        self.repeat = repeat

        self.clfType = clf
        self.adpt = adpt
        self.fsType = fs

        self.clf_loc = clf_loc
        self.earlystop = earlystop

        # Universal
        self.pvalue = pvalue
        # TCA
        self.kernelType = kernelType
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        # PCAmining
        self.pcaDim = pcaDim
        # GIS
        self.mProb = mProb
        self.mCount = mCount
        self.popsize = popsize
        self.chrmsize = chrmsize
        self.numgens = numgens
        self.numparts = numparts
        # NNfilter
        self.NNn_neighbors = NNn_neighbors
        self.NNmetric = NNmetric
        # CDE-SMOTE
        self.CDE_k = CDE_k
        self.CDE_metric = CDE_metric
        # CLIFE-MORPH
        self.Cliff_alpha = Cliff_alpha
        self.Clife_beta = Clife_beta
        self.Clife_n = Clife_n
        self.percentage = percentage
        # FeSCH
        self.Fesch_nt = Fesch_nt
        self.Fesch_strategy = Fesch_strategy
        # FSS_bagging
        self.FSS_topn = FSS_topn
        self.FSS_ratio = FSS_ratio
        self.FSS_score_thre = FSS_score_thre
        # HISNN
        self.HISNNminham = HISNNminham
        # MCWs
        self.MCW_k = MCW_k
        self.MCW_lamb = MCW_lamb
        self.MCW_sigmma = MCW_sigmma
        # TD
        self.TD_num = TD_num
        self.TD_strategy = TD_strategy
        # VCB
        self.VCB_lamb = VCB_lamb
        self.VCB_M = VCB_M
        # SVM
        self.SVCkernel = 'linear'
        self.coef0 = 0
        self.gamma = 1
        self.degree = 3
        self.svmC = 1
        self.svmMaxiter = -1
        # KNN
        self.KNNneighbors = KNNneighbors
        self.KNNp = KNNp
        # RF
        self.RFn_estimators = RFn_estimators
        self.RFcriterion = RFcriterion
        self.RFmax_features = RFmax_features
        self.RFmin_samples_split = RFmin_samples_split
        self.RFmin_samples_leaf = RFmin_samples_leaf
        # NB
        self.NBType = 'gaussian'
        self.alpha = 1.0
        self.norm = True
        # Decision Tree
        self.DTsplitter = DTsplitter
        self.DTcriterion = DTcriterion
        self.DTmax_features = DTmax_features
        self.DTmin_samples_leaf = DTmin_samples_leaf
        self.DTmin_samples_split = DTmin_samples_split
        # LR
        self.penalty = penalty
        self.lrC = lrC
        self.maxiter = maxiter
        self.fit_intercept = fit_intercept
        # Ridge
        self.Ridge_alpha = Ridge_alpha
        self.Ridge_fit = Ridge_fit
        self.Ridge_tol = Ridge_tol
        # PAC
        self.PAC_c = PAC_c
        self.PAC_fit = PAC_fit
        self.PAC_tol = PAC_tol
        self.PAC_loss = PAC_loss
        # Perceptron
        self.Per_penalty = Per_penalty
        self.Per_fit = Per_fit
        self.Per_alpha = Per_alpha
        self.Per_tol = Per_tol
        # MLP
        self.MLP_hidden = MLP_hidden
        self.MLP_activate = MLP_activation
        self.MLP_maxiter = MLP_maxiter
        # RNC
        self.RNC_radius = RNC_radius
        self.RNC_weight = RNC_weights
        # NCC
        self.NCC_metric = NCC_metric
        self.NCC_shrink_thre = NCC_shrink_thre
        # EXtree
        self.EX_criterion = EX_criterion
        self.EX_splitter = EX_splitter
        self.EX_max_feature = EX_max_feature
        self.EX_min_leaf = EX_min_leaf
        self.EX_min_split = EX_min_split
        # adaBoost
        self.ada_n = ada_n
        self.ada_learning_rate = ada_learning_rate
        # bagging
        self.bag_n = bag_n
        self.bag_max_feature = bag_max_features
        self.bag_max_samples = bag_max_samples
        # EXs
        self.EXs_criterion = EXs_criterion
        self.EXs_max_feature = EXs_max_feature
        self.EXs_min_leaf = EXs_min_samples_leaf
        self.EXs_min_split = EXs_min_samples_split
        self.EXs_n_estimator = EXs_n_estimator
        # LASSO
        self.LASSOC = LASSOC
        self.LASSOPenalty = LASSOPenalty
        # RandomForestImportance
        self.RF_n_estimators = RF_n_estimators
        self.RFmax_depth = RFmax_depth
        self.RFImportance_threshold = RFImportance_threshold
        # FSFilter
        self.FS_threshold = FS_threshold
        self.FS_Strategy = FS_Strategy
        self.res = []



    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])



    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out


    def set_params_es(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        if 'svmC' in params:
            self.svmC = params['svmC']
            self.SVCkernel = params['SVCkernel']['kernel']
            self.svmMaxiter = params['SVCkernel']['max_iter']

            if self.SVCkernel == 'rbf':
                self.gamma = params['SVCkernel']['rbfgamma']
            elif self.SVCkernel == 'sigmoid':
                self.gamma = params['SVCkernel']['siggamma']
                self.coef0 = params['SVCkernel']['sigcoef0']
            elif self.SVCkernel == 'poly':
                self.gamma = params['SVCkernel']['polygamma']
                self.degree = params['SVCkernel']['degree']
                self.coef0 = params['SVCkernel']['polycoef0']
            params.pop('SVCkernel')
            params.pop('svmC')

        if 'NBparam' in params:
            self.NBType = params['NBparam']['NBType']
            if self.NBType == 'multinomial':
                self.alpha = params['NBparam']['malpha']
            elif self.NBType == 'complement':
                self.alpha = params['NBparam']['calpha']
                self.norm = params['NBparam']['norm']
            params.pop('NBparam')

        if 'TDparam' in params:
            self.TD_strategy = params['TDparam']['TD_strategy']
            if self.TD_strategy == 'NN':
                self.TD_num = params['TDparam']['TD_num']
            params.pop('TDparam')

        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self



    def classifier_ensemble(self):
        classifiers = []
        classifiers_name = []

        if any(clf == 'RF' for clf in self.clfType):
            classifiers.append(RandomForestClassifier(n_estimators=self.RFn_estimators, criterion=self.RFcriterion,
                                                      max_features=self.RFmax_features,
                                                      min_samples_split=self.RFmin_samples_split,
                                                      min_samples_leaf=self.RFmin_samples_leaf))
            classifiers_name.append('RF')

        if any(clf == 'SVM' for clf in self.clfType):
            classifiers.append(SVC(kernel=self.SVCkernel, C=self.svmC, degree=self.degree, coef0=self.coef0,
                                   gamma=self.gamma, max_iter=10))
            classifiers_name.append('SVM')

        if any(clf == 'KNN' for clf in self.clfType):
            classifiers.append(KNeighborsClassifier(n_neighbors=self.KNNneighbors, p=self.KNNp))
            classifiers_name.append('KNN')

        if any(clf == 'NB' for clf in self.clfType):
            if self.NBType == 'gaussian':
                classifiers.append(GaussianNB())
            elif self.NBType == 'multinomial':
                classifiers.append(MultinomialNB(alpha=self.alpha))
            elif self.NBType == 'complement':
                classifiers.append(ComplementNB(alpha=self.alpha, norm=self.norm))
            classifiers_name.append('NB')

        if any(clf == 'DT' for clf in self.clfType):
            classifiers.append(DecisionTreeClassifier(criterion=self.DTcriterion, splitter=self.DTsplitter,
                                                      max_features=self.DTmax_features,
                                                      min_samples_split=self.DTmin_samples_split,
                                                      min_samples_leaf=self.DTmin_samples_leaf))
            classifiers_name.append('DT')

        if any(clf == 'LR' for clf in self.clfType):
            classifiers.append(LogisticRegression(C=self.lrC, penalty=self.penalty, fit_intercept=self.fit_intercept,
                                                  max_iter=self.maxiter))
            classifiers_name.append('LR')

        if any(clf == 'Ridge' for clf in self.clfType):
            classifiers.append(RidgeClassifier(alpha=self.Ridge_alpha, fit_intercept=self.Ridge_fit, tol=self.Ridge_tol))
            classifiers_name.append('Ridge')


        if any(clf == 'PAC' for clf in self.clfType):
            classifiers.append(PassiveAggressiveClassifier(C=self.PAC_c, fit_intercept=self.PAC_fit,
                                                           tol=self.PAC_tol, loss=self.PAC_loss))
            classifiers_name.append('PAC')

        if any(clf == 'Perceptron' for clf in self.clfType):
            classifiers.append(Perceptron(penalty=self.Per_penalty, fit_intercept=self.Per_fit,
                                          tol=self.Per_tol, alpha=self.Per_alpha))
            classifiers_name.append('Perceptron')

        if any(clf == 'MLP' for clf in self.clfType):
            classifiers.append(MLPClassifier(hidden_layer_sizes=self.MLP_hidden, activation=self.MLP_activate,
                                             max_iter=self.MLP_maxiter))
            classifiers_name.append('MLP')

        if any(clf == 'RNC' for clf in self.clfType):
            classifiers.append(RadiusNeighborsClassifier(radius=self.RNC_radius, weights=self.RNC_weight))
            classifiers_name.append('RNC')

        if any(clf == 'NCC' for clf in self.clfType):
            classifiers.append(NearestCentroid(metric=self.NCC_metric, shrink_threshold=self.NCC_shrink_thre))
            classifiers_name.append('NCC')

        if any(clf == 'EXtree' for clf in self.clfType):
            classifiers.append(ExtraTreeClassifier(criterion=self.EX_criterion, splitter=self.EX_splitter,
                                                   max_features=self.EX_max_feature,
                                                   min_samples_leaf=self.EX_min_leaf, min_samples_split=self.EX_min_split))
            classifiers_name.append('EXtree')

        if any(clf == 'adaBoost' for clf in self.clfType):
            classifiers.append(AdaBoostClassifier(n_estimators=self.ada_n, learning_rate=self.ada_learning_rate))
            classifiers_name.append('adaBoost')

        if any(clf == 'bagging' for clf in self.clfType):
            classifiers.append(BaggingClassifier(n_estimators=self.bag_n, max_features=self.bag_max_feature,
                                                 max_samples=self.bag_max_samples))
            classifiers_name.append('bagging')

        if any(clf == 'EXs' for clf in self.clfType):
            classifiers.append(ExtraTreesClassifier(n_estimators=self.EXs_n_estimator, criterion=self.EXs_criterion,
                                                    max_features=self.EXs_max_feature,
                                                    min_samples_split=self.EXs_min_split, min_samples_leaf=self.EXs_min_leaf))
            classifiers_name.append('EXs')

        return classifiers, classifiers_name



    def fs_ensemble(self, Xsource, Ysource, Xtarget, Ytarget):
        Xs, Ys, Xt, Yt = Xsource, Ysource, Xtarget, Ytarget

        if self.fsType == 'LASSO':
            self.FS = LASSO.LASSO(LASSOC=self.LASSOC, LASSOPenalty=self.LASSOPenalty)


        if self.fsType == 'PCAmining':
            self.FS = PCAmining.PCAmining(dim=self.pcaDim)


        if self.fsType == 'RFImportance':
            self.FS = RFImportance.RFImportance(RF_n_estimators=self.RF_n_estimators,
                                                RFmax_depth=self.RFmax_depth,
                                                RFImportance_threshold=self.RFImportance_threshold)


        if self.fsType == 'FeSCH':
            self.FS = FeSCH.FeSCH(strategy=self.Fesch_strategy, nt=self.Fesch_nt)


        if self.fsType == 'FSFilter':
            self.FS = FSFilter.FSFilter(FS_threshold=self.FS_threshold, FS_Strategy=self.FS_Strategy)


        if self.fsType != 'None':
            Xs, Ys, Xt, Yt = self.FS.run(Xsource, Ysource, Xtarget, Ytarget)

        return Xs, Ys, Xt, Yt



    def ds_ensemble(self, sclf):

        DA = None

        if self.adpt == 'UM':
            DA = UM.Universal(pvalue=self.pvalue)


        if self.adpt == 'TCAplus':
            DA = TCAplus.TCA(kernelType=self.kernelType, dim=self.dim, lamb=self.lamb, gamma=self.gamma)


        if self.adpt == 'CLIFE':
            DA = CLIFE_MORPH.CLIFE_MORPH(n=self.Clife_n,
                                         alpha=self.Cliff_alpha, beta=self.Clife_beta,
                                         percentage=self.percentage)

        if self.adpt == 'TD':
            DA = Training_Data_Selection.TDS(strategy=self.TD_strategy, expected_num=self.TD_num)

        if self.adpt == 'NNfilter':
            DA = NNfilter.NNfilter(n_neighbors=self.NNn_neighbors, metric=self.NNmetric)


        if self.adpt == 'GIS':
            DA = GIS.GIS(model=sclf, model_name=self.clfType, mProb=self.mProb, mCount=self.mCount,
                         popsize=self.popsize,
                         chrmsize=self.chrmsize,
                         numgens=self.numgens, numparts=self.numparts)


        if self.adpt == 'CDE_SMOTE':
            DA = CDE_SMOTE.CDE_SMOTE(model=sclf, k=self.CDE_k, metric=self.CDE_metric)


        if self.adpt == 'FSS_bagging':
            DA = FSS_bagging.FSS_bagging(model=sclf, FSS=self.FSS_ratio,
                                         topN=self.FSS_topn,
                                         score_thre=self.FSS_score_thre)


        if self.adpt == 'HISNN':
            DA = HISNN.HISNN(model=sclf, MinHam=self.HISNNminham)


        if self.adpt == 'MCWs':
            DA = multiple_components_weights.MCWs(model=sclf, k=self.MCW_k, lamb=self.MCW_lamb,
                                                  sigmma=self.MCW_sigmma)

        if self.adpt == 'VCB':
            DA = Value_Cognitive_Boosting.VCB(model=sclf, M=self.VCB_M, lamb=self.VCB_lamb)

        if self.adpt == 'None':
            DA = None

        return DA



    def run_ensemble(self, Xsource, Ysource, Xtarget, Ytarget, loc, ESSelected_Size, MultiObj=False):

        if len(self.clf_loc) > 1:

            meta = LogisticRegression()
            classifiers, classifiers_name = self.classifier_ensemble()


            predictions = {}
            score = {}
            for classifier in classifiers[:]:
                classifier.fit(Xsource, Ysource)
                predictions[classifier] = classifier.predict(Xsource)
                score[classifier] = roc_auc_score(Ysource, classifier.predict(Xsource))


                if score[classifier] < self.earlystop:
                    classifiers.remove(classifier)
                    del predictions[classifier]
                    del score[classifier]

                    if len(classifiers) < 1:
                        break


            name_mapping = dict(zip(classifiers, classifiers_name))
            sorted_classifiers = sorted(score, key=score.get, reverse=True)
            sorted_classifiers_name = [name_mapping[clf] for clf in sorted_classifiers if clf in name_mapping]

            clf_name_to_loc = {clf_name: self.clf_loc[self.clfType.index(clf_name)] for clf_name in
                               sorted_classifiers_name}
            self.clf_loc = [clf_name_to_loc[clf] for clf in sorted_classifiers_name]

            if len(self.clf_loc) <= ESSelected_Size:
                classifiers = sorted_classifiers
            else:
                num_best_classifiers = max(1, ESSelected_Size // 2)
                best_classifiers = sorted_classifiers[:num_best_classifiers]
                remaining_classifiers = [clf for clf in sorted_classifiers if clf not in best_classifiers]
                selected_classifiers = list(best_classifiers)

                while len(selected_classifiers) < ESSelected_Size and remaining_classifiers:
                    max_q_statistic = 0
                    best_classifier_for_diversity = None

                    for classifier1, classifier2 in combinations(remaining_classifiers, 2):
                        mutual_info = mutual_info_score(predictions[classifier1], predictions[classifier2])
                        q_statistic = mutual_info / (1 - mutual_info)

                        if q_statistic > max_q_statistic:
                            max_q_statistic = q_statistic
                            best_classifier_for_diversity = classifier1 if score[classifier1] > score[
                                classifier2] else classifier2

                    if best_classifier_for_diversity:
                        selected_classifiers.append(best_classifier_for_diversity)
                        remaining_classifiers.remove(best_classifier_for_diversity)

                classifiers = selected_classifiers

            sclf = StackingCVClassifier(classifiers=deepcopy(classifiers), meta_classifier=meta, cv=10,
                                        use_features_in_secondary=True)
            final_classifier_names = []

            for classifier in classifiers:
                classifier_name = type(classifier).__name__
                final_classifier_names.append(classifier_name)

            sorted_classifiers_name1 = [name_mapping[clf] for clf in classifiers if clf in name_mapping]
            self.clf_loc = [clf_name_to_loc[clf] for clf in sorted_classifiers_name1]

            DA = self.ds_ensemble(sclf)
            self.DA = DA

        else:
            classifiers, final_classifier_names = self.classifier_ensemble()
            DA = self.ds_ensemble(classifiers[0])
            self.DA = DA
            sclf = classifiers[0]



        if self.adpt in ['GIS', 'CDE_SMOTE', 'HISNN', 'VCB']:
            for i in range(self.repeat):
                Xs, Ys, Xt, Yt = self.fs_ensemble(Xsource, Ysource, Xtarget, Ytarget)

                res = self.DA.run(Xs=Xs, Xt=Xt, Ys=Ys, Yt=Yt, MultiObj=MultiObj)
                self.res.append(res)
                if res[0] < self.earlystop:
                    return np.mean(np.asarray(self.res), axis=0), self.clf_loc

            return np.mean(np.asarray(self.res), axis=0), self.clf_loc


        if self.adpt == 'FSS_bagging':
            for i in range(self.repeat):
                Xs, Ys, Xt, Yt = self.fs_ensemble(Xsource, Ysource, Xtarget, Ytarget)
                res = self.DA.run(Xsource=Xs, Xtarget=Xt, Ysource=Ys, Ytarget=Yt,
                                  loc=loc, MultiObj=MultiObj)

                self.res.append(res)

                if res[0] < self.earlystop:
                    return np.mean(np.asarray(self.res), axis=0), self.clf_loc

            return np.mean(np.asarray(self.res), axis=0), self.clf_loc


        if self.adpt == 'MCWs':
            for i in range(self.repeat):
                Xs, Ys, Xt, Yt = self.fs_ensemble(Xsource, Ysource, Xtarget, Ytarget)
                Train, Xt, Ltrain, Yt = train_test_split(Xt, Yt, test_size=0.9)

                res = self.DA.run(Xs=Xs, Ys=Ys, test=Xt, l_test=Yt, train=Train,
                                  l_train=Ltrain, loc=loc, MultiObj=MultiObj)
                self.res.append(res)

                if res[0] < self.earlystop:
                    return np.mean(np.asarray(self.res), axis=0), self.clf_loc


            return np.mean(np.asarray(self.res), axis=0), self.clf_loc



        for i in range(self.repeat):

            Xs, Ys, Xt, Yt = self.fs_ensemble(Xsource, Ysource, Xtarget, Ytarget)
            if self.adpt in ['UM', 'TD', 'FSS_bagging']:
                Xs, Ys, Xt, Yt = self.DA.run(Xs, Ys, Xt, Yt, loc)
            else:
                if self.adpt == 'NNfilter':
                    Train, Xt, Ltrain, Yt = train_test_split(Xt, Yt, test_size=0.9)
                    Xs, Ys, train, ltrain = self.DA.run(Xs, Ys, Train, Ltrain)
                    Xt = np.log(Xt + 1)
                elif self.adpt == 'None':
                    pass
                else:
                    Xs, Ys, Xt, Yt = self.DA.run(Xs, Ys, Xt, Yt)

            if MultiObj == False:
                if np.asarray(Xs).ndim <= 1 or len(np.unique(Ys)) <= 1:
                    res = [0]
                    self.res.append(res)
                else:
                    try:
                        sclf.fit(Xs, Ys)
                        predict = sclf.predict(Xt)
                        res = [roc_auc_score(Yt, predict)]
                        self.res.append(res)
                    except:
                        res = [0]
                        self.res.append(res)

            else:
                if np.asarray(Xs).ndim <= 1 or len(np.unique(Ys)) <= 1:
                    res = [0, 0, 0, 0, 0, 0, 0]
                    self.res.append(res)
                else:
                    try:
                        sclf.fit(Xs, Ys)
                        predict = sclf.predict(Xt)

                        if -1 in Yt and 0 in predict:
                            Yt[Yt == -1] = 0
                        elif 0 in Yt and -1 in predict:
                            predict[predict == -1] = 0

                        # AUC, F1, ACC, Recall, ERR, PREC, MCC
                        res = [0, 0, 0, 0, 0, 0, 0]

                        try:
                            res[0] = roc_auc_score(Yt, predict)
                        except Exception:
                            pass

                        try:
                            res[1] = f1_score(Yt, predict)
                        except Exception:
                            pass

                        try:
                            res[2] = accuracy_score(Yt, predict)
                        except Exception:
                            pass

                        try:
                            res[3] = recall_score(Yt, predict)
                        except Exception:
                            pass

                        if res[2] != 0:
                            res[4] = 1 - res[2]

                        try:
                            res[5] = precision_score(Yt, predict)
                        except Exception:
                            pass

                        try:
                            res[6] = matthews_corrcoef(Yt, predict)
                        except Exception:
                            pass

                        self.res.append(res)

                    except:
                        res = [0, 0, 0, 0, 0, 0, 0]
                        self.res.append(res)


            if res[0] < self.earlystop:
                return np.mean(np.asarray(self.res), axis=0), self.clf_loc

        return np.mean(np.asarray(self.res), axis=0), self.clf_loc


