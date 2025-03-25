import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile

"""Filter Method"""

class FSFilter():
    def __init__(self, FS_threshold=0.8, FS_Strategy='Variance'):
        self.FS_threshold = FS_threshold
        self.FS_Strategy = FS_Strategy


    def constant_feature_detect(self, data, threshold=0.8):
        """ detect features that show the same value for the
        majority/all of the observations (constant/quasi-constant features)

        Parameters
        ----------
        data : pd.Dataframe
        threshold : threshold to identify the variable as constant

        Returns
        -------
        list of variables names
        """

        data_copy = pd.DataFrame(data)
        quasi_constant_feature = []
        for feature in data_copy.columns:
            predominant = (data_copy[feature].value_counts() / np.float(
                len(data_copy))).sort_values(ascending=False).values[0]
            if predominant >= threshold:
                quasi_constant_feature.append(feature)

        Xsource_len = data.shape[1]
        feats = np.ones(Xsource_len, dtype=bool)
        feats[quasi_constant_feature] = False
        return feats


    def mutual_info(self, X, y, select_k=10):

        Xsource_len = X.shape[1]
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        if select_k >= 1:
            sel_ = SelectKBest(mutual_info_classif, k=select_k).fit(X, y)
            col = X.columns[sel_.get_support()]

        elif 0 < select_k < 1:
            sel_ = SelectPercentile(mutual_info_classif, percentile=select_k * 100).fit(X, y)
            col = X.columns[sel_.get_support()]

        else:
            raise ValueError("select_k must be a positive number")

        feats = np.zeros(Xsource_len, dtype=bool)
        feats[col] = True
        return col


    def chi_square_test(self, X, y, select_k=10):

        """
        Compute chi-squared stats between each non-negative feature and class.
        This score should be used to evaluate categorical variables in a classification task
        """
        Xsource_len = X.shape[1]
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        if select_k >= 1:
            sel_ = SelectKBest(chi2, k=select_k).fit(X, y)
            col = X.columns[sel_.get_support()]
        elif 0 < select_k < 1:
            sel_ = SelectPercentile(chi2, percentile=select_k * 100).fit(X, y)
            col = X.columns[sel_.get_support()]
        else:
            raise ValueError("select_k must be a positive number")

        feats = np.zeros(Xsource_len, dtype=bool)
        feats[col] = True

        return feats


    def FSFilter_featureselect(self, Xs, Ys):
        X_train = Xs
        y_train = Ys

        # Variance method
        if self.FS_Strategy =='Variance':
            feats = self.constant_feature_detect(data=X_train, threshold=self.FS_threshold)

        # Mutual Information Filter
        if self.FS_Strategy == 'MIF':
            feats = self.mutual_info(X=X_train, y=y_train, select_k=self.FS_threshold)

        # Chi-Square Filter
        if self.FS_Strategy == 'CSF':
            feats = self.chi_square_test(X=X_train, y=y_train, select_k=self.FS_threshold)

        return feats


    def run(self, Xs, Ys, Xt, Yt):
        feats = self.FSFilter_featureselect(Xs, Ys)
        xs = Xs[:, feats]
        Xt = Xt[:, feats]

        return xs, Ys, Xt, Yt



