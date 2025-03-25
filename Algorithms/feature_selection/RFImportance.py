from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class RFImportance():
    def __init__(self, RF_n_estimators=10, RFmax_depth =10, RFImportance_threshold='mean'):
        self.RF_n_estimators = RF_n_estimators
        self.RFmax_depth = RFmax_depth
        self.RFImportance_threshold = RFImportance_threshold


    def RFImportance_featureselect(self, Xs, Ys, RF_n_estimators, RFmax_depth, RFImportance_threshold):
        model = RandomForestClassifier(n_estimators=RF_n_estimators, max_depth=RFmax_depth,
                                       n_jobs=-1)
        model.fit(Xs, Ys)


        feature_selection = SelectFromModel(model, threshold=RFImportance_threshold, prefit=True)
        feats = feature_selection.get_support()

        return feats


    def run(self, Xs, Ys, Xt, Yt):
        feats = self.RFImportance_featureselect(Xs, Ys, self.RF_n_estimators,
                                                self.RFmax_depth,
                                                self.RFImportance_threshold)
        xs = Xs[:, feats]
        Xt = Xt[:, feats]

        return xs, Ys, Xt, Yt



