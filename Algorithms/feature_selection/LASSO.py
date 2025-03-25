from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel as SF



class LASSO():

    def __init__(self, LASSOC=1, LASSOPenalty='l1'):
        self.LASSOC = LASSOC
        self.LASSOPenalty = LASSOPenalty


    def Lasso_featureselect(self, Xs, Ys):
        X_train = Xs
        y_train = Ys

        # linear models benefit from feature scaling
        scaler = RobustScaler()
        scaler.fit(X_train)

        # fit the LR model
        sel_ = SF(LogisticRegression(C=self.LASSOC, penalty=self.LASSOPenalty))  # penalty=l1 or l2
        sel_.fit(scaler.transform(X_train), y_train)

        # make a list with the selected features
        selected_feat = sel_.get_support()

        feats = selected_feat == True

        return feats


    def run(self, Xs, Ys, Xt, Yt):
        feats = self.Lasso_featureselect(Xs, Ys)
        xs = Xs[:, feats]
        Xt = Xt[:, feats]

        return xs, Ys, Xt, Yt



