

class XySet(object):
    def __init__(self, X, y, X_train, y_train, X_test, y_test, X_submit,
                 id_submit, cv_folds, y_eval_list, y_eval, i_eval):
        """
        this class stores the various data required for analysis and hyepr-optimisation

        :param X: X features for the complete dataset (excluding bechnmark/test and submit)
        :param y: y for this complete set
        :param X_train: X features for the train set after holdout
        :param y_train: y for this set
        :param X_test: X features for the holdout set or test set in benchmark
        :param y_test: y for this set
        :param X_submit: X features for the submit set (competition mode only)
        :param id_submit: y for this set
        :param cv_folds: cross validation folds
        :param y_eval_list: list of folds, with indexes for each fold
        :param y_eval: y for the eval set
        :param i_eval: indexes for the eval set
        """
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_submit = X_submit
        self.id_submit = id_submit
        self.cv_folds = cv_folds
        self.y_eval_list = y_eval_list
        self.y_eval = y_eval
        self.i_eval = i_eval