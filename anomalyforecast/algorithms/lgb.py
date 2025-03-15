from typing import Optional, Dict

import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit

VALIDATION_SIZE = 0.3

PARAMETER_SET = \
    dict(num_leaves=[5, 10, 15, 30],
         max_depth=[-1, 3, 5, 10, 15],
         lambda_l1=[0, 0.1, 1, 100],
         lambda_l2=[0, 0.1, 1, 100],
         learning_rate=[0.05, 0.1, 0.2],
         min_child_samples=[15, 30, 50, 100],
         n_jobs=[1],
         linear_tree=[True, False],
         boosting_type=['gbdt'])

CLF_PARAMS = {}


class LGBMRegression(BaseEstimator, RegressorMixin):
    """
    Implementation of lightgbm regression model, which includes optuna-based optimization
    """

    def __init__(self, params: Optional[Dict] = None, n_trials: int = 10):
        self.model = None
        self.n_trials = n_trials
        self.params = params

    def fit(self, X, y):
        if self.params is None:
            self.params = self.optimize_params_regr(X, y, n_trials=self.n_trials)

        dtrain = lgb.Dataset(X, label=y)

        self.model = lgb.train(self.params, dtrain)

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def objective_r(trial, X, y):
        """
        Training plus validation cycle of optuna for optimization
        """
        train_x, valid_x, train_y, valid_y = \
            train_test_split(X, y, test_size=VALIDATION_SIZE, shuffle=True)
        dtrain = lgb.Dataset(train_x, label=train_y)

        param = {
            'objective': 'regression',
            'metric': 'mean_absolute_error',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'linear_tree': True,
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        gbm = lgb.train(param, dtrain)
        preds = gbm.predict(valid_x)

        err = mean_absolute_error(valid_y, preds)

        return err

    @classmethod
    def optimize_params_regr(cls, X, y, n_trials: int):
        """
        Parameter optimization using optuna

        :param X: Explanatory variables (lagged features)
        :param y: Target variable
        :param n_trials: Number of trials for optimization

        :return: Best configuration found
        """
        func = lambda trial: cls.objective_r(trial, X, y)

        study = optuna.create_study(direction='minimize')
        study.optimize(func, n_trials=n_trials)

        trial = study.best_trial

        return trial.params


class LGBMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, iters: int = 50, params=CLF_PARAMS):
        self.model = None
        self.iters = iters
        self.estimator = lgb.LGBMClassifier(n_jobs=1)
        self.params = params
        self.parameters = \
            dict(num_leaves=[5, 10, 15, 30],
                 max_depth=[-1, 3, 5, 10],
                 lambda_l1=[0, 0.1, 1, 100],
                 lambda_l2=[0, 0.1, 1, 100],
                 learning_rate=[0.05, 0.1, 0.2],
                 min_child_samples=[15, 30, 50, 100],
                 n_jobs=[1],
                 linear_tree=[True, False],
                 boosting_type=['gbdt'],
                 num_boost_round=[25, 50, 100])

    def fit(self, X, y=None):
        if self.params is None:
            self.model = RandomizedSearchCV(estimator=self.estimator,
                                            param_distributions=self.parameters,
                                            scoring='roc_auc',
                                            n_iter=self.iters,
                                            n_jobs=1,
                                            refit=True,
                                            verbose=0,
                                            cv=ShuffleSplit(n_splits=1, test_size=0.3),
                                            random_state=123)

            self.model.fit(X, y)
        else:
            self.model = lgb.LGBMClassifier(**self.params, verbose=-1)
            self.model.fit(X, y)

    def predict(self, X):
        y_hat = self.model.predict(X)

        return y_hat

    def predict_proba(self, X):
        y_hat = self.model.predict_proba(X)
        y_hat_df = pd.DataFrame(y_hat, columns=self.model.classes_)

        y_hat_df = y_hat_df.values

        return y_hat_df

    @classmethod
    def optimize_params(cls, X, y):

        mod = cls(iters=200, params=None)

        mod.fit(X, y)

        return mod.model.best_params_
