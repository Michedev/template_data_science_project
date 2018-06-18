from sklearn.model_selection import validation_curve
from constants.dataset import TARGETVAR
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb


def validate_param(param_name, param_values, train, model=lgb.LGBMRegressor(n_jobs=-1), scoring='accuracy', cv=None):
    X, y = train.drop(columns=TARGETVAR).values, train[TARGETVAR].values
    scorename = scoring if isinstance(scoring, str) else scoring.__name__
    train_scores, valid_scores = validation_curve(model, X, y, param_name, param_values, scoring=scoring, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.title("Validation Curve with " + model.__class__.__name__)
    plt.xlabel(param_name)
    plt.ylabel(scorename)
    plt.xticks(param_values)
    plt.semilogx(param_values, train_scores_mean, label="Training score", color='#78A8F3')
    plt.fill_between(param_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="#78A8F3")
    plt.semilogx(param_values, valid_scores_mean, label="Cross-validation score",
                 color="#F37878")
    plt.fill_between(param_values, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.2,
                     color="#F37878")
    plt.legend(loc="best")
    plt.show()
    return train_scores, valid_scores


__all__ = ['validate_param']