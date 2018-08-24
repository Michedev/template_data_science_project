import pandas as pd
from constants.dataset import TARGETVAR
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


def pca_plot_train_test(train: pd.DataFrame, test: pd.DataFrame, **lmplot_args):
    assert len(train.columns) == len(test.columns)
    train_test = pd.concat([train, test], copy=False)
    train_test.reset_index(inplace=True, drop=True)
    train_test['_is_train_'] = train_test.index < len(train)
    dfpca = pca_plot(df=train_test, target='_is_train_', **lmplot_args)
    del train_test
    return dfpca


def pca_plot(df, target=TARGETVAR, **lmplot_args):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.drop(columns=target).values)
    df_pca = pd.DataFrame(X_pca, columns=['a', 'b'])
    df_pca[target] = df[target]
    sns.lmplot(x='a', y='b', data=df_pca, hue=target, **lmplot_args)
    plt.show()
    return df_pca


__all__ = ['pca_plot', 'pca_plot_train_test']
