import pandas as pd
from constants.dataset import TARGETVAR
from sklearn.decomposition import PCA
import seaborn as sns


def pca_plot(df: pd.DataFrame, target=TARGETVAR, **lmplot_args):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.drop(columns=target).values)
    df_pca = pd.DataFrame(X_pca, columns=['a', 'b'])
    df_pca[target] = df[target]
    sns.lmplot(x='a', y='b', data=df_pca, hue=target, **lmplot_args)
    plt.show()
    return df_pca


__all__ = ['pca_plot']