import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad

if __name__ == "__main__":
    df = pd.DataFrame(np.random.randint(100, size=(100, 3)), columns=['A', 'B', 'C'])
    fit = umap.UMAP()
    u = fit.fit_transform(df)
    plt.scatter(u[:, 0], u[:, 1])
    plt.title('UMAP embedding of random colours')

    obs = pd.DataFrame(index=df.index)
    var = pd.DataFrame(index=df.columns)

    #obsm = {"X_umap": u, "Y_map:": u}
    obs['Y'] = u[:, 0]
    obs['X'] = u[:, 1]

    uns = dict()

    adata = ad.AnnData(df.to_numpy(), var=var, obs=obs, uns=uns)
    adata.write('./src/H5AD/test.h5ad')

    plt.show()
