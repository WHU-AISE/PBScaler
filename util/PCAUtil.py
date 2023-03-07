from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca(x, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)
    return principal_components

def explained_variance_ratio(x, n):
    pca = PCA(n_components=n)
    pca.fit_transform(x)
    print(pca.explained_variance_ratio_)
    importance = pca.explained_variance_ratio_
    plt.scatter(range(1,5),importance)
    plt.plot(range(1,5),importance)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

    
