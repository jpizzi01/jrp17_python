import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import kneed
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from preprocessing import explained_var, scat_coef, eig_eval


def optimal_kmeans_plus(data, plot_dim):
    n = len(data)
    max_k = int(np.sqrt(n))
    inertia = np.zeros(max_k)
    cluster_iterator = range(1, max_k + 1)
    # indexing through k=1 to k=kmax, extracting inertia for each k
    for i in cluster_iterator:
        result = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=17).fit(data)
        inertia[i - 1] = result.inertia_
    # finding elbow, then performing kmeans at the elbow value
    knee = kneed.KneeLocator(cluster_iterator, inertia, curve='convex', direction='decreasing')
    elbow = knee.knee
    print('The chosen number of clusters based off of the elbow method is %.1f' % elbow)
    best = KMeans(n_clusters=elbow, init='k-means++', n_init=15, random_state=17).fit(data)
    labels = best.labels_

    # plotting if plot_dim is 2 or 3, ending if not

    if isinstance(plot_dim, int) and not isinstance(plot_dim, bool):
        if plot_dim > 2:
            fig1 = plt.figure(1)
            ax = fig1.add_subplot(projection='3d')
        else:
            fig1 = plt.figure(1)
            ax = fig1.add_subplot()
        k_indices = [[]]
        for i in range(elbow - 1):
            k_indices.append([])
        for i in range(elbow):
            plotting_group = [[]]
            for go in range(plot_dim-1):
                plotting_group.append([])
            for j in range(len(labels)):
                if i == labels[j]:
                    point = data.iloc[j]
                    for q in range(plot_dim):
                        plotting_group[q].append(list(point)[q])
                        k_indices[i].append(j)
            if plot_dim == 3:
                ax.scatter(plotting_group[0], plotting_group[1], plotting_group[2])
            else:
                ax.scatter(plotting_group[0], plotting_group[1])
        ax.set_xlabel('Axis 1')
        ax.set_ylabel('Axis 2')
        if plot_dim > 2:
            ax.set_zlabel('Axis 3')
        ax.set_title('K-Means++ Clustering of Dataset, k=%.1f' % elbow)
        plt.show()
    return


# Like for tSNE, the usefulness of PCA (to me) is as a preprocessing technique, where high-parameter datasets are
# simplified to principal components which (in ascending) explain the most variance in the set
# Mathematically, principal component axes are the eigenvectors of the eigenvalues of the covariance matrix
# The larger the eigenvalue, the more variation the eigenvector, or PC captures.

def pick_pca_transform(data, validate):

    # defining variables, n of observations, max k value as sqrt(n)
    features = []
    n = len(data)
    for col in data.columns:
        features.append(col)
    m = len(features)

    # standard scaling (z score) prior to pca and clustering
    ss = StandardScaler()
    ss.fit(data)
    normalized = ss.transform(data)

    # determining PCA suitability

    # first, explained variance, which is the det of the covariance matrix. Smaller is better!
    if validate:
        ev = explained_var(normalized)
        print('The explained variance of this dataset is %.9f' % ev)

    # then, scatter coef, which is det of correlation matrix. Smaller is better!
        scat, cor = scat_coef(data)
        print('The scatter coefficient of this dataset is %.9f' % scat)

    # try psi index and information statistic, where larger values are better
        eig_eval(cor)

    # creating test/train split of normalized data

    # setting maximum number of pcs as the minimum choice between n and # of parameters
    if m > n:
        start_comp = n
    else:
        start_comp = m

    # run PCA with maximum number of pcs and fit it to normalized data
    pca = PCA(n_components=start_comp)
    pca.fit(normalized)
    pc_choice = 0
    # from 0 to the max PC, cumulatively sum the amount of variance (starting with PC1, then PC1+PC2, etc.)
    for i in range(0, start_comp):
        var = sum(pca.explained_variance_ratio_[0:i])
        # when the cumulative variance achieves 80% explained, perform PCA again with that number of PCs
        if var >= 0.8:
            pc_choice = i+1
            pca = PCA(n_components=pc_choice)
            pca.fit(normalized)
            # end the loop to prevent it from adding more PCs
            break
    print('The chosen number of PCs is %.1f' % pc_choice)

    # transform the normalized data on the final PC selection, store variance explained and loadings
    pca1 = pca.transform(normalized)
    pc_variance = sum(pca.explained_variance_ratio_ * 100)
    print('The variance explained by this choice is %.2f percent' % pc_variance)

    # exporting pca transformed data as a pd.df with appropriate column names
    pcs = []
    for i in (range(1, pc_choice+1)):
        label = 'PC' + str(i)
        pcs.append(label)
    transformed = pd.DataFrame(pca1, columns=pcs)

    # pca transformed data
    return transformed

# moving on to another unsupervised dimensionality reduction tool - tSNE (T-Distributed Stochastic Neighbor Embedding)


# tsne (to me) seems like a pre-processing tool for other algorithms like random forest, where the reduction in d
# speeds up computation/ processing time. SO, it should be applied on the entire dataset rather than a test/train split

def tsne_bot(data_df, normalize, target):
    # determine sample size
    n = len(data_df)

    # first, normalize with z score. Turn this off if input is already normalized, or doesn't need it (PCA)
    if normalize:
        ss = StandardScaler()
        ss.fit(data_df)
        normalized = ss.transform(data_df)
    else:
        normalized = data_df

    # optimize learning rate based on sample size: Kobak & Berens 2019. Corresponds with the early exaggeration def (12)
    possible_learn = [200, n/12]
    learning_rate = max(possible_learn)

    # first optimize perplexity: reflects the number of nearest neighbors that is used in other manifold algorithms
    # Use S parameter detailed by Cao and Wang: Automatic Selection of t-SNE perplexity
    # This accounts for the fact that KL divergence will naturally decrease with increasing perplexity, always favoring
    # a higher value. The offset is the perplexity's relationship to the sample size, as the max perplexity value is n

    # n/3 taken from OpenTSNE documentation
    perplexity = np.arange(5, n/3, n/30)
    s_param = []

    # offset part of S will not change regardless of perplexity
    offset = np.log10(n)*(1/n)

    # calculate S using KL divergence for each perplexity
    for i in perplexity:
        model = TSNE(n_components=2, perplexity=i, learning_rate=learning_rate, random_state=17)
        model.fit(normalized)
        kl1 = model.kl_divergence_
        s_param.append((2*kl1) + (offset*i))

    # choosing the perplexity value associated with minimal S
    best_perplexity = perplexity[np.argmin(s_param)]
    print('The chosen perplexity value is %.1f' % int(best_perplexity))

    # now that perplexity is fixed, range through n_iter and minimize KL divergence
    n_iter = [500, 1000, 2000, 3000, 5000, 7500, 10000]
    working_kl = []
    for j in n_iter:
        the_model = TSNE(n_components=2, perplexity=best_perplexity, learning_rate=learning_rate, n_iter=j,
                         random_state=17)
        the_model.fit(normalized)
        kl2 = the_model.kl_divergence_
        working_kl.append(kl2)

    # best n is the value that minimizes KL divergence
    best_n = n_iter[np.argmin(working_kl)]
    print('The number of iterations that minimizes KL divergence is %.1f' % best_n)

    # fit with tuned parameters and transform data to tSNE space
    best_model = TSNE(n_components=2, perplexity=best_perplexity, learning_rate=learning_rate, n_iter=best_n,
                      random_state=17)
    tsne_transformed = best_model.fit_transform(normalized)

    # dealing with non-numerical targets in plotting tsne 1+2, mapping unique classifiers to numbers
    if type(target[0]) is str:
        classes = np.unique(target)
        for i in range(len(classes)):
            for j in range(len(target)):
                if target[j] == classes[i]:
                    target[j] = i

    # visualizing the results of the operation
    plt.scatter(tsne_transformed[:, 0], tsne_transformed[:, 1], c=target)
    plt.xlabel('First tSNE')
    plt.ylabel('Second tSNE')
    plt.title('Dimensionality Reduction with t-SNE, perplexity = %.1f' % int(best_perplexity))
    plt.show()
    return
