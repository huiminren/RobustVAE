from dimredu.eRPCAviaADMMFast import eRPCA as eRPCASparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

def denseToSparse(M, E):
    assert M.shape == E.shape, 'shape mismatch'
    m = M.shape[0]
    n = M.shape[1]

    u = np.empty([m * n])
    v = np.empty([m * n])
    vecM = np.empty([m * n])
    vecE = np.empty([m * n])

    k = 0
    for i in range(m):
        for j in range(n):
            u[k] = i
            v[k] = j
            vecM[k] = M[i, j]
            vecE[k] = E[i, j]
            k += 1

    return m, n, u, v, vecM, vecE


def eRPCA(M, E, **kw):
    m, n, u, v, vecM, vecE = denseToSparse(M, E)
    maxRank = np.min(M.shape)
    return eRPCASparse(m, n, u, v, vecM, vecE, maxRank, **kw)

def plot_faces(faces):
    fig, axes = plt.subplots(1, 10, figsize=(12, 12),
                              subplot_kw={'xticks': [], 'yticks': []},
                              gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate (axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')
    plt.show()


# def test_small():
#     X = np.random.random(size=[5, 15])
#     E = np.ones(X.shape)*1e-6
#     eRPCA(X, E)


if __name__ == '__main__':
    # test_small()
    faces = fetch_lfw_people()
    random_indexes = np.random.permutation (len(faces.data))
    X = faces.data[random_indexes]
    # example_faces = X[:36, :]
    # plot_faces (example_faces)

    import random
    random.seed (2)

    faces2 = fetch_lfw_people (min_faces_per_person=250)
    random_indexes = np.random.permutation(len(faces2.data))
    X = faces2.data[random_indexes]
    example_faces2 = X[:10, :]
    test = example_faces2[0]
    for _ in test:
        print(_)
    plot_faces(example_faces2)
    from sklearn.decomposition import PCA
    pca = PCA(svd_solver='randomized')
    pca.fit(example_faces2)
    plot_faces(pca.components_[:10, :])

    E = np.ones(example_faces2.shape) * 1e-6
    eRPCA (example_faces2, E)
    result = eRPCA(example_faces2, E)

    # test_small()
    U = result[0]
    E = np.diag(result[1])
    VT = result[2]
    S = result[3]
    low_Rank_Matrix = U.dot(E).dot(VT)
    from scipy.sparse import csr_matrix

    sparse_Matrix = csr_matrix(S).todense()
    recover_Matrix = sparse_Matrix + low_Rank_Matrix
    plot_faces(low_Rank_Matrix)
    plot_faces (sparse_Matrix)
    plot_faces(recover_Matrix)
    plt.show()


    # application 2
    # from time import time
    # import logging
    # import matplotlib.pyplot as plt
    #
    # from sklearn.model_selection import train_test_split
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.datasets import fetch_lfw_people
    # from sklearn.metrics import classification_report
    # from sklearn.metrics import confusion_matrix
    # from sklearn.decomposition import PCA
    # from sklearn.svm import SVC
    #
    # lfw_people = fetch_lfw_people (min_faces_per_person=70, resize=0.4)
    # n_samples, h, w = lfw_people.images.shape
    # X = lfw_people.data
    # n_features = X.shape[1]
    #
    # y = lfw_people.target
    # target_names = lfw_people.target_names
    # n_classes = target_names.shape[0]
    #
    # print ("Total dataset size:")
    # print ("n_samples: %d" % n_samples)
    # print ("n_features: %d" % n_features)
    # print ("n_classes: %d" % n_classes)
    #
    # X_train, X_test, y_train, y_test = train_test_split (
    #     X, y, test_size=0.25, random_state=42)
    #
    # n_components = 150
    #
    # print ("Extracting the top %d eigenfaces from %d faces"
    #        % (n_components, X_train.shape[0]))
    # t0 = time ()
    #
    # E = np.ones (X_train.shape) * 1e-6
    # train_result = eRPCA(X_train, E)
    # U = train_result[0]
    # E = np.diag(train_result[1])
    # VT = train_result[2]
    # S = train_result[3]
    # x_train_low_Rank_Matrix = U.dot(E).dot(VT)
    #
    #
    # E = np.ones (X_test.shape) * 1e-6
    # test_result = eRPCA(X_test, E)
    # U = test_result[0]
    # E = np.diag(test_result[1])
    # VT = test_result[2]
    # S = test_result[3]
    # x_test_low_Rank_Matrix = U.dot(E).dot(VT)
    #
    # pca = PCA (n_components=n_components, svd_solver='randomized',
    #            whiten=True).fit(x_train_low_Rank_Matrix)
    # print("done in %0.3fs" % (time () - t0))
    # eigenfaces = pca.components_.reshape((n_components, h, w))
    #
    # print ("Projecting the input data on the eigenfaces orthonormal basis")
    # t0 = time ()
    # # X_train_pca = pca.transform (X_train)
    # X_train_lowRan_pca = pca.transform(x_train_low_Rank_Matrix)
    # X_test_lowrank_pca = pca.transform(x_test_low_Rank_Matrix)
    # # X_test_pca = pca.transform (X_test)
    # print ("done in %0.3fs" % (time () - t0))
    #
    # print ("Fitting the classifier to the training set")
    # t0 = time ()
    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # clf = GridSearchCV (SVC (kernel='rbf', class_weight='balanced'),
    #                     param_grid, cv=5)
    # clf = clf.fit (X_train_lowRan_pca, y_train)
    # print ("done in %0.3fs" % (time () - t0))
    # print ("Best estimator found by grid search:")
    # print (clf.best_estimator_)
    #
    # # #############################################################################
    # # Quantitative evaluation of the model quality on the test set
    #
    # print ("Predicting people's names on the test set")
    # t0 = time ()
    # y_pred = clf.predict (X_test_lowrank_pca)
    # print ("done in %0.3fs" % (time () - t0))
    #
    # print (classification_report (y_test, y_pred, target_names=target_names))
    # print (confusion_matrix (y_test, y_pred, labels=range (n_classes)))
    #
    #
    # # #############################################################################
    # # Qualitative evaluation of the predictions using matplotlib
    #
    # def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    #     """Helper function to plot a gallery of portraits"""
    #     plt.figure (figsize=(1.8 * n_col, 2.4 * n_row))
    #     plt.subplots_adjust (bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    #     for i in range (n_row * n_col):
    #         plt.subplot (n_row, n_col, i + 1)
    #         plt.imshow (images[i].reshape ((h, w)), cmap=plt.cm.gray)
    #         plt.title (titles[i], size=12)
    #         plt.xticks (())
    #         plt.yticks (())
    #
    #
    # # plot the result of the prediction on a portion of the test set
    #
    # def title(y_pred, y_test, target_names, i):
    #     pred_name = target_names[y_pred[i]].rsplit (' ', 1)[-1]
    #     true_name = target_names[y_test[i]].rsplit (' ', 1)[-1]
    #     return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
    #
    #
    # prediction_titles = [title (y_pred, y_test, target_names, i)
    #                      for i in range (y_pred.shape[0])]
    #
    # plot_gallery (X_test, prediction_titles, h, w)
    #
    # # plot the gallery of the most significative eigenfaces
    #
    # eigenface_titles = ["eigenface %d" % i for i in range (eigenfaces.shape[0])]
    # plot_gallery (eigenfaces, eigenface_titles, h, w)
    #
    # plt.show ()
    #
    #
    #
    #




