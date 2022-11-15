from sklearn.mixture import GaussianMixture
import numpy as np
from P1 import bayes_classifier
import matplotlib.pyplot as plt


def run_on_trainset(train, name):
    class0_points = train[train[:,2] == 0, :]
    class1_points = train[train[:,2] == 1, :]
    prior_0 = class0_points.shape[0] / (train.shape[0])
    prior_1 = class1_points.shape[0] / (train.shape[0])

    class0_gmm = GaussianMixture(n_components=2)
    class1_gmm = GaussianMixture(n_components=1)

    class0_gmm.fit(class0_points[:,0:2])
    class1_gmm.fit(class1_points[:,0:2])

    est_m01 = class0_gmm.means_[0]
    est_m02 = class0_gmm.means_[1]
    est_c01 = class0_gmm.covariances_[0]
    est_c02 = class0_gmm.covariances_[1]
    est_w1 = class0_gmm.weights_[0]
    est_w2 = class0_gmm.weights_[1]
    est_m1 = class1_gmm.means_[0]
    est_c1 = class1_gmm.covariances_[0]

    est_classifier = bayes_classifier(est_m01, est_m02, est_c01, est_c02, est_m1, est_c1, est_w1, est_w2)

    val_data = np.load("validation.npy")
    gamma_range = np.arange(start=0, stop=25, step=.2)
    points_x = []
    points_y = []
    min_error = 1
    min_error_x = 0
    min_error_y = 0
    for g in gamma_range:
        print(name, g)
        tp, fp, tn, fn, _ = est_classifier.classify_dataset(val_data, gamma=g)
        points_x.append(fp/(fp + tn))
        points_y.append(tp/(tp + fn))
        err = fp/(fp + tn) * prior_0 + (1 - tp/(tp + fn)) * prior_1
        if err < min_error:
            min_error = err
            min_error_x = fp/(fp + tn)
            min_error_y = tp/(tp + fn)

    plt.clf()
    plt.plot(points_x, points_y)
    plt.scatter(min_error_x, min_error_y, marker='x', c='red', label= "min err = " + str(min_error)[:6])
    plt.legend()
    plt.title("ROC" + name)
    plt.savefig("q1p2roc_" + name + ".png", dpi=1000)

run_on_trainset(np.load("train_10k.npy"), "10k")
run_on_trainset(np.load("train_1k.npy"), "1k")
run_on_trainset(np.load("train_100.npy"), "100")

