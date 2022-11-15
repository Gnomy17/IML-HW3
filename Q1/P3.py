import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class quadratic():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.w = np.zeros(6)
    # def nll(self):
        
    def nll(self, w=None):
        if w is None:
            w = self.w
        xs_one = self.dataset[self.dataset[:, 2] == 1, 0:2]
        zs_one = np.concatenate((np.ones((xs_one.shape[0], 1)), xs_one, (xs_one[:,0] ** 2).reshape(-1,1),
         (xs_one[:,0] * xs_one[:,1]).reshape(-1,1), (xs_one[:,1]**2).reshape(-1,1)), axis=1)
        # print(zs_one.shape)
        xs_o = self.dataset[self.dataset[:, 2] == 0, 0:2]
        zs_o = np.concatenate((np.ones((xs_o.shape[0], 1)), xs_o, (xs_o[:,0] ** 2).reshape(-1,1),
         (xs_o[:,0] * xs_o[:,1]).reshape(-1,1), (xs_o[:,1]**2).reshape(-1,1)), axis=1)
        l = 0
        for i in range(xs_one.shape[0]):
            l += np.log(1/ (1 + np.exp(-w.dot(zs_one[i]))))
        for i in range(xs_o.shape[0]):
            l += np.log(1 - 1/ (1 + np.exp(-w.dot(zs_o[i]))))
        return -l

    def likelihood(self, x):
        # print(([1], x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2))
        z = np.array([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])
        return 1/(1 + np.exp(-self.w.dot(z)))

    def classify_dataset(self, dataset, gamma):
        labels = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        labels = []
        for i in range(dataset.shape[0]):
            l = self.likelihood(dataset[i,0:2])
            #print(l)
            label = 1 if l > gamma else 0
            labels.append(label)
            true_label = dataset[i,2]
            if true_label == 1:
                if label == 1:
                    tp += 1
                else:
                    fn += 1
            elif true_label == 0:
                if label == 1:
                    fp += 1
                else:
                    tn += 1
        return tp, fp, tn, fn, labels

class linear():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.w = np.zeros(3)
    # def nll(self):
        
    def nll(self, w=None):
        if w is None:
            w = self.w
        xs_one = self.dataset[self.dataset[:, 2] == 1, 0:2]
        zs_one = np.concatenate((np.ones((xs_one.shape[0], 1)), xs_one), axis=1)
        xs_o = self.dataset[self.dataset[:, 2] == 0, 0:2]
        zs_o = np.concatenate((np.ones((xs_o.shape[0], 1)), xs_o), axis=1)
        l = 0
        for i in range(xs_one.shape[0]):
            l += np.log(1/ (1 + np.exp(-w.dot(zs_one[i]))))
        for i in range(xs_o.shape[0]):
            l += np.log(1 - 1/ (1 + np.exp(-w.dot(zs_o[i]))))
        return -l

    def likelihood(self, x):
        z = np.concatenate(([1], x))
        return 1/(1 + np.exp(-self.w.dot(z)))

    def classify_dataset(self, dataset, gamma):
        labels = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        labels = []
        for i in range(dataset.shape[0]):
            l = self.likelihood(dataset[i,0:2])
            #print(l)
            label = 1 if l > gamma else 0
            labels.append(label)
            true_label = dataset[i,2]
            if true_label == 1:
                if label == 1:
                    tp += 1
                else:
                    fn += 1
            elif true_label == 0:
                if label == 1:
                    fp += 1
                else:
                    tn += 1
        return tp, fp, tn, fn, labels

def run_on_set(classifier, name):
    val_data = np.load("validation.npy")
    gamma_range = np.arange(start=0, stop=1, step=.01)
    points_x = []
    points_y = []
    min_error = 1
    min_gamma = 0.5
    min_error_x = 0
    min_error_y = 0
    labels = []
    for g in gamma_range:
        print(name, g)
        tp, fp, tn, fn, l = classifier.classify_dataset(val_data, gamma=g)
        points_x.append(fp/(fp + tn))
        points_y.append(tp/(tp + fn))
        err = fp/(fp + tn) * 0.6 + (1 - tp/(tp + fn)) * 0.4
        if err < min_error:
            labels = l
            min_gamma = g
            min_error = err
            min_error_x = fp/(fp + tn)
            min_error_y = tp/(tp + fn)
    labels = np.array(labels).reshape(-1)
    plt.clf()
    plt.scatter(val_data[labels==1,0], val_data[labels==1,1], c="green", label= "l=1")
    plt.scatter(val_data[labels==0,0], val_data[labels==0,1], c="red", label= "l=0")
    plt.legend()
    plt.savefig("q1p3datavis/" +name + "_val.png", dpi=100)
    _, _, _, _, labels = classifier.classify_dataset(classifier.dataset, gamma= min_gamma)
    labels = np.array(labels).reshape(-1)
    plt.clf()
    plt.scatter(classifier.dataset[labels==1,0], classifier.dataset[labels==1,1], c="green", label= "l=1")
    plt.scatter(classifier.dataset[labels==0,0], classifier.dataset[labels==0,1], c="red", label= "l=0")
    plt.legend()
    plt.savefig("q1p3datavis/" +name + "_train.png", dpi=100)
    plt.clf()
    plt.plot(points_x, points_y)
    plt.scatter(min_error_x, min_error_y, marker='x', c='red', label= "min err = " + str(min_error)[:6])
    plt.legend()
    plt.title("ROC" + name)
    plt.savefig("q1p3datavis/roc_" + name + ".png", dpi=100)

d = np.load("train_10k.npy",)
lc = linear(dataset=d)
lc.w = minimize(lc.nll, lc.w).x
run_on_set(lc, "linear_10k")
qc = quadratic(dataset=d)
qc.w = minimize(qc.nll, qc.w).x
run_on_set(qc, "quadratic_10k")

d = np.load("train_1k.npy",)
lc = linear(dataset=d)
lc.w = minimize(lc.nll, lc.w).x
run_on_set(lc, "linear_1k")
qc = quadratic(dataset=d)
qc.w = minimize(qc.nll, qc.w).x
run_on_set(qc, "quadratic_1k")

d = np.load("train_100.npy",)
lc = linear(dataset=d)
lc.w = minimize(lc.nll, lc.w).x
run_on_set(lc, "linear_100")
qc = quadratic(dataset=d)
qc.w = minimize(qc.nll, qc.w).x
run_on_set(qc, "quadratic_100")

