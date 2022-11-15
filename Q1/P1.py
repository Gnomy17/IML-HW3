from scipy.stats import multivariate_normal as mvn
from generate_data import m01, m02, c01, c02, c1, m1
import numpy as np
import matplotlib.pyplot as plt

class bayes_classifier():
    # This classifier will use the simple Bayes decision rule, as the conditional probability
    # of each label given an image is proportional to its joint probability over some normalizing probability
    # that's the same for all labels w just calculate the joint posterior using the conditional pdfs and the priors
    def __init__(self, m01, m02, c01, c02, m1, c1, w1, w2):
        self.g01 = mvn(mean=m01, cov=c01)
        self.g02 = mvn(mean=m02, cov=c02)
        self.g1 = mvn(mean=m1, cov=c1)
        self.w1 = w1
        self.w2 = w2
    
    def classify(self, x, gamma):
        pl0_x = (self.w1 * self.g01.pdf(x) + self.w2 * self.g02.pdf(x))
        pl1_x = self.g1.pdf(x) 
        # print(gamma)
        # print(pl0_x, pl1_x)
        return 1 if pl1_x/pl0_x > gamma else 0
    
    def classify_dataset(self, dataset, gamma):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        labels = []
        for i in range(dataset.shape[0]):
            label = self.classify(dataset[i,0:2], gamma=gamma)
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
classifier = bayes_classifier(m01, m02, c01, c02, m1, c1, 0.5, 0.5)
val_data = np.load("validation.npy")
gamma_range = (np.arange(start=0,stop=25, step=.05))
optimal_gamma = (1 - 0) / (1 - 0) * 0.6 / 0.4 # The optimal decision rule assuming 1-0 loss and using the priors 0.6 for l0 and 0.4 for l1
points_x = []
points_y = []
min_error = 1
min_error_x = 0
min_error_y = 0
for g in gamma_range:
    print(g)
    tp, fp, tn, fn, _ = classifier.classify_dataset(val_data, gamma=g)
    points_x.append(fp/(fp + tn))
    points_y.append(tp/(tp + fn))
    err = fp/(fp + tn) * 0.6 + (1 - tp/(tp + fn)) * 0.4
    if err < min_error:
        min_error = err
        min_error_x = fp/(fp + tn)
        min_error_y = tp/(tp + fn)
    
tp, fp, tn, fn, optimal_labels = classifier.classify_dataset(val_data, gamma=optimal_gamma)
optimal_labels = np.array(optimal_labels).reshape(-1)
optimal_x = fp/(fp + tn)
optimal_y = tp/(tp + fn)
optimal_err = fp/(fp + tn) * 0.6 + (1 - tp/(tp + fn)) * 0.4

plt.clf()
plt.scatter(val_data[optimal_labels==1,0], val_data[optimal_labels==1,1], c="green", label= "l=1")
plt.scatter(val_data[optimal_labels==0,0], val_data[optimal_labels==0,1], c="red", label= "l=0")
plt.legend()
plt.savefig("q1p1datavis.png", dpi=1000)
plt.clf()
plt.plot(points_x, points_y)
plt.scatter(min_error_x, min_error_y, marker='x', c='red', label= "min err = " + str(min_error)[:6])
plt.scatter(optimal_x, optimal_y, marker='*', c='green', label='optimal gamma = ' + str(0.4/0.6)[:6] +  " with err = " + str(optimal_err)[:6])
plt.legend()
plt.title("ROC")
plt.savefig("q1p1roc.png", dpi=1000)








