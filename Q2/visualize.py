import numpy as np
import matplotlib.pyplot as plt

def generate_landmarks(K):
    # Generate landmarks
    angles = [(i/K)*2*np.pi for i in range(K)]
    return np.array([np.array([np.cos(t), np.sin(t)]) for t in angles]) 

def generate_measurements(marks, x_t, sn):
    # Calculate distances and generate
    # measurements until they are all non-negative
    dist_t = np.linalg.norm(marks - x_t, axis=1)
    res = -1 * np.ones_like(dist_t)
    while np.sum(res < 0) > 0:
        res = dist_t + np.random.normal(0, sn, dist_t.shape)
    return res

def generate_true_pos(sx, sy):
    # Generate the true vehicle position
    return np.random.multivariate_normal(mean = [0,0], cov=[[sx**2, 0], [0, sy**2]])

def calc_post(x_c, measures, marks, sx, sy, sn):
    ncis = measures - np.linalg.norm(marks - x_c, axis=1)
    exp_term = -(np.sum((ncis/sn)**2) + (x_c[0]/ sx)**2 + (x_c[1]/ sy)**2)/2
    # We don't actually need the denom term since it's not dependant on the candidate positions
    # denom_term = np.power(2*np.pi, marks.shape[0]/2 + 1) * sx * sy * (sn ** marks.shape[0])
    # We don't even need to exponentiate since it is a monotonically increasing function but we'll do it anyways
    return np.exp(exp_term) # / denom_term

Ks = range(1,5)
sx = 0.25
sy = 0.25
sn = 0.3
x_true = generate_true_pos(sx, sy)
spacing = 1000
x_s = np.linspace(-2, 2, spacing)
y_s = np.linspace(-2, 2, spacing)
for k in Ks:
    landmarks = generate_landmarks(k)
    measures = generate_measurements(landmarks, x_true, sn)
    points = np.zeros((spacing,spacing))
    for i, x in enumerate(x_s):
        for j, y in enumerate(y_s):
            points[i, j] = calc_post(np.array([x, y]), measures, landmarks, sx, sy, sn)
    plt.clf()
    plt.contourf(x_s, y_s, points)
    plt.scatter(x_true[0], x_true[1], marker="P", c='red')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker="o", c='yellow')
    plt.savefig("contours_" + str(k) + ".png")