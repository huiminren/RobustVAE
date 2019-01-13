#!/usr/bin/env python3
import fid
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt
plt.switch_backend('agg')


# load precalculated training set statistics
f = np.load("fid_stats_mnist.npz")
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()


lambdas = [0.1,1,5,10,15,20,25,50,70,100]

noise_factors = [round(0.1*i,1) for i in range(11)]
noise_factors.extend([round(0.01*i,2) for i in range(51,60)])
noise_factors.extend([round(0.01*i,2) for i in range(61,70)])
noise_factors.sort()


fid_scores = []
for l in lambdas:
    ls = []
    for n in noise_factors:
        path = "./fid_precalc/fid_stats_lambda_"+str(l)+"noise_"+str(n)+".npz"
        f_g = np.load(path)
        mu_gen, sigma_gen = f_g['mu'][:], f_g['sigma'][:]
        fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        print("FID: %s" % fid_value)
        ls.append(fid_value)
    fid_scores.append(ls)

fid_scores = np.array(fid_scores)
np.save("fid_scores.npy",fid_scores)
ax = sns.heatmap(fid_scores, linewidth=0.5)
plt.xlabel("noise factor")
plt.ylabel("lambda")
plt.yticks(lambdas)
plt.xticks(noise_factors)
plt.show()
plt.savefig("fid_heatmap.png")
