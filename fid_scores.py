#!/usr/bin/env python3
import fid
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt
plt.switch_backend('agg')


# load precalculated training set statistics
f = np.load("fid_stats_mnist_1w.npz")
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

lambdas = [1,5,10,15,20,25,50,70,100,250]
noise_factors = [i*0.01 for i in range(1,30,2)]


fid_scores = []
for l in lambdas:
    ls = []
    for n in noise_factors:
        path = "./fid_precalc/fid_stats_lambda_"+str(l)+"noise_"+str(n)+".npz"
        f_g = np.load(path)
        mu_gen, sigma_gen = f_g['mu'][:], f_g['sigma'][:]
        try: 
            fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        except:
            fid_value = 9999999
        print("FID: %s" % fid_value)
        ls.append(fid_value)
    fid_scores.append(ls)

fid_scores = np.array(fid_scores)
np.save("fid_scores_rvae.npy",fid_scores)
ax = sns.heatmap(fid_scores, linewidth=0.5)
plt.xlabel("noise factor")
plt.ylabel("lambda")
plt.yticks(lambdas)
plt.xticks(noise_factors)
plt.show()
plt.savefig("fid_rvae_heatmap.png")
