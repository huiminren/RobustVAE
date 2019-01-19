#!/usr/bin/env python3
import fid
import numpy as np

# load precalculated training set statistics
f = np.load("fid_stats_mnist_1w.npz")
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

noise_factors = [i*0.01 for i in range(1,52,2)]


fid_scores = []
for n in noise_factors:
    path = "./fid_precalc_1w/fid_stats_noise_"+str(n)+".npz"
    f_g = np.load(path)
    mu_gen, sigma_gen = f_g['mu'][:], f_g['sigma'][:]
    try: 
        fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    except:
        fid_value = 9999999
    print("FID: %s" % fid_value)
    fid_scores.append(fid_value)

fid_scores = np.array(fid_scores)
np.save("fid_scores_vae_51.npy",fid_scores)
