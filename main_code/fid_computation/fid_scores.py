#!/usr/bin/env python3
import fid
import numpy as np


def main(model, data_source, noise_method):
    # load precalculated training set statistics
    if data_source == 'MNIST':
        f = np.load("fid_stats_mnist_1w.npz")
    else:
        f = np.load("fid_stats_fashion_1w.npz")
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()
    
    lambdas = [1,5,10,15,20,25,50,70,100,250]
    if noise_method == 'sp':
        noise_factors = [round(i*0.01,2) for i in range(1,52,2)]
    else:
        noise_factors = [round(i*0.1,1) for i in range(1,10)]
    
    fid_scores = []
    if model == 'RVAE':
        for l in lambdas:
            ls = []
            for nr in noise_factors:
                path = "fid_precalc/"+model+"_"+data_source+"_"+noise_method+"/"+'fid_stats_lambda_'+str(l)+'noise_'+str(nr)+".npz"
                f_g = np.load(path)
                mu_gen, sigma_gen = f_g['mu'][:], f_g['sigma'][:]
                fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
                print("FID: %s" % fid_value)
                ls.append(fid_value)
            fid_scores.append(ls)
        fid_scores = np.array(fid_scores)
        np.save("fid_scores_"+model+"_"+data_source+"_"+noise_method+".npy",fid_scores)
        
    else:
        for nr in noise_factors:
            path = "fid_precalc/"+model+"_"+data_source+"_"+noise_method+"/"+'fid_stats_noise_'+str(nr)+".npz"
            f_g = np.load(path)
            mu_gen, sigma_gen = f_g['mu'][:], f_g['sigma'][:]
            fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            print("FID: %s" % fid_value)
            fid_scores.append(fid_value)
        fid_scores = np.array(fid_scores)
        np.save("fid_scores_"+model+"_"+data_source+"_"+noise_method+".npy",fid_scores)

if __name__ == "__main__":
    model = "RVAE"
    data_source = "MNIST"
    noise_method = "sp"
    main(model, data_source, noise_method)