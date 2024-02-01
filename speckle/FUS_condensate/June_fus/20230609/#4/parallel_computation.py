import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import math

from pandas import read_csv
from scipy.optimize import curve_fit
import pickle

import czifile
import cv2
from multiprocessing import Pool


def vectorized_compute_g2t_wo_kernel(px_indices, laglist_g2t, start_t, max_t_range, data, normalize=True, plot=False):
    # Extract the intensity values for all px indices at once
    # Assuming 'data' is a 5D array and px_indices is a 2D array with shape (n, 2)
    intensities = data[0, :, 0, 0, px_indices[:, 0], px_indices[:, 1], 0]
    
    intensities = intensities.astype(np.uint64)
    # print(intensities.shape)
    
    g2t_results = np.zeros((px_indices.shape[0], len(laglist_g2t)))

    # Loop over laglist_g2t
    for idx, lag in enumerate(laglist_g2t):
        # Vectorized computation for numerator and denominator
        numerator = (intensities[:, start_t:start_t + max_t_range] * intensities[:, start_t + lag:start_t + max_t_range + lag]).mean(axis=1)
        
        denominator1 = intensities[:, start_t:start_t + max_t_range].mean(axis=1)
        denominator2 = intensities[:, start_t + lag:start_t + max_t_range + lag].mean(axis=1)
        g2t_results[:, idx] = (np.array(numerator[:]) / (np.array(denominator1[:]) * np.array(denominator2[:]))) - 1

    # Normalization
    if normalize:
        first = g2t_results[:, 0].reshape(-1, 1)  # Reshape 'first' to be a column vector
        g2t_results /= first
    # if normalize:
    #     first = g2t_results[:, 0]
    #     for idx, lag in enumerate(laglist_g2t):
    #         g2t_results[:, idx] /= first


    # Plotting (if required)
    if plot:
        plt.figure(figsize=(8,5))
        for intensity in intensities:
            plt.plot(intensity[:200], 'k-')
        plt.xlabel("Time point", fontsize=18)
        plt.ylabel("Intensity", fontsize=18)
        plt.tick_params(direction='in')
        plt.show()

    return g2t_results

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def stretched_exponential(t, tau, beta):
    return np.exp(-(t / tau)**beta)

def fit_stretched_exponential(time_data, y_data):

    # Perform the curve fitting
    params, covariance = curve_fit(stretched_exponential, time_data, y_data,p0=[0.5,0.4])

    # Extracting the parameters
    tau, beta = params
    return tau, beta







# tau_map = np.zeros((h,w))
# beta_map = np.zeros((h,w))
# time_data = [ele *framespeed for ele in laglist_g2t]
# for i in range(100,101):
#     for j in range(110,111):
#         if mask[i,j]:
#             tau_map[i,j]=fit_stretched_exponential(time_data, corrmaps_v[i][j])[0]
#             beta_map[i,j]=fit_stretched_exponential(time_data, corrmaps_v[i][j])[1]

def process_pixel(args):
    i, j, time_data, corrmaps_v, mask = args
    if mask[i,j]:
        print(i,j)
        tau = fit_stretched_exponential(time_data, corrmaps_v[i][j])[0]
        beta = fit_stretched_exponential(time_data, corrmaps_v[i][j])[1]
        return i, j, tau, beta
    else:
        return i, j, None, None

def main():
    video_time_length=60 #s
    imgname='#4_fus_639nm_75h_1min_1116p2um'
    foldername='/Volumes/T7_Shield/Speckle_more/June_FUS/20230609/'

    # Read the CZI file
    data = czifile.imread(foldername+imgname+'.czi')
    framespeed=video_time_length/data.shape[1]

    # apply a gaussian blur
    for t in range(0,data.shape[1],1):
        data[0,t,0,0,:,:,0]=cv2.GaussianBlur(data[0,t,0,0,:,:,0], (3, 3), 1.6)

    # Assuming 'data' is a NumPy array and 'compute_g2t_wo_kernel' is vectorized
    h, w = 250,250  # Assuming data has at least 4 dimensions and spatial dimensions are at 2nd and 3rd
    corrmaps_v = np.empty((h, w), dtype=object)  # or the appropriate dtype for your data
    max_lag=200
    laglist_g2t = np.arange(0, max_lag, 1)

    # Create a grid of indices for the entire array
    i_indices, j_indices = np.indices((h, w))

    print('computing')
    # Vectorized computation for the whole array
    results = vectorized_compute_g2t_wo_kernel(np.stack((i_indices, j_indices), axis=-1).reshape(-1, 2),
                                    laglist_g2t, 0, 900, data)

    # # Reshape results to match the 2D structure of corrmaps
    corrmaps_v = results.reshape(h, w, max_lag)
    h=250
    w=250
    mask = create_circular_mask(h, w, [134,106],106)
    tau_map = np.zeros((h,w))
    beta_map = np.zeros((h,w))
    time_data = [ele * framespeed for ele in laglist_g2t]  # define time_data here

    # Prepare arguments for multiprocessing
    args = [(i, j, time_data, corrmaps_v, mask) for i in range(100, 105) for j in range(110, 115)]

    # Use a multiprocessing pool to parallelize the computation
    with Pool() as pool:
        results = pool.map(process_pixel, args)

    # Store the results in the maps
    for i, j, tau, beta in results:
        if tau is not None and beta is not None:
            tau_map[i, j] = tau
            beta_map[i, j] = beta
    np.savetxt("tau_map.csv", tau_map, delimiter=",")
    np.savetxt("beta_map.csv", beta_map, delimiter=",")

if __name__ == "__main__":
    main()