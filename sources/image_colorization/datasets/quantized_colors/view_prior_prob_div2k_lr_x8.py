#!/usr/bin/env python
# coding: utf-8

# # BEGIN

from __future__ import absolute_import, division, print_function
from IPython import get_ipython

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from config import *
from common import *

# Parameters
# parser
parser = argparse.ArgumentParser(description='VIEW PRIOR PROBABILITIES')

parser.add_argument('--pts_in_hull_path', type=str, 
                    default=f'{data_dir}/colorization_richard_zhang/pts_in_hull.npy')
parser.add_argument('--ab_hist_path', type=str, 
                    default=f'{dataset_dir}/div2k/outputs/preprocessing/ab_hist_train_div2k.npy')
parser.add_argument('--prior_prob_path', type=str, 
                    default=f'{dataset_dir}/div2k/outputs/preprocessing/prior_prob_train_div2k.npy')
parser.add_argument('--prior_prob_smoothed_path', type=str,  
                    default=f'{dataset_dir}/div2k/outputs/preprocessing/prior_prob_smoothed_train_div2k.npy')
parser.add_argument('--prior_prob_factor_path', type=str,  
                    default=f'{dataset_dir}/div2k/outputs/preprocessing/prior_prob_factor_train_div2k.npy')

args, _ = parser.parse_known_args()
params = vars(args)


pts_in_hull = np.load(params["pts_in_hull_path"])
q_ab        = pts_in_hull
ab_hist     = np.load(params["ab_hist_path"])
prior_prob  = np.load(params["prior_prob_path"])
prior_prob_smoothed = np.load(params["prior_prob_smoothed_path"])
prior_prob_factor = np.load(params["prior_prob_factor_path"])


# ## ab_hist

print("---------------------------")
print("Prior Distribution in ab space")
print("---------------------------")

plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])
for i in range(q_ab.shape[0]):
    ax.scatter(q_ab[:, 0], q_ab[:, 1])
    ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
    ax.set_xlim([-110, 110])
    ax.set_ylim([-110, 110])
# for

plt.title("Prior Distribution in ab space\n", fontsize=16)
plt.imshow(ab_hist.transpose(), norm=LogNorm(), cmap=plt.cm.jet, extent = (-128, 127, -128, 127), origin = "uper")
plt.xlim([-120, 120])
plt.ylim([-120, 120])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("b channel", fontsize = 14)
plt.ylabel("a channel", fontsize = 14)
plt.colorbar()
plt.show()
plt.clf()
plt.close()


# ## prior_prob

print("---------------------------")
print("Prior Probability Histogram")
print("---------------------------")

plt.figure(figsize=(20, 7))
plt.subplot(1,3,1), plt.hist(prior_prob, bins=100), plt.xlabel("Prior probability"), plt.ylabel("Frequency"), plt.yscale("log")
plt.subplot(1,3,2), plt.hist(prior_prob_smoothed, bins=100), plt.xlabel("Prior probability"), plt.ylabel("Frequency"), plt.yscale("log")
plt.subplot(1,3,3), plt.hist(prior_prob_factor, bins=100), plt.xlabel("Prior probability"), plt.ylabel("Frequency"), plt.yscale("log")
plt.show()


print("---------------------------")
print("Prior Probability View")
print("---------------------------")

plt.figure(figsize=(20, 7))
plt.subplot(1,3,1), plt.plot(prior_prob), plt.title("prior_prob");
plt.subplot(1,3,2), plt.plot(prior_prob_smoothed), plt.title("prior_prob_smoothed");
plt.subplot(1,3,3), plt.plot(prior_prob_factor), plt.title("prior_prob_factor");
plt.show()


plt.figure(figsize=(30, 7))

plt.subplot(1,3,1), plt.bar(range(len(prior_prob)), prior_prob), plt.title("Prior quantized-color distribution\n", fontsize=16), 
plt.xticks(fontsize=12), plt.yticks(fontsize=12), plt.xlabel("Label", fontsize = 14), plt.ylabel("Frequency", fontsize = 14)

plt.subplot(1,3,2), plt.bar(range(len(prior_prob_smoothed)), prior_prob_smoothed), plt.title("Smoothness Prior quantized-color distribution\n", fontsize=16), 
plt.xticks(fontsize=12), plt.yticks(fontsize=12), plt.xlabel("Label", fontsize = 14), plt.ylabel("Frequency", fontsize = 14)

plt.subplot(1,3,3), plt.bar(range(len(prior_prob_factor)), prior_prob_factor), plt.title("Weighted Smoothness Prior quantized-color distribution\n", fontsize=16), 
plt.xticks(fontsize=12), plt.yticks(fontsize=12), plt.xlabel("Label", fontsize = 14), plt.ylabel("Weight", fontsize = 14)

plt.show()


print(f"prior_prob: min = {np.min(prior_prob)}, max = {np.max(prior_prob)}, sum = {np.sum(prior_prob)}, avg = {np.average(prior_prob)} ")
print(f"prior_prob_smoothed: min = {np.min(prior_prob_smoothed)}, max = {np.max(prior_prob_smoothed)}, sum = {np.sum(prior_prob_smoothed)}, avg = {np.average(prior_prob_smoothed)} ")
print(f"prior_prob_factor: min = {np.min(prior_prob_factor)}, max = {np.max(prior_prob_factor)}, sum = {np.sum(prior_prob_factor)}, avg = {np.average(prior_prob_factor)} ")
print("")


# ## prior_smooth

from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve
sigma = 5

plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plt.plot(prior_prob, label="prior_prob")
plt.plot(prior_prob_smoothed, "g--", label="prior_prob_smoothed")
plt.yscale("log")
plt.legend()

f = interp1d(np.arange(prior_prob.shape[0]), prior_prob)
xx = np.linspace(0, prior_prob.shape[0] - 1, 1000)
yy = f(xx)
window = gaussian(2000, sigma)  # 2000 pts in the window, sigma=5
smoothed = convolve(yy, window / window.sum(), mode='same')

plt.subplot(2, 2, 2)
plt.plot(prior_prob, label="prior_prob")
plt.plot(xx, smoothed, "r-", label="smoothed")
plt.yscale("log")
plt.legend()

plt.subplot(2, 2, 3)
plt.hist(prior_prob, bins=100)
plt.xlabel("Prior probability")
plt.ylabel("Frequency")
plt.yscale("log")

plt.subplot(2, 2, 4)
plt.hist(prior_prob_smoothed, bins=100)
plt.xlabel("Prior probability smoothed")
plt.ylabel("Frequency")
plt.yscale("log")

plt.show()


# ## prior_factor

plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.hist(prior_prob)
plt.xlabel("Prior probability")
plt.ylabel("Frequency")
plt.yscale("log")

plt.subplot(1, 3, 2)
plt.hist(prior_prob_smoothed)
plt.xlabel("Prior probability smoothed")
plt.ylabel("Frequency")
plt.yscale("log")

plt.subplot(1, 3, 3)
plt.hist(prior_prob_factor)
plt.xlabel("Prior probability smoothed factor")
plt.ylabel("Frequency")
plt.yscale("log")

plt.show()


# # END
