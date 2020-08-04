"""
Functions:
    def read_image(img_path, is_resize = True, width = 224, height = 224, interpolation = cv2.INTER_AREA)
    
    def cielab_color_space()
    def view_db_info(db_root, db_files, db_name)
    
    def compute_prior_prob(image_files, width, height, do_plot, pts_in_hull_path, prior_prob_path)
    def compute_prior_prob_v1(image_files, is_resize, width, height, do_plot, pts_in_hull_path, prior_prob_path, ab_hist_path):
    
    def compute_prior_prob_smoothed(prior_prob_path, prior_prob_smoothed_path, sigma, do_plot = True, verbose = 1)
    def compute_prior_factor(prior_prob_path, prior_prob_smoothed_path, prior_prob_factor_path, gamma = 0.5, alpha = 1, do_plot = True, verbose = 1)
    
Main:
    def compute_prior_prob_export(db_root, db_file, db_name, column_image = "image", column_type = "type", process_types = ["train"], 
                                  pts_in_hull_path = os.path.join(module_dir, "data", "prior_prob_train_div2k.npy").replace("\\", "/"),
                                  export_prior_prob_path = None, 
                                  export_ab_hist_path = None, 
                                  is_resize = False, width = 112, height = 112, 
                                  do_plot = True, verbose = 1, )
                              
    def main()
    def main_index_data(**input_params)
    def main_compute_prior_prob(**input_params)
    def main_compute_prior_prob_smoothed(**input_params)
    def main_cielab_color_space()
"""
from __future__ import absolute_import, division, print_function
import click, os, pandas as pd, glob, tqdm, cv2, numpy as np, sys

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm

import time
from skimage import color
from console_progressbar import ProgressBar

import sklearn.neighbors as nn

from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve

def read_image(img_path, is_resize = True, width = 224, height = 224, interpolation = cv2.INTER_AREA):
    """
    Load img with opencv and reshape
    """
    result = {}
    
    org_img_color = cv2.imread(img_path)
    if len(org_img_color.shape)==2: # grayscale
        org_img_color = np.dstack([org_img_color, org_img_color, org_img_color])
    else:
        org_img_color = org_img_color[:, :, ::-1] # RGB convert
    # if
    org_img_gray  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    org_img_Lab   = cv2.cvtColor(org_img_color, cv2.COLOR_RGB2Lab)
    
    result.update(dict(org_img_color=org_img_color, org_img_gray=org_img_gray, org_img_Lab=org_img_Lab))

    if is_resize == True:
        res_img_color = cv2.resize(org_img_color, (width, height), interpolation=interpolation)
        res_img_gray  = cv2.resize(org_img_gray, (width, height), interpolation=interpolation)
        res_img_Lab   = cv2.cvtColor(res_img_color, cv2.COLOR_RGB2Lab)
        result.update(dict(res_img_color=res_img_color, res_img_gray=res_img_gray, res_img_Lab=res_img_Lab))
    # if

    return result
# read_image

def compute_prior_prob(image_files, width, height, do_plot, pts_in_hull_path, prior_prob_path):
    """
    Compute color prior probabilities for pts in hull
    Reference: https://github.com/foamliu/Colorful-Image-Colorization/blob/master/class_rebal.py
    Usage:
        df_data        = pd.read_hdf(os.path.join(data_dir, "preprocessing", "DIV2K", "div2k.hdf5"), "data")
        list_types     = ["'train'"]
        df_select_data = df_data.query("type in [" + ",".join(list_types) + "]")
        image_dir      = os.path.join(dataset_dir, "DIV2K").replace("\\", "/")

        image_files    = image_dir + "/" + df_select_data["path"].values
        image_files[0:3], len(image_files)

        info = dict(
            image_files      = image_files,
            pts_in_hull_path = os.path.join(data_dir, "colorization_richard_zhang", "pts_in_hull.npy"),
            prior_prob_path  = os.path.join(data_dir, "preprocessing", "DIV2K", "prior_prob_train_div2k.npy"),
            width   = 112,
            height  = 112,
            do_plot = True
        )
        locals().update(**info)
        prior_prob = compute_prior_prob(**info)
    """
    # Load ab image
    X_ab = []
    for image_path in tqdm.tqdm(image_files):
        result     = read_image(image_path, is_resize = True, width = width, height = height)
        X_ab.append(result["res_img_Lab"][:, :, 1:])
    # for
    X_ab = np.array(X_ab)
    X_ab = X_ab - 128.0
    
    # Load the gamut points location
    q_ab = np.load(pts_in_hull_path)

    if do_plot:
        plt.figure(figsize=(8, 8))
        plt.title("ab quantize")
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        for i in range(q_ab.shape[0]):
            ax.scatter(q_ab[:, 0], q_ab[:, 1])
            ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
            ax.set_xlim([-110, 110])
            ax.set_ylim([-110, 110])
        # for
    # if
        
    npts, c, h, w = X_ab.shape
    X_a_ravel = np.ravel(X_ab[:, :, :, 0])
    X_b_ravel = np.ravel(X_ab[:, :, :, 1])
    X_ab_ravel = np.vstack((X_a_ravel, X_b_ravel)).T
        
    if do_plot:
        plt.title("Prior Distribution in ab space\n", fontsize=16)
        plt.hist2d(X_ab_ravel[:, 0], X_ab_ravel[:, 1], bins=100, density=True, norm=LogNorm(), cmap=plt.cm.jet)
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
    # if
        
    # Create nearest neighbord instance with index = q_ab
    NN = 1
    nearest = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(q_ab)
    # Find index of nearest neighbor for X_ab
    dists, ind = nearest.kneighbors(X_ab_ravel)

    # We now count the number of occurrences of each color
    ind = np.ravel(ind)
    counts = np.bincount(ind)
    idxs = np.nonzero(counts)[0]
    prior_prob = np.zeros((q_ab.shape[0]))
    
    prior_prob[idxs] = counts[idxs] 
    
    # We turn this into a color probability
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Save
    if prior_prob_path is not None:
        save_dir = os.path.dirname(prior_prob_path)
        if save_dir != ""  and os.path.exists(save_dir) == False: os.makedirs(save_dir)
        pts_in_hull_name = os.path.basename(pts_in_hull_path)
        safe_copy(pts_in_hull_path, os.path.join(save_dir, pts_in_hull_name))
        np.save(prior_prob_path, prior_prob)
    # if

    if do_plot:
        plt.hist(prior_prob, bins=100)
        plt.xlabel("Prior probability")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.show()
    # if
    
    return prior_prob
    pass
# compute_prior_prob

def compute_prior_prob_v1(image_files, is_resize, width, height, do_plot, pts_in_hull_path, prior_prob_path, ab_hist_path):
    """
    Compute color prior probabilities for pts in hull
    Reference: https://github.com/foamliu/Colorful-Image-Colorization/blob/master/class_rebal.py
    Usage:
        df_data        = pd.read_hdf(os.path.join(dataset_dir, "DIV2K", "div2k.hdf5"), "data")
        list_types     = ["'train'"]
        df_select_data = df_data.query("type in [" + ",".join(list_types) + "]")
        image_dir      = os.path.join(dataset_dir, "DIV2K").replace("\\", "/")

        image_files    = image_dir + "/" + df_select_data["path"].values
        image_files[0:3], len(image_files)

        info = dict(
            image_files      = image_files,
            pts_in_hull_path = os.path.join(module_dir, "data", "pts_in_hull.npy"),
            prior_prob_path  = os.path.join(module_dir, "data", "prior_prob_train_div2k.npy"),
            ab_hist_path     = os.path.join(data_dir, "preprocessing", "DIV2K", "ab_hist_train_div2k.npy"),
            
            is_resize = False,
            width     = 112,
            height    = 112,
            
            do_plot = True
        )
        locals().update(**info)
        prior_prob = compute_prior_prob(**info)
    """
    # Load ab image
    ab_hist = np.zeros((256, 256), dtype = np.uint64)
    for image_path in tqdm.tqdm(image_files):
        result     = read_image(image_path, is_resize = is_resize, 
                                width = width, height = height)
        I_ab = result["res_img_Lab"][:, :, 1:] if is_resize==True else result["org_img_Lab"][:, :, 1:] 
        I_ab = I_ab.reshape(-1, 2).astype(np.uint)

        (ab_vals, ab_cnts) = np.unique(I_ab, return_counts = True, axis=0)
        ab_hist[ab_vals[:, 0], ab_vals[:, 1]] += ab_cnts.astype(np.uint64)
    # for
        
    # Load the gamut points location
    q_ab = np.load(pts_in_hull_path)
        
    if do_plot:
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
    # if
    
    X_ab_ravel_h = np.vstack(np.nonzero(ab_hist)).T
    X_ab_ravel_h = X_ab_ravel_h - 128
        
    # Create nearest neighbord instance with index = q_ab
    NN = 1
    nearest = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(q_ab)
    # Find index of nearest neighbor for X_ab
    dists, ind = nearest.kneighbors(X_ab_ravel_h)

    # We now count the number of occurrences of each color
    ind = np.ravel(ind)
    
    counts = np.zeros(np.max(ind) + 1, np.uint64)
    for idx, (a,b) in enumerate(X_ab_ravel_h):
        counts[ind[idx]] = counts[ind[idx]] + ab_hist[(a + 128,b + 128)]
        pass
    # for
    
    idxs = np.nonzero(counts)[0]
    prior_prob = np.zeros((q_ab.shape[0]))
    prior_prob[idxs] = counts[idxs]
        
    # We turn this into a color probability
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Save
    if prior_prob_path is not None:
        save_dir = os.path.dirname(prior_prob_path)
        if save_dir != ""  and os.path.exists(save_dir) == False: os.makedirs(save_dir)
        np.save(prior_prob_path, prior_prob)
    # if
    
    # Save
    if ab_hist_path is not None:
        save_dir = os.path.dirname(ab_hist_path)
        if save_dir != ""  and os.path.exists(save_dir) == False: os.makedirs(save_dir)
        np.save(ab_hist_path, ab_hist)
    # if

    if do_plot:
        plt.hist(prior_prob, bins=100)
        plt.xlabel("Prior probability")
        plt.ylabel("Frequency")
        plt.yscale("log")
        plt.show()
    # if
    
    return prior_prob, ab_hist
    pass
# compute_prior_prob_v1

def compute_prior_prob_smoothed(prior_prob_path, prior_prob_smoothed_path, sigma = 5, do_plot = True, verbose = 1):
    """
    Interpolation on prior prob, next using interpolation to smoothness path, and normalize again
    Reference: https://github.com/foamliu/Colorful-Image-Colorization/blob/master/class_rebal.py
    Usage:
        info = dict(
            prior_prob_path = os.path.join(module_dir, "data", "prior_prob_train_div2k.npy"),
            prior_prob_smoothed_path = os.path.join(module_dir, "data", "prior_prob_smoothed_train_div2k.npy"),
            sigma = 5,
            do_plot = True,
            verbose = True,
        )
        locals().update(**info)
        prior_prob_smoothed = compute_prior_prob_smoothed(**info)
    """
    # load prior probability
    
    if verbose==1: print("\n=== Compute Prior Probability Smoothed === ")
    prior_prob = np.load(prior_prob_path)
    
    # add an epsilon to prior prob to avoid 0 vakues and possible NaN
    prior_prob += 1E-3 * np.min(prior_prob)
    # renormalize
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Smooth with gaussian
    f = interp1d(np.arange(prior_prob.shape[0]), prior_prob)
    xx = np.linspace(0, prior_prob.shape[0] - 1, 1000)
    yy = f(xx)
    window = gaussian(2000, sigma)  # 2000 pts in the window, sigma=5
    smoothed = convolve(yy, window / window.sum(), mode='same')
    fout = interp1d(xx, smoothed)
    prior_prob_smoothed = np.array([fout(i) for i in range(prior_prob.shape[0])])
    prior_prob_smoothed = prior_prob_smoothed / np.sum(prior_prob_smoothed)

    # Save
    if prior_prob_smoothed_path is not None:
        save_dir = os.path.dirname(prior_prob_smoothed_path)
        if save_dir != ""  and os.path.exists(save_dir) == False: os.makedirs(save_dir)
        np.save(prior_prob_smoothed_path, prior_prob_smoothed)
    # if

    if do_plot:
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.plot(prior_prob, label="prior_prob")
        plt.plot(prior_prob_smoothed, "g--", label="prior_prob_smoothed")
        plt.yscale("log")
        plt.legend()
        
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
    # if
    
    return prior_prob_smoothed
# compute_prior_prob_smoothed

def compute_prior_factor(prior_prob_path, prior_prob_smoothed_path, prior_prob_factor_path, gamma = 0.5, alpha = 1, do_plot = True, verbose = 1):
    """
    Calculate prior probability factorization
    Reference: https://github.com/foamliu/Colorful-Image-Colorization/blob/master/class_rebal.py
    Usage:
        info = dict(
            prior_prob_path = os.path.join(data_dir, "preprocessing", "DIV2K", "prior_prob_train_div2k.npy"),
            prior_prob_smoothed_path = os.path.join(data_dir, "preprocessing", "DIV2K", "prior_prob_smoothed_train_div2k.npy"),
            prior_prob_factor_path = os.path.join(data_dir, "preprocessing", "DIV2K", "prior_prob_factor_train_div2k.npy"),
            gamma = 0.5,
            alpha = 1,
            do_plot = True,
            verbose = 1,
        )
        locals().update(**info)
        prior_factor = compute_prior_factor(**info)
    """
    if verbose==1: print("\n=== Compute Prior Factor === ")
    prior_prob = np.load(prior_prob_path)
    prior_prob_smoothed = np.load(prior_prob_smoothed_path)

    u = np.ones_like(prior_prob_smoothed)
    u = u / np.sum(1.0 * u)

    prior_factor = (1 - gamma) * prior_prob_smoothed + gamma * u
    prior_factor = np.power(prior_factor, -alpha)

    # renormalize
    prior_factor = prior_factor / (np.sum(prior_factor * prior_prob_smoothed))

    # Save
    if prior_prob_factor_path is not None:
        save_dir = os.path.dirname(prior_prob_factor_path)
        if save_dir != ""  and os.path.exists(save_dir) == False: os.makedirs(save_dir)
        np.save(prior_prob_factor_path, prior_factor)
    # if
    
    if do_plot:
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
        plt.hist(prior_factor)
        plt.xlabel("Prior probability smoothed factor")
        plt.ylabel("Frequency")
        plt.yscale("log")
        
        plt.show()
    # if

    return prior_factor
# def

def cielab_color_space():
    print('SkImage:')
    start = time.time()
    L = [0] * 256 ** 3
    a = [0] * 256 ** 3
    b = [0] * 256 ** 3
    i = 0
    pb = ProgressBar(total=256, prefix='SkImage converting images', suffix='', decimals=3, length=50, fill='=')
    for r in range(256):
        for g in range(256):
            for bb in range(256):
                im = np.array((bb, g, r), np.uint8).reshape(1, 1, 3)
                color.rgb2lab(im)  # transform it to LAB
                L[i] = im[0, 0, 0]
                a[i] = im[0, 0, 1]
                b[i] = im[0, 0, 2]
                i += 1
            # for
        # for
        pb.print_progress_bar(r)
    # for
            
    print("")
    print(min(L), '<=L<=', max(L))
    print(min(a), '<=a<=', max(a))
    print(min(b), '<=b<=', max(b))
    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds.'.format(elapsed))
    ##############################################
    
    print('OpenCV:')
    start = time.time()
    L = [0] * 256 ** 3
    a = [0] * 256 ** 3
    b = [0] * 256 ** 3
    i = 0
    pb = ProgressBar(total=256, prefix='OpenCV converting images', suffix='', decimals=3, length=50, fill='=')
    for r in range(256):
        for g in range(256):
            for bb in range(256):
                im = np.array((bb, g, r), np.uint8).reshape(1, 1, 3)
                cv2.cvtColor(im, cv2.COLOR_BGR2LAB, im)  # transform it to LAB
                L[i] = im[0, 0, 0]
                a[i] = im[0, 0, 1]
                b[i] = im[0, 0, 2]
                i += 1
            # for
        # for
        pb.print_progress_bar(r)
    # for

    print("")
    print(min(L), '<=L<=', max(L))
    print(min(a), '<=a<=', max(a))
    print(min(b), '<=b<=', max(b))
    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds.'.format(elapsed))
# cielab_color_space

def view_db_info(db_root, db_files, db_name):
    df_data = pd.read_hdf(db_files, key = db_name)

    print("Dataset info: ")
    print("+ Image Path: ", db_root)
    print("+ Index Path: ", db_files)
    print("+ Columns: ", df_data.keys())
    print("+ Rows: ", len(df_data))
    info_types = df_data.groupby("type").count().reset_index()[["type", "image"]].values
    print("+ Types: \n", info_types)
    print()
# view_db_info

def compute_prior_prob_export(db_root, db_file, db_name, column_image = "image", column_type = "type", process_types = ["train"], 
                              pts_in_hull_path = "prior_prob_train_div2k.npy",
                              export_prior_prob_path = None, 
                              export_ab_hist_path = None, 
                              is_resize = False, width = 112, height = 112, 
                              do_plot = True, verbose = 1, ):       
    print("\n=== Compute Prior Probability === ")
    df_data        = pd.read_hdf(db_file, key = db_name)
    select_expr    = f'{column_type} in ["%s"]'%('", "'.join(list(process_types)))
    df_select_data = df_data.query(select_expr)
    image_files    = db_root + "/" + df_select_data[column_image].values
    
    if verbose==1: 
        view_db_info(db_root, db_file, db_name)
        print(f'Select_expr: {select_expr}')
        print(f'Rows after select: {len(df_select_data)}')
        print()
        print("Images: ", image_files[0:5], " ... ")
        print()
        print("Caluculate prior probability")
    # if
    
    prior_prob, ab_hist = compute_prior_prob_v1(image_files = image_files, 
                                                pts_in_hull_path = pts_in_hull_path, 
                                                prior_prob_path  = export_prior_prob_path,
                                                ab_hist_path  = export_ab_hist_path,
                                                is_resize = is_resize, 
                                                width = width, height = height, 
                                                do_plot = do_plot)
    
    if verbose==1:
        print()
        print(f'prior_prob: shape={prior_prob.shape}')
        print(prior_prob)
        print()
        print(f'ab_hist: shape={ab_hist.shape}')
        print(ab_hist)
    # if
    
    return prior_prob, ab_hist
# compute_prior_prob_export