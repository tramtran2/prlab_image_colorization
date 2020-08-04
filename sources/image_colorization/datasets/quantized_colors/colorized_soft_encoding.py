"""
Soft-Encoding Utils
+ Convert ab channels to categorical data as Zhang et al. paper
References:
+ https://github.com/foamliu/Colorful-Image-Colorization

"""
import numpy as np
import sklearn.neighbors as nn

class ColorizedSoftEncoding(object):
    """
    Class Convert Channeld ab in Lab to Categorical Data as Zhang et al. paper
    pts_in_hull.npy --> array of pts in colorspaces ab for categorical data
    (shape: (??, 2))
    Usage:
        soft_encoding = ColorizedSoftEncoding(pts_in_hull_path = "pts_in_hull.npy", 
                                              nb_neighbors = 5, sigma_neighbor = 5)
        image_Lab = read_image(...)["org_image_Lab"]
        y = soft_encoding(image_Lab);
    """
    def __init__(self, pts_in_hull_path, nb_neighbors = 5, sigma_neighbor = 5):
        self.pts_in_hull_path = pts_in_hull_path
        self.nb_neighbors = nb_neighbors
        self.sigma_neighbor = sigma_neighbor

        self.q_ab, self.nn_finder = load_nn_finder(self.pts_in_hull_path, self.nb_neighbors)
        self.nb_q = self.q_ab.shape[0]
        pass
    # __init__

    def __call__(self, image_Lab):
        self.input = image_Lab
        self.output = get_soft_encoding(self.input, self.nn_finder, self.nb_q, self.sigma_neighbor)
        return self.output
    # __call__
# ColorizedSoftEncoding

def load_nn_finder(pts_in_hull_path, nb_neighbors = 5):
    # Load the array of quantized ab value
    q_ab = np.load(pts_in_hull_path)
    nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)
    return q_ab, nn_finder
# load_nn_finder
    
def get_soft_encoding(image_Lab, nn_finder, nb_q, sigma_neighbor = 5):
    """
    image_Lab = read_image("...")["res_image_Lab"]
    q_ab, nn_finder = load_nn_finder("pts_in_hull.npy", nb_neighbors = 5)
    y = get_soft_encoding(image_Lab, nn_finder, nb_q = q_ab.shape[0], sigma_neighbor = 5)
    """
    # get and normalize image_ab
    # due to preprocessing weighted with minus 128
    image_ab = image_Lab[:, :, 1:].astype(np.int32) - 128 

    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    ab = np.vstack((a, b)).T
    
    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    
    # Smooth the weights with a gaussian kernel
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    
    # format the target
    y = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts

    y = y.reshape(h, w, nb_q)

    return y
# get_soft_encoding