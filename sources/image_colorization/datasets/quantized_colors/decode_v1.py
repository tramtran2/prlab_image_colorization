"""
v1 vs v:
+ decode ab truoc
+ phong lon bang x_bact
+ moi ghep vao
--> gia su y_image co dang (h, w, nb_q)
--> gia su y_bact co dang  (batch, h, w, nb_q)
"""
import numpy as np, cv2, os
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni
from .utils import read_image
import tqdm

__all__ = ["decode_soft_encoding_list_rgb", "decode_soft_encoding_batch_rgb", 
           "decode_soft_encoding_image_rgb", "decode_soft_encoding_batch_ab"]

def decode_soft_encoding_list_rgb(list_x_image, y_batch, q_ab, epsilon = 1e-8, T = 0.38, x_post_fn = None, version = 0, usegpu = False, verbose = 1, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_batch: list of L channel (height, width, 1), (height, width) --> gray  or list of string
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """    
    list_image_rgb = []
    tqdm_list_x_image = tqdm.tqdm(enumerate(list_x_image), total=len(list_x_image)) if verbose==1 else enumerate(list_x_image)
    for idx, x_image in tqdm_list_x_image:
        x_L_image = None
        if type(x_image) is str or type(x_image) is np.str_:
            result = read_image(x_image, is_resize = False)
            x_L_image = result["org_img_Lab"][..., 0] # L channel
        elif type(x_image) is np.ndarray:
            if len(x_image.shape)==2:
                x_image = x_image.reshape(x_image.shape + (1,))
                x_L_image = x_image[..., 0]
            elif len(x_image.shape)==3:
                x_L_image = x_image[..., 0]
            # if
            pass
        # if
        image_rgb = decode_soft_encoding_image_rgb(x_L_image, y_batch[idx, ...], q_ab, epsilon = epsilon, T = T, 
                                                   x_post_fn = x_post_fn, version=version, usegpu=usegpu,**kwargs)
        list_image_rgb.append(image_rgb)                                                   
    # for
    return list_image_rgb
# decode_soft_encoding_list_rgb

def decode_soft_encoding_batch_rgb(x_batch, y_batch, q_ab, epsilon = 1e-8, T = 0.38, x_post_fn = None, version = 0, usegpu = False, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_batch: L channel (batchsize, height, width, 1), (height, width) --> gray 
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """    
    assert len(x_batch.shape)==3 or (len(x_batch.shape)==4 and x_batch.shape[3]==1), "x_batch is not valid!"
    if len(x_batch.shape) == 3: x_batch = x_batch.reshape(x_batch.shape[0:3] + (1,))

    if x_post_fn is not None: x_batch = x_post_fn(x_batch)
    
    y_batch_ab = decode_soft_encoding_batch_ab(y_batch, q_ab, epsilon = epsilon, T = T, 
                                               image_size = (x_batch.shape[2],x_batch.shape[1]), # width, height
                                               version = version, usegpu = usegpu, **kwargs)

    batch_Lab  = np.concatenate([x_batch, y_batch_ab], axis = 3)
    batch_rgb = np.array([cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_Lab2RGB) for image in batch_Lab])
    return batch_rgb
    pass
# decode_soft_encoding_batch_rgb

def decode_soft_encoding_image_rgb(x_image, y_image, q_ab, epsilon = 1e-8, T = 0.38, x_post_fn = None, version = 0, usegpu = False, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_image: L channel (height, width, 1), (height, width) --> gray 
    y_image: softencoding of ab channel (height, width, nb_q)
    """    
    assert len(x_image.shape)==2 or (len(x_image.shape)==3 and x_image.shape[2]==1), "x_image is not valid!"

    if len(x_image.shape) == 2: x_image = x_image.reshape(x_image.shape[0:2] + (1,))
    if x_post_fn is not None: x_image = x_post_fn(x_image)
    
    y_image_ab = decode_soft_encoding_image_ab(y_image, q_ab, epsilon = epsilon, T = T, 
                                               image_size = (x_image.shape[1],x_image.shape[0]), # width, height
                                               version = version, usegpu = usegpu, **kwargs)
    image_Lab  = np.concatenate([np.uint8(x_image), y_image_ab], axis = 2)
    image_rgb  = cv2.cvtColor(image_Lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    return image_rgb
# decode_soft_encoding_image_rgb

def decode_soft_encoding_image_ab(y_image, q_ab, epsilon = 1e-8, T = 0.38, image_size = None, version = 0, usegpu = False, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    image_size = (width, height)
    y_image: softencoding of ab channel (height, width, nb_q)
    """    
    y_batch    = y_image.reshape((1,) + y_image.shape[0:3])
    y_batch_ab = decode_soft_encoding_batch_ab(y_batch, q_ab, 
                                               epsilon = epsilon, T = T, image_size = image_size,
                                               version = version, usegpu = usegpu, **kwargs)

    return y_batch_ab.reshape(y_batch_ab.shape[1:])
# decode_soft_encoding_image

def decode_soft_encoding_batch_ab(y_batch, q_ab, epsilon = 1e-8, T = 0.38, image_size = None, version = 0, usegpu = False, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    image_size = (width, height)
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """    
    if version==0:
        y_batch_ab = decode_soft_encoding_batch_ab_v0(y_batch, q_ab, **kwargs)
    elif version==1:
        if usegpu==False:
            y_batch_ab = decode_soft_encoding_batch_ab_v1(y_batch, q_ab, epsilon=epsilon, T = T, **kwargs)
        else:
            y_batch_ab = tf2_decode_soft_encoding_batch_ab_v1(y_batch, q_ab, epsilon=epsilon, T = T, **kwargs)
        # if
    # if
    
    if image_size is not None:
        y_h, y_w = y_batch.shape[1:3]
        n_w, n_h = image_size
        y_batch_ab = sni.zoom(y_batch_ab, [1, 1.*n_h/y_h, 1.*n_w/y_w, 1])
    # if

    return y_batch_ab
# decode_soft_encoding_batch_ab

def tf2_decode_soft_encoding_batch_ab_v1(y_batch, q_ab, epsilon = 1e-8, T = 0.38, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_batch: L channel (batchsize, height, width, 1) --> gray 
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """    
    import tensorflow as tf

    nb_q = q_ab.shape[0]
    
    assert len(y_batch.shape)==4, "Invalid y_batch shape (batchsize, height, width, nb_q)"
    
    y_batch_size, y_height, y_width, y_nb_q = y_batch.shape
    
    assert y_nb_q == nb_q, "y_nb_q is not equal to q_ab"

    tf_y_batch = tf.convert_to_tensor(y_batch)

    tf_batch_reshape = tf.reshape(tf_y_batch, (-1, nb_q))
    tf_batch_reshape = tf.exp(tf.math.log(tf_batch_reshape + epsilon) / T);
    tf_batch_reshape = tf_batch_reshape / tf.reshape(tf.math.reduce_sum(tf_batch_reshape, axis = 1), (-1, 1))

    q_a = q_ab[:, 0].reshape((1, nb_q))
    q_b = q_ab[:, 1].reshape((1, nb_q))

    tf_batch_a = tf.reshape(tf.math.reduce_sum(tf_batch_reshape * q_a, 1), (y_batch_size, y_height, y_width, 1)) + 128
    tf_batch_b = tf.reshape(tf.math.reduce_sum(tf_batch_reshape * q_b, 1), (y_batch_size, y_height, y_width, 1)) + 128


    tf_batch_ab = tf.cast(tf.concat([tf_batch_a, tf_batch_b], axis = 3), dtype=tf.uint8)
    y_batch_ab  = np.array(tf_batch_ab)
    return y_batch_ab
# tf2_decode_soft_encoding_batch_ab_v1

def decode_soft_encoding_batch_ab_v1(y_batch, q_ab, epsilon = 1e-8, T = 0.38, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_batch: L channel (batchsize, height, width, 1) --> gray 
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """
    # if usegpu == True: 
    #     return tf2_decode_soft_encoding_batch_image_v1(x_batch, y_batch, q_ab, epsilon, T, 
    #                                                    x_post_fn, **kwargs)
    nb_q = q_ab.shape[0]
    
    assert len(y_batch.shape)==4, "Invalid y_batch shape (batchsize, height, width, nb_q)"
    
    y_batch_size, y_height, y_width, y_nb_q = y_batch.shape
    
    assert y_nb_q == nb_q, "y_nb_q is not equal to q_ab"

    y_batch_reshape = y_batch.reshape(-1, nb_q)
    y_batch_reshape = np.exp(np.log(y_batch_reshape + epsilon) / T)
    y_batch_reshape = y_batch_reshape / np.sum(y_batch_reshape, 1)[:, np.newaxis]
    
    q_a = q_ab[:, 0].reshape((1, nb_q))
    q_b = q_ab[:, 1].reshape((1, nb_q))
    
    y_batch_a = np.sum(y_batch_reshape * q_a, 1).reshape((y_batch_size, y_height, y_width, 1)) + 128
    y_batch_b = np.sum(y_batch_reshape * q_b, 1).reshape((y_batch_size, y_height, y_width, 1)) + 128
    
    y_batch_ab= np.concatenate([y_batch_a, y_batch_b], axis = 3)
    return y_batch_ab
# decode_batch_image_v1

def decode_soft_encoding_batch_ab_v0(y_batch, q_ab, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """
    nb_q = q_ab.shape[0]

    assert len(y_batch.shape)==4, "Invalid y_batch shape (batchsize, height, width, nb_q)"
    
    y_batch_size, y_height, y_width, y_nb_q = y_batch.shape

    assert y_nb_q == nb_q, "y_nb_q is not equal to q_ab"

    y_batch_reshape = y_batch.reshape(-1, nb_q)
    y_batch_reshape = q_ab[np.argmax(y_batch_reshape, 1)]
    y_batch_a       = y_batch_reshape[:, 0].reshape((y_batch_size, y_height, y_width, 1)) + 128
    y_batch_b       = y_batch_reshape[:, 1].reshape((y_batch_size, y_height, y_width, 1)) + 128
    y_batch_ab      = np.concatenate([y_batch_a, y_batch_b], axis = 3)

    return y_batch_ab
# decode_soft_encoding_batch_ab_v0
