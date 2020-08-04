import numpy as np, cv2
import scipy.ndimage.interpolation as sni
from . import read_image
import tqdm

__all__ = ["decode_regression_list_image", "decode_regression_image", "decode_regression_batch_image"]

def decode_regression_list_image(list_x_image, y_batch, x_post_fn = None, y_post_fn = None, verbose = 1, **kwargs):
    """
    x_image: list str to image or (batchsize, height, width, 1) or (batchsize, height, width)
    y_batch: ab channel (batchsize, height, width, 2) 
    """
    list_image_rgb = []
    tqdm_list_x_image = tqdm.tqdm(enumerate(list_x_image), total=len(list_x_image)) if verbose==1 else enumerate(list_x_image)
    for idx, x_image in tqdm_list_x_image:
        image_rgb = decode_regression_image(x_image, y_batch[idx, ...], x_post_fn=x_post_fn, y_post_fn=y_post_fn)
        list_image_rgb.append(image_rgb)
    # for
    return list_image_rgb
# decode_regression_list_image

def decode_regression_image(x_image, y_image, x_post_fn = None, y_post_fn = None, **kwargs):
    """
    x_image: str to image or (height, width, 1) or (height, width)
    y_image: ab channel (height, width, 2) 
    """
    x_L_image = None # (height, width, 1)
    if type(x_image) is str or type(x_image) is np.str_:
        result = read_image(x_image, is_resize = False)
        x_L_image = result["org_img_Lab"][..., 0] # L channel
        x_L_image = x_L_image.reshape(x_L_image.shape + (1,))
    elif type(x_image) is np.ndarray:
        if len(x_image.shape)==2:
            x_image = x_image.reshape(x_image.shape + (1,))
            x_L_image = x_image[..., 0]
        elif len(x_image.shape)==3:
            x_L_image = x_image[..., 0]
        # if
        pass
    # if
    assert x_L_image is not None, "x_image is not valid!"
    assert len(y_image.shape)==3 and y_image.shape[2]==2, "y_image is not valid!"

    x_batch = x_L_image.reshape((1, ) + x_L_image.shape)
    y_batch = y_image.reshape((1, ) + y_image.shape)

    
    y_height, y_width = y_batch.shape[1:3]
    x_height, x_width = x_batch.shape[1:3]
    if x_height != y_height or x_width != y_width:
        y_batch = sni.zoom(y_batch, [1, 1.*x_height/y_height, 1.*x_width/y_width, 1])
    # if

    x_batch = x_post_fn(x_batch) if x_post_fn is not None else x_batch
    y_batch = y_post_fn(y_batch) if y_post_fn is not None else y_batch

    y_batch_Lab = np.concatenate([x_batch, y_batch], axis = 3)
    y_batch_RGB = np.array([cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_Lab2RGB) for image in y_batch_Lab])
    y_batch_RGB = y_batch_RGB.reshape(y_batch_RGB.shape[1:])
    return y_batch_RGB
# decode_regression_image

def decode_regression_batch_image(x_batch, y_batch, x_post_fn = None, y_post_fn = None, **kwargs):
    """
    x_batch: L or gray  (batch_size, height, width, 1)
    y_batch: ab channel (batch_size, height, width, 2) 
    
    x_post_fn: decode function of x_batch
    y_post_fn: decode function of y_batch
    """
    assert len(y_batch.shape)==4 and y_batch.shape[3]==2, "Invalid y_batch shape (batchsize, height, width, 2)"
    assert len(x_batch.shape)==3 and x_batch.shape[3]==1, "Invalid y_batch shape (batchsize, height, width, 1)"

    y_height, y_width = y_batch.shape[1:3]
    x_height, x_width = x_batch.shape[1:3]
    if x_height != y_height or x_width != y_width:
        y_batch = sni.zoom(y_batch, [1, 1.*x_height/y_height, 1.*x_width/y_width, 1])
    # if

    x_batch = x_post_fn(x_batch) if x_post_fn is not None else x_batch
    y_batch = y_post_fn(y_batch) if y_post_fn is not None else y_batch

    y_batch_Lab = np.concatenate([y_batch_L, y_batch_ab], axis = 3)
    y_batch_RGB = np.array([cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_Lab2RGB) for image in y_batch_Lab])

    return y_batch_RGB
# decode_regression_batch_image
