import numpy as np, cv2

def decode_regression_image(x_image, y_image, x_post_fn = None, y_post_fn = None):
    if len(x_image.shape) == 2:
        x_batch_image = x_image.reshape((1,) + x_image.shape[0:2] + (1,))
    elif len(x_image.shape) == 3 and x_image.shape[2]==1:
        x_batch_image = x_image.reshape((1,) + x_image.shape[0:3])
    else: 
        assert("Invalid Gray Image with shape (w, h) or (w, h, 1)")
    # if
    y_batch_image = y_image.reshape((1,) + y_image.shape)
    y_batch_RGB = decode_regression_batch_image(x_batch_image, y_batch_image, x_post_fn, y_post_fn)
    
    return y_batch_RGB.reshape(y_batch_RGB.shape[1:])
# decode_regression_image

def decode_regression_batch_image(x_batch, y_batch, x_post_fn = None, y_post_fn = None, **kwargs):
    """
    x_batch: L or gray  (batch_size, height, width, 1)
    y_batch: ab channel (batch_size, height, width, 2) 
    
    x_post_fn: decode function of x_batch
    y_post_fn: decode function of y_batch
    """
    batch_size, height, width = x_batch.shape[0:3]
    y_batch = y_batch.reshape(-1, height, width, 2)
    
    x_batch = x_post_fn(x_batch) if x_post_fn is not None else x_batch
    y_batch = y_post_fn(y_batch) if y_post_fn is not None else y_batch

    y_batch_L  = np.uint8(x_batch)
    y_batch_ab = np.uint8(y_batch)
    
    y_batch_Lab = np.concatenate([y_batch_L, y_batch_ab], axis = 3).reshape(batch_size, height, width, 3)
    y_batch_RGB = np.array([cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_Lab2BGR)[..., ::-1] for image in y_batch_Lab])

    return y_batch_RGB
# decode_regression_batch_image

