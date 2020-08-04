import numpy as np, cv2, os
import matplotlib.pyplot as plt

def decode_soft_encoding_image_v1(x_image, y_image, q_ab, epsilon = 1e-8, T = 0.38, x_post_fn = None, usegpu = False, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_image: L channel (batchsize, height, width, 1) --> gray 
    y_image: softencoding of ab channel (height, width, nb_q)
    """    
    if len(x_image.shape) == 2:
        x_batch_image = x_image.reshape((1,) + x_image.shape[0:2] + (1,))
    elif len(x_image.shape) == 3 and x_image.shape[2]==1:
        x_batch_image = x_image.reshape((1,) + x_image.shape[0:3])
    else: 
        assert("Invalid Gray Image with shape (w, h) or (w, h, 1)")
    # if
    y_batch_image = y_image.reshape((1,) + y_image.shape[0:3])
    y_batch_RGB = decode_soft_encoding_batch_image_v1(x_batch_image, y_batch_image, q_ab, epsilon, T, x_post_fn, usegpu = usegpu)
    return y_batch_RGB.reshape(y_batch_RGB.shape[1:])
# decode_image_v1

def decode_soft_encoding_image_v0(x_image, y_image, q_ab, x_post_fn = None, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_image: L channel (batchsize, height, width, 1) --> gray 
    y_image: softencoding of ab channel (height, width, nb_q)
    """    
    if len(x_image.shape) == 2:
        x_batch_image = x_image.reshape((1,) + x_image.shape[0:2] + (1,))
    elif len(x_image.shape) == 3 and x_image.shape[2]==1:
        x_batch_image = x_image.reshape((1,) + x_image.shape[0:3])
    else: 
        assert("Invalid Gray Image with shape (w, h) or (w, h, 1)")
    # if
    y_batch_image = y_image.reshape((1,) + y_image.shape[0:3])
    y_batch_RGB = decode_soft_encoding_batch_image_v0(x_batch_image, y_batch_image, q_ab, x_post_fn)
    return y_batch_RGB.reshape(y_batch_RGB.shape[1:])
# decode_image_v0

def decode_soft_encoding_batch_image_v1(x_batch, y_batch, q_ab, epsilon = 1e-8, T = 0.38, 
                                        x_post_fn = None, usegpu = False, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_batch: L channel (batchsize, height, width, 1) --> gray 
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """
    if usegpu == True: 
        return tf2_decode_soft_encoding_batch_image_v1(x_batch, y_batch, q_ab, epsilon, T, 
                                                       x_post_fn, **kwargs)
    nb_q = q_ab.shape[0]
    
    batch_size, height, width = x_batch.shape[0:3]

    y_batch_reshape = y_batch.reshape(batch_size * height * width, nb_q)
    
    y_batch_reshape = np.exp(np.log(y_batch_reshape + epsilon) / T)
    y_batch_reshape = y_batch_reshape / np.sum(y_batch_reshape, 1)[:, np.newaxis]
    
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))
    
    y_batch_a = np.sum(y_batch_reshape * q_a, 1).reshape((batch_size, height, width, 1)) + 128
    y_batch_b = np.sum(y_batch_reshape * q_b, 1).reshape((batch_size, height, width, 1)) + 128

    x_batch = x_post_fn(x_batch) if x_post_fn is not None else x_batch
    
    y_batch_Lab = np.concatenate([np.uint8(x_batch), y_batch_a, y_batch_b], axis = 3).reshape(batch_size, height, width, 3)
    y_batch_RGB = np.array([cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_Lab2BGR)[..., ::-1] for image in y_batch_Lab])
    
    return y_batch_RGB
# decode_batch_image_v1

def decode_soft_encoding_batch_image_v0(x_batch, y_batch, q_ab, x_post_fn = None, **kwargs):
    """
    q_ab = np.load(os.path.join(module_dir, "data", "pts_in_hull.npy").replace("\\", "/"))
    x_batch: L channel (batchsize, height, width, 1) --> gray
    y_batch: softencoding of ab channel (batchsize, height, width, nb_q)
    """
    nb_q = q_ab.shape[0]
    
    batch_size, height, width = x_batch.shape[0:3]

    y_batch_reshape = y_batch.reshape(batch_size * height * width, nb_q)
    y_batch_reshape = q_ab[np.argmax(y_batch_reshape, 1)]
    y_batch_a       = y_batch_reshape[:, 0].reshape((batch_size, height, width, 1)) + 128
    y_batch_b       = y_batch_reshape[:, 1].reshape((batch_size, height, width, 1)) + 128

    x_batch = x_post_fn(x_batch) if x_post_fn is not None else x_batch

    y_batch_Lab = np.concatenate([np.uint8(x_batch), y_batch_a, y_batch_b], axis = 3).reshape(batch_size, height, width, 3)
    y_batch_RGB = np.array([cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_Lab2BGR)[..., ::-1] for image in y_batch_Lab])
    
    return y_batch_RGB
# decode_batch_image_v0

def tf2_decode_soft_encoding_batch_image_v1(x_batch, y_batch, q_ab, epsilon = 1e-8, T = 0.38, x_post_fn = None, **kwargs):
    import tensorflow as tf

    nb_q = q_ab.shape[0]
    batch_size, height, width = x_batch.shape[0:3]
    
    # tf_x_batch = tf.compat.v1.placeholder(tf.float32, shape=(None, ) + x_batch.shape[1:])
    # tf_y_batch = tf.compat.v1.placeholder(tf.float32, shape=(None, ) + y_batch.shape[1:])
    tf_x_batch = tf.convert_to_tensor(x_post_fn(x_batch))
    tf_y_batch = tf.convert_to_tensor(y_batch)

    tf_batch_reshape = tf.reshape(tf_y_batch, (-1, nb_q))

    tf_batch_reshape = tf.exp(tf.math.log(tf_batch_reshape + epsilon) / T);
    tf_batch_reshape = tf_batch_reshape / tf.reshape(tf.math.reduce_sum(tf_batch_reshape, axis = 1), (-1, 1))

    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    tf_batch_a = tf.reshape(tf.math.reduce_sum(tf_batch_reshape * q_a, 1), x_batch.shape) + 128
    tf_batch_b = tf.reshape(tf.math.reduce_sum(tf_batch_reshape * q_b, 1), x_batch.shape) + 128


    tf_batch_Lab = tf.cast(tf.concat([tf_x_batch, tf_batch_a, tf_batch_b], axis = 3), dtype=tf.uint8)
    
    # y_batch_Lab = sess.run(tf_batch_Lab, feed_dict={tf_x_batch:x_post_fn(x_batch), tf_y_batch:y_batch})
    y_batch_Lab = np.array(tf_batch_Lab)
    y_batch_RGB = np.array([cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_Lab2BGR)[..., ::-1] for image in y_batch_Lab])
    
    return y_batch_RGB
# tf2_decode_soft_encoding_batch_image_v1