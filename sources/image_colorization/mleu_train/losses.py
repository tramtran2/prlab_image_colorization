import cv2 as cv
import tensorflow.keras.backend as K
import numpy as np, os
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
import matplotlib.pyplot as plt

smooth  = 1.
epsilon = 1e-7

def build_categorical_crossentropy_color_loss(prior_factor_path = "prior_factor.npy", nb_q = 313, train_session = None):
    # Load the color prior factor that encourages rare colors
    prior_factor = None
    if prior_factor_path is None:
        prior_factor = np.ones(nb_q).astype(np.float32)
    elif os.path.exists(prior_factor_path) == False:
        assert(f'prior_factor_path: {prior_factor_path} is not exists!')
    else:
        prior_factor = np.load(prior_factor_path).astype(np.float32)
        print("Init build_categorical_crossentropy_color_loss: ")
        if train_session is not None:
            plt.figure(figsize=(6,4))
            plt.plot(prior_factor)
            plt.savefig(os.path.join(train_session["logs_dir"], "loss_balance.png"))
            plt.close()
            print("+ Save figure: ", os.path.join(train_session["logs_dir"], "loss_balance.png"))
        # if
    # if
    q = len(prior_factor) # number of bins
    
    def categorical_crossentropy_color(y_true, y_pred):
        y_true = K.reshape(y_true, (-1, q))
        y_pred = K.reshape(y_pred, (-1, q))

        idx_max = K.argmax(y_true, axis=1)
        weights = K.gather(prior_factor, idx_max)
        weights = K.reshape(weights, (-1, 1))

        # multiply y_true by weights
        y_true = y_true * weights

        # cross_ent = K.categorical_crossentropy(y_pred, y_true)
        cross_ent = K.categorical_crossentropy(y_true, y_pred)
        cross_ent = K.mean(cross_ent, axis=-1)

        return cross_ent
    # wrapper
    
    return categorical_crossentropy_color
# build_categorical_crossentropy_color_loss

def soft_dice_loss(y_true, y_pred): 
    """ 
    https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - K.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch
# soft_dice_loss

def soft_dice(y_true, y_pred): 
    return 1 - soft_dice_loss(y_true, y_pred)
# soft_dice_loss

def dice_coef_multi_acc(y_true, y_pred):
    y_true_f = K.flatten(y_true[..., 1])
    y_pred_f = K.flatten(y_pred[..., 1])
    
    y_true_sum = K.sum(K.cast(y_true_f > epsilon, dtype="float32"))
    y_pred_sum = K.sum(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
# dice_coef_multi_acc

def dice_coef_multi(y_true, y_pred):
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    
    y_true_sum = K.sum(K.cast(y_true_f > epsilon, dtype="float32"))
    y_pred_sum = K.sum(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
# dice_coef_multi

def dice_coef_multi_loss(y_true, y_pred):
    return 1.0 - dice_coef_multi(y_true, y_pred)
# dice_coef_multi_loss


def dice_coef_float(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    y_true_sum = K.sum(K.cast(y_true_f > epsilon, dtype="float32"))
    y_pred_sum = K.sum(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
# dice_coef_float

def dice_coef_float_loss(y_true, y_pred):
    return 1.0 - dice_coef_float(y_true, y_pred)
# dice_coef_float_loss

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# dice_coef

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
# dice_coef_loss

def bce_dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return binary_crossentropy(y_true_f, y_pred_f) + dice_coef_loss(y_true_f, y_pred_f)
# bce_dice_loss

def bce_logdice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return binary_crossentropy(y_true_f, y_pred_f) - K.log(1. - dice_coef_loss(y_true_f, y_pred_f))
# bce_logdice_loss

def weighted_bce_loss(y_true, y_pred, weight):    
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) + 
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)
# weighted_bce_loss

def weighted_dice_loss(y_true, y_pred, weight):
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss
# weighted_dice_loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss
# weighted_bce_dice_loss
