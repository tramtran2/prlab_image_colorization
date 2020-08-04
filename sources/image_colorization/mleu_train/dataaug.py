from prlab.data_augment.albumentations_keras_tf import KerasDataAugment

import numpy as np, cv2

from albumentations import (
    RandomBrightnessContrast, LongestMaxSize, RandomScale, Rotate, 
    SmallestMaxSize, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ToFloat, 
    Resize, Normalize, Rotate, RandomCrop, Crop, CenterCrop, DualTransform, PadIfNeeded, RandomCrop, 
    IAAFliplr, IAASuperpixels, VerticalFlip, RandomGamma, ElasticTransform, ImageOnlyTransform
)

from prlab.data_augment.albumentations import (
    RandomResizedCrop, HorizontalShear, MaskThreshold, BrightnessShift, BrightnessMultiply, DoGama, ShiftScaleRotateHeng, ElasticTransformHeng
)

dict_keras_colorization_data_augmentation = dict(
    shear_range        = 0.2,
    zoom_range         = 0.2,
    rotation_range     = 30,
    horizontal_flip    = True,
    
    #featurewise_center = True,  # set input mean to 0 over the dataset
    #samplewise_center  = True,  # set each sample mean to 0
    
    featurewise_std_normalization = False,  # divide inputs by std of the dataset
    samplewise_std_normalization  = False,  # divide each input by its std
    zca_whitening                 = False,  # apply ZCA whitening
    
    fill_mode                     = 'nearest',
    cval                          = 0,
    
    # rotation_range     = 40,   # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range  = 0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.2,  # randomly shift images vertically (fraction of total height)
    # vertical_flip      = True
)

def colorized_train_aug(image_size, p=1.0):
    return Compose([
        SmallestMaxSize(max_size = image_size + 50, interpolation = cv2.INTER_CUBIC, p=1),
        Rotate(limit=15, p = 0.5),
        RandomScale(scale_limit=0.1, p = 0.5),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p = 0.5),
        HorizontalFlip(p=0.5),    
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.5),
        HueSaturationValue(p=0.3),
        RandomCrop(image_size, image_size, p=1),
        PadIfNeeded(min_height=image_size, min_width=image_size, p=1)
    ], p=p)
# colorized_train_aug

def colorized_train_aug_soft(image_size, p=1.0):
    return Compose([
        SmallestMaxSize(max_size = image_size + 50, interpolation = cv2.INTER_CUBIC, p=1),
        Rotate(limit=15, p = 0.5),
        RandomScale(scale_limit=0.1, p = 0.5),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p = 0.5),
        HorizontalFlip(p=0.5),    
        # OneOf([
        #     CLAHE(clip_limit=2),
        #     IAASharpen(),
        #     IAAEmboss(),
        #     RandomContrast(),
        #     RandomBrightness(),
        # ], p=0.5),
        # HueSaturationValue(p=0.3),
        RandomCrop(image_size, image_size, p=1),
        PadIfNeeded(min_height=image_size, min_width=image_size, p=1)
    ], p=p)
# colorized_train_aug

def colorized_valid_aug(image_size, p=1.0):
    return Compose([
        Resize(image_size, image_size, p = 1),
        PadIfNeeded(min_height=image_size, min_width=image_size, border_mode = cv2.BORDER_CONSTANT, value = (0, 0, 0), p=1)
    ], p=p)
# colorized_valid_aug


""" 
PREPROCESSING INPUT
"""
def normal_preprocessing_fn(x):
    x = x / 255.0
    return x
# normal_preprocessing_fn

def normal_postprocessing_fn(x):
    x = x * 255.0
    return x
# normal_postprocessing_fn

def tf_normal_preprocessing_rgb_fn(x): # RGB
    x = x / 127.5
    x = x - 1.0
    return x
# normal_preprocessing_fn

def tf_normal_postprocessing_rgb_fn(x): # RGB
    x  = x + 1.0
    x  = x * 127.5
    return x
# normal_postprocessing_fn