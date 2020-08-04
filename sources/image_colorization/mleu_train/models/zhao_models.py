import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.utils import plot_model

l2_reg = l2(1e-3)

__all__ = ["zhao_vgg16_normal_build"]

def zhao_vgg16_normal_build(
    input_shape = (256, 256, 1), # output default 64
    kernel = 3,
    n_softencoding_class = 313,
    n_segmentation_class = 183,
    model_name = "m",
):
    input_tensor = Input(shape=input_shape)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(input_tensor)
    x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg, strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg,
               strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_3', kernel_initializer="he_normal",
               strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    
    x1 = UpSampling2D(size=(2, 2))(x)
    x1 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv5_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x1)
    x1 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv5_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x1)
    x1 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv5_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x1)               
    x1 = BatchNormalization()(x1)


    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    
    x2 = UpSampling2D(size=(2, 2))(x)
    x2 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv6_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x2)
    x2 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv6_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x2)
    x2 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv6_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x2)               
    x2 = BatchNormalization()(x2)    

    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    
    x3 = UpSampling2D(size=(2, 2))(x)
    x3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv7_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x3)
    x3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv7_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x3)
    x3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='deconv7_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x3)               
    x3 = BatchNormalization()(x3) 


    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_1',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_2',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_3',
               kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x4 = Concatenate()([x1, x2, x3])
    x4 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv9_1', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x4)
    x4 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv9_2', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x4)
    x4 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv9_3', kernel_initializer="he_normal",
               kernel_regularizer=l2_reg)(x4)               
    x4 = BatchNormalization()(x4) 
    output_seg = Conv2D(n_segmentation_class, (1, 1), activation='softmax', padding='same', name='segmentation')(x4)
    
    output_softencoding = Conv2D(n_softencoding_class, (1, 1), activation='softmax', padding='same', name='softencoding')(x)
    
    outputs = []
    outputs.append(output_softencoding)
    outputs.append(output_seg)

    model = Model(inputs=input_tensor, outputs=outputs, name=model_name)
    return model
# zhang_vgg16_normal_build