from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dropout, SpatialDropout2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import Convolution2D, BatchNormalization
from prlab.tf2.layers.instancenormalization import InstanceNormalization

def unet_sub_conv(input_tensor, nfilters, kernel, 
                
                padding = "same", 
                kernel_initializer = "he_normal", 
                kernel_regularizer = None,
                dilation_rate = (1,1), 

                activation = "relu",
                
                first_strides = (1,1), 
                default_strides = (1,1), 
                last_strides  = (1,1),
                
                batchnorm   = False, last_batchnorm = True,
                instantnorm = False, last_instantnorm = False, 
                
                last_dropout=0.0, 

                n_conv_layers = 2,

                name = "unet_sub_conv", 
                ):    
    conv = input_tensor
    for i_conv in range(n_conv_layers):
        cur_strides = first_strides if i_conv == 0 else \
                      last_strides  if i_conv == (n_conv_layers - 1) else default_strides
        conv = Convolution2D(nfilters, (kernel, kernel), 
                            kernel_initializer = kernel_initializer, 
                            kernel_regularizer = kernel_regularizer,
                            padding=padding,
                            strides = cur_strides,
                            dilation_rate = dilation_rate,
                            name = f"{name}_conv2d_{i_conv}")(conv)
        if batchnorm==True: conv = BatchNormalization(name = f"{name}_batchnorm_{i_conv}")(conv)
        if instantnorm==True: conv = InstanceNormalization(name = f"{name}_instantnorm_{i_conv}")(conv)
        conv = Activation(activation, name = f"{name}_activation_{i_conv}")(conv)
    # for
    # print("---")
    if last_batchnorm==True: conv = BatchNormalization(name = f"{name}_batchnorm")(conv)
    if last_instantnorm==True: conv = InstanceNormalization(name = f"{name}_instantnorm")(conv)
    if last_dropout > 0: conv = SpatialDropout2D(dropout, name = f"{name}_dropout")(conv)
    return conv
# unet_sub_conv

def segclas_colorized_unet_v0(
    input_shape = (None, None, 1), # Gray Image with Free Size

    has_regression        = True,     # regression channel a, b
    n_regression_class    = 2,        # 
    drop_out_regression   = 0.0,
    output_regression_mode= "tanh",
    output_regression_name= "regression",

    has_softencoding         = True,       # soft encoding (n bins)
    n_softencoding_class     = 313,        # 
    drop_out_softencoding    = 0.0,
    output_softencoding_mode = "softmax",
    output_softencoding_name = "softencoding",

    has_segmentation         = True,       # semantic segmentation
    n_segmentation_class     = 183,   
    drop_out_segmentation    = 0.0,   
    output_segmentation_mode = "softmax",  # dice-score
    output_segmentation_name = "segmentation",
    
    
    has_classification        = True,  # classification image scence
    n_classification_class    = 365,       
    n_classification_feature  = 1024,
    drop_out_classification   = [0.0, 0.0],
    output_classification_mode= "softmax",
    output_classification_name= "classification",

    has_pooling     = True,
    has_shortcut    = True,

    nfilters = 64,
    nblocks  =  4,
    kernel   =  3,
    kernel_initializer = "he_normal", 
    kernel_regularizer = None,
    dilation_rate      = (1,1),
    activation         = "relu",
    pool_size = (2, 2),

    # 2 * nblocks + 1: 5 * 2 + 1 = 11
    last_batchnorm  =[True, True, True, True, True, True, True, True, True, True, True],
    batchnorm       =[False, False, False, False, False, False, False, False, False, False, False, False, False],

    last_instantnorm=[False, False, False, False, False, False, False, False, False, False, False, False, False],
    instantnorm     =[False, False, False, False, False, False, False, False, False, False, False, False, False],
    last_dropout    =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    model_name       = "segclas-colorized-unet",

    # **kwargs
    ):

    inputs = Input(shape=input_shape)

    idx_layer = 0
    strides = (1,1) if has_pooling==True else (2,2)

    ##############
    # Encoder
    ##############
    x = inputs
    branch_encoder = []
    cur_filters = nfilters
    for i in range(nblocks):        
        x = unet_sub_conv(x, cur_filters, kernel, 
                padding = "same", 
                kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, 
                dilation_rate      = dilation_rate, activation = activation,
                
                first_strides = strides if i > 0 else (1,1), # only first Conv2D at Block k>=1
                default_strides = (1,1), 
                last_strides  = (1,1),
                
                batchnorm   = batchnorm[idx_layer], last_batchnorm = last_batchnorm[idx_layer],
                instantnorm = instantnorm[idx_layer], last_instantnorm = last_instantnorm[idx_layer], 
                
                last_dropout=last_dropout[idx_layer], 
                n_conv_layers = 2,
                
                name = f'{model_name}_encoder_block{idx_layer}', 
        )
        branch_encoder.append(x)
        print(f'Encoder {i}: {x.shape}')
        if has_pooling==True: x = MaxPooling2D(pool_size, name = f'{model_name}_encoder_block{idx_layer}_pooling')(x)
        cur_filters = cur_filters * 2
        idx_layer = idx_layer + 1
    # for
    # #############
    
    ##############
    # Features
    ##############
    # between encoder and decoder
    x = unet_sub_conv(x, cur_filters, kernel, 
        padding = "same", 
        kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, 
        dilation_rate      = dilation_rate, activation = activation,
                
        first_strides = (1, 1), # down-sampling
        last_strides  = strides,
        default_strides = (1,1), 
                
        batchnorm   = batchnorm[idx_layer], last_batchnorm = last_batchnorm[idx_layer],
        instantnorm = instantnorm[idx_layer], last_instantnorm = last_instantnorm[idx_layer], 
                
        last_dropout=last_dropout[idx_layer], 
        n_conv_layers = 2,
                
        name = f'{model_name}_features', 
    )
    print(f"Features: {x.shape}")
    idx_layer = idx_layer + 1

    if has_classification == True:
        output_classification = GlobalAveragePooling2D(name = f'{model_name}_classification_GlobalAveragePooling2D')(x)
        if drop_out_classification[0]>0: 
            output_classification = Dropout(drop_out_classification[0], 
                name = f'{model_name}_classification_dropout_0')(output_classification)
        if n_classification_feature>0  : 
            output_classification = Dense(n_classification_feature, activation='relu', 
                name = f'{model_name}_classification_feature')(output_classification)
        if drop_out_classification[1]>0: 
            output_classification = Dropout(drop_out_classification[1], 
                name = f'{model_name}_classification_dropout_1')(output_classification)
        output_classification = Dense(n_classification_class, activation=output_classification_mode, 
            name = f'{output_classification_name}')(output_classification)
    # if

    # #############
    # Decoder
    # #############
    for i in range(nblocks):
        cur_filters = cur_filters // 2
        
        x = UpSampling2D(pool_size, name = f"{model_name}_decoder_block{idx_layer}_upsampling")(x)

        x = unet_sub_conv(x, cur_filters, kernel,  # 1 conv
                padding = "same", 
                kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, 
                dilation_rate      = dilation_rate, activation = activation,
                
                first_strides = (1, 1), default_strides = (1,1), last_strides  = (1,1),
                
                batchnorm   = batchnorm[idx_layer], last_batchnorm = last_batchnorm[idx_layer],
                instantnorm = instantnorm[idx_layer], last_instantnorm = last_instantnorm[idx_layer], 
                
                last_dropout=last_dropout[idx_layer], 
                n_conv_layers = 1,
                
                name = f'{model_name}_decoder_block{idx_layer}a', 
        )
        if has_shortcut == True: 
            print(f'Decoder {i} : {x.shape}')
            print(f'+ Concat Encoder {nblocks - i - 1}: {branch_encoder[nblocks - i - 1].shape}')
            x = Concatenate(name = f"{model_name}_decoder_block{idx_layer}_concatenate")([x, branch_encoder[nblocks - i - 1]])


        x = unet_sub_conv(x, cur_filters, kernel,  # 1 conv
                padding = "same", 
                kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, 
                dilation_rate      = dilation_rate, activation = activation,
                
                first_strides = (1, 1), default_strides = (1,1), last_strides  = (1,1),
                
                batchnorm   = batchnorm[idx_layer], last_batchnorm = last_batchnorm[idx_layer],
                instantnorm = instantnorm[idx_layer], last_instantnorm = last_instantnorm[idx_layer], 
                
                last_dropout=last_dropout[idx_layer], 
                n_conv_layers = 2,
                
                name = f'{model_name}_decoder_block{idx_layer}b', 
        )
        idx_layer = idx_layer + 1
    # for
    print(f"End : {x.shape}")
    
    if has_regression: 
        x1 = Convolution2D(n_regression_class, (1, 1), padding="valid", name = f'{model_name}_regression_conv2d')(x)
        if drop_out_regression>0: x1 = SpatialDropout2D(drop_out_regression)(x1)
        x1 = Reshape((-1, n_regression_class))(x1)
        output_regression = Activation(output_regression_mode, name = output_regression_name)(x1)
    # if

    if has_softencoding: 
        x1 = Convolution2D(n_softencoding_class, (1, 1), padding="valid", name = f'{model_name}_softencoding_conv2d')(x)
        if drop_out_softencoding>0: x1 = SpatialDropout2D(drop_out_softencoding)(x1)
        x1 = Reshape((-1, n_softencoding_class))(x1)
        output_softencoding = Activation(output_softencoding_mode, name = output_softencoding_name)(x1)
    # if    
    
    if has_segmentation: 
        x1 = Convolution2D(n_segmentation_class, (1, 1), padding="valid", name = f'{model_name}_segmentation_conv2d')(x)
        if drop_out_segmentation>0: x1 = SpatialDropout2D(drop_out_segmentation)(x1)
        x1 = Reshape((-1, n_segmentation_class))(x1)
        output_segmentation = Activation(output_segmentation_mode, name = output_segmentation_name)(x1)
    # if    

    outputs = []
    if has_regression: outputs.append(output_regression)
    if has_softencoding: outputs.append(output_softencoding)
    if has_classification: outputs.append(output_classification)
    if has_segmentation: outputs.append(output_segmentation)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    return model
# segclas_colorized_unet_v0

segclas_colorized_unet_v0_cfg = {}
segclas_colorized_unet_v0_cfg["soft_colorized"] = dict(
    input_shape = (None, None, 1), # Gray Image with Free Size

    has_regression         = False,     # regression channel a, b
    # n_regression_class    = 2,        # 
    # drop_out_regression   = 0.0,
    # output_regression_mode= "tanh",
    # output_regression_name= "regression",

    has_softencoding         = True,       # soft encoding (n bins)
    n_softencoding_class     = 313,        # 
    drop_out_softencoding    = 0.0,
    output_softencoding_mode = "softmax",
    output_softencoding_name = "softencoding",

    has_segmentation         = False,       # semantic segmentation
    # n_segmentation_class     = 184,   
    # drop_out_segmentation    = 0.0,   
    # output_segmentation_mode = "sigmoid",  # dice-score
    # output_segmentation_name = "segmentation",
    
    
    has_classification        = True,  # classification image scence
    # n_classification_class    = 10,       
    # n_classification_feature  = 1024,
    # drop_out_classification   = [0.0, 0.0],
    # output_classification_mode= "softmax",
    # output_classification_name= "classification",

    has_pooling     = False,
    has_shortcut    = True,

    nfilters = 64,
    nblocks  =  4,
    kernel   =  3,
    kernel_initializer = "he_normal", 
    kernel_regularizer = None,
    dilation_rate      = (1,1),
    activation         = "relu",
    pool_size = (2, 2),

    # 2 * nblocks + 1: 5 * 2 + 1 = 11
    last_batchnorm  =[True, True, True, True, True, True, True, True, True, True, True],
    batchnorm       =[False, False, False, False, False, False, False, False, False, False, False, False, False],

    last_instantnorm=[False, False, False, False, False, False, False, False, False, False, False, False, False],
    instantnorm     =[False, False, False, False, False, False, False, False, False, False, False, False, False],
    last_dropout    =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    model_name       = "m",
)

segclas_colorized_unet_v0_cfg["reg_colorized"] = dict(
    input_shape = (None, None, 1), # Gray Image with Free Size

    has_regression         = False,     # regression channel a, b
    n_regression_class     = 2,        # 
    drop_out_regression    = 0.0,
    output_regression_mode = "tanh",
    output_regression_name = "regression",

    # has_softencoding         = True,       # soft encoding (n bins)
    # n_softencoding_class     = 313,        # 
    # drop_out_softencoding    = 0.0,
    # output_softencoding_mode = "softmax",
    # output_softencoding_name = "softencoding",

    has_segmentation         = False,       # semantic segmentation
    # n_segmentation_class     = 184,   
    # drop_out_segmentation    = 0.0,   
    # output_segmentation_mode = "sigmoid",  # dice-score
    # output_segmentation_name = "segmentation",
    
    
    has_classification        = True,  # classification image scence
    # n_classification_class    = 10,       
    # n_classification_feature  = 1024,
    # drop_out_classification   = [0.0, 0.0],
    # output_classification_mode= "softmax",
    # output_classification_name= "classification",

    has_pooling     = False,
    has_shortcut    = True,

    nfilters = 64,
    nblocks  =  4,
    kernel   =  3,
    kernel_initializer = "he_normal", 
    kernel_regularizer = None,
    dilation_rate      = (1,1),
    activation         = "relu",
    pool_size = (2, 2),

    # 2 * nblocks + 1: 5 * 2 + 1 = 11
    last_batchnorm  =[True, True, True, True, True, True, True, True, True, True, True],
    batchnorm       =[False, False, False, False, False, False, False, False, False, False, False, False, False],

    last_instantnorm=[False, False, False, False, False, False, False, False, False, False, False, False, False],
    instantnorm     =[False, False, False, False, False, False, False, False, False, False, False, False, False],
    last_dropout    =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    model_name       = "m",
)

segclas_colorized_unet_v0_cfg["regsoft_colorized"] = dict(
    input_shape = (None, None, 1), # Gray Image with Free Size

    has_regression         = True,     # regression channel a, b
    n_regression_class     = 2,        # 
    drop_out_regression    = 0.0,
    output_regression_mode = "tanh",
    output_regression_name = "regression",

    has_softencoding         = True,       # soft encoding (n bins)
    n_softencoding_class     = 313,        # 
    drop_out_softencoding    = 0.0,
    output_softencoding_mode = "softmax",
    output_softencoding_name = "softencoding",

    has_segmentation         = False,       # semantic segmentation
    # n_segmentation_class     = 184,   
    # drop_out_segmentation    = 0.0,   
    # output_segmentation_mode = "sigmoid",  # dice-score
    # output_segmentation_name = "segmentation",
    
    
    has_classification        = False,  # classification image scence
    # n_classification_class    = 10,       
    # n_classification_feature  = 1024,
    # drop_out_classification   = [0.0, 0.0],
    # output_classification_mode= "softmax",
    # output_classification_name= "classification",

    has_pooling     = False,
    has_shortcut    = True,

    nfilters = 64,
    nblocks  =  4,
    kernel   =  3,
    kernel_initializer = "he_normal", 
    kernel_regularizer = None,
    dilation_rate      = (1,1),
    activation         = "relu",
    pool_size = (2, 2),

    # 2 * nblocks + 1: 5 * 2 + 1 = 11
    last_batchnorm  =[True, True, True, True, True, True, True, True, True, True, True],
    batchnorm       =[False, False, False, False, False, False, False, False, False, False, False, False, False],

    last_instantnorm=[False, False, False, False, False, False, False, False, False, False, False, False, False],
    instantnorm     =[False, False, False, False, False, False, False, False, False, False, False, False, False],
    last_dropout    =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    model_name       = "m",
)

segclas_colorized_unet_v0_cfg["clasregsoft_colorized"] = dict(
    input_shape = (None, None, 1), # Gray Image with Free Size

    has_regression         = True,     # regression channel a, b
    n_regression_class     = 2,        # 
    drop_out_regression    = 0.0,
    output_regression_mode = "tanh",
    output_regression_name = "regression",

    has_softencoding         = True,       # soft encoding (n bins)
    n_softencoding_class     = 313,        # 
    drop_out_softencoding    = 0.0,
    output_softencoding_mode = "softmax",
    output_softencoding_name = "softencoding",

    has_segmentation         = False,       # semantic segmentation
    # n_segmentation_class     = 184,   
    # drop_out_segmentation    = 0.0,   
    # output_segmentation_mode = "sigmoid",  # dice-score
    # output_segmentation_name = "segmentation",
    
    
    has_classification        = False,  # classification image scence
    n_classification_class    = 365,       
    n_classification_feature  = 1024,
    drop_out_classification   = [0.0, 0.0],
    output_classification_mode= "softmax",
    output_classification_name= "classification",

    has_pooling     = False,
    has_shortcut    = True,

    nfilters = 64,
    nblocks  =  4,
    kernel   =  3,
    kernel_initializer = "he_normal", 
    kernel_regularizer = None,
    dilation_rate      = (1,1),
    activation         = "relu",
    pool_size = (2, 2),

    # 2 * nblocks + 1: 5 * 2 + 1 = 11
    last_batchnorm  =[True, True, True, True, True, True, True, True, True, True, True],
    batchnorm       =[False, False, False, False, False, False, False, False, False, False, False, False, False],

    last_instantnorm=[False, False, False, False, False, False, False, False, False, False, False, False, False],
    instantnorm     =[False, False, False, False, False, False, False, False, False, False, False, False, False],
    last_dropout    =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    model_name       = "m",
)

segclas_colorized_unet_v0_cfg["segclasregsoft_colorized"] = dict(
    input_shape = (None, None, 1), # Gray Image with Free Size

    has_regression         = True,     # regression channel a, b
    n_regression_class     = 2,        # 
    drop_out_regression    = 0.0,
    output_regression_mode = "tanh",
    output_regression_name = "regression",

    has_softencoding         = True,       # soft encoding (n bins)
    n_softencoding_class     = 313,        # 
    drop_out_softencoding    = 0.0,
    output_softencoding_mode = "softmax",
    output_softencoding_name = "softencoding",

    has_segmentation         = False,       # semantic segmentation
    n_segmentation_class     = 183,   
    drop_out_segmentation    = 0.0,   
    output_segmentation_mode = "softmax",  # dice-score
    output_segmentation_name = "segmentation",
    
    
    has_classification        = False,  # classification image scence
    n_classification_class    = 365,       
    n_classification_feature  = 1024,
    drop_out_classification   = [0.0, 0.0],
    output_classification_mode= "softmax",
    output_classification_name= "classification",

    has_pooling     = False,
    has_shortcut    = True,

    nfilters = 64,
    nblocks  =  4,
    kernel   =  3,
    kernel_initializer = "he_normal", 
    kernel_regularizer = None,
    dilation_rate      = (1,1),
    activation         = "relu",
    pool_size = (2, 2),

    # 2 * nblocks + 1: 5 * 2 + 1 = 11
    last_batchnorm  =[True, True, True, True, True, True, True, True, True, True, True],
    batchnorm       =[False, False, False, False, False, False, False, False, False, False, False, False, False],

    last_instantnorm=[False, False, False, False, False, False, False, False, False, False, False, False, False],
    instantnorm     =[False, False, False, False, False, False, False, False, False, False, False, False, False],
    last_dropout    =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    model_name       = "m",
)